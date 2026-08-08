"""
vfc: Compute a Vector Field Convolution (VFC) field from a NIfTI image,
or run VFC-based image registration.

**Field-export mode** (``-i`` / ``--input``)::

    vfc -i image.nii.gz -o vfc_field.nii.gz --vfc-radius 15 --vfc-beta 2.0

    Output is a 4-D NIfTI (D×H×W×3) whose vectors point inward toward edge
    features.  Pass ``--for-slicer`` for a 5-D Slicer-compatible displacement
    vector field (intent_code=1006).

**Registration mode** (``-f`` / ``--fixed`` + ``-m`` / ``--moving``)::

    vfc -f fixed.nii.gz -m moving.nii.gz \\
        -o moving_warped.nii.gz -w warpfield.nii.gz

    Runs the pTVreg optimiser with the VFC similarity metric (equivalent to
    ``ptvreg --metric vfc``) using an automatic multi-scale pyramid
    (n_levels=None, MATLAB-style auto-depth).

References
----------
Li, B. & Acton, S. T. (2007). Active contour external force using vector
field convolution for image segmentation. IEEE TIP 16(8), 2096–2106.
"""

import argparse
import sys
import json
import numpy as np
import torch
import nibabel as nib
import scipy.ndimage as ndimage

from pTVreg.metrics import _vfc_build_kernel, _vfc_fft_field, _vfc_image_edge_map
from pTVreg.registration import PTVRegistration
from pTVreg.utils import save_nifti
from pTVreg.deformation import warp_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_edge(edge: torch.Tensor) -> torch.Tensor:
    """Percentile-clip and normalise an edge map to [0, 1]."""
    flat = edge.flatten()
    n = flat.numel()
    k = max(1, min(n, int(round(0.999 * (n - 1))) + 1))
    e_max = torch.kthvalue(flat, k).values.clamp(min=1e-8)
    return (edge / e_max).clamp(0.0, 1.0)


def _to_slicer_dvf(
    data: np.ndarray,
    affine: np.ndarray,
    flip_xy: bool = False,
) -> nib.Nifti1Image:
    """
    Package a (X, Y, Z, 3) VFC field (vectors in voxel-axis space) as a NIfTI
    that 3D Slicer / ITK recognises as a displacement vector field.

    VFC component k is expressed along the k-th voxel axis direction. Slicer
    expects displacement vectors in **LPS physical-space**. The conversion is:

        v_lps = diag([-1, -1, 1]) @ R_unit @ v_vox

    where R_unit[:,k] = unit direction vector of voxel axis k in RAS world
    (= column k of the affine, normalised to unit length).

    For a standard axial scan whose affine diagonal starts with a negative entry
    (axis 0 = –R = +L direction), the rotation and the RAS→LPS flip cancel on
    that axis, leaving it unchanged. For an identity affine the net effect is
    diag([-1,-1,1]).  The formula handles arbitrary orientations correctly.

    Slicer also requires:
      - shape  (X, Y, Z, 1, 3)   — 5-D with a singleton *t* dimension
      - intent_code 1006          — NIFTI_INTENT_DISPVECT
      - dtype  float32

    Parameters
    ----------
    data     : (X, Y, Z, 3) float32 VFC field (voxel-axis components).
    affine   : 4×4 NIfTI affine of the source image.
    flip_xy  : Apply an additional X/Y negation after the main conversion
               (escape hatch for non-standard situations).
    """
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f"Expected (X,Y,Z,3) array, got {data.shape}")

    # Direction cosines: column k = unit direction of voxel axis k in RAS world.
    col_norms = np.linalg.norm(affine[:3, :3], axis=0, keepdims=True).clip(min=1e-8)
    R_unit = (affine[:3, :3] / col_norms).astype(np.float64)   # (3, 3)

    # Combined transform: voxel-axis space → LPS world space.
    lps_from_ras = np.diag([-1., -1., 1.])
    M = (lps_from_ras @ R_unit).astype(np.float32)             # (3, 3)

    # Apply M to every voxel vector in one einsum: (X,Y,Z,3) → (X,Y,Z,3)
    out = np.einsum('ij,...j->...i', M, data.astype(np.float32))

    if flip_xy:
        out[..., 0] *= -1
        out[..., 1] *= -1

    out = out[..., np.newaxis, :]  # (X, Y, Z, 1, 3)

    img = nib.Nifti1Image(out, affine)
    img.header.set_data_dtype(np.float32)
    img.header["intent_code"] = 1006   # NIFTI_INTENT_DISPVECT
    img.header["dim"][0] = 5
    img.header["dim"][4] = 1
    img.header["dim"][5] = 3
    return img


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="vfc",
        description=(
            "vfc: compute a VFC field from a NIfTI image (field-export mode, -i), "
            "or run VFC-based registration (registration mode, -f / -m)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Shared: input images ─────────────────────────────────────────────────
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="[Field-export] Input intensity NIfTI image. Ignored when --edgemap is provided.",
    )
    parser.add_argument(
        "-f", "--fixed",
        default=None,
        help="[Registration] Fixed image path.",
    )
    parser.add_argument(
        "-m", "--moving",
        default=None,
        help="[Registration] Moving image path.",
    )

    # ── Shared: output ───────────────────────────────────────────────────────
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path. Field-export default: ./vfc_field.nii.gz. "
             "Registration default: ./moving_warped.nii.gz.",
    )
    parser.add_argument(
        "-w", "--warp",
        nargs="?",
        const="./warpfield.nii.gz",
        default=None,
        help="[Registration] Output warp field path. "
             "If flag is set without a value, defaults to ./warpfield.nii.gz.",
    )
    parser.add_argument(
        "--edgemap",
        dest="edgemap",
        default=None,
        help="[Field-export] Pre-computed edge-magnitude NIfTI (same geometry as -i). "
             "When supplied the intensity image is not used for edge extraction.",
    )

    # ── Shared: preprocessing ────────────────────────────────────────────────
    parser.add_argument(
        "--clip",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="[Shared] Clip input intensities to [MIN, MAX] (e.g. 80 900 for lung CT).",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.0,
        help="[Shared] Downsample scale factor (e.g. 0.5) to save memory.",
    )

    # ── Shared: VFC kernel ───────────────────────────────────────────────────
    parser.add_argument(
        "--vfc-radius",
        dest="vfc_radius",
        type=float,
        default=15.0,
        help="[Shared] VFC 1/e characteristic length in mm. Weight = exp(-1)\u22480.37 at r=R "
             "regardless of beta. Effective kernel support = R\u00b7(ln1000)^(1/beta): "
             "~2.6R for beta=2 (Gaussian), ~1.6R for beta=4, ~6.9R for beta=1 (capped at 4R).",
    )
    parser.add_argument(
        "--vfc-beta",
        dest="vfc_beta",
        type=float,
        default=2.0,
        help="[Shared] Weibull shape exponent \u03b2 > 0 for VFC attenuation. Controls kernel compactness: "
             "beta=2=Gaussian (support\u22482.6R); beta=1=exponential (support\u22486.9R); "
             "beta=4=super-Gaussian (support\u22481.6R); beta\u21920=long-range tails (capped at 4R).",
    )
    parser.add_argument(
        "--vfc-sign-invariant",
        dest="vfc_sign_invariant",
        action="store_true",
        help="[Shared] Use sign-invariant kernel (cos² alignment; useful for label-boundary edge maps).",
    )

    # ── Field-export mode: output options ────────────────────────────────────
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="[Field-export] L2-normalize each voxel's VFC vector to unit length.",
    )
    parser.add_argument(
        "--zero-threshold",
        dest="zero_threshold",
        type=float,
        default=1e-3,
        help="[Field-export] Zero out voxels whose VFC L2 magnitude is below this threshold. "
             "~50%% of voxels zeroed at default while retaining >99.9999%% of field energy. "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--for-slicer",
        dest="for_slicer",
        action="store_true",
        help="[Field-export] Write a Slicer-compatible DVF NIfTI: "
             "shape (X,Y,Z,1,3) with intent_code=1006 (NIFTI_INTENT_DISPVECT).",
    )
    parser.add_argument(
        "--flip-xy",
        dest="flip_xy",
        action="store_true",
        help="[Field-export] Additional X/Y sign flip on top of the RAS→LPS correction "
             "(escape hatch for non-standard situations).",
    )

    # ── Shared: hardware ─────────────────────────────────────────────────────
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage if available.",
    )

    # ── Registration mode: masks ─────────────────────────────────────────────
    parser.add_argument(
        "--fixed_mask",
        default=None,
        help="[Registration] Fixed image mask NIfTI (metric evaluated where mask > 0).",
    )
    parser.add_argument(
        "--mask_labels",
        type=int,
        nargs="+",
        default=None,
        help="[Registration] Integer labels to extract from a multi-label mask.",
    )
    parser.add_argument(
        "--border_mask",
        type=int,
        default=5,
        help="[Registration] Voxels to ignore at image boundaries.",
    )

    # ── Registration mode: optimiser ─────────────────────────────────────────
    parser.add_argument(
        "--spacing",
        type=int,
        default=8,
        help="[Registration] Grid spacing at finest level, in voxels.",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        nargs="+",
        default=[0.15],
        help="[Registration] Regularization weight, or per-level list (coarse→fine).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[100],
        help="[Registration] Iterations per pyramid level (coarse→fine). "
             "Pyramid depth is auto-computed from image size (n_levels=None).",
    )
    parser.add_argument(
        "--metric_param",
        type=float,
        default=2.1,
        help="[Registration] LCC sigma in mm for the VFC alignment metric.",
    )
    parser.add_argument(
        "--lambda_jac",
        type=float,
        default=0.0,
        help="[Registration] Weight for Jacobian determinant penalty (discourages folding). "
             "Indicative range: 0.5–5.0.",
    )

    # ── Registration mode: VFC-specific ──────────────────────────────────────
    parser.add_argument(
        "--fixed-edgemap",
        dest="fixed_edgemap",
        default=None,
        help="[Registration] Pre-computed edge-magnitude NIfTI for the fixed image.",
    )
    parser.add_argument(
        "--moving-edgemap",
        dest="moving_edgemap",
        default=None,
        help="[Registration] Pre-computed edge-magnitude NIfTI for the moving image.",
    )
    parser.add_argument(
        "--vfc-normalize",
        dest="vfc_normalize",
        action="store_true",
        help="[Registration] L2-normalise VFC fields to unit vectors before the "
             "alignment loss. Makes the metric purely directional.",
    )

    # ── Registration mode: label-guided soft-Dice ────────────────────────────
    parser.add_argument(
        "--fixed-labels",
        dest="fixed_labels",
        default=None,
        help="[Registration] Multi-label segmentation NIfTI for fixed image "
             "(enables soft-Dice guidance on top of VFC).",
    )
    parser.add_argument(
        "--moving-labels",
        dest="moving_labels",
        default=None,
        help="[Registration] Multi-label segmentation NIfTI for moving image.",
    )
    parser.add_argument(
        "--dice-weight",
        dest="dice_weight",
        type=float,
        default=0.5,
        help="[Registration] Weight for the soft-Dice label loss term.",
    )
    parser.add_argument(
        "--dice-labels",
        dest="dice_labels",
        type=int,
        nargs="+",
        default=None,
        help="[Registration] Label IDs to include in the soft-Dice loss. "
             "If not set, all shared non-zero labels are used.",
    )

    # ── JSON config override (same pattern as ptvreg) ────────────────────────
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to JSON configuration file to override/extend CLI arguments.",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # ── JSON config merging (same logic as ptvreg CLI) ────────────────────────
    if args.config:
        print(f"Loading configuration from {args.config}")
        specified_args = set()
        for arg in sys.argv[1:]:
            for action in parser._actions:
                for opt in action.option_strings:
                    if arg == opt or arg.startswith(opt + "="):
                        specified_args.add(action.dest)

        with open(args.config, "r") as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(args, key) and key != "config" and key not in specified_args:
                    setattr(args, key, value)

    # ── Mode detection ────────────────────────────────────────────────────────
    has_input = args.input is not None
    has_fixed_moving = args.fixed is not None and args.moving is not None

    if has_input and has_fixed_moving:
        parser.error("Specify either -i/--input (field-export mode) or -f/-m (registration mode), not both.")
    if not has_input and not has_fixed_moving:
        parser.error("Specify -i/--input for field-export mode, or -f and -m for registration mode.")

    # ── Device selection ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU detected and used by default.")
    else:
        device = "cpu"
        print("GPU not available, using CPU.")
    print(f"Using device: {device}")

    # =========================================================================
    # FIELD-EXPORT MODE
    # =========================================================================
    if has_input:
        if args.output is None:
            args.output = "./vfc_field.nii.gz"

        print(f"Loading image: {args.input}")
        img_nib = nib.load(args.input)
        img_data = img_nib.get_fdata().astype(np.float32)
        pix_res = np.array(img_nib.header.get_zooms()[:3], dtype=np.float32)

        if args.clip:
            cl_min, cl_max = args.clip
            print(f"Clipping intensities to [{cl_min}, {cl_max}]")
            img_data = np.clip(img_data, cl_min, cl_max)

        orig_shape = img_data.shape

        if args.scale_factor != 1.0:
            print(f"Scaling image by {args.scale_factor}…")
            img_data = ndimage.zoom(img_data, args.scale_factor, order=1)
            pix_res = pix_res / max(args.scale_factor, 1e-8)
            print(f"Scaled shape: {img_data.shape}")

        if args.edgemap:
            print(f"Loading pre-computed edgemap: {args.edgemap}")
            edge_np = nib.load(args.edgemap).get_fdata().astype(np.float32)
            if args.scale_factor != 1.0:
                edge_np = ndimage.zoom(edge_np, args.scale_factor, order=1)
            edge_t = torch.tensor(edge_np, device=device)
        else:
            print("Computing gradient-magnitude edge map from input image…")
            img_t = torch.tensor(img_data, device=device)
            pix_res_t = torch.tensor(pix_res, device=device)
            edge_t = _vfc_image_edge_map(img_t, pix_res_t)

        print(
            f"Building VFC kernel  1/e-radius={args.vfc_radius} mm, "
            f"beta={args.vfc_beta}, sign_invariant={args.vfc_sign_invariant}"
        )
        pix_res_t = torch.tensor(pix_res, dtype=torch.float32, device=device)
        kernel = _vfc_build_kernel(
            radius_mm=args.vfc_radius,
            pix_res=pix_res_t,
            beta=args.vfc_beta,
            device=device,
        )
        if args.vfc_sign_invariant:
            kernel = kernel.abs()

        print("Applying VFC (FFT convolution)…")
        edge_norm = _normalise_edge(edge_t)
        with torch.no_grad():
            vfc_field = _vfc_fft_field(edge_norm, kernel)   # (3, D, H, W)

        vfc_np = vfc_field.cpu().float().numpy()             # (3, D, H, W)

        if args.scale_factor != 1.0:
            print(f"Upsampling VFC field back to original shape {orig_shape}…")
            zoom_factors = tuple(orig_shape[i] / vfc_np.shape[i + 1] for i in range(3))
            vfc_up = np.stack(
                [ndimage.zoom(vfc_np[c], zoom_factors, order=1) for c in range(3)],
                axis=0,
            )
            vfc_up = vfc_up / max(args.scale_factor, 1e-8)
            vfc_np = vfc_up

        vfc_np = np.moveaxis(vfc_np, 0, -1)                  # (D, H, W, 3)

        if args.zero_threshold > 0.0:
            mag = np.linalg.norm(vfc_np, axis=-1, keepdims=True)
            n_zeroed = (mag[..., 0] < args.zero_threshold).sum()
            vfc_np[mag[..., 0] < args.zero_threshold] = 0.0
            print(f"Zero-threshold {args.zero_threshold:.0e}: zeroed {n_zeroed:,} / {mag.size:,} voxels "
                  f"({100 * n_zeroed / mag.size:.1f}%)")

        if args.normalize:
            print("Normalising VFC field to unit vectors…")
            mag = np.linalg.norm(vfc_np, axis=-1, keepdims=True)
            vfc_np = vfc_np / np.maximum(mag, 1e-8)

        if args.for_slicer:
            print("Formatting output for 3D Slicer (shape X,Y,Z,1,3 · intent DISPVECT)…")
            out_img = _to_slicer_dvf(vfc_np, img_nib.affine, flip_xy=args.flip_xy)
        else:
            out_img = nib.Nifti1Image(vfc_np, img_nib.affine)
            out_img.header.set_data_dtype(np.float32)

        nib.save(out_img, args.output)
        print(f"VFC field saved to: {args.output}  shape={vfc_np.shape}")

        json_path = args.output.replace(".nii.gz", ".json").replace(".nii", ".json")
        metadata = vars(args).copy()
        metadata["device_used"] = device
        metadata["output_shape"] = list(vfc_np.shape)
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata written to: {json_path}")
        return

    # =========================================================================
    # REGISTRATION MODE  (equivalent to: ptvreg --metric vfc)
    # =========================================================================
    if args.output is None:
        args.output = "./moving_warped.nii.gz"

    print(f"Loading fixed: {args.fixed}")
    fixed_img = nib.load(args.fixed)
    fixed_data = fixed_img.get_fdata().astype(np.float32)
    fixed_pix_resolution = np.array(fixed_img.header.get_zooms()[:3], dtype=np.float32)

    print(f"Loading moving: {args.moving}")
    moving_img = nib.load(args.moving)
    moving_data = moving_img.get_fdata().astype(np.float32)

    fixed_mask = None
    if args.fixed_mask:
        print(f"Loading fixed mask: {args.fixed_mask}")
        mask_data = nib.load(args.fixed_mask).get_fdata()
        if args.mask_labels is not None:
            print(f"Filtering mask for labels: {args.mask_labels}")
            fixed_mask = np.isin(mask_data, args.mask_labels).astype(np.float32)
        else:
            fixed_mask = (mask_data > 0).astype(np.float32)

    if args.clip:
        cl_min, cl_max = args.clip
        print(f"Clipping intensities to [{cl_min}, {cl_max}]")
        fixed_data = np.clip(fixed_data, cl_min, cl_max)
        moving_data = np.clip(moving_data, cl_min, cl_max)

    orig_shape = fixed_data.shape

    # Normalize intensities strictly to [0, 1]
    fixed_data = (fixed_data - np.min(fixed_data)) / (np.ptp(fixed_data) + 1e-8)
    moving_min = np.min(moving_data)
    moving_ptp = np.ptp(moving_data) + 1e-8
    moving_data = (moving_data - moving_min) / moving_ptp

    if args.scale_factor != 1.0:
        print(f"Scaling images by {args.scale_factor} for memory optimization...")
        fixed_data = ndimage.zoom(fixed_data, args.scale_factor, order=1)
        moving_data = ndimage.zoom(moving_data, args.scale_factor, order=1)
        fixed_pix_resolution = fixed_pix_resolution / max(args.scale_factor, 1e-8)
        print(f"Scaled shapes: {fixed_data.shape}")

    fixed_edgemap_data = None
    if args.fixed_edgemap:
        print(f"Loading fixed edgemap: {args.fixed_edgemap}")
        fixed_edgemap_data = nib.load(args.fixed_edgemap).get_fdata().astype(np.float32)
        if args.scale_factor != 1.0:
            fixed_edgemap_data = ndimage.zoom(fixed_edgemap_data, args.scale_factor, order=1)

    moving_edgemap_data = None
    if args.moving_edgemap:
        print(f"Loading moving edgemap: {args.moving_edgemap}")
        moving_edgemap_data = nib.load(args.moving_edgemap).get_fdata().astype(np.float32)
        if args.scale_factor != 1.0:
            moving_edgemap_data = ndimage.zoom(moving_edgemap_data, args.scale_factor, order=1)

    fixed_labels_data = None
    if args.fixed_labels:
        print(f"Loading fixed labels: {args.fixed_labels}")
        fixed_labels_data = np.round(nib.load(args.fixed_labels).get_fdata()).astype(np.int32)
        if args.scale_factor != 1.0:
            fixed_labels_data = ndimage.zoom(fixed_labels_data, args.scale_factor, order=0)

    moving_labels_data = None
    if args.moving_labels:
        print(f"Loading moving labels: {args.moving_labels}")
        moving_labels_data = np.round(nib.load(args.moving_labels).get_fdata()).astype(np.int32)
        if args.scale_factor != 1.0:
            moving_labels_data = ndimage.zoom(moving_labels_data, args.scale_factor, order=0)

    # n_levels=None triggers MATLAB-style auto-computation from image size;
    # the iterations list is padded to however many levels are auto-selected.
    reg = PTVRegistration(fixed_data, moving_data,
                          fixed_mask=fixed_mask,
                          border_mask=args.border_mask,
                          grid_spacing=args.spacing,
                          pix_resolution=fixed_pix_resolution.tolist(),
                          device=device,
                          lambda_reg=args.lambda_reg,
                          n_levels=None,
                          metric='vfc',
                          metric_param=args.metric_param,
                          fixed_edgemap=fixed_edgemap_data,
                          moving_edgemap=moving_edgemap_data,
                          vfc_radius=args.vfc_radius,
                          vfc_beta=args.vfc_beta,
                          vfc_sign_invariant=args.vfc_sign_invariant,
                          vfc_normalize=args.vfc_normalize,
                          fixed_labels=fixed_labels_data,
                          moving_labels=moving_labels_data,
                          dice_weight=args.dice_weight,
                          dice_labels=args.dice_labels,
                          lambda_jac=args.lambda_jac)

    print(f"Starting VFC registration (Metric=vfc, Reg={args.lambda_reg}, Iters={args.iterations})...")
    warped, flow = reg.optimize(iterations=args.iterations)

    if flow.dim() == len(orig_shape) + 1:
        flow = flow.unsqueeze(0)

    if args.scale_factor != 1.0:
        print(f"Upsampling warp field back to original size {orig_shape}...")
        mode = 'trilinear' if len(orig_shape) == 3 else 'bilinear'
        full_flow = torch.nn.functional.interpolate(
            flow, size=orig_shape, mode=mode, align_corners=True
        ).squeeze(0)
        full_flow = full_flow / args.scale_factor
    else:
        full_flow = flow.squeeze(0)

    print("Warping original resolution moving image with flow field...")
    moving_orig_tensor = (
        torch.tensor(moving_img.get_fdata().astype(np.float32), device=device)
        .unsqueeze(0).unsqueeze(0)
    )
    full_flow_tensor = full_flow.unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            warped_np = warp_image(moving_orig_tensor, full_flow_tensor).squeeze().cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        print("VRAM limit exceeded during final full-resolution warp — falling back to CPU...")
        moving_orig_tensor = moving_orig_tensor.cpu()
        full_flow_tensor = full_flow_tensor.cpu()
        with torch.no_grad():
            warped_np = warp_image(moving_orig_tensor, full_flow_tensor).squeeze().numpy()

    flow_np = full_flow.detach().cpu().numpy()

    target_dtype = moving_img.get_data_dtype()
    if np.issubdtype(target_dtype, np.floating):
        warped_np = warped_np.astype(np.float16).astype(np.float32)
    else:
        warped_np = np.round(warped_np).astype(target_dtype)

    print(f"Saving warped image to: {args.output}")
    save_nifti(warped_np, args.output, affine=fixed_img.affine)

    if args.warp:
        print(f"Saving warp field to: {args.warp}")
        flow_save = np.moveaxis(flow_np, 0, -1)   # (D, H, W, 3)
        scale = 0.01
        flow_quantized = np.clip(np.round(flow_save / scale), -32768, 32767).astype(np.int16)
        warp_nib = nib.Nifti1Image(flow_quantized, fixed_img.affine)
        warp_nib.header.set_data_dtype(np.int16)
        warp_nib.header['scl_slope'] = scale
        warp_nib.header['scl_inter'] = 0.0
        nib.save(warp_nib, args.warp)

    json_path = args.output.replace('.nii.gz', '.json').replace('.nii', '.json')
    print(f"Saving registration metadata to: {json_path}")
    metadata = vars(args).copy()
    metadata['device_used'] = device
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
