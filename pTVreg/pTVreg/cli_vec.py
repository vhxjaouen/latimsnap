"""
ptvreg-vec: pTVreg registration driven by vector-valued images.

Typical use-cases:
  - Align two VFC fields produced by the `vfc` CLI (e.g. from two MRI scans)
    and obtain a warp field applicable via `ptvwarp` to the original scalar images.
  - Align two gradient vector fields (D×H×W×3 NIfTI) more generally.

Registration metric: mean per-component LCC (MultiChannelLCC).
Each spatial component is correlated independently and results are averaged.
This preserves directional information without requiring Jacobian reorientation.
The warp field output is compatible with `ptvwarp` (pTVreg format, int16-quantized).
"""
import argparse
import sys
import json
import torch
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from pTVreg.registration import PTVRegistration
from pTVreg.utils import save_nifti
from pTVreg.deformation import warp_image


def _load_vector_nifti(path):
    """Load a 4-D NIfTI vector field (D,H,W,C) and return (D,H,W,C) float32 array.
    Also handles 5-D Slicer-format fields (D,H,W,1,C) by squeezing the singleton dim.
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    # Squeeze singleton Slicer dim: (X,Y,Z,1,C) -> (X,Y,Z,C)
    if data.ndim == 5 and data.shape[3] == 1:
        data = data[:, :, :, 0, :]
    if data.ndim != 4:
        raise ValueError(f"Expected 4-D vector field in {path}, got shape {data.shape}")
    return img, data


def _normalize_channels(data):
    """Normalize each channel of (D,H,W,C) independently to [0, 1]."""
    out = np.empty_like(data)
    for c in range(data.shape[-1]):
        ch = data[..., c]
        mn, mx = ch.min(), ch.max()
        out[..., c] = (ch - mn) / (mx - mn + 1e-8)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='ptvreg-vec: pTVreg registration for vector-valued images (e.g. gradient fields)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-c', '--config', help='Path to JSON config file')
    parser.add_argument('-f', '--fixed', help='Fixed vector field (4-D NIfTI, D×H×W×C)')
    parser.add_argument('-m', '--moving', help='Moving vector field (4-D NIfTI, D×H×W×C)')
    parser.add_argument('--orig', default=None,
                        help='Original scalar moving image (3-D NIfTI) to warp with the estimated field. '
                             'If omitted, the moving vector field magnitude is warped instead.')
    parser.add_argument('-o', '--output', default='./moving_warped.nii.gz',
                        help='Output warped image path')
    parser.add_argument('-w', '--warp', nargs='?', const='./warpfield.nii.gz', default=None,
                        help='Output warp field path. If set without value, defaults to ./warpfield.nii.gz')

    parser.add_argument('--fixed_mask', default=None,
                        help='Path to fixed image mask (NIfTI, evaluated where mask > 0)')
    parser.add_argument('--mask_labels', type=int, nargs='+', default=None,
                        help='Integers to extract from a multi-label mask')
    parser.add_argument('--border_mask', type=int, default=5,
                        help='Voxels to ignore at image boundaries')

    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--spacing', type=int, default=8,
                        help='Grid spacing at finest level (voxels)')
    parser.add_argument('--scale_factor', type=float, default=1.0,
                        help='Downsample scale factor (e.g. 0.5) to save memory')
    parser.add_argument('--lambda_reg', type=float, nargs='+', default=[0.15],
                        help='Regularization weight, or per-level list (coarse→fine)')
    parser.add_argument('--iterations', type=int, nargs='+', default=[160, 120, 80],
                        help='Iterations per pyramid level (coarse→fine)')
    parser.add_argument('--metric_param', type=float, default=2.1,
                        help='LCC sigma for the per-component metric')
    parser.add_argument('--lambda_jac', type=float, default=0.0,
                        help='Weight for the Jacobian determinant penalty (discourages folding). '
                             'Default 0.0 = disabled. Indicative range: 0.5–5.0.')
    parser.add_argument('--clip', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='Clip each vector component to [MIN, MAX] before normalizing.')
    parser.add_argument('--no-normalize', dest='no_normalize', action='store_true',
                        help='Skip per-component [0,1] normalization. '
                             'Recommended for VFC fields whose components already share '
                             'a common magnitude scale set by the kernel normalization.')
    parser.add_argument('--vfs', dest='vfs', action='store_true',
                        help='Use Vector Field Similarity (VFS) metric: maximise cosine similarity '
                             'between fixed and warped-moving vector fields (NGF-style). '
                             'More principled than per-component LCC for VFC/gradient fields. '
                             'Combine with --no-normalize for pre-scaled VFC inputs.')
    parser.add_argument('--vfs-sign-invariant', dest='vfs_sign_invariant', action='store_true',
                        help='Use cos\u00b2 variant of VFS (sign-invariant). '
                             'Useful when VFC fields from boundary maps may be globally flipped.')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # --- Config file override ---
    if args.config:
        print(f"Loading configuration from {args.config}")
        specified_args = set()
        for arg in sys.argv[1:]:
            for action in parser._actions:
                for opt in action.option_strings:
                    if arg == opt or arg.startswith(opt + '='):
                        specified_args.add(action.dest)
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(args, key) and key != 'config' and key not in specified_args:
                    setattr(args, key, value)

    if not args.fixed or not args.moving:
        parser.error('--fixed and --moving are required')

    # --- Load vector fields ---
    print(f"Loading fixed vector field: {args.fixed}")
    fixed_img, fixed_vec = _load_vector_nifti(args.fixed)

    print(f"Loading moving vector field: {args.moving}")
    moving_img, moving_vec = _load_vector_nifti(args.moving)

    if fixed_vec.shape != moving_vec.shape:
        raise ValueError(f"Fixed and moving vector field shapes differ: "
                         f"{fixed_vec.shape} vs {moving_vec.shape}")

    n_components = fixed_vec.shape[-1]
    print(f"Vector field: {fixed_vec.shape[:3]}, {n_components} components")

    # --- Optional scalar original image ---
    orig_scalar = None
    if args.orig:
        print(f"Loading original scalar moving image: {args.orig}")
        orig_img = nib.load(args.orig)
        orig_scalar = orig_img.get_fdata().astype(np.float32)
        if orig_scalar.shape != fixed_vec.shape[:3]:
            raise ValueError(f"Original image shape {orig_scalar.shape} does not match "
                             f"vector field spatial shape {fixed_vec.shape[:3]}")

    # --- Optional mask ---
    fixed_mask = None
    if args.fixed_mask:
        print(f"Loading fixed mask: {args.fixed_mask}")
        mask_img = nib.load(args.fixed_mask)
        mask_data = mask_img.get_fdata()
        if args.mask_labels is not None:
            fixed_mask = np.isin(mask_data, args.mask_labels).astype(np.float32)
        else:
            fixed_mask = (mask_data > 0).astype(np.float32)

    orig_shape = fixed_vec.shape[:3]
    fixed_pix_resolution = np.array(fixed_img.header.get_zooms()[:3], dtype=np.float32)

    # --- Optional clip ---
    if args.clip:
        cl_min, cl_max = args.clip
        print(f"Clipping vector components to [{cl_min}, {cl_max}]")
        fixed_vec  = np.clip(fixed_vec,  cl_min, cl_max)
        moving_vec = np.clip(moving_vec, cl_min, cl_max)

    # --- Normalize each component independently (skip for VFC fields) ---
    if args.no_normalize:
        print("Skipping per-component normalization (--no-normalize).")
    else:
        fixed_vec  = _normalize_channels(fixed_vec)
        moving_vec = _normalize_channels(moving_vec)

    # --- Optional spatial downsampling ---
    if args.scale_factor != 1.0:
        print(f"Scaling vector fields by {args.scale_factor}...")
        zoom = [args.scale_factor] * 3 + [1.0]
        fixed_vec = ndimage.zoom(fixed_vec, zoom, order=1)
        moving_vec = ndimage.zoom(moving_vec, zoom, order=1)
        if fixed_mask is not None:
            fixed_mask = ndimage.zoom(fixed_mask, args.scale_factor, order=0)
        fixed_pix_resolution = fixed_pix_resolution / max(args.scale_factor, 1e-8)
        print(f"Scaled spatial shape: {fixed_vec.shape[:3]}")

    # --- Device ---
    if torch.cuda.is_available():
        device = 'cuda'
        print("GPU detected and used by default.")
    else:
        device = 'cpu'
        print("GPU not available, using CPU.")

    # n_levels=None: let PTVRegistration auto-compute pyramid depth from image size
    # (MATLAB-style). The iterations list is padded to however many levels are chosen.
    n_levels = None

    selected_metric = 'vfs' if args.vfs else 'mclcc'

    # PTVRegistration expects (D,H,W,C); the 4D disambiguation heuristic in registration.py
    # routes last-dim-small arrays to the (1,C,D,H,W) layout automatically.
    reg = PTVRegistration(
        fixed_vec, moving_vec,
        fixed_mask=fixed_mask,
        border_mask=args.border_mask,
        grid_spacing=args.spacing,
        pix_resolution=fixed_pix_resolution.tolist(),
        device=device,
        lambda_reg=args.lambda_reg,
        n_levels=n_levels,
        metric=selected_metric,
        metric_param=args.metric_param,
        lambda_jac=args.lambda_jac,
        vfs_sign_invariant=args.vfs_sign_invariant,
    )

    print(f"Starting optimization (Metric={selected_metric}, {n_components} components, "
          f"Reg={args.lambda_reg}, Iters={args.iterations})...")
    warped_vec, flow = reg.optimize(iterations=args.iterations)

    # --- Upsample warp field back to original resolution if downsampled ---
    if flow.dim() == len(orig_shape) + 1:
        flow = flow.unsqueeze(0)

    if args.scale_factor != 1.0:
        print(f"Upsampling warp field to original shape {orig_shape}...")
        full_flow = torch.nn.functional.interpolate(
            flow, size=orig_shape, mode='trilinear', align_corners=True
        ).squeeze(0)
        full_flow = full_flow / args.scale_factor
    else:
        full_flow = flow.squeeze(0)

    full_flow_tensor = full_flow.unsqueeze(0).to(device)

    # --- Warp target: original scalar or vector magnitude fallback ---
    if orig_scalar is not None:
        target_tensor = torch.tensor(orig_scalar, device=device).unsqueeze(0).unsqueeze(0)
        print("Warping original scalar image with estimated field...")
        try:
            with torch.no_grad():
                warped_np = warp_image(target_tensor, full_flow_tensor).squeeze().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            print("VRAM limit exceeded during warp — falling back to CPU...")
            with torch.no_grad():
                warped_np = warp_image(target_tensor.cpu(), full_flow_tensor.cpu()).squeeze().numpy()
        target_dtype = orig_img.get_data_dtype()
        if np.issubdtype(target_dtype, np.floating):
            warped_np = warped_np.astype(np.float16).astype(np.float32)
        else:
            warped_np = np.round(warped_np).astype(target_dtype)
        print(f"Saving warped image to: {args.output}")
        save_nifti(warped_np, args.output, affine=fixed_img.affine)
    else:
        print("No --orig provided; warping all components of the moving vector field...")
        # Reload original (pre-normalization, pre-downsampling) moving vector field
        _, moving_vec_orig = _load_vector_nifti(args.moving)
        # (D, H, W, C) -> (1, C, D, H, W)
        moving_vec_tensor = torch.tensor(
            np.moveaxis(moving_vec_orig, -1, 0), device=device
        ).unsqueeze(0)
        try:
            with torch.no_grad():
                warped_vec = warp_image(moving_vec_tensor, full_flow_tensor).squeeze(0).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            print("VRAM limit exceeded during warp — falling back to CPU...")
            with torch.no_grad():
                warped_vec = warp_image(
                    moving_vec_tensor.cpu(), full_flow_tensor.cpu()
                ).squeeze(0).numpy()
        # (C, D, H, W) -> (D, H, W, C)
        warped_np = np.moveaxis(warped_vec, 0, -1).astype(np.float32)
        print(f"Saving warped vector field ({warped_np.shape[-1]} components) to: {args.output}")
        save_nifti(warped_np, args.output, affine=fixed_img.affine)

    # --- Save warp field ---
    if args.warp:
        print(f"Saving warp field to: {args.warp}")
        flow_np = full_flow.detach().cpu().numpy()
        flow_save = np.moveaxis(flow_np, 0, -1)  # (D,H,W,3)
        scale = 0.01
        flow_quantized = np.clip(np.round(flow_save / scale), -32768, 32767).astype(np.int16)
        warp_img = nib.Nifti1Image(flow_quantized, fixed_img.affine)
        warp_img.header.set_data_dtype(np.int16)
        warp_img.header['scl_slope'] = scale
        warp_img.header['scl_inter'] = 0.0
        nib.save(warp_img, args.warp)

    # --- JSON sidecar ---
    json_path = args.output.replace('.nii.gz', '.json').replace('.nii', '.json')
    metadata = vars(args).copy()
    metadata['device_used'] = device
    metadata['n_components'] = int(n_components)
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {json_path}")


if __name__ == '__main__':
    main()
