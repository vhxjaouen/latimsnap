"""
ITK-SNAP bridge CLI for pTVreg.

Streamlined variant of ``cli.py`` designed to be spawned by ITK-SNAP as a
QProcess. Emits machine-parsable ``SNAP_*`` marker lines on stdout so the
host application can display live progress, refresh a preview overlay after
each pyramid level, and optionally refresh again inside a level every N
LBFGS iterations.

Markers (one per line, unbuffered):

  SNAP_META:n_levels=<L>:grid_spacing=<S>:device=<cpu|cuda>
  SNAP_LEVEL_START:level=<i>:iters=<N>:shape=<X>x<Y>x<Z>
  SNAP_ITER:level=<i>:iter=<k>:metric=<m>
  SNAP_LEVEL_END:level=<i>:preview=<abs_path>
  SNAP_DONE:warped=<abs>:warp=<abs>
  SNAP_ERROR:<message>

Any non-marker line printed by pTVreg is forwarded verbatim to the host log
widget.
"""

import argparse
import os
import sys
import traceback

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import torch

from pTVreg.registration import PTVRegistration
from pTVreg.utils import save_nifti
from pTVreg.deformation import warp_image


def _emit(line):
    """Write a marker line to stdout, unbuffered."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _parse_args():
    p = argparse.ArgumentParser(
        description="pTVreg bridge for ITK-SNAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs / outputs (subset of cli.py, aligned to what SNAP passes)
    p.add_argument("-f", "--fixed", required=True)
    p.add_argument("-m", "--moving", required=True)
    p.add_argument("-o", "--output", default="./moving_warped.nii.gz")
    p.add_argument("-w", "--warp", default=None)
    p.add_argument("--fixed_mask", default=None)

    # Algorithm
    p.add_argument("--spacing", type=int, default=8)
    p.add_argument("--scale_factor", type=float, default=1.0)
    p.add_argument("--lambda_reg", type=float, nargs="+", default=[0.15])
    p.add_argument("--iterations", type=int, nargs="+", default=[100])
    p.add_argument(
        "--metric",
        choices=["lcc", "ssd", "nuclear", "emse", "vfc"],
        default="lcc",
    )
    p.add_argument("--metric_param", type=float, default=2.1)
    p.add_argument("--border_mask", type=int, default=5)
    p.add_argument("--clip", type=float, nargs=2, default=None)

    # VFC-specific flags
    p.add_argument("--vfc-radius", dest="vfc_radius", type=float, default=15.0)
    p.add_argument("--vfc-beta", dest="vfc_beta", type=float, default=2.0)
    p.add_argument("--vfc-sign-invariant", dest="vfc_sign_invariant",
                   action="store_true")
    p.add_argument("--vfc-normalize", dest="vfc_normalize", action="store_true")
    p.add_argument("--fixed-edgemap", dest="fixed_edgemap", default=None)
    p.add_argument("--moving-edgemap", dest="moving_edgemap", default=None)

    # Label-guided soft-Dice
    p.add_argument("--fixed-labels", dest="fixed_labels", default=None)
    p.add_argument("--moving-labels", dest="moving_labels", default=None)
    p.add_argument("--dice-weight", dest="dice_weight", type=float, default=0.5)
    p.add_argument("--dice-labels", dest="dice_labels", type=int, nargs="+",
                   default=None)

    # Penalties & sparsification
    p.add_argument("--lambda_jac", type=float, default=0.0)
    p.add_argument("--dvf_epsilon", type=float, default=0.1)

    # SNAP-specific
    p.add_argument("--snap-preview-dir", dest="snap_preview_dir", required=True)
    p.add_argument(
        "--snap-every-iter",
        dest="snap_every_iter",
        type=int,
        default=0,
        help="If > 0 emit finest-level per-iteration previews every N iterations.",
    )
    return p.parse_args()


def _resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    args = _parse_args()

    try:
        os.makedirs(args.snap_preview_dir, exist_ok=True)

        print(f"Loading fixed: {args.fixed}")
        fixed_img = nib.load(args.fixed)
        fixed_data = fixed_img.get_fdata().astype(np.float32)
        fixed_pix_resolution = np.array(
            fixed_img.header.get_zooms()[:3], dtype=np.float32
        )

        print(f"Loading moving: {args.moving}")
        moving_img = nib.load(args.moving)
        moving_data = moving_img.get_fdata().astype(np.float32)

        fixed_mask = None
        if args.fixed_mask:
            print(f"Loading fixed mask: {args.fixed_mask}")
            fixed_mask = (nib.load(args.fixed_mask).get_fdata() > 0).astype(
                np.float32
            )

        # Intensity clipping (typical use: lung CT --clip 80 900).
        # When --clip is supplied we normalise using the clip window itself as
        # the intensity range, matching MATLAB's img_thr(vol, cl_min, cl_max, 1)
        # (see pTVreg/matlab/examples_ptv/COPD_final.m). Without --clip we
        # fall back to the per-image min/max range.
        if args.clip:
            cl_min, cl_max = args.clip
            print(f"Clipping intensities to [{cl_min}, {cl_max}] and normalising to [0,1]")
            fixed_data = np.clip(fixed_data, cl_min, cl_max)
            moving_data = np.clip(moving_data, cl_min, cl_max)
            denom = max(float(cl_max) - float(cl_min), 1e-8)
            fixed_data = (fixed_data - float(cl_min)) / denom
            moving_data = (moving_data - float(cl_min)) / denom
        else:
            # Normalise using each image's own min/max so intensities lie in [0,1].
            fixed_data = (fixed_data - np.min(fixed_data)) / (np.ptp(fixed_data) + 1e-8)
            moving_data = (moving_data - np.min(moving_data)) / (
                np.ptp(moving_data) + 1e-8
            )

        orig_shape = fixed_data.shape

        if args.scale_factor != 1.0:
            print(f"Scaling images by {args.scale_factor}...")
            fixed_data = ndimage.zoom(fixed_data, args.scale_factor, order=1)
            moving_data = ndimage.zoom(moving_data, args.scale_factor, order=1)
            fixed_pix_resolution = fixed_pix_resolution / max(
                args.scale_factor, 1e-8
            )

        device = _resolve_device()
        print(f"Using device: {device}")

        # Optional edge maps and label maps
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

        reg = PTVRegistration(
            fixed_data,
            moving_data,
            fixed_mask=fixed_mask,
            border_mask=args.border_mask,
            grid_spacing=args.spacing,
            pix_resolution=fixed_pix_resolution.tolist(),
            device=device,
            lambda_reg=args.lambda_reg,
            n_levels=None,
            metric=args.metric,
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
            lambda_jac=args.lambda_jac,
        )

        n_levels = reg.n_levels
        _emit(
            f"SNAP_META:n_levels={n_levels}:grid_spacing={args.spacing}:device={device}"
        )

        # We keep the full-resolution moving image on the compute device so
        # every preview can be produced by warping the original data with the
        # current flow field, matching what the final CLI does at the end.
        moving_orig_tensor = (
            torch.tensor(
                moving_img.get_fdata().astype(np.float32), device=device
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        def _save_preview(level_idx, flow_tensor, tag):
            """Warp the full-resolution moving image with ``flow_tensor`` and
            save it as a NIfTI at the fixed image resolution."""
            if flow_tensor.dim() == len(orig_shape) + 1:
                flow_tensor = flow_tensor.unsqueeze(0)

            if args.scale_factor != 1.0:
                mode = "trilinear" if len(orig_shape) == 3 else "bilinear"
                full_flow = torch.nn.functional.interpolate(
                    flow_tensor, size=orig_shape, mode=mode, align_corners=True
                ).squeeze(0)
                full_flow = full_flow / args.scale_factor
            else:
                full_flow = flow_tensor.squeeze(0)

            full_flow_batched = full_flow.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    warped_np = (
                        warp_image(moving_orig_tensor, full_flow_batched)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
            except torch.cuda.OutOfMemoryError:
                warped_np = (
                    warp_image(
                        moving_orig_tensor.cpu(), full_flow_batched.cpu()
                    )
                    .squeeze()
                    .numpy()
                )

            warped_np = warped_np.astype(np.float32)
            preview_path = os.path.join(
                args.snap_preview_dir, f"preview_level_{level_idx}_{tag}.nii.gz"
            )
            save_nifti(warped_np, preview_path, affine=fixed_img.affine)
            return preview_path

        def level_callback(level_idx, warped_tensor, flow_tensor):
            preview_path = _save_preview(level_idx, flow_tensor, "end")
            # Metric value is already logged by pTVreg; recompute a cheap value
            # here would double compute, so we omit metric= in the marker and
            # rely on the parser to tolerate missing fields.
            _emit(
                f"SNAP_LEVEL_END:level={level_idx}:preview={preview_path}"
            )

        finest_level = n_levels - 1

        def iter_callback(level_idx, iter_idx, metric_value):
            _emit(
                f"SNAP_ITER:level={level_idx}:iter={iter_idx}:metric={metric_value:.6f}"
            )
            if (
                args.snap_every_iter
                and args.snap_every_iter > 0
                and level_idx == finest_level
                and iter_idx > 0
                and (iter_idx % args.snap_every_iter) == 0
            ):
                # Extra preview inside the finest level.
                with torch.no_grad():
                    _, flow = reg.forward(level_idx)
                preview_path = _save_preview(
                    level_idx, flow, f"iter{iter_idx}"
                )
                _emit(
                    f"SNAP_LEVEL_END:level={level_idx}:preview={preview_path}"
                )

        print(
            f"Starting optimization (metric={args.metric}, "
            f"lambda={args.lambda_reg}, iterations={args.iterations})..."
        )

        warped, flow = reg.optimize(
            iterations=args.iterations,
            level_callback=level_callback,
            iter_callback=iter_callback,
        )

        # Final full-resolution warped image + warp field
        final_preview = _save_preview(n_levels - 1, flow, "final")
        # Also copy to the requested output path.
        if args.output != final_preview:
            try:
                import shutil
                shutil.copyfile(final_preview, args.output)
            except Exception:
                pass

        warp_path = ""
        if args.warp:
            if flow.dim() == len(orig_shape) + 1:
                flow_out = flow.unsqueeze(0)
            else:
                flow_out = flow
            if args.scale_factor != 1.0:
                mode = "trilinear" if len(orig_shape) == 3 else "bilinear"
                full_flow = torch.nn.functional.interpolate(
                    flow_out, size=orig_shape, mode=mode, align_corners=True
                ).squeeze(0)
                full_flow = full_flow / args.scale_factor
            else:
                full_flow = flow_out.squeeze(0)

            flow_np = full_flow.detach().cpu().numpy()
            flow_save = np.moveaxis(flow_np, 0, -1)
            scale = 0.01
            flow_quantized = np.clip(
                np.round(flow_save / scale), -32768, 32767
            ).astype(np.int16)
            warp_img = nib.Nifti1Image(flow_quantized, fixed_img.affine)
            warp_img.header.set_data_dtype(np.int16)
            warp_img.header["scl_slope"] = scale
            warp_img.header["scl_inter"] = 0.0
            nib.save(warp_img, args.warp)
            warp_path = args.warp

        _emit(f"SNAP_DONE:warped={args.output}:warp={warp_path}")
        return 0

    except Exception as exc:  # pragma: no cover
        tb = traceback.format_exc()
        # Print traceback to stderr for the log widget.
        sys.stderr.write(tb + "\n")
        sys.stderr.flush()
        _emit(f"SNAP_ERROR:{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
