# sweep.py — run several ptvreg variants for one subject.
# In VS Code: just press the ▶ Run button. Flip the toggles below to change behaviour.
import subprocess
from pathlib import Path

# ----------------------------------------------------------------------
# TOGGLES — flip these, then press Run.
# ----------------------------------------------------------------------
DRY_RUN   = True    # True = just print the commands (start here!). False = actually run.
SAVE_WARP = False   # True = also save the warp field per variant.

# ----------------------------------------------------------------------
# 1. WHAT AM I ALIGNING?  (edit these per subject)
# ----------------------------------------------------------------------
SUBJECT = "sub-004"
FIXED   = Path("sub-004/features/nacpet.nii.gz")            # -f  (target space)
MOVING  = Path("sub-004/features/mri_combined_sum.nii.gz")  # -m  (gets warped)

RUNS_DIR = Path("runs") / SUBJECT

# ----------------------------------------------------------------------
# 2. VARIANTS — your experiment list. Comment out to skip.
# ----------------------------------------------------------------------
VARIANTS = [
    {"name": "lcc_s05",  "flags": ["--metric", "lcc",  "--scale_factor", "0.5"]},
    {"name": "lcc_s07",  "flags": ["--metric", "lcc",  "--scale_factor", "0.7"]},
    {"name": "edge_s07", "flags": ["-e",                "--scale_factor", "0.7"]},
    {"name": "emse_s05", "flags": ["--metric", "emse", "--scale_factor", "0.5"]},
    # {"name": "vfc_s05", "flags": ["--metric", "vfc",  "--scale_factor", "0.5"]},
]

# ----------------------------------------------------------------------
# 3. Machinery — you shouldn't need to touch below here.
# ----------------------------------------------------------------------
def run(cmd):
    print("  " + " ".join(str(c) for c in cmd))
    if not DRY_RUN:
        subprocess.run(cmd, check=True)


def main():
    mode = "DRY RUN (nothing will execute)" if DRY_RUN else "RUNNING FOR REAL"
    print(f"### {mode} ###")

    for v in VARIANTS:
        out_dir = RUNS_DIR / v["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        warped = out_dir / "warped.nii.gz"

        print(f"\n=== {SUBJECT} / {v['name']} ===")
        cmd = ["ptvreg", "-f", FIXED, "-m", MOVING, "-o", warped, *v["flags"]]
        if SAVE_WARP:
            cmd += ["-w", out_dir / "warp.nii.gz"]
        run(cmd)

    print(f"\nDone. Outputs in: {RUNS_DIR}/<variant>/warped.nii.gz")


if __name__ == "__main__":
    main()