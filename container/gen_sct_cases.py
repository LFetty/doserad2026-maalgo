"""Generate sCT case dirs for the MR variant: MR -> sCT (nnUNet Dataset038) written
as image/ct.mha, with plan JSON + reference dose/ symlinked from the real case. The
existing evaluate_doserad_proton_case.py can then be run on these dirs unchanged, so
we measure the dosimetric impact of using sCT instead of the true CT.

    uv run python container/gen_sct_cases.py \
        --src-root <DoseRAD2026>/proton/training --cases 1ABB006 1THB002 --out sct_cases
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import SimpleITK as sitk

from standalone_regression_inference import StandaloneRegressionInference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, required=True,
                    help="DoseRAD proton training root holding the source cases")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("out/sct_cases"))
    ap.add_argument("--bundle", type=Path, default=Path("model_mri/mrtoct_bundle"))
    args = ap.parse_args()

    pred = StandaloneRegressionInference(str(args.bundle), device="cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    for cid in args.cases:
        src = args.src_root / cid
        mr = sitk.ReadImage(str(src / "image" / "mr.mha"))
        ct = sitk.ReadImage(str(src / "image" / "ct.mha"))
        t0 = time.time()
        sct_a = np.asarray(pred.predict(sitk.GetArrayFromImage(mr).astype(np.float32))).astype(np.float32).squeeze()
        dt = time.time() - t0
        # sanity vs true CT
        ct_a = sitk.GetArrayFromImage(ct).astype(np.float32)
        mae = float(np.abs(sct_a[ct_a > -500] - ct_a[ct_a > -500]).mean()) if sct_a.shape == ct_a.shape else float("nan")

        sct = sitk.GetImageFromArray(sct_a)
        sct.CopyInformation(ct)  # sCT on the (registered) CT/MR grid

        dst = args.out / cid
        (dst / "image").mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sct, str(dst / "image" / "ct.mha"), useCompression=True)
        # symlink plan json + dose reference dir from the real case
        plan = src / f"{cid}.json"
        link_plan = dst / f"{cid}.json"
        if link_plan.exists() or link_plan.is_symlink():
            link_plan.unlink()
        link_plan.symlink_to(plan)
        link_dose = dst / "dose"
        if link_dose.exists() or link_dose.is_symlink():
            link_dose.unlink()
        link_dose.symlink_to(src / "dose")
        print(f"{cid}: sCT MAE(body)={mae:.1f}HU  predict {dt:.1f}s -> {dst}", flush=True)

    print("SCT CASES DONE", flush=True)


if __name__ == "__main__":
    main()
