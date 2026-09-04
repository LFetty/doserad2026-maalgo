"""Convert MHA reference beamlet doses to bbox-cropped blosc2 (.b2nd) files.

Each .mha is read, the nonzero bounding box (with PAD voxels margin) is extracted,
and saved alongside the original as <name>.b2nd with the bbox stored in vlmeta.
The original .mha files are NOT deleted.

Usage:
    # single case
    uv run python scripts/convert_ref_doses_b2nd.py --case-dir /data/proton/training/1ABB006

    # all cases under a root
    uv run python scripts/convert_ref_doses_b2nd.py --data-root /data/proton/training

    # write to a different directory (mirrors structure)
    uv run python scripts/convert_ref_doses_b2nd.py --data-root /data/proton/training \\
        --out-root /fast/b2nd_doses
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import blosc2
import numpy as np
import SimpleITK as sitk

PAD = 2
CODEC = blosc2.Codec.ZSTD
CLEVEL = 3


def _b2nd_out_path(mha_path: Path, out_root: Path | None, data_root: Path | None) -> Path:
    if out_root is None:
        return mha_path.with_suffix(".b2nd")
    if data_root is not None:
        rel = mha_path.relative_to(data_root)
        return (out_root / rel).with_suffix(".b2nd")
    return out_root / mha_path.with_suffix(".b2nd").name


def convert_one(
    mha_path: Path,
    out_root: Path | None,
    data_root: Path | None,
    overwrite: bool,
) -> tuple[Path, str]:
    out_path = _b2nd_out_path(mha_path, out_root, data_root)
    if out_path.exists() and not overwrite:
        return out_path, "skip"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mha_path))).astype(np.float32, copy=False)
    nz = np.nonzero(arr)

    if len(nz[0]) == 0:
        z0, z1, y0, y1, x0, x1 = 0, 1, 0, 1, 0, 1
        bbox_arr = np.zeros((1, 1, 1), dtype=np.float32)
    else:
        z0 = max(int(nz[0].min()) - PAD, 0)
        z1 = min(int(nz[0].max()) + PAD + 1, arr.shape[0])
        y0 = max(int(nz[1].min()) - PAD, 0)
        y1 = min(int(nz[1].max()) + PAD + 1, arr.shape[1])
        x0 = max(int(nz[2].min()) - PAD, 0)
        x1 = min(int(nz[2].max()) + PAD + 1, arr.shape[2])
        bbox_arr = np.ascontiguousarray(arr[z0:z1, y0:y1, x0:x1])

    na = blosc2.asarray(
        bbox_arr,
        urlpath=str(out_path),
        mode="w",
        cparams={"codec": CODEC, "clevel": CLEVEL},
    )
    na.vlmeta["bbox"] = np.array([z0, z1, y0, y1, x0, x1], dtype=np.int32).tobytes()
    return out_path, "converted"


def _collect_mha_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    roots: list[Path] = []
    if args.data_root:
        roots.append(Path(args.data_root))
    for cd in args.case_dir or []:
        roots.append(Path(cd))
    for root in roots:
        files.extend(sorted(root.rglob("Dose_B*_R*_L*.mha")))
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None,
                   help="Root containing patient subdirs; recurse for Dose_*.mha")
    p.add_argument("--case-dir", type=Path, action="append", default=None,
                   help="Single case directory (repeatable)")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Write .b2nd here (mirrors input tree). Default: alongside each .mha")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--blosc2-threads", type=int, default=2,
                   help="blosc2 threads per worker (workers × threads should not exceed CPU cores)")
    args = p.parse_args()

    files = _collect_mha_files(args)
    if not files:
        print("No Dose_*.mha files found.", file=sys.stderr)
        sys.exit(1)

    blosc2.set_nthreads(args.blosc2_threads)
    data_root = Path(args.data_root) if args.data_root else None
    out_root = Path(args.out_root) if args.out_root else None

    total = len(files)
    print(f"Found {total} .mha files  workers={args.workers}  blosc2_threads={args.blosc2_threads}")

    converted = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(convert_one, f, out_root, data_root, args.overwrite): f
            for f in files
        }
        for i, fut in enumerate(as_completed(futs), 1):
            src = futs[fut]
            try:
                out_path, status = fut.result()
                if status == "converted":
                    converted += 1
                else:
                    skipped += 1
                if i % 100 == 0 or i == total:
                    print(f"  [{i}/{total}] converted={converted} skipped={skipped} errors={errors}")
            except Exception as exc:
                errors += 1
                print(f"  ERROR {src}: {exc}", file=sys.stderr)

    print(f"\nDone: {converted} converted, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
