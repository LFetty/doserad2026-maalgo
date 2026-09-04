"""Overlay proton LUT parameters (Z, sigma, sigma1, sigma2, weight) across multiple
machine .mat files for a grid of energies, to compare a new fit against the previous
fit and the pyRadPlan reference.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pydosert-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_proton_lut import load_reference_entry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LUTS = [
    ("new 3D fit", ROOT / "example_data" / "mc_fit_smooth" / "lut_fast_3d.mat"),
    ("previous fit", ROOT / "example_data" / "mc_fit_smooth" / "lut_fast_tail_safe_halo_mc_v1.mat"),
    ("pyRadPlan", ROOT / "example_data" / "pyradplan" / "protons_Generic.mat"),
]
DEFAULT_ENERGIES = (72.3349, 101.9976, 131.5856, 159.8591, 184.7095, 200.7966)
PARAMS = [
    ("Z", "Z / cm", "depth dose"),
    ("sigma_mm", "sigma [mm]", "single sigma"),
    ("sigma1_mm", "sigma1 [mm]", "double sigma1"),
    ("sigma2_mm", "sigma2 [mm]", "double sigma2"),
    ("weight", "weight", "halo weight"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lut",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="LUT to overlay as 'label=path'. Repeatable. Defaults to new/previous/pyradplan.",
    )
    parser.add_argument(
        "--energies-mev",
        type=float,
        nargs="+",
        default=list(DEFAULT_ENERGIES),
        help="Energies (MeV) to plot; nearest LUT entry is used.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "out" / "lut_compare",
        help="Output directory for comparison figures.",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=float,
        default=None,
        help="Optional x-axis upper limit in mm.",
    )
    return parser.parse_args()


def resolve_luts(lut_args: list[str] | None) -> list[tuple[str, Path]]:
    if not lut_args:
        return [(label, path) for label, path in DEFAULT_LUTS if path.exists()]
    resolved: list[tuple[str, Path]] = []
    for item in lut_args:
        if "=" not in item:
            raise ValueError(f"--lut must be 'label=path', got {item!r}")
        label, path_str = item.split("=", 1)
        resolved.append((label, Path(path_str)))
    return resolved


def main() -> None:
    args = parse_args()
    luts = resolve_luts(args.lut)
    if not luts:
        raise FileNotFoundError("No LUT files found to compare (defaults missing?)")
    for label, path in luts:
        if not path.exists():
            raise FileNotFoundError(f"LUT '{label}' not found: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    colors = [f"C{i}" for i in range(len(luts))]

    print("Comparing LUTs:")
    for label, path in luts:
        print(f"  {label:14s} {path}")
    print(f"Energies: {args.energies_mev}")

    for energy in args.energies_mev:
        entries = [(label, color, load_reference_entry(path, energy)) for (label, path), color in zip(luts, colors)]

        fig, axes = plt.subplots(1, len(PARAMS), figsize=(4.2 * len(PARAMS), 4.0))
        matched = entries[0][2]["energy_mev"]
        for ax, (key, ylabel, title) in zip(axes, PARAMS):
            for label, color, ref in entries:
                depths = np.asarray(ref["depths_mm"], dtype=np.float64)
                values = np.asarray(ref[key], dtype=np.float64)
                ax.plot(depths, values, color=color, label=f"{label} (E={float(ref['energy_mev']):.1f})", lw=1.4)
            ax.set_title(title)
            ax.set_xlabel("depth [mm]")
            ax.set_ylabel(ylabel)
            if args.max_depth_mm is not None:
                ax.set_xlim(0.0, args.max_depth_mm)
            ax.grid(True, alpha=0.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"LUT comparison @ ~{energy:.2f} MeV (matched {matched:.2f} MeV)")
        fig.tight_layout()
        safe = f"{energy:.4f}".replace(".", "p")
        out_path = args.out_dir / f"lut_compare_{safe}MeV.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"wrote {out_path}")

    print(f"\nWrote {len(args.energies_mev)} comparison figures to {args.out_dir}")


if __name__ == "__main__":
    main()
