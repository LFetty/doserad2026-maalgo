"""Direct per-energy comparison of the pencil-beam dose engine vs Monte-Carlo dose
in a pure water phantom (0 HU). For every MC energy sim we run a single beamlet
through the IonDoseEngine on a matching 1 mm water grid and compare laterally
integrated depth dose (IDD), the central depth-lateral plane, and per-energy MAE.

    uv run python scripts/benchmark_pb_vs_mc_water.py [--limit N]
"""
from __future__ import annotations

import argparse, glob, os, re, sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pydosert-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch, json, SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pydose_rt.data.machine_config import MachineConfig
from pydose_rt.data.ion_beam import IonSpotBeam, IonSpotBeamSequence
from pydose_rt.engine import ion_dose_engine
from pydose_rt.engine.ion_dose_engine import IonDoseEngine
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT
from pydose_rt.utils.gamma import local_gamma_pass_rate

LATERAL_MODEL = "gauss"   # set from CLI
HET_MCS = False           # heterogeneity-aware MCS (Fuchs dH); set from CLI
MAT_X0 = False            # per-material radiation length; set from CLI

ROOT = Path(__file__).parent.parent
MC_DIR: Path | None = None  # set from --mc-dir
LUT_MAT = ROOT / "example_data" / "mc_fit_smooth" / "lut_fast_3d.mat"
BEAM_PARAMS = json.loads((ROOT / "example_data" / "beam_parameters.json").read_text())
OUT = ROOT / "out" / "pb_vs_mc_water"; OUT.mkdir(parents=True, exist_ok=True)

N_LAT = 80           # lateral voxels (1 mm) -> +/-40 mm, >= the +/-37 mm kernel window
N_DEPTH = 1500       # depth voxels (set from --dz in main); native 0.2 mm -> 300 mm
DZ = 0.2             # depth voxel size [mm] (set from --dz in main)
DEPTH_FACTOR = 1     # MC depth rebin factor; DZ = 0.2 * DEPTH_FACTOR (set in main)
C = N_LAT // 2       # lateral centre index
W = 37               # +/- kernel half-window (mm == voxels at 1 mm)
WIN = slice(C - W, C + W)
PART = 1e7           # particles per beamlet (MC events)


def sigma0_mm(energy: float) -> float:
    t = BEAM_PARAMS["proton"]["energy_table"]
    e = np.array([x["energy_mev"] for x in t]); s = np.array([x["sigma_spot_mm"] for x in t])
    return float(np.interp(energy, e, s))


def load_mc(path: Path) -> np.ndarray:
    """Return MC dose rebinned to 1 mm lateral and DZ mm depth -> (zlat, ylat, depth)."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))  # (z=500, y=500, x=1500) @0.2mm
    arr = arr[50:450, 50:450, :1500]                          # central +/-40 mm lateral, full depth
    f = DEPTH_FACTOR
    nd = 1500 // f
    return arr.reshape(80, 5, 80, 5, nd, f).mean(axis=(1, 3, 5))  # rebin lateral + depth -> (80,80,nd)


def run_engine(energy: float, device, dtype) -> np.ndarray:
    """Single beamlet through pure water; returns dose (zlat, ydepth, xlat)."""
    lut = PyRadPlanIonLUT(LUT_MAT)
    beam = IonSpotBeam.create(
        gantry_angle_deg=0.0,  # beam axis -> +y, so y is depth
        spot_positions_mm=torch.zeros((1, 2), device=device, dtype=dtype),
        spot_weights=torch.full((1,), PART, device=device, dtype=dtype),
        spot_layer_index=torch.zeros((1,), device=device, dtype=torch.long),
        layer_energies_mev=torch.tensor([energy], device=device, dtype=dtype),
        layer_sigmas_mm=torch.tensor([sigma0_mm(energy)], device=device, dtype=dtype),
        # MC beam sits at physical 0 -> half-voxel boundary; centre PB there too (C-0.5)
        iso_center=(float(C) - 0.5, float(N_DEPTH // 2 * DZ), float(C) - 0.5),  # (z,y,x) mm
        sad_mm=1.0e5, requires_grad=False,
    )
    seq = IonSpotBeamSequence.from_beams([beam])
    eng = IonDoseEngine(
        machine_config=MachineConfig(tpr_20_10=0.7, number_of_leaf_pairs=40),
        lut=lut, dose_grid_spacing=(1.0, DZ, 1.0),
        dose_grid_shape=(N_LAT, N_DEPTH, N_LAT), beam_template=seq,
        device=device, dtype=dtype, field_size=(160, 160),
        lateral_model=LATERAL_MODEL,
        heterogeneous_mcs=HET_MCS, material_radiation_length=MAT_X0,
    )
    water = torch.ones((1, N_LAT, N_DEPTH, N_LAT), device=device, dtype=dtype)
    with torch.inference_mode():
        dose = eng.compute_dose_bev_lattice_sparse_batch(
            seq, water, mass_density_image=water, overwrite=False)[0]
    return dose.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--match", type=str, default=None, help="only energies whose filename contains this substring")
    ap.add_argument("--lut", type=str, default=None, help="override LUT .mat path")
    ap.add_argument("--lateral-model", type=str, default="gauss", choices=["gauss", "gauss_double"])
    ap.add_argument("--split-mode", type=str, default=None, choices=["single", "split"],
                    help="override engine SPLITTING_MODE")
    ap.add_argument("--dz", type=float, default=0.2,
                    help="depth voxel size mm (multiple of 0.2); 1.0 -> 1x1x1 mm grid (faster)")
    ap.add_argument("--gamma", type=str, default="3/3,2/2",
                    help="comma list of dd%%/dta_mm local-gamma criteria, e.g. '3/3,2/2'")
    ap.add_argument("--heterogeneous-mcs", action="store_true",
                    help="enable Fermi-Eyges/Kanematsu heterogeneity-aware MCS (Fuchs dH)")
    ap.add_argument("--material-radiation-length", action="store_true",
                    help="use per-material radiation length for MCS; needs --heterogeneous-mcs")
    ap.add_argument("--mc-dir", type=str, default=None,
                    help="Directory of water-phantom MC edep files (default: the 1e8 output dir).")
    args = ap.parse_args()
    global LUT_MAT, LATERAL_MODEL, DZ, N_DEPTH, DEPTH_FACTOR, HET_MCS, MAT_X0, MC_DIR
    if args.mc_dir:
        MC_DIR = Path(args.mc_dir)
    HET_MCS = args.heterogeneous_mcs
    MAT_X0 = args.material_radiation_length
    if args.lut:
        LUT_MAT = Path(args.lut)
    LATERAL_MODEL = args.lateral_model
    if args.split_mode:
        ion_dose_engine.SPLITTING_MODE = args.split_mode
    DEPTH_FACTOR = max(1, round(args.dz / 0.2))
    DZ = 0.2 * DEPTH_FACTOR
    N_DEPTH = 1500 // DEPTH_FACTOR
    gamma_crits = [tuple(float(v) for v in c.split("/")) for c in args.gamma.split(",") if c.strip()]
    print(f"lateral_model={LATERAL_MODEL}  split_mode={ion_dose_engine.SPLITTING_MODE}  "
          f"lut={LUT_MAT.name}  grid=1x{DZ:g}x1mm  gamma={args.gamma}  "
          f"het_mcs={HET_MCS}  material_x0={MAT_X0}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    # All metrics are IDD-/peak-normalised, so edep == dose in uniform water (dose =
    # edep / voxel_mass, one global scalar). Prefer edep (complete set under output/);
    # fall back to dose for output_1e7.
    files = glob.glob(str(MC_DIR / "*__edep.mhd")) or glob.glob(str(MC_DIR / "*__dose.mhd"))
    files = sorted(files, key=lambda p: float(re.search(r"_([\d.]+)MeV", p).group(1)))
    if args.match:
        tokens = [t.strip() for t in args.match.split(",") if t.strip()]
        files = [f for f in files if any(t in f for t in tokens)]
    if args.limit:
        files = files[:: max(1, len(files) // args.limit)][: args.limit]

    rows = []
    for path in files:
        energy = float(re.search(r"_([\d.]+)MeV", path).group(1))
        mc = load_mc(Path(path))                              # (zlat, ylat, depth)
        pb = run_engine(energy, device, dtype)               # (zlat, ydepth, xlat)
        # laterally integrated depth dose over the +/-37 mm kernel window
        idd_mc = mc[WIN, WIN, :].sum((0, 1))
        idd_pb = pb[WIN, :, WIN].sum((0, 2))
        smc, spb = idd_mc.max(), idd_pb.max()
        if smc <= 0 or spb <= 0:
            continue
        # central depth-lateral plane (mean of the two central voxels straddling the
        # beam centre, so MC and PB are sampled at the same physical axis), IDD-norm -> (lateral, depth)
        pmc = 0.5 * (mc[:, C - 1, :] + mc[:, C, :]) / smc
        ppb = 0.5 * (pb[:, :, C - 1] + pb[:, :, C]) / spb
        diff = ppb - pmc
        peak = pmc[WIN, :].max()
        mask = pmc[WIN, :] > 0.1 * peak
        mae = float(np.abs(diff[WIN, :][mask]).mean()) / peak * 100.0  # % of peak dose
        maxerr = float(np.abs(diff[WIN, :][mask]).max()) / peak * 100.0  # peak (max) error % of peak
        # central-axis depth profile in shared IDD-normalised units (reveals on-axis overdose)
        ax_mc = 0.5 * (pmc[C - 1, :] + pmc[C, :])
        ax_pb = 0.5 * (ppb[C - 1, :] + ppb[C, :])

        # 3D local-gamma pass rate over the full volume. Align PB (zlat, depth, xlat)
        # to MC layout (zlat, lat, depth); normalise each to its own peak (shape gamma).
        pb_g = np.transpose(pb, (0, 2, 1))                    # (zlat, lat, depth)
        mc_n3 = (mc / mc.max()).astype(np.float32)
        pb_n3 = (pb_g / pb_g.max()).astype(np.float32)
        gammas = [
            local_gamma_pass_rate(
                mc_n3, pb_n3, voxel_size_mm=(1.0, 1.0, DZ),
                dose_threshold_pct=dd, dist_threshold_mm=dta,
                prescription_gy=1.0, lower_cutoff_pct=10.0, device=device,
            )
            for dd, dta in gamma_crits
        ]
        rows.append((energy, mae, maxerr, gammas))

        z = np.arange(N_DEPTH) * DZ
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        ax[0, 0].plot(z, idd_mc / smc, label="MC"); ax[0, 0].plot(z, idd_pb / spb, "--", label="PB")
        ax[0, 0].set_title(f"IDD (+/-37mm)  {energy:.1f} MeV"); ax[0, 0].legend(); ax[0, 0].set_xlabel("depth [mm]")
        ax[0, 1].plot(z, ax_mc, label="MC"); ax[0, 1].plot(z, ax_pb, "--", label="PB")
        ax[0, 1].set_title("central-axis depth (IDD-norm)"); ax[0, 1].legend(); ax[0, 1].set_xlabel("depth [mm]")
        ax[0, 2].axis("off"); ax[0, 2].text(0.1, 0.5, f"MAE = {mae:.2f}% of peak", fontsize=14)
        vmax = max(pmc.max(), ppb.max())
        ext = [0, N_DEPTH * DZ, 0, N_LAT]  # depth [mm], lateral [mm]
        ax[1, 0].imshow(pmc, aspect="auto", vmin=0, vmax=vmax, cmap="inferno", extent=ext); ax[1, 0].set_title("MC")
        ax[1, 1].imshow(ppb, aspect="auto", vmin=0, vmax=vmax, cmap="inferno", extent=ext); ax[1, 1].set_title("PB")
        dm = np.abs(diff).max()
        im = ax[1, 2].imshow(diff, aspect="auto", vmin=-dm, vmax=dm, cmap="bwr", extent=ext); ax[1, 2].set_title("PB - MC")
        fig.colorbar(im, ax=ax[1, 2])
        for a in ax[1]: a.set_xlabel("depth [mm]"); a.set_ylabel("lateral [mm]")
        fig.suptitle(f"PB vs MC water phantom  |  {energy:.2f} MeV  |  MAE={mae:.2f}%")
        fig.tight_layout(); fig.savefig(OUT / f"e{energy:07.2f}.png", dpi=120); plt.close(fig)
        gtxt = "  ".join(f"g{dd:g}/{dta:g}={g:5.1f}%" for (dd, dta), g in zip(gamma_crits, gammas))
        print(f"{energy:8.2f} MeV   MAE={mae:5.2f}%peak   peakErr={maxerr:5.2f}%peak   {gtxt}")

    rows.sort()
    e = [r[0] for r in rows]; m = [r[1] for r in rows]; pk = [r[2] for r in rows]
    gmat = np.array([r[3] for r in rows]) if rows else np.zeros((0, len(gamma_crits)))
    fig, a = plt.subplots(figsize=(10, 5))
    a.bar(range(len(e)), m); a.set_xticks(range(len(e))); a.set_xticklabels([f"{x:.0f}" for x in e], rotation=90)
    a.set_ylabel("central-plane MAE [% of peak dose]"); a.set_xlabel("energy [MeV]")
    a.set_title(f"PB vs MC water phantom  |  mean MAE = {np.mean(m):.2f}%")
    fig.tight_layout(); fig.savefig(OUT / "summary_mae.png", dpi=120); plt.close(fig)
    print(f"\nmean MAE over {len(rows)} energies = {np.mean(m):.2f}% of peak")
    print(f"mean peak(max) error over {len(rows)} energies = {np.mean(pk):.2f}% of peak  (worst {np.max(pk):.2f}%)")
    for j, (dd, dta) in enumerate(gamma_crits):
        print(f"mean gamma {dd:g}%/{dta:g}mm over {len(rows)} energies = {gmat[:, j].mean():.1f}%  (worst {gmat[:, j].min():.1f}%)")
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
