"""Differentiable LUT calibration against MC in a water phantom (single energy).

Makes the LUT depth curves (Z, sigma, sigma1, sigma2, weight) learnable torch
parameters and minimises |engine(LUT) - MC| on the 3D water dose by backprop
through the dose engine, with a depth-smoothness prior. Writes an optimised LUT.

    uv run python scripts/optimize_lut_water.py --energy 164.4532 --iters 300
"""
from __future__ import annotations

import argparse, os, re, sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pydosert-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch, glob, SimpleITK as sitk

# The engine wraps its forward in `with torch.inference_mode()`, which hard-disables
# autograd. Neutralise it so gradients flow to the LUT parameters during calibration.
torch.inference_mode = torch.enable_grad

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pydose_rt.data.machine_config import MachineConfig
from pydose_rt.data.ion_beam import IonSpotBeam, IonSpotBeamSequence
from pydose_rt.engine.ion_dose_engine import IonDoseEngine
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT
from fit_proton_lut import load_reference_entry, write_dense_lut_mat

ROOT = Path(__file__).parent.parent
MC_DIR: Path | None = None  # set from --mc-dir
BASE_LUT = ROOT / "example_data" / "mc_fit_smooth" / "lut_fast_3d_1e9.mat"
BEAM_PARAMS = __import__("json").loads((ROOT / "example_data" / "beam_parameters.json").read_text())
OUT = ROOT / "out" / "pb_vs_mc_water_1e8"; OUT.mkdir(parents=True, exist_ok=True)

# Lateral grid (1 mm bins); depth grid set from CLI. N_LAT is the engine/MC lateral extent
# in mm-bins, C its centre, W the half-width of the IDD normalisation window in `widx`.
# Defaults reproduce the historical setting: MC cropped to +/-40 mm of its +/-50 mm phantom,
# and the IDD window at +/-37 mm to match the LUT's --kernel-width-mm 74. Both are widened
# together by --lat-half-mm / --widx-half-mm so a wider-window LUT is calibrated against a
# target that can actually see the halo it added; leaving them at 37 would re-fit Z back
# down to the +/-37 mm integral and cancel the wider window entirely.
N_LAT, C, W = 80, 40, 37
MASK_FLOOR = 0.005  # loss mask: MC voxels below this fraction of peak are excluded


def sigma0_mm(e):
    t = BEAM_PARAMS["proton"]["energy_table"]
    return float(np.interp(e, [x["energy_mev"] for x in t], [x["sigma_spot_mm"] for x in t]))


def load_mc(path, f):  # -> (zlat, ylat, depth); depth rebinned by factor f (DZ=0.2*f mm)
    # MC is 500x500 lateral voxels at 0.2 mm (+/-50 mm). Keep the central 2*N_LAT mm and
    # rebin 5 voxels -> 1 mm. lo/hi are voxel indices, symmetric about the phantom centre.
    lo, hi = 250 - 5 * N_LAT // 2, 250 + 5 * N_LAT // 2
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))[lo:hi, lo:hi, :1500]
    nd = 1500 // f
    return arr.reshape(N_LAT, 5, N_LAT, 5, nd, f).mean(axis=(1, 3, 5))


def inv_softplus(y):  # y>0
    return torch.log(torch.expm1(y.clamp_min(1e-6)))


def widx(vol):  # windowed IDD peak normaliser; vol (lat,lat,depth)
    return vol[C - W:C + W, C - W:C + W, :].sum((0, 1))


def optimize_energy(lut, energy, mc_path, args, dev, dt):
    """Calibrate one energy entry against its water-phantom MC. Returns
    (depths, z, sig, s1, s2, w, let, init_L1, best_L1). Init = base LUT entry."""
    mc = torch.from_numpy(load_mc(Path(mc_path), args.depth_factor).astype(np.float32)).to(dev)  # (zlat,ylat,depth)
    mc_n = mc / widx(mc).max()
    mask = mc_n > MASK_FLOOR * mc_n.max()

    idx = int(np.argmin([abs(e - energy) for e in lut._energies]))
    depth_t = torch.tensor(lut._depths[idx], device=dev, dtype=dt)
    z0 = torch.tensor(lut._idd[idx], device=dev, dtype=dt)
    sig0 = torch.tensor(lut._sigma[idx], device=dev, dtype=dt)
    s10 = torch.tensor(lut._sigma1[idx], device=dev, dtype=dt)
    s20 = torch.tensor(lut._sigma2[idx], device=dev, dtype=dt)
    w0 = torch.tensor(lut._weight[idx], device=dev, dtype=dt)

    # learnable raw params (positivity / [0,1] via transforms; init = base curves)
    p = {
        "z": inv_softplus(z0).clone().requires_grad_(),
        "sig": inv_softplus(sig0).clone().requires_grad_(),
        "s1": inv_softplus(s10).clone().requires_grad_(),
        "gap": inv_softplus((s20 - s10).clamp_min(1e-3)).clone().requires_grad_(),
        "w": torch.logit(w0.clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(),
    }

    def curves():
        z = torch.nn.functional.softplus(p["z"])
        sig = torch.nn.functional.softplus(p["sig"])
        s1 = torch.nn.functional.softplus(p["s1"])
        s2 = s1 + torch.nn.functional.softplus(p["gap"])
        w = torch.sigmoid(p["w"])
        return z, sig, s1, s2, w

    # override LUT getters to interpolate the learnable curves (energy hits this entry)
    def patch():
        z, sig, s1, s2, w = curves()
        lut.get_edep_curve = lambda e, **k: (depth_t, z)
        lut.get_sigma_curve = lambda e, **k: (depth_t, sig)
        lut.get_double_gauss_curves = lambda e, **k: (depth_t, s1, s2, w)

    # Build the beam/engine/water ONCE per energy (not per iteration): the dose engine
    # rebuilds rotation + radiological-depth sampling grids in __init__, which is pure
    # CPU overhead that otherwise starves the GPU when repeated every optimiser step.
    # patch() only rebinds the LUT curve getters (closures over the live params), so the
    # cached engine reads the updated curves each forward.
    beam = IonSpotBeam.create(
        gantry_angle_deg=0.0,
        spot_positions_mm=torch.zeros((1, 2), device=dev, dtype=dt),
        spot_weights=torch.full((1,), 1e7, device=dev, dtype=dt),
        spot_layer_index=torch.zeros((1,), device=dev, dtype=torch.long),
        layer_energies_mev=torch.tensor([energy], device=dev, dtype=dt),
        layer_sigmas_mm=torch.tensor([sigma0_mm(energy)], device=dev, dtype=dt),
        iso_center=(C - 0.5, N_DEPTH // 2 * DZ, C - 0.5), sad_mm=1e5, requires_grad=False,
    )
    seq = IonSpotBeamSequence.from_beams([beam])
    eng = IonDoseEngine(
        machine_config=MachineConfig(tpr_20_10=0.7, number_of_leaf_pairs=40),
        lut=lut, dose_grid_spacing=(1.0, DZ, 1.0), dose_grid_shape=(N_LAT, N_DEPTH, N_LAT),
        beam_template=seq, device=dev, dtype=dt, lateral_model="gauss_double", field_size=(160, 160),
    )
    water = torch.ones((1, N_LAT, N_DEPTH, N_LAT), device=dev, dtype=dt)

    def forward():
        patch()
        dose = eng.compute_dose_bev_lattice_sparse_batch(seq, water, mass_density_image=water, overwrite=False)[0]
        return dose.permute(0, 2, 1)  # (zlat, xlat, ydepth) -> (lat, lat, depth), matches mc

    def loss_fn(pb):
        pb_n = pb / widx(pb).max().clamp_min(1e-12)
        data = (pb_n - mc_n).abs()[mask].mean()
        reg = sum((c[2:] - 2 * c[1:-1] + c[:-2]).pow(2).mean() for c in curves())
        return data + args.smooth * reg, data

    opt = torch.optim.Adam(p.values(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.lr * 0.02)
    hist = []
    best = float("inf"); best_state = None; checked_grad = False
    for it in range(args.iters):
        opt.zero_grad()
        pb = forward()
        loss, data = loss_fn(pb)
        # Skip non-finite steps so one unstable energy can't poison its LUT entry with
        # NaN (a stepped NaN grad corrupts every param). If an energy is non-finite for
        # every iter, params stay at init -> base curves are written (no worse than base).
        if not torch.isfinite(loss):
            if it % 50 == 0:
                print(f"iter {it:4d}  non-finite loss; skipping step")
            sched.step()
            continue
        loss.backward()
        if not checked_grad:
            if p["z"].grad is None or float(p["z"].grad.norm()) == 0.0:
                raise RuntimeError("no gradient flows to LUT params - engine forward is not differentiable here")
            checked_grad = True
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in p.values()):
            opt.zero_grad(); sched.step()
            continue
        opt.step(); sched.step()
        d = float(data); hist.append(d)
        if d < best:  # keep best-so-far (loss is noisy near convergence)
            best = d; best_state = {k: v.detach().clone() for k, v in p.items()}
        if it % 50 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  data_L1={d:.6f}  best={best:.6f}  lr={sched.get_last_lr()[0]:.4f}")
    if best_state is not None:
        for k in p:
            p[k].data.copy_(best_state[k])

    z, sig, s1, s2, w = (c.detach().cpu().numpy() for c in curves())
    ref = load_reference_entry(args.base_lut, energy)
    let = np.interp(lut._depths[idx], ref["depths_mm"], ref["LET"])
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return lut._depths[idx], z, sig, s1, s2, w, let, hist[0], best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=str, default="164.4532", help="single-energy mode: MC filename substring")
    ap.add_argument("--all", action="store_true", help="optimize every MC energy into one new LUT")
    ap.add_argument("--match", type=str, default=None,
                    help="comma-separated filename substrings; optimize only these energies "
                         "(implies incremental merge into an existing --out-lut)")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing --out-lut instead of overwriting it from --base-lut "
                         "(other energies' entries are preserved). Auto-enabled by --match.")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--smooth", type=float, default=2e-4, help="depth-smoothness weight")
    ap.add_argument("--depth-factor", type=int, default=2, help="MC depth rebin factor; DZ=0.2*f mm (1->0.2mm, 2->0.4mm, 5->1.0mm)")
    ap.add_argument("--base-lut", type=str, default=str(BASE_LUT))
    ap.add_argument("--out-lut", type=str, default=str(ROOT / "example_data/mc_fit_smooth/lut_fast_3d_1e8_opt.mat"))
    ap.add_argument("--lat-half-mm", type=int, default=40,
                    help="Half-width (mm) of the MC/engine lateral grid. Max 50 (the phantom "
                         "is +/-50 mm). Must be raised with the LUT's --kernel-width-mm.")
    ap.add_argument("--widx-half-mm", type=int, default=37,
                    help="Half-width (mm) of the IDD normalisation window in widx(). Must "
                         "match half the LUT's --kernel-width-mm, else the calibration re-fits "
                         "Z back to the narrower integral and cancels a wider window.")
    ap.add_argument("--mask-floor", type=float, default=0.005,
                    help="Loss mask: MC voxels below this fraction of peak are excluded. The "
                         "default hides the far halo from the calibration entirely.")
    ap.add_argument("--mc-dir", type=str, required=True,
                    help="Directory of water-phantom MC edep files (default: the 1e8 output dir).")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32
    global N_DEPTH, DZ, N_LAT, C, W, MASK_FLOOR, MC_DIR
    if args.mc_dir:
        MC_DIR = Path(args.mc_dir)
    DZ = 0.2 * args.depth_factor
    N_DEPTH = 1500 // args.depth_factor
    if not 1 <= int(args.lat_half_mm) <= 50:
        raise SystemExit(f"--lat-half-mm must be in 1..50 (phantom is +/-50 mm), got {args.lat_half_mm}")
    if int(args.widx_half_mm) > int(args.lat_half_mm):
        raise SystemExit(f"--widx-half-mm ({args.widx_half_mm}) cannot exceed --lat-half-mm ({args.lat_half_mm})")
    N_LAT = 2 * int(args.lat_half_mm)
    C = int(args.lat_half_mm)
    W = int(args.widx_half_mm)
    MASK_FLOOR = float(args.mask_floor)
    print(f"lateral grid +/-{C} mm ({N_LAT} bins) | widx window +/-{W} mm | mask floor {MASK_FLOOR:g}", flush=True)

    lut = PyRadPlanIonLUT(args.base_lut)
    # Loss is peak-normalised (pb_n / mc_n), so edep and dose are equivalent in a
    # uniform water phantom (dose = edep / voxel_mass, one global scalar). Prefer
    # edep (the complete MC set under output/); fall back to dose for output_1e7.
    files = glob.glob(str(MC_DIR / "*__edep.mhd")) or glob.glob(str(MC_DIR / "*__dose.mhd"))
    files = sorted(files, key=lambda p: float(re.search(r"_([\d.]+)MeV", p).group(1)))
    merge = args.merge or bool(args.match)
    if args.match:
        tokens = [t.strip() for t in args.match.split(",") if t.strip()]
        files = [f for f in files if any(t in Path(f).name for t in tokens)]
        if not files:
            raise SystemExit(f"--match {args.match!r} matched no MC files in {MC_DIR}")
    elif not args.all:
        files = [f for f in files if args.energy in f][:1]

    # Seed the output LUT. In merge mode keep the existing --out-lut (preserve already-optimized
    # entries; write_dense_lut_mat updates only the processed energies in place). Otherwise the
    # base curves are the init/reference for a full (re)build.
    import shutil
    if not merge:
        shutil.copy(args.base_lut, args.out_lut)
    elif not Path(args.out_lut).exists():
        shutil.copy(args.base_lut, args.out_lut)  # nothing to merge into yet; start from base
    rows = []
    for n, mc_path in enumerate(files, 1):
        energy = float(re.search(r"_([\d.]+)MeV", mc_path).group(1))
        print(f"\n=== [{n}/{len(files)}] {energy:.4f} MeV ===")
        depths, z, sig, s1, s2, w, let, init_L1, best_L1 = optimize_energy(lut, energy, mc_path, args, dev, dt)
        write_dense_lut_mat(Path(args.base_lut), Path(args.out_lut), energy, depths, z, sig, s1, s2, w, let)
        rows.append((energy, init_L1, best_L1))
        print(f"  {energy:.2f} MeV  L1 {init_L1:.6f} -> {best_L1:.6f}  written")
    print(f"\nwrote {args.out_lut}  ({len(rows)} entries; init = {Path(args.base_lut).name})")

    e = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(e, [r[1] for r in rows], "o-", label="init (base LUT)")
    ax.plot(e, [r[2] for r in rows], "s-", label="optimized")
    ax.set_xlabel("energy [MeV]"); ax.set_ylabel("masked L1 (norm)"); ax.set_yscale("log"); ax.legend()
    ax.set_title(f"Differentiable LUT calibration ({len(rows)} energies)")
    fig.tight_layout(); fig.savefig(OUT / "opt_all_energies.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
