# Proton Split-Kernel Halo Restructure & Water LUT Calibration

Why the beam-splitting kernel is shaped the way it is, and how the shipped machine LUT
was calibrated against water-phantom Monte Carlo. Two problems drove the design:

1. The per-split-ray double-Gaussian halo was inaccurate and slow.
2. The water LUT optimization plateaued at ~0.7% error, which is high — in water the
   pencil beam should reproduce the Monte-Carlo dose to near zero.

Both came from the same subsystem: the **split (beam-splitting) path** of
`IonDoseEngine.compute_layer_edep`, which `optimize_lut_water.py` and
`benchmark_pb_vs_mc_water.py` both run under `SPLITTING_MODE="split"` +
`lateral_model="gauss_double"`.

## What was wrong, and the fixes

### 1. Halo restructured: one broad Gaussian over the full crop

`compute_layer_edep` split branch (`src/pydose_rt/engine/ion_dose_engine.py`).

Previously every sub-beam built BOTH a narrow and a broad Gaussian over the full
`(S, D, H, W)` lattice. The broad nuclear halo is wide and smooth, so resolving it per
sub-beam adds a second huge tensor and approximation error for no accuracy gain.

Now only the **narrow core** is resolved per sub-beam. The **halo is added once over
the full crop** as a single Gaussian centred on the beamlet, convolved with the full
initial spot sigma (`broad σ = sqrt(σ2_center² + σ_spot²)`), with an IDD-conserving
amplitude `Σ_s sub_w · ray_edep_s · w_s`. Energy conservation is exact and identical to
the old per-(S,D) renormalization; in homogeneous media it reduces to the single-mode
double Gaussian.

Microbenchmark at real shapes (S=81, D=1500, 80×80):

| kernel build | time/call | peak VRAM |
|---|---|---|
| old (broad per sub-beam) | 209.7 ms | 15.75 GB (near-OOM on 16 GB 5080) |
| new (halo once) | 37.6 ms | 9.45 GB |

→ ~5.6× faster kernel build, −40% VRAM (this is the kernel-build portion; it dominates
at real multi-beamlet patient scale).

### 2. Sub-beam variance identity bug (the real water-degradation cause)

`_make_subbeam_grid`. The halo restructure alone did **not** fix the worst water case
(36.8 MeV stayed at 2.54%) because at low energy the halo weight ≈ 0 — the error was in
the **narrow core** reconstruction.

The Gaussian-splitting identity requires `σ_spot² = σ_envelope² + σ_sub²`, where
σ_envelope is the width of the sub-beam weight grid and σ_sub is each sub-beam's
residual width. The old code used `σ_envelope = σ_spot` with `σ_sub = σ_spot/√n`,
violating the identity and **over-broadening the reconstructed core by √(1+1/n)**
(~5.4% at n=9) — worst at low energy where the spot is wide and the transport σ small.

Fix: set the weight envelope to `σ_envelope = sqrt(σ_spot² − σ_sub²)`, keeping
`σ_sub = σ_spot/√n`. The reconstruction is then exactly `σ_spot² + σ1_lut²`, matching
single mode.

### 3. erf cell-integration in the split kernels

The split kernels point-sampled `exp()`; single mode uses erf cell-integration
(`IonPencilBeamModel._gaussian_cell_1d`). The split narrow/broad now use the same
`_gauss_cell_1d` so split ≡ single in the homogeneous limit (negligible at 1 mm voxels,
but correct).

### Water benchmark, base LUT (central-plane MAE %peak, gauss_double)

Attribution is against an oracle: `benchmark_pb_vs_mc_water.py --split-mode single` is the
exact double-Gaussian kernel (erf cell-integration, no sub-beam splitting). In a homogeneous
water phantom the split path must reduce to it, so single mode is the ground-truth floor.

| Energy | single (oracle) | split OLD | split NEW |
|---|---|---|---|
| 36.8 MeV | 0.52 | 2.54 | **0.80** |
| 142 MeV | 0.96 | 1.23 | **0.87** |
| 200 MeV | 1.41 | 1.30 | **1.30** |

Full 43-energy sweep (base LUT, new split): mean 0.89%, **NaN-free** (the old halo path
produced a NaN-poisoned mean).

## "Near zero" — it is reachable; the limiter was depth-grid resolution

At 164 MeV (split, gauss_double), benchmarked at native 0.2 mm:

| LUT | MAE %peak |
|---|---|
| base | 0.77 |
| optimized, 1.0 mm depth fit | 0.54 |
| optimized, 0.4 mm depth fit | **0.20** |

The PB−MC residual image for the 1 mm fit showed a sharp strip **at the Bragg-peak
depth** (the distal edge); the 0.4 mm re-fit collapsed the error scale ~7×
(±2e-4 → ±3e-5) and erased the strip, leaving only faint lateral tail-mismatch. A 1 mm
fit cannot sharpen the sub-mm distal falloff that a 0.2 mm benchmark resolves. So the
~0.5% "floor" was **depth-grid resolution in the fit, not a lateral double-Gaussian
limit**. The lateral double Gaussian is not the limiter.

## All-43 optimized result

LUT: `example_data/mc_fit_smooth/lut_fast_3d_1e8_opt.mat` — the shipped machine model
(`optimize_lut_water.py --all --iters 150 --depth-factor 2`; L1 plateaus by ~iter 100).
A later refit on higher-statistics 1e9-history MC is in the same directory as
`lut_fast_3d_1e9_opt.mat`.

Benchmark: optimized LUT, split + gauss_double, **1×1×1 mm** grid, **1%/1mm** local
gamma (10% cutoff).

**Means over 43 energies:** MAE **0.26 %** of peak · peak(max) error **2.1 %**
(worst 4.5 %) · **γ 1%/1mm = 98.8 %** (worst 97.2 %).

The 42 cleanly-optimized energies: MAE 0.16–0.48 %, γ 1%/1mm 97–100 %. Peak error
(~2 %) is dominated by the Bragg-peak distal edge and rises with energy as the peak
sharpens.

### Optimizer performance

`optimize_lut_water.forward()` was rebuilding the entire `IonDoseEngine` (rotation and
radiological-depth sampling grids) every iteration — hoisted to once per energy. The
dominant remaining cost is (a) loading each 1.5 GB MC volume off the (slow) MC data mount
(I/O, GPU ~4%) and (b) 150 forward+backward passes over a 1.5 GB intermediate;
~80 s/energy, ~40 min for all 43.

## Known issue: 142.0583 MeV optimizer NaN

The optimizer goes **non-finite at every iteration** for this single energy (its
nuclear-halo σ₂ reaches 154 mm, destabilizing the backward). The engine **forward** at
this energy/grid is finite (verified: base-LUT forward nan=0), so this is an
optimizer/gradient issue, not an engine bug.

Mitigations applied:
- Added a non-finite guard to `optimize_energy`: it skips steps with non-finite loss or
  gradients, so one unstable energy can no longer poison its LUT entry with NaN (a
  stepped NaN gradient corrupts every parameter). If an energy is non-finite for every
  iteration, parameters stay at init → the base curves are written (no worse than base).
- The first `--all` run (pre-guard) left 142.06 as an all-NaN entry; the clean **base**
  LUT 142.06 entry was injected into the optimized LUT. It therefore benchmarks at the
  base level (0.81 % MAE, γ 97.2 %) rather than the ~0.2 % of its optimized neighbours.

The backward NaN at 142.06 MeV is not root-caused; the likely candidates are the 154 mm
broad σ, a degenerate `w(depth)`, or an unsafe `sqrt`/division gradient at this energy.

## Limits of this calibration

- **It validates the variance fix, speed and memory — not heterogeneous accuracy.** In
  water the split path reduces to single mode by construction. The halo-once
  approximation takes the broad σ at the central sub-beam's depth, which only matters in
  heterogeneous tissue; the heterogeneity behaviour of the lateral model is covered
  separately in [`proton_heterogeneous_mcs.md`](proton_heterogeneous_mcs.md).
- Going below ~0.2 % would need a 0.2 mm (`--depth-factor 1`) re-fit.
