# Heterogeneity-aware lateral scattering (Fermi–Eyges / Fuchs dH)

Adds a physically-correct, heterogeneity-aware **multiple-Coulomb-scattering (MCS)**
lateral spread to the ion pencil-beam engine, replacing the water-equivalent-depth σ
lookup that under-broadens the beam in low-density tissue (lung). Implemented in
`src/pydose_rt/engine/ion_dose_engine.py`, off by default, enabled per-engine.

## Why

The engine's lateral transport σ was a pure water-equivalent-depth lookup,
`σ = lut.get_sigma(WEQ)`. That uses **water's** geometric scaling for the MCS lever
arm, so wherever the beam crosses low-density lung (a longer *geometric* path per unit
WEQ) it **under-predicts** the lateral spread. Measured directly on patient 1THB002
(thorax), the engine under-broadened the lateral RMS by ~2 mm, and the 1%/1mm gamma
sat well below the homogeneous-case level.

This is a known limitation of WEQ-only pencil beams (Schaffner 1999, Szymanowski &
Oelfke 2002) and is exactly what Fuchs et al. 2012 (the algorithm this engine descends
from) fixed with Gottschalk's non-local scattering power inside Fermi–Eyges transport.

## What it does

Two orthogonal layers (Gottschalk/Fuchs split):

1. **MCS core** — the narrow Gaussian. Its σ now gets a heterogeneity correction
   (this feature). Reduces to the LUT σ exactly in homogeneous tissue.
2. **Nuclear halo** — the broad Gaussian (double-Gaussian model). A *different*
   physical process (nuclear secondaries); unchanged, still from the LUT.

### The correction (excess form)

The lateral variance is the Fermi–Eyges spatial integral
`σ_x²(z) = ∫₀ᶻ (z−u)² T(u) du`, with geometric lever arm `(z−u)` and scattering power
`T`. We keep the LUT σ as the water baseline and add only the **geometric-vs-WEQ
lever-arm excess**:

```
σ_het²(z_d) = σ_LUT²(w_d) + Σ_{i≤d} [ (z_d − z_i)² − (w_d − w_i)² ] · ΔΘ²_i
```

- `z` = geometric depth (`index · dose_grid_spacing[1]`), `w` = WEQ depth.
- The bracket is **identically 0 where geometric == WEQ** (homogeneous tissue), so
  **water and homogeneous patients are unchanged by construction** — not approximately.
- In low-density lung the geometric gap exceeds the WEQ gap → bracket > 0 → broadens.
- Fully vectorized as five `cumsum`s over depth.

### The scattering power ΔΘ² (Kanematsu differential-Highland, closed form)

No LUT-σ inversion, no curve interpolation — `ΔΘ²` is computed directly from the local
residual range and WEQ (Kanematsu, NIM B 2008, Eq. 14/16/17; the form Fuchs uses):

```
ΔΘ²_i = f_dH(ℓ_i) · (E_s² / pv_i²) · (ΔWEQ_i,cm / ρX₀)
E_s   = 15.0 MeV
f_dH(ℓ) ≈ 0.970 · (1 + ln ℓ / 20.7) · (1 + ln ℓ / 22.7)      # non-local single-scatter correction
ℓ      = WEQ_cm / ρX₀                                          # radiative path length
(pv/MeV)² = (R / 4.67e-4 cm)^1.08 ,  R = R₀ − WEQ              # residual range -> momentum·velocity
R₀     = 4.67e-4 · (pv₀²)^(1/1.08) ,  pv₀ = E(E+2m_p)/(E+m_p)  # beamlet water range from energy
```

The stopping-point divergence (`pv → 0` past the range) is capped (`R.clamp_min(0.1cm)`
and scattering zeroed beyond the range); harmless anyway since `ray_edep ≈ 0` there.

### Material radiation length (optional sub-feature)

By default `ρX₀ = 36.08 g/cm²` (water). With `material_radiation_length=True` the engine
uses the **true per-material** ρX₀ from the DoseRAD Geant4 compositions
(`DICOMphantom.cc`), computed via PDG element radiation lengths + the compound rule and
stored in `materials.py` (`GEANT4_DENSITY_GRID` / `GEANT4_RHOX0_GRID`). The local
material is read from the WEQ gradient (`ΔWEQ/Δz` = physical density, since the engine's
WEQ = ∫ρ dx), so **no extra BEV channel or input is needed**. Bone (ρX₀ ≈ 26) scatters
more per g/cm² than water; soft tissue (ρX₀ ≈ 40, H/C-rich) less.

## Usage

```python
engine = IonDoseEngine(
    ...,
    lateral_model="gauss_double",   # keep the nuclear halo
    heterogeneous_mcs=True,         # the MCS heterogeneity correction
    material_radiation_length=False # optional: True for thick-bone sites
)
```

CLI (both `scripts/evaluate_doserad_proton_case.py` and
`scripts/benchmark_pb_vs_mc_water.py`):

```
--heterogeneous-mcs [--material-radiation-length]
```

## Validation (local γ 1%/1mm, full plan)

| configuration | thorax 1THB002 | abdomen 1ABB006 |
|---|---|---|
| base LUT, no correction | 74.87% | 94.08% |
| heterogeneous_mcs, gauss, base LUT | 80.79% | 94.21% |
| **gauss_double + optimized LUT + heterogeneous_mcs** | **81.81%** | **95.44%** |
| + material_radiation_length | 81.77% | 95.45% |

- **Thorax (heterogeneous): 74.87 → 81.81%** (+6.9).
- **Abdomen (homogeneous): unchanged** (94.08 → 95.44 from the LUT/halo, +0.0 from MCS) —
  the excess is 0 by construction.
- **material_radiation_length** is negligible here (lung is water-equivalent in X₀, ribs
  are thin); its payoff is on **thick-bone sites** (H&N, pelvis), not thorax.

The gate that matters: validate the **excess/scattering-power magnitude offline across
energies**. The water benchmark cannot catch a magnitude bug here, because the excess is
identically 0 in water — an earlier LUT-inversion attempt passed the water benchmark with
a wrong magnitude for exactly this reason.

## Performance

Closed-form, no inversion. Five `cumsum`s + per-voxel `pow` over (sub-beams × depth).
At the 0.2 mm/D=1500 water grid `lattice_pb` 148 → 217 ms (+68 ms); scales with depth
voxels, so ~10–15 ms on clinical patient grids (negligible). The `pow((R/λ),0.54)` is the
hot op and can be replaced by `exp(log)`/a fit if needed.

## Physics references

- B. Gottschalk, *On the scattering power of radiotherapy protons*, Med. Phys. 37 (2010) —
  the non-local scattering power (differential Molière/Highland).
- N. Kanematsu, *Alternative scattering power for the Gaussian beam model…*, NIM B (2008) —
  the differential-Highland `T_dH` and the R–pv relation used here.
- H. Fuchs et al., *A pencil beam algorithm for helium ion beam therapy*, Med. Phys. 39
  (2012) — uses Gottschalk non-local scattering + Fermi–Eyges; this engine's lineage.
- H. Szymanowski & U. Oelfke, *Two-dimensional pencil beam scaling…*, PMB 47 (2002) —
  the proton analogue (water-σ scaling for heterogeneity).

See also `docs/proton_split_kernel_water_calibration.md` (the variance fix, halo
restructure, and all-43 water LUT optimization that this builds on).
