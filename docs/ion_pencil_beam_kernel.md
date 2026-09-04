# Ion pencil-beam kernel

The proton kernel is the Hong/pyRadPlan analytic pencil beam, evaluated on the BEV
lattice by `IonPencilBeamModel` (`src/pydose_rt/physics/kernels/ion_pencil_beam_model.py`)
and driven by `IonDoseEngine.compute_layer_edep`.

| Component | Source |
|---|---|
| Machine model | `PyRadPlanIonLUT` over `example_data/pyradplan/protons_Generic.mat`, refit against water-phantom MC by `scripts/export_proton_lut_fast.py` + `scripts/optimize_lut_water.py` |
| Depth dose | `LUT.get_edep(energy, kernel_depth)`, with `kernel_depth` taken per BEV column from the full-BEV WEQ (see [`weq_computation.md`](weq_computation.md)) |
| Lateral spread | `LUT.get_sigma(energy, kernel_depth)`, combined in quadrature with the spot sigma and integrated over each voxel cell with `erf` rather than point-sampled |
| Lateral model | `gauss` (single) or `gauss_double` (narrow MCS core + broad nuclear halo). The shipped model uses `gauss_double`. |
| Heterogeneity | optional Fermi–Eyges correction on the core sigma — see [`proton_heterogeneous_mcs.md`](proton_heterogeneous_mcs.md) |
| Patient input | `input_kind="doserad"`, i.e. density derived from the challenge HLUT |
| Dose conversion | fixed water convention (`MEV_CM2_PER_G_TO_GY_MM2`) |

The sub-beam splitting path, the halo restructure and the water calibration that produced
the shipped LUT are described in
[`proton_split_kernel_water_calibration.md`](proton_split_kernel_water_calibration.md).
