# WEQ Computation in PyDoseRT

## Dense Ion (Proton) Engine — Lateral WEQ for Depth Dose

1. `RadiologicalDepthLayer.forward(density_image)` traces **one ray per beam** through
   the patient CT via precomputed indices along the beam axis (through the isocenter).
   It samples density via trilinear interpolation at D sample points, then computes:
   ```
   WEQ = cumsum(density) * step_size
   ```
   Result: `[B*G, D]` — one WEQ profile per beam.

2. `RadiologicalDepthLayer.forward_bev(density_image)` also computes a full BEV
   WEQ volume `[B*G, D, H, W]`. The dense ion engine uses this per-column WEQ
   for the depth-dose term:
   ```
   kernel_depth(h,w) = WEQ_bev(d,h,w) + offset - kernel_offset(energy)
   edep_map(d,h,w) = LUT.get_edep(energy, kernel_depth(h,w))
   fluence_volume(d,h,w) = fluence(h,w) * edep_map(d,h,w)
   ```

3. `IonPencilBeamModel.evaluate_lateral_cell_weights` (called from
   `IonDoseEngine.compute_layer_edep`) evaluates the lateral term **directly on the
   BEV lattice**, one weight per `(d, h, w)` cell, from the *same* per-column
   `kernel_depth` — there is no separate convolution kernel and no central-ray
   `[G, D]` WEQ in this path:
   ```
   sigma_transport(d,h,w) = LUT.get_sigma(energy, kernel_depth(d,h,w))
   sigma_total_{x,y}      = sqrt(sigma_transport^2 + sigma_spot_{x,y}^2)
   cell_weights(d,h,w)    = gaussian_cell(sigma_total, x_mm, y_mm)   # cell-integrated
   contribution           = spot_weight * edep_map * cell_weights / sum_hw(cell_weights)
   ```
   Each column therefore carries its own sigma, so lateral spread and Bragg-peak
   depth respond to the same tissue. The double-Gaussian core+halo and the
   heterogeneous-MCS sub-beam split (`_make_subbeam_grid`, `_fermi_eyges_excess`)
   both act inside this step.

4. The result is accumulated per beamlet in BEV and rotated back to the patient
   grid (`_rotate_beamlet_to_patient`, `_finalize_patient_dose_multi_crop`).
   The convolution-based `BeamWiseConvolutionalLayer` path is the **photon** engine's
   (see below); the ion engine does not use it.

**No central-ray lateral assumption in the dense ion path.** An earlier version of
this engine built one lateral kernel per depth slice from the central-ray WEQ and
applied it as a grouped convolution, which left lateral scatter blind to
off-axis heterogeneity. That is no longer the case: sigma is evaluated per column.

## Photon Engine — Central Ray Only

The photon engine still uses the original central-ray convolution pattern:

1. `RadiologicalDepthLayer.forward(density_image)` → `[B*G, D]` central-ray WEQ
   (same layer, same code).
2. `PencilBeamKernelLayer(radiological_depth)` → `[kH, kW, B*G, D]` kernels.
   Uses TPR(20/10) instead of a LUT, but same concept: one kernel per depth,
   shared laterally.
3. `BeamWiseConvolutionalLayer` convolves fluence with kernels — same grouped
   convolution.

**Photons remain central-ray-only for WEQ.** The dense ion lateral-WEQ depth-dose
correction is not applied to photon kernels.

## Full-BEV WEQ via `forward_bev`

`RadiologicalDepthLayer.forward_bev()` computes WEQ at **every BEV voxel**
`[B*G, D, H, W]` by inverse-rotating patient density to BEV and doing cumsum per
column. It is also what the correction network's input features are built from.
The dense ion PB dose computation uses this full-BEV WEQ for **both** the
depth-dose/`edep` term and the lateral sigma, which is what dropping the grouped
convolution in favour of direct per-cell evaluation bought.
