from __future__ import annotations

import math
import os
import time

import torch
from torch import nn
import torch.nn.functional as F


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _gauss_cell_1d(x_mm: torch.Tensor, sigma_mm: torch.Tensor, half_width_mm: float) -> torch.Tensor:
    """Integral of a unit-area 1-D Gaussian over a cell of width ``2*half_width_mm``
    centred ``x_mm`` from the beam axis. Matches ``IonPencilBeamModel._gaussian_cell_1d``
    so the split path is the exact single-mode kernel in the homogeneous limit."""
    inv = _INV_SQRT2 / sigma_mm
    return 0.5 * (torch.erf((x_mm + half_width_mm) * inv) - torch.erf((x_mm - half_width_mm) * inv))

from pydose_rt.data.ion_beam import IonSpotBeam, IonSpotBeamSequence
from pydose_rt.data.machine_config import MachineConfig
from pydose_rt.layers.BeamRotationLayer import BeamRotationLayer
from pydose_rt.layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from pydose_rt.physics.constants import MEV_CM2_PER_G_TO_GY_MM2
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT, _interp1d
from pydose_rt.physics.kernels.ion_pencil_beam_model import IonPencilBeamModel
from pydose_rt.sparse.ions import IonSparseHooks


SPLITTING_MODE = "split"   # or "single"
N_PER_DIM = 9


class IonDoseEngine(nn.Module):
    """BEV lattice pencil-beam ion dose engine.

    Computes proton pencil-beam dose using per-ray heterogeneous water-equivalent
    depth (WEQ) on a beam's-eye-view (BEV) lattice cropped around each beamlet.
    The main entry point is ``compute_dose_bev_lattice_sparse_batch``.
    """

    def __init__(
        self,
        machine_config: MachineConfig,
        lut: PyRadPlanIonLUT,
        dose_grid_spacing: tuple[float, float, float],
        dose_grid_shape: tuple[int, int, int],
        beam_template: IonSpotBeamSequence | IonSpotBeam | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        lateral_model: str = "gauss",
        transport_step_mm: float | None = None,
        sparse_hooks: IonSparseHooks | None = None,
        field_size: tuple[int, int] = (400, 400),
        heterogeneous_mcs: bool = False,
        material_radiation_length: bool = False,
    ) -> None:
        super().__init__()
        self.machine_config = machine_config
        self.lut = lut
        self.dose_grid_spacing = dose_grid_spacing
        self.dose_grid_shape = dose_grid_shape
        self.device = device
        self.dtype = dtype
        self.lateral_model = lateral_model
        self.transport_step_mm = transport_step_mm
        self.field_size = field_size
        # Heterogeneity-aware multiple-Coulomb-scattering (MCS) lateral spread: adds the
        # Fermi-Eyges geometric lever-arm correction (Kanematsu/Fuchs differential-Highland
        # scattering power) to the narrow core sigma. Off by default.
        self.heterogeneous_mcs = bool(heterogeneous_mcs)
        # Use true per-material radiation length (DoseRAD compositions) instead of water
        # (36.08) for the MCS scattering power. Only relevant when heterogeneous_mcs is on.
        self.material_radiation_length = bool(material_radiation_length)
        object.__setattr__(self, "sparse_hooks", sparse_hooks or IonSparseHooks())
        self._patient_dose_density_threshold_g_cm3 = 0.03
        # Optional explicit dose-scoring mask; see set_patient_dose_mask.
        self._patient_dose_mask: torch.Tensor | None = None

        self.number_of_beams: int | None = None
        self.gantry_angles: torch.Tensor | None = None
        self.iso_center: tuple[float, float, float] | None = None
        self.iso_centers: torch.Tensor | None = None
        self.sad_values_mm: torch.Tensor | None = None
        self.layers_initialized = False

        self._initialize_layers(beam_template)

    def _set_device_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.device is None:
            self.device = device
        if self.dtype is None:
            self.dtype = dtype

    def _initialize_layers(
        self,
        new_beam_data: IonSpotBeamSequence | IonSpotBeam | None,
        overwrite: bool = False,
    ) -> None:
        if new_beam_data is None:
            return

        if isinstance(new_beam_data, IonSpotBeam):
            new_beam_data = IonSpotBeamSequence.from_beams([new_beam_data])

        self._set_device_dtype(new_beam_data.device, new_beam_data.dtype)

        if not overwrite and self.layers_initialized:
            same_beam_count = self.number_of_beams == len(new_beam_data)
            same_angles = (
                self.gantry_angles is not None
                and self.gantry_angles.shape == new_beam_data.gantry_angles.shape
                and torch.allclose(self.gantry_angles, new_beam_data.gantry_angles.to(self.gantry_angles))
            )
            same_iso = (
                self.iso_centers is not None
                and self.iso_centers.shape == new_beam_data.iso_centers.shape
                and torch.allclose(
                    self.iso_centers,
                    new_beam_data.iso_centers.to(device=self.iso_centers.device, dtype=self.iso_centers.dtype),
                )
            )
            same_sad = (
                self.sad_values_mm is not None
                and self.sad_values_mm.shape == new_beam_data.sad_values_mm.shape
                and torch.allclose(
                    self.sad_values_mm,
                    new_beam_data.sad_values_mm.to(device=self.sad_values_mm.device, dtype=self.sad_values_mm.dtype),
                )
            )
            if same_beam_count and same_angles and same_iso and same_sad:
                return

        self.number_of_beams = len(new_beam_data)
        self.gantry_angles = new_beam_data.gantry_angles.to(device=self.device, dtype=self.dtype)
        self.iso_center = new_beam_data.iso_center
        self.iso_centers = new_beam_data.iso_centers.to(device=self.device, dtype=self.dtype)
        self.sad_values_mm = new_beam_data.sad_values_mm.to(device=self.device, dtype=self.dtype)
        self._initialize_dense_layers(new_beam_data)
        self.layers_initialized = True

    def _initialize_dense_layers(self, beam_data: IonSpotBeamSequence) -> None:
        _H, _D, _W = self.dose_grid_shape
        res_h, _res_d, res_w = self.dose_grid_spacing

        if self.iso_center is None:
            iso_for_layers = self.iso_centers.to(device=self.device, dtype=self.dtype)
        else:
            iso_for_layers = self.iso_center

        self.rad_depth_layer = RadiologicalDepthLayer(
            machine_config=self.machine_config,
            resolution=self.dose_grid_spacing,
            ct_array_shape=self.dose_grid_shape,
            gantry_angles=self.gantry_angles.tolist(),
            iso_center=iso_for_layers,
            depth_origin="entry",
            device=self.device,
            dtype=self.dtype,
        )
        self.bev_lattice_model = IonPencilBeamModel(
            lut=self.lut,
            energy_mev=float(self.lut.available_energies[0]),
            resolution=(res_h, float(self.transport_step_mm or _res_d), res_w),
            lateral_model=self.lateral_model,
        )
        self.rotation_layer = BeamRotationLayer(
            self.machine_config,
            ct_array_shape=self.dose_grid_shape,
            iso_center=iso_for_layers,
            resolution=self.dose_grid_spacing,
            gantry_angles=self.gantry_angles,
            depth_origin="entry",
            device=self.device,
            dtype=self.dtype,
        )

    @staticmethod
    def _dense_crop_slices(center: float, full_size: int, crop_size: int) -> tuple[slice, slice, slice]:
        crop_size = int(crop_size)
        center_i = int(round(float(center)))
        target_lo = center_i - crop_size // 2
        target_hi = target_lo + crop_size
        src_lo = max(target_lo, 0)
        src_hi = min(target_hi, int(full_size))
        dst_lo = src_lo - target_lo
        dst_hi = dst_lo + max(src_hi - src_lo, 0)
        return slice(src_lo, src_hi), slice(dst_lo, dst_hi), slice(target_lo, target_hi)

    def _build_dense_bev_crop(
        self,
        center_h: float,
        center_w: float,
        size_h: int,
        size_w: int,
    ) -> dict:
        full_h, _full_d, full_w = self.dose_grid_shape
        size_h = max(1, min(int(size_h), int(full_h)))
        size_w = max(1, min(int(size_w), int(full_w)))
        h_src, h_dst, h_target = self._dense_crop_slices(center_h, int(full_h), size_h)
        w_src, w_dst, w_target = self._dense_crop_slices(center_w, int(full_w), size_w)
        return {
            "shape_hw": (size_h, size_w),
            "full_shape_hw": (int(full_h), int(full_w)),
            "h_src": h_src,
            "h_dst": h_dst,
            "h_target": h_target,
            "w_src": w_src,
            "w_dst": w_dst,
            "w_target": w_target,
            "target_h_start": int(h_target.start),
            "target_w_start": int(w_target.start),
        }

    def _dense_forward_bev_multi_crop(
        self,
        density_image: torch.Tensor,
        crops: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, full_h, D, full_w = density_image.shape
        G = len(crops)
        if B != 1:
            raise ValueError("multi-crop dense BEV currently supports B=1")
        out_h, out_w = crops[0]["shape_hw"]
        density_bev = density_image.new_zeros((B, G, D, out_h, out_w))

        for g_idx, crop in enumerate(crops):
            if tuple(crop["shape_hw"]) != (out_h, out_w):
                raise ValueError("all multi-crop BEV crops must share shape_hw")
            h_src = crop["h_src"]
            h_dst = crop["h_dst"]
            w_src = crop["w_src"]
            w_dst = crop["w_dst"]
            if h_src.stop <= h_src.start or w_src.stop <= w_src.start:
                continue

            density_src = density_image[:, h_src, :, :]
            src_h = h_src.stop - h_src.start
            density_flat = density_src.reshape(B * src_h, 1, D, full_w)
            grid_g = self.rad_depth_layer._inv_rot_grid[0, g_idx, 0, :, w_src].to(
                device=density_image.device,
                dtype=density_image.dtype,
            )
            grid = grid_g.unsqueeze(0).expand(B * src_h, -1, -1, -1)
            sampled = F.grid_sample(
                density_flat,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.reshape(B, src_h, D, w_src.stop - w_src.start)
            sampled = sampled.permute(0, 2, 1, 3).contiguous()
            density_bev[:, g_idx, :, h_dst, w_dst] = sampled

        ry = self.dose_grid_spacing[1]
        segment_weq = density_bev.clamp_min(0.0) * ry
        boundary_end = torch.cumsum(segment_weq, dim=2)
        weq_bev = boundary_end - 0.5 * segment_weq
        return (
            density_bev.reshape(B * G, D, out_h, out_w),
            weq_bev.reshape(B * G, D, out_h, out_w),
        )

    def _resolve_rad_depth_offset(
        self,
        rad_depth_offset_mm: torch.Tensor | float | None,
        ssd_mm: torch.Tensor | float | None,
        num_beams: int,
    ) -> torch.Tensor | None:
        """Return a per-beam rad_depth_offset tensor on the engine device."""
        device = self.device
        dtype = self.dtype
        if rad_depth_offset_mm is not None:
            tensor = torch.as_tensor(rad_depth_offset_mm, device=device, dtype=dtype)
            if tensor.ndim == 0:
                tensor = tensor.expand(num_beams).clone()
            return tensor
        if ssd_mm is None:
            return None
        ssd_tensor = torch.as_tensor(ssd_mm, device=device, dtype=dtype)
        if ssd_tensor.ndim == 0:
            ssd_tensor = ssd_tensor.expand(num_beams).clone()
        if ssd_tensor.shape != (num_beams,):
            raise ValueError(
                f"ssd_mm must be scalar or [{num_beams}], got {tuple(ssd_tensor.shape)}"
            )
        if self.sad_values_mm is None:
            raise RuntimeError("sad_values_mm is not initialized")
        sad_tensor = self.sad_values_mm.to(device=device, dtype=dtype)
        bams = float(getattr(self.machine_config, "bams_to_iso_dist_mm", 0.0))
        fit_air = float(getattr(self.machine_config, "fit_air_offset_mm", 0.0))
        nozzle_to_skin = (ssd_tensor + bams) - sad_tensor
        return 0.0011 * (nozzle_to_skin - fit_air)

    def set_patient_dose_mask(self, mask: torch.Tensor | None) -> None:
        """Set the voxels where dose is scored, overriding the density threshold.

        Pass ``pydose_rt.physics.spr.patient_dose_mask(mass_density)`` to keep dose in
        internal air cavities, which the plain threshold zeroes even though the MC
        reference has real dose there. ``None`` restores the threshold. The mask must
        match the patient grid layout of ``mass_density_image`` (``[H, D, W]`` or
        ``[1, H, D, W]``); it is beam-independent, so set it once per case.
        """
        if mask is None:
            self._patient_dose_mask = None
            return
        mask = mask.to(device=self.device, dtype=torch.bool)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        self._patient_dose_mask = mask

    def _convert_mev_to_gy(
        self,
        deposited_energy_mev: torch.Tensor,
        mass_density_g_cm3: torch.Tensor,
        patient_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        voxel_volume_mm3 = (
            float(self.dose_grid_spacing[0])
            * float(self.dose_grid_spacing[1])
            * float(self.dose_grid_spacing[2])
        )
        if patient_mask is None:
            patient_mask = mass_density_g_cm3 > self._patient_dose_density_threshold_g_cm3
        lateral_area_mm2 = voxel_volume_mm3 / float(self.transport_step_mm or self.dose_grid_spacing[1])
        dose_gy = deposited_energy_mev * (MEV_CM2_PER_G_TO_GY_MM2 / lateral_area_mm2)
        return torch.where(patient_mask, dose_gy, torch.zeros_like(dose_gy))

    def _patient_mask_slab(self, h_src: slice | None = None) -> torch.Tensor | None:
        """The configured dose mask, sliced like the mass-density slabs the finalizers cut."""
        mask = getattr(self, "_patient_dose_mask", None)
        if mask is None:
            return None
        return mask if h_src is None else mask[:, h_src, :, :]

    def _rotate_beamlet_to_patient(self, edep_bev, crops, g_idx, full_h, full_w):
        """Rotate one beamlet's cropped BEV edep into patient-frame (B, src_h, D, full_w),
        returning (rotated_g, h_src) or (None, None) if the crop is empty. Shared by the
        summed and per-beamlet finalizers so the geometry stays identical."""
        B, G, D, cH, cW = edep_bev.shape
        crop = crops[g_idx]
        h_src = crop["h_src"]
        h_dst = crop["h_dst"]
        if h_src.stop <= h_src.start:
            return None, None
        src_h = h_src.stop - h_src.start
        dose_g = edep_bev[:, g_idx, :, h_dst, :]
        dose_g = dose_g.permute(0, 2, 1, 3).contiguous().reshape(B * src_h, 1, D, cW)
        grid_full = self.rotation_layer.rot_grid[0, g_idx, 0].to(device=edep_bev.device, dtype=edep_bev.dtype)
        full_x = ((grid_full[..., 0] + 1.0) * float(full_w) - 1.0) * 0.5
        crop_x = full_x - float(crop["target_w_start"])
        crop_grid_x = (2.0 * (crop_x + 0.5) / float(cW)) - 1.0
        grid_g = torch.stack((crop_grid_x, grid_full[..., 1]), dim=-1)
        grid_g = grid_g.unsqueeze(0).expand(B * src_h, -1, -1, -1)
        rotated = F.grid_sample(dose_g, grid_g, mode="bilinear", padding_mode="zeros", align_corners=False)
        return rotated.reshape(B, src_h, D, full_w), h_src

    def _finalize_per_beamlet(self, edep_bev, mass_density_image, crops):
        """Return a list (len G) of per-beamlet patient-frame Gy volumes, each bbox-cropped
        to its nonzero support with a patient-grid offset. Same physics as the summed
        finalizer (linear Gy conversion), just not accumulated -- so Σ per-beamlet == summed.
        Kept cropped so memory is Σ(beamlet bboxes), never full_grid × G."""
        B, G, D, cH, cW = edep_bev.shape
        full_h, full_w = crops[0]["full_shape_hw"]
        out: list[dict | None] = []
        for g_idx in range(G):
            rotated_g, h_src = self._rotate_beamlet_to_patient(edep_bev, crops, g_idx, full_h, full_w)
            if rotated_g is None:
                out.append(None)
                continue
            dose_slab = self._convert_mev_to_gy(
                rotated_g,
                mass_density_image[:, h_src, :, :],
                self._patient_mask_slab(h_src),
            )
            nz = (dose_slab[0] > 0).nonzero()
            if nz.numel() == 0:
                out.append(None)
                continue
            mins = nz.min(dim=0).values
            maxs = nz.max(dim=0).values + 1
            z0, y0, x0 = int(mins[0]), int(mins[1]), int(mins[2])
            z1, y1, x1 = int(maxs[0]), int(maxs[1]), int(maxs[2])
            out.append({
                "dose": dose_slab[:, z0:z1, y0:y1, x0:x1].contiguous(),
                "offset": (h_src.start + z0, y0, x0),  # (z, y, x) in patient grid
                "full_shape": (full_h, D, full_w),
            })
        return out

    def _finalize_patient_dose_multi_crop(
        self,
        edep_bev: torch.Tensor,
        mass_density_image: torch.Tensor,
        crops: list[dict],
        chunk_size: int = 4,
        return_per_beamlet: bool = False,
    ):
        """Rotate per-beamlet cropped BEV deposited energy into patient frame and convert to Gy.

        With ``return_per_beamlet=True`` returns the list of per-beamlet cropped Gy volumes
        instead of the summed patient dose (same single batched BEV pass)."""
        B, G, D, cH, cW = edep_bev.shape
        if B != 1:
            raise ValueError("multi-crop dense finalization currently supports B=1")
        if return_per_beamlet:
            return self._finalize_per_beamlet(edep_bev, mass_density_image, crops)
        full_h, full_w = crops[0]["full_shape_hw"]
        total_edep = edep_bev.new_zeros((B, full_h, D, full_w))
        chunk_size = max(1, int(chunk_size))

        for start in range(0, G, chunk_size):
            end = min(start + chunk_size, G)
            dose_chunks = []
            grid_chunks = []
            chunk_meta = []

            for g_idx in range(start, end):
                crop = crops[g_idx]
                if tuple(crop["shape_hw"]) != (cH, cW):
                    raise ValueError("all multi-crop finalized BEV crops must match edep_bev shape")

                h_src = crop["h_src"]
                h_dst = crop["h_dst"]
                if h_src.stop <= h_src.start:
                    continue
                src_h = h_src.stop - h_src.start
                if h_dst.stop - h_dst.start != src_h:
                    raise ValueError("BEV crop h_src/h_dst lengths do not match")

                dose_g = edep_bev[:, g_idx, :, h_dst, :]
                dose_g = dose_g.permute(0, 2, 1, 3).contiguous().reshape(B * src_h, 1, D, cW)

                grid_full = self.rotation_layer.rot_grid[0, g_idx, 0].to(
                    device=edep_bev.device,
                    dtype=edep_bev.dtype,
                )
                full_x = ((grid_full[..., 0] + 1.0) * float(full_w) - 1.0) * 0.5
                crop_x = full_x - float(crop["target_w_start"])
                crop_grid_x = (2.0 * (crop_x + 0.5) / float(cW)) - 1.0
                grid_g = torch.stack((crop_grid_x, grid_full[..., 1]), dim=-1)
                grid_g = grid_g.unsqueeze(0).expand(B * src_h, -1, -1, -1)

                dose_chunks.append(dose_g)
                grid_chunks.append(grid_g)
                chunk_meta.append((h_src, src_h))

            if not dose_chunks:
                continue

            dose_batch = torch.cat(dose_chunks, dim=0)
            grid_batch = torch.cat(grid_chunks, dim=0)
            rotated_batch = F.grid_sample(
                dose_batch,
                grid_batch,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            offset = 0
            for h_src, src_h in chunk_meta:
                count = B * src_h
                rotated_g = rotated_batch[offset : offset + count].reshape(B, src_h, D, full_w)
                total_edep[:, h_src, :, :] += rotated_g
                offset += count

        return self._convert_mev_to_gy(total_edep, mass_density_image, self._patient_mask_slab())

    def _fermi_eyges_excess(self, energy_g, e_val, kernel_depth_s):
        """Heterogeneity lever-arm excess added to sigma_transport^2 (S,D), in mm^2:
            E(z_d) = sum_{i<=d} [ (z_d-z_i)^2 - (w_d-w_i)^2 ] * dTheta2_i
        z = geometric depth, w = WEQ depth. Identically 0 where geometric==WEQ
        (homogeneous) -> water/abdomen unchanged by construction; >0 in low-density
        lung (geometric gap > WEQ gap) -> broadens.

        The per-step angular variance dTheta2_i is Kanematsu's differential-Highland
        (dH) scattering power (NIM B 2008, Fuchs et al. 2012), closed-form from local
        residual range and WEQ -- no LUT-sigma inversion, no interpolation:
            T_dH = f_dH(l) * (E_s^2 / X0) * (z_charge/pv)^2 ,  E_s = 15.0 MeV
            f_dH(l) ~ 0.970 (1 + ln l / 20.7)(1 + ln l / 22.7)
            l = WEQ / 36.08cm  (radiative path, water radiation length, water-equiv)
            (pv/MeV)^2 = (R / 4.67e-4cm)^1.08 ,  R = R0 - WEQ  (residual range, water)
        """
        S, D = kernel_depth_s.shape
        w = kernel_depth_s                                          # (S,D) WEQ depth [mm]
        mp = 938.272
        E = float(e_val)
        pv0 = E * (E + 2.0 * mp) / (E + mp)                         # MeV at incidence
        r0_cm = 4.67e-4 * (pv0 * pv0) ** (1.0 / 1.08)               # water range [cm]
        w_cm = w / 10.0
        r_resid = r0_cm - w_cm                                      # residual range [cm]
        r_cm = r_resid.clamp_min(0.1)                              # floor: scattering power
        pv = (r_cm / 4.67e-4) ** 0.54                               # (S,D) MeV (bounded as pv->0)
        ell = (w_cm / 36.08).clamp_min(1e-6)                        # radiative path length
        ln_ell = torch.log(ell)
        f_dh = (0.970 * (1.0 + ln_ell / 20.7) * (1.0 + ln_ell / 22.7)).clamp_min(0.0)
        dw = torch.zeros_like(w)                                    # WEQ increment per voxel [mm]
        dw[:, 0] = w[:, 0]
        dw[:, 1:] = (w[:, 1:] - w[:, :-1])
        dw = dw.clamp_min(0.0)
        in_range = (r_resid > 0.1).to(dw.dtype)                     # no scattering past the range
        # Mass radiation length rho*X0 [g/cm^2] for the scattering rate. Default = water
        # (36.08). With material_radiation_length, use the true DoseRAD material value via
        # the local density (= WEQ gradient dw/res_d, since WEQ = integral rho dx) -> no
        # extra BEV channel. Bone (rho*X0~26) scatters more, soft tissue (~40) less.
        res_d = float(self.dose_grid_spacing[1])
        rhox0 = 36.08
        if self.material_radiation_length:
            from pydose_rt.physics.materials import GEANT4_DENSITY_GRID, GEANT4_RHOX0_GRID
            dgrid = torch.as_tensor(GEANT4_DENSITY_GRID, device=self.device, dtype=self.dtype)
            xgrid = torch.as_tensor(GEANT4_RHOX0_GRID, device=self.device, dtype=self.dtype)
            local_rho = (dw / res_d).clamp_min(0.0)                 # (S,D) physical density
            rhox0 = _interp1d(dgrid, xgrid, local_rho).clamp_min(1.0)   # (S,D) g/cm^2
        # dTheta2 = f_dH * (E_s^2 / pv^2) * (dWEQ_cm / rho*X0);  E_s^2=225
        dth = in_range * f_dh * (225.0 / (pv * pv)) * (dw / 10.0 / rhox0)   # (S,D), dimensionless
        z = torch.arange(D, device=self.device, dtype=self.dtype) * res_d   # (D,) geometric [mm]
        c0 = dth.cumsum(1)
        cz1 = (dth * z).cumsum(1)
        cw1 = (dth * w).cumsum(1)
        cz2 = (dth * z * z).cumsum(1)
        cw2 = (dth * w * w).cumsum(1)
        return (z * z - w * w) * c0 - 2.0 * z * cz1 + 2.0 * w * cw1 + cz2 - cw2

    def _make_subbeam_grid(self, sigma_x_mm, sigma_y_mm, n_per_dim, device, dtype):
        """Sub-beam offsets (mm) on a symmetric quarter-FWHM grid + initial-fluence weights.

        Returns:
            offsets_yx_mm: (S, 2) offsets in (y, x), mm
            sub_w:         (S,)  normalized weights from the initial spot Gaussian
            sub_sigma_xy:  (2,)  residual per-sub-beam sigma (x, y), mm
        """
        import math
        fwhm_x = 2.354820045 * sigma_x_mm
        fwhm_y = 2.354820045 * sigma_y_mm
        g = torch.arange(n_per_dim, device=device, dtype=dtype) - (n_per_dim - 1) / 2.0
        ox = g * (fwhm_x / 4.0)              # (n,)
        oy = g * (fwhm_y / 4.0)              # (n,)
        oy2, ox2 = torch.meshgrid(oy, ox, indexing="ij")   # (n, n)
        offsets_yx = torch.stack([oy2.reshape(-1), ox2.reshape(-1)], dim=-1)  # (S, 2)

        # Gaussian-splitting variance identity: the superposition of sub-beams of
        # width sigma_sub, placed with a weight envelope of width sigma_env, has
        # width^2 = sigma_env^2 + sigma_sub^2. To reconstruct the spot exactly we
        # therefore pick sigma_sub = sigma_spot/sqrt(n) and set the *envelope* (the
        # weights) to sigma_env = sqrt(sigma_spot^2 - sigma_sub^2) -- NOT sigma_spot,
        # which would over-broaden the core by sqrt(1+1/n) (worst at low energy).
        sub_sigma_x = (sigma_x_mm / math.sqrt(n_per_dim)).clamp_min(1e-3)
        sub_sigma_y = (sigma_y_mm / math.sqrt(n_per_dim)).clamp_min(1e-3)
        env_x_sq = (sigma_x_mm.square() - sub_sigma_x.square()).clamp_min(torch.finfo(dtype).eps)
        env_y_sq = (sigma_y_mm.square() - sub_sigma_y.square()).clamp_min(torch.finfo(dtype).eps)
        if os.environ.get("PYDOSERT_OLD_SUBBEAM_ENVELOPE", "").lower() in {"1", "true", "yes", "on"}:
            # Diagnostic only: the pre-fix (over-broadened) envelope = sigma_spot.
            env_x_sq = sigma_x_mm.square()
            env_y_sq = sigma_y_mm.square()

        w = torch.exp(
            -(oy2.reshape(-1) ** 2) / (2.0 * env_y_sq)
            - (ox2.reshape(-1) ** 2) / (2.0 * env_x_sq)
        )
        w = w / w.sum().clamp_min(torch.finfo(dtype).eps)

        return offsets_yx, w, torch.stack([sub_sigma_x, sub_sigma_y])

    def compute_layer_edep(
        self,
        g,
        energy_g, e_val,
        sigma_x, sigma_y, spot_weight,
        weq_bev, resolved_offset,
        crop_centers_hw, res_w, res_h,
        h_coords, w_coords, H, W,
        valid_lateral, spot_mask, layer_mask,
        splitting_mode="single",   # "single" | "split"
        n_per_dim=9,
    ):
        """Returns edep map (1, H, W) for layer g. Switchable single vs split."""
        eps = torch.finfo(self.dtype).eps

        kernel_offset = self.lut.get_kernel_offset(energy_g, energy_value_hint=e_val)
        kernel_offset = torch.as_tensor(kernel_offset, device=self.device, dtype=self.dtype)

        valid = valid_lateral[g].unsqueeze(0)                       # (1, H, W)
        active = valid & spot_mask[0, g, 0] & layer_mask[0, g, 0]   # (1, H, W)

        if splitting_mode == "single":
            # ---- your original path, unchanged ----
            kernel_depth = (weq_bev[0, g] + resolved_offset[g] - kernel_offset).clamp_min(0.0)  # (D,H,W) or (H,W)
            ray_edep = self.lut.get_edep(energy_g, kernel_depth, energy_value_hint=e_val).clamp_min(0.0)
            sigma_transport = self.lut.get_sigma(energy_g, kernel_depth, energy_value_hint=e_val).clamp_min(0.0)
            sigma_total_x = (sigma_transport.square() + sigma_x.square()).sqrt().clamp_min(1e-6)
            sigma_total_y = (sigma_transport.square() + sigma_y.square()).sqrt().clamp_min(1e-6)

            x_mm = (w_coords.view(1, 1, W) - crop_centers_hw[g, 1]) * res_w
            y_mm = (h_coords.view(1, H, 1) - crop_centers_hw[g, 0]) * res_h
            cell_weights = self.bev_lattice_model.evaluate_lateral_cell_weights(
                depth_water_mm=kernel_depth, x_mm=x_mm, y_mm=y_mm,
                sigma_x_mm=sigma_total_x, sigma_y_mm=sigma_total_y,
                cell_width_x_mm=kernel_depth.new_tensor(res_w),
                cell_width_y_mm=kernel_depth.new_tensor(res_h),
                energy_mev=energy_g,
            )
            cell_weights = torch.where(active, cell_weights, torch.zeros_like(cell_weights))
            weight_norm = cell_weights.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
            return spot_weight * ray_edep * (cell_weights / weight_norm)

        # ---- splitting path (vectorized over sub-beams) ----
        offsets_yx, sub_w, sub_sigma = self._make_subbeam_grid(
            sigma_x, sigma_y, n_per_dim, self.device, self.dtype
        )
        S = offsets_yx.shape[0]

        # Per-sub-beam centers (in voxel units), (S,)
        cy = crop_centers_hw[g, 0] + offsets_yx[:, 0] / res_h
        cx = crop_centers_hw[g, 1] + offsets_yx[:, 1] / res_w

        # Sample WEQ at each sub-beam's column.
        # weq_bev[0, g] is (D, H, W). Gather the (H,W) column-stack at (cy, cx) -> (S, D).
        weq_col = weq_bev[0, g]                       # (D, H, W)
        Dd = weq_col.shape[0]
        iy = cy.round().long().clamp(0, H - 1)        # (S,)
        ix = cx.round().long().clamp(0, W - 1)        # (S,)
        weq_s = weq_col[:, iy, ix].transpose(0, 1).contiguous()   # (S, D)

        kernel_depth_s = (weq_s + resolved_offset[g] - kernel_offset).clamp_min(0.0)  # (S, D)
        ray_edep_s = self.lut.get_edep(energy_g, kernel_depth_s, energy_value_hint=e_val).clamp_min(0.0)      # (S, D)

        # Lateral coords per sub-beam: (S, 1, W) and (S, H, 1)
        x_mm = (w_coords.view(1, 1, W) - cx.view(S, 1, 1)) * res_w   # (S, 1, W)
        y_mm = (h_coords.view(1, H, 1) - cy.view(S, 1, 1)) * res_h   # (S, H, 1)

        # Lateral kernel. Only the NARROW core is resolved per sub-beam: it is sharp
        # and heterogeneity-sensitive, so the sub-beam superposition matters there.
        # The broad nuclear halo is wide and smooth -- splitting it per sub-beam adds
        # cost (a second (S,D,H,W) tensor) and approximation error (the crude per-beam
        # sub_sigma residual) for no accuracy gain. So we add the halo ONCE over the
        # full crop as a single Gaussian centred on the beamlet, convolved with the
        # FULL initial spot sigma. In homogeneous media this reduces exactly to the
        # single-mode double Gaussian (broad sigma = sqrt(s2^2 + sigma_spot^2)).
        use_double = self.lateral_model == "gauss_double" and getattr(self.lut, "has_double_gauss", False)
        active_b = active.view(1, 1, H, W)

        if use_double:
            s1_s, s2_s, w_s = self.lut.get_double_gauss(
                energy_g, kernel_depth_s, energy_value_hint=e_val
            )  # (S,D) each
        else:
            s1_s = self.lut.get_sigma(energy_g, kernel_depth_s, energy_value_hint=e_val).clamp_min(0.0)  # (S,D)
            w_s = torch.zeros_like(s1_s)

        # Heterogeneity-aware lateral spread: add the geometric lever-arm excess to the
        # narrow transport sigma (Fermi-Eyges / Szymanowski-Oelfke / Fuchs non-local).
        # Reduces to the LUT sigma exactly in homogeneous tissue.
        if self.heterogeneous_mcs:
            excess = self._fermi_eyges_excess(energy_g, e_val, kernel_depth_s)   # (S,D)
            s1_s = (s1_s.square() + excess).clamp_min(1e-12).sqrt()

        # ---- narrow core: per sub-beam, each plane normalized to sum 1 ----
        hw_x = 0.5 * res_w
        hw_y = 0.5 * res_h
        s1x = (s1_s.square() + sub_sigma[0].square()).sqrt().clamp_min(1e-6)   # (S,D)
        s1y = (s1_s.square() + sub_sigma[1].square()).sqrt().clamp_min(1e-6)
        gx1 = _gauss_cell_1d(x_mm.view(S, 1, 1, W), s1x.view(S, Dd, 1, 1), hw_x)
        gy1 = _gauss_cell_1d(y_mm.view(S, 1, H, 1), s1y.view(S, Dd, 1, 1), hw_y)
        narrow = gx1 * gy1                                                     # (S,D,H,W)
        narrow = torch.where(active_b, narrow, torch.zeros_like(narrow))
        nnorm = narrow.sum(dim=(2, 3), keepdim=True).clamp_min(eps)            # (S,D,1,1)
        narrow = narrow / nnorm
        core_amp = (sub_w.view(S, 1) * ray_edep_s * (1.0 - w_s)).view(S, Dd, 1, 1)
        edep = (core_amp * narrow).sum(dim=0)                                  # (D,H,W)

        # ---- broad halo: one Gaussian over the full crop, IDD-conserving ----
        if use_double:
            halo_amp = (sub_w.view(S, 1) * ray_edep_s * w_s).sum(dim=0)        # (D,)
            c = S // 2                                                         # central sub-beam (offset 0)
            s2c = s2_s[c]                                                      # (D,) broad sigma at beamlet centre
            s2x = (s2c.square() + sigma_x.square()).sqrt().clamp_min(1e-6)     # (D,)
            s2y = (s2c.square() + sigma_y.square()).sqrt().clamp_min(1e-6)
            bx = (w_coords.view(1, W) - crop_centers_hw[g, 1]) * res_w         # (1,W)
            by = (h_coords.view(H, 1) - crop_centers_hw[g, 0]) * res_h         # (H,1)
            gbx = _gauss_cell_1d(bx.view(1, 1, W), s2x.view(Dd, 1, 1), hw_x)   # (D,1,W)
            gby = _gauss_cell_1d(by.view(1, H, 1), s2y.view(Dd, 1, 1), hw_y)   # (D,H,1)
            broad = gbx * gby                                                  # (D,H,W)
            broad = torch.where(active, broad, torch.zeros_like(broad))        # active (1,H,W) broadcasts
            bnorm = broad.sum(dim=(1, 2), keepdim=True).clamp_min(eps)         # (D,1,1)
            broad = broad / bnorm
            edep = edep + halo_amp.view(Dd, 1, 1) * broad

        return spot_weight * edep

    def compute_dose_bev_lattice_sparse_batch(
        self,
        beam_sequence: IonSpotBeamSequence,
        density_image: torch.Tensor,
        mass_density_image: torch.Tensor | None = None,
        overwrite: bool = False,
        rad_depth_offset_mm: torch.Tensor | float | None = None,
        ssd_mm: torch.Tensor | float | None = None,
        finalize_chunk_size: int = 4,
        return_per_beamlet: bool = False,
    ):
        """Beamlet-batched BEV lattice PB with per-ray heterogeneous WEQ.

        ``beam_sequence`` is expected to contain one spot/energy layer per beam.
        The beam dimension is treated as the beamlet batch dimension.
        """
        profile_timing = os.environ.get("PYDOSERT_DENSE_ENGINE_TIMING", "").lower() in {"1", "true", "yes", "on"}
        timings: dict[str, float] = {}

        def _sync_time(label: str, start_time: float) -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings[label] = timings.get(label, 0.0) + time.perf_counter() - start_time

        _t_init = time.perf_counter()
        self._initialize_layers(beam_sequence, overwrite=overwrite)
        if density_image.dim() == 3:
            density_image = density_image.unsqueeze(0)
        density_image = density_image.to(device=self.device, dtype=self.dtype)
        if mass_density_image is None:
            mass_density_image = density_image
        elif mass_density_image.dim() == 3:
            mass_density_image = mass_density_image.unsqueeze(0)
        mass_density_image = mass_density_image.to(device=self.device, dtype=self.dtype)
        if profile_timing:
            _sync_time("init_layers_inputs", _t_init)

        _t_stack = time.perf_counter()
        (
            spot_positions_mm,
            spot_weights,
            spot_layer_index,
            spot_mask,
            layer_energies_mev,
            layer_sigmas_mm,
            layer_mask,
        ) = IonSpotBeamSequence.stack([beam_sequence])
        spot_positions_mm = spot_positions_mm.to(device=self.device, dtype=self.dtype)
        spot_weights = spot_weights.to(device=self.device, dtype=self.dtype)
        spot_layer_index = spot_layer_index.to(device=self.device, dtype=torch.long)
        spot_mask = spot_mask.to(device=self.device, dtype=torch.bool)
        layer_energies_mev = layer_energies_mev.to(device=self.device, dtype=self.dtype)
        layer_sigmas_mm = layer_sigmas_mm.to(device=self.device, dtype=self.dtype)
        layer_mask = layer_mask.to(device=self.device, dtype=torch.bool)
        if profile_timing:
            _sync_time("stack_sequence", _t_stack)

        B, G, S, _ = spot_positions_mm.shape
        if B != 1:
            raise ValueError("BEV-lattice sparse path currently supports one patient")
        if S != 1 or layer_energies_mev.shape[2] != 1:
            raise ValueError("BEV-lattice sparse path expects one spot and one energy layer per beam")

        resolved_offset = self._resolve_rad_depth_offset(
            rad_depth_offset_mm=rad_depth_offset_mm,
            ssd_mm=ssd_mm,
            num_beams=G,
        )
        if resolved_offset is None:
            resolved_offset = torch.zeros(G, device=self.device, dtype=self.dtype)

        res_h, _res_d, res_w = (float(v) for v in self.dose_grid_spacing)
        field_h, field_w = (int(self.field_size[0]), int(self.field_size[1]))
        iso = self.iso_centers.to(device=self.device, dtype=self.dtype)
        center_h_vox = iso[:, 0] / res_h + spot_positions_mm[0, :, 0, 1] / res_h
        center_w_vox = iso[:, 2] / res_w + spot_positions_mm[0, :, 0, 0] / res_w
        crops = [
            self._build_dense_bev_crop(
                center_h=float(center_h_vox[g].detach()),
                center_w=float(center_w_vox[g].detach()),
                size_h=field_h,
                size_w=field_w,
            )
            for g in range(G)
        ]
        crop_centers_hw = torch.stack(
            (
                center_h_vox - torch.as_tensor([c["target_h_start"] for c in crops], device=self.device, dtype=self.dtype),
                center_w_vox - torch.as_tensor([c["target_w_start"] for c in crops], device=self.device, dtype=self.dtype),
            ),
            dim=1,
        )

        with torch.inference_mode():
            _t_bev = time.perf_counter()
            density_bev_flat, weq_bev_flat = self._dense_forward_bev_multi_crop(density_image, crops)
            D = density_bev_flat.shape[1]
            H, W = crops[0]["shape_hw"]
            density_bev = density_bev_flat.view(B, G, D, H, W)
            weq_bev = weq_bev_flat.view(B, G, D, H, W)
            if profile_timing:
                _sync_time("density_weq_bev", _t_bev)

            _t_lattice = time.perf_counter()
            valid_lateral = torch.zeros((G, H, W), device=self.device, dtype=torch.bool)
            for g, crop in enumerate(crops):
                if crop["h_dst"].stop > crop["h_dst"].start and crop["w_dst"].stop > crop["w_dst"].start:
                    valid_lateral[g, crop["h_dst"], crop["w_dst"]] = True

            h_coords = torch.arange(H, device=self.device, dtype=self.dtype)
            w_coords = torch.arange(W, device=self.device, dtype=self.dtype)
            edep_layers = []
            weq_depths = []
            for g in range(G):
                energy_g = layer_energies_mev[0, g, 0]
                e_val = float(energy_g.detach())
                sigma_x = layer_sigmas_mm[0, g, 0, 0]
                sigma_y = layer_sigmas_mm[0, g, 0, 1]
                spot_weight = spot_weights[0, g, 0]

                edep_layers.append(self.compute_layer_edep(g, energy_g, e_val,
                                    sigma_x, sigma_y, spot_weight,
                                    weq_bev, resolved_offset,
                                    crop_centers_hw, res_w, res_h,
                                    h_coords, w_coords, H, W,
                                    valid_lateral, spot_mask, layer_mask,
                                    splitting_mode=SPLITTING_MODE,
                                    n_per_dim=N_PER_DIM,
                                ))

                h = torch.clamp(crop_centers_hw[g, 0], 0.0, float(H - 1))
                w = torch.clamp(crop_centers_hw[g, 1], 0.0, float(W - 1))
                h0 = int(torch.floor(h).item())
                w0 = int(torch.floor(w).item())
                h1 = min(h0 + 1, H - 1)
                w1 = min(w0 + 1, W - 1)
                dh = h - h0
                dw = w - w0
                v00 = weq_bev[:, g, :, h0, w0]
                v01 = weq_bev[:, g, :, h0, w1]
                v10 = weq_bev[:, g, :, h1, w0]
                v11 = weq_bev[:, g, :, h1, w1]
                weq_depths.append(
                    v00 * (1.0 - dh) * (1.0 - dw)
                    + v01 * (1.0 - dh) * dw
                    + v10 * dh * (1.0 - dw)
                    + v11 * dh * dw
                )

            edep_5d = torch.stack(edep_layers, dim=0).view(B, G, D, H, W)
            weq_depths_flat = torch.stack(weq_depths, dim=1).view(B * G, D)
            if profile_timing:
                _sync_time("lattice_pb", _t_lattice)

        voxel_volume_mm3 = (
            float(self.dose_grid_spacing[0])
            * float(self.dose_grid_spacing[1])
            * float(self.dose_grid_spacing[2])
        )
        lateral_area_mm2 = voxel_volume_mm3 / float(self.transport_step_mm or self.dose_grid_spacing[1])
        edep_to_gy = edep_5d.new_tensor(MEV_CM2_PER_G_TO_GY_MM2 / lateral_area_mm2)
        _t_hook = time.perf_counter()
        dense_payload = self.sparse_hooks.apply_dense_bev(
            {
                "edep_bev": edep_5d,
                "density_bev": density_bev,
                "weq_bev": weq_bev,
                "weq_depths": weq_depths_flat,
                "density_image": density_image,
                "resolved_offset": resolved_offset,
                "spot_positions_mm": spot_positions_mm,
                "spot_weights": spot_weights,
                "spot_layer_index": spot_layer_index,
                "spot_mask": spot_mask,
                "layer_energies_mev": layer_energies_mev,
                "layer_sigmas_mm": layer_sigmas_mm,
                "layer_mask": layer_mask,
                "edep_to_gy": edep_to_gy,
                "bev_crop": crops,
                "crop_centers_hw": crop_centers_hw,
            },
            engine=self,
        )
        if profile_timing:
            _sync_time("correction_hook", _t_hook)
        corrected_edep = dense_payload["edep_bev"]
        _t_finalize = time.perf_counter()
        dose = self._finalize_patient_dose_multi_crop(
            corrected_edep,
            mass_density_image,
            crops=crops,
            chunk_size=finalize_chunk_size,
            return_per_beamlet=return_per_beamlet,
        )
        if profile_timing:
            _sync_time("finalize_rotation", _t_finalize)
            summary = " ".join(f"{name}={value * 1e3:.2f}ms" for name, value in sorted(timings.items()))
            print(f"[dense_engine_timing] beams={G} field={H}x{W} {summary}")
        return dose
