from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import torch


def _normalize_layer_sigmas(
    layer_energies_mev: torch.Tensor,
    layer_sigmas_mm: torch.Tensor,
) -> torch.Tensor:
    """Return per-layer sigmas as [..., 2] ordered [sigma_x_mm, sigma_y_mm]."""
    expected_shape = tuple(layer_energies_mev.shape)
    if tuple(layer_sigmas_mm.shape) == expected_shape:
        return torch.stack((layer_sigmas_mm, layer_sigmas_mm), dim=-1)
    if tuple(layer_sigmas_mm.shape) == expected_shape + (2,):
        return layer_sigmas_mm
    raise ValueError(
        "layer_sigmas_mm must match layer_energies_mev shape or have a trailing "
        f"xy component axis, got {tuple(layer_sigmas_mm.shape)} vs {expected_shape}"
    )


def _normalize_iso_centers(
    iso_center: tuple[float, float, float] | torch.Tensor | None,
    num_beams: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if iso_center is None:
        base = torch.zeros((3,), device=device, dtype=dtype)
        return base.unsqueeze(0).expand(num_beams, -1).clone()

    iso_tensor = torch.as_tensor(iso_center, device=device, dtype=dtype)
    if iso_tensor.shape == (3,):
        return iso_tensor.unsqueeze(0).expand(num_beams, -1).clone()
    if iso_tensor.shape == (num_beams, 3):
        return iso_tensor.clone()
    raise ValueError(
        f"iso_center must be shape (3,) or ({num_beams}, 3), got {tuple(iso_tensor.shape)}"
    )


@dataclass
class IonSpotBeam:
    """Single proton spot-scanning beam."""

    gantry_angle: float
    spot_positions_mm: torch.Tensor
    spot_weights: torch.Tensor
    spot_layer_index: torch.Tensor
    layer_energies_mev: torch.Tensor
    layer_sigmas_mm: torch.Tensor
    field_size: tuple[int, int] = (401, 401)
    iso_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sad_mm: float = 10_000.0

    def __post_init__(self) -> None:
        if self.spot_positions_mm.dim() != 2 or self.spot_positions_mm.shape[1] != 2:
            raise ValueError(
                f"spot_positions_mm must be [S, 2], got {tuple(self.spot_positions_mm.shape)}"
            )

        num_spots = self.spot_positions_mm.shape[0]
        if self.spot_weights.shape != (num_spots,):
            raise ValueError(
                f"spot_weights must be [{num_spots}], got {tuple(self.spot_weights.shape)}"
            )
        if self.spot_layer_index.shape != (num_spots,):
            raise ValueError(
                f"spot_layer_index must be [{num_spots}], got {tuple(self.spot_layer_index.shape)}"
            )
        if self.spot_layer_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("spot_layer_index must use an integer dtype")

        if self.layer_energies_mev.dim() != 1:
            raise ValueError(
                f"layer_energies_mev must be [L], got {tuple(self.layer_energies_mev.shape)}"
            )
        self.layer_sigmas_mm = _normalize_layer_sigmas(self.layer_energies_mev, self.layer_sigmas_mm)
        if self.layer_energies_mev.numel() == 0:
            raise ValueError("At least one energy layer is required")

        if num_spots > 0:
            if int(self.spot_layer_index.min()) < 0:
                raise ValueError("spot_layer_index must be non-negative")
            if int(self.spot_layer_index.max()) >= self.layer_energies_mev.numel():
                raise ValueError("spot_layer_index exceeds the available energy layers")

        float_tensors = [
            self.spot_positions_mm,
            self.spot_weights,
            self.layer_energies_mev,
            self.layer_sigmas_mm,
        ]
        devices = {tensor.device for tensor in float_tensors}
        dtypes = {tensor.dtype for tensor in float_tensors}
        if len(devices) != 1:
            raise ValueError(f"IonSpotBeam tensors must share one device, got {devices}")
        if len(dtypes) != 1:
            raise ValueError(f"IonSpotBeam tensors must share one dtype, got {dtypes}")

        if self.spot_layer_index.device != self.spot_positions_mm.device:
            raise ValueError("spot_layer_index must live on the same device as the spot tensors")

    @classmethod
    def create(
        cls,
        gantry_angle_deg: float,
        spot_positions_mm: torch.Tensor,
        layer_energies_mev: torch.Tensor,
        layer_sigmas_mm: torch.Tensor,
        spot_layer_index: torch.Tensor | None = None,
        spot_weights: torch.Tensor | None = None,
        field_size: tuple[int, int] = (401, 401),
        iso_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sad_mm: float = 10_000.0,
        requires_grad: bool = True,
        sigma_requires_grad: bool = False,
    ) -> "IonSpotBeam":
        """Build one beam from spot tensors."""
        if spot_weights is None:
            spot_weights = torch.ones(
                spot_positions_mm.shape[0],
                device=spot_positions_mm.device,
                dtype=spot_positions_mm.dtype,
            )
        if spot_layer_index is None:
            spot_layer_index = torch.zeros(
                spot_positions_mm.shape[0],
                device=spot_positions_mm.device,
                dtype=torch.long,
            )

        spot_positions_mm = spot_positions_mm.clone()
        spot_weights = spot_weights.clone()
        layer_energies_mev = layer_energies_mev.clone()
        layer_sigmas_mm = _normalize_layer_sigmas(layer_energies_mev, layer_sigmas_mm.clone())
        spot_layer_index = spot_layer_index.clone().to(dtype=torch.long)

        if requires_grad:
            spot_positions_mm.requires_grad_(True)
            spot_weights.requires_grad_(True)
            layer_energies_mev.requires_grad_(True)
        if sigma_requires_grad:
            layer_sigmas_mm.requires_grad_(True)

        return cls(
            gantry_angle=math.radians(gantry_angle_deg),
            spot_positions_mm=spot_positions_mm,
            spot_weights=spot_weights,
            spot_layer_index=spot_layer_index,
            layer_energies_mev=layer_energies_mev,
            layer_sigmas_mm=layer_sigmas_mm,
            field_size=field_size,
            iso_center=iso_center,
            sad_mm=float(sad_mm),
        )

    @property
    def gantry_angle_deg(self) -> float:
        return math.degrees(self.gantry_angle)

    @property
    def device(self) -> torch.device:
        return self.spot_positions_mm.device

    @property
    def dtype(self) -> torch.dtype:
        return self.spot_positions_mm.dtype

    @property
    def num_spots(self) -> int:
        return self.spot_positions_mm.shape[0]

    @property
    def num_layers(self) -> int:
        return self.layer_energies_mev.numel()

    @property
    def layer_sigmas_x_mm(self) -> torch.Tensor:
        return self.layer_sigmas_mm[..., 0]

    @property
    def layer_sigmas_y_mm(self) -> torch.Tensor:
        return self.layer_sigmas_mm[..., 1]

    @property
    def requires_grad(self) -> bool:
        return (
            self.spot_positions_mm.requires_grad
            or self.spot_weights.requires_grad
            or self.layer_energies_mev.requires_grad
            or self.layer_sigmas_mm.requires_grad
        )

    def detach(self) -> "IonSpotBeam":
        return IonSpotBeam(
            gantry_angle=self.gantry_angle,
            spot_positions_mm=self.spot_positions_mm.detach(),
            spot_weights=self.spot_weights.detach(),
            spot_layer_index=self.spot_layer_index.detach(),
            layer_energies_mev=self.layer_energies_mev.detach(),
            layer_sigmas_mm=self.layer_sigmas_mm.detach(),
            field_size=self.field_size,
            iso_center=self.iso_center,
            sad_mm=self.sad_mm,
        )

    def clone(self) -> "IonSpotBeam":
        return IonSpotBeam(
            gantry_angle=self.gantry_angle,
            spot_positions_mm=self.spot_positions_mm.clone(),
            spot_weights=self.spot_weights.clone(),
            spot_layer_index=self.spot_layer_index.clone(),
            layer_energies_mev=self.layer_energies_mev.clone(),
            layer_sigmas_mm=self.layer_sigmas_mm.clone(),
            field_size=self.field_size,
            iso_center=self.iso_center,
            sad_mm=self.sad_mm,
        )

    def to(self, target: torch.device | str | torch.dtype) -> "IonSpotBeam":
        if isinstance(target, torch.dtype):
            spot_layer_index = self.spot_layer_index.clone()
        else:
            spot_layer_index = self.spot_layer_index.to(target)

        return IonSpotBeam(
            gantry_angle=self.gantry_angle,
            spot_positions_mm=self.spot_positions_mm.to(target),
            spot_weights=self.spot_weights.to(target),
            spot_layer_index=spot_layer_index,
            layer_energies_mev=self.layer_energies_mev.to(target),
            layer_sigmas_mm=self.layer_sigmas_mm.to(target),
            field_size=self.field_size,
            iso_center=self.iso_center,
            sad_mm=self.sad_mm,
        )


@dataclass
class IonSpotBeamSequence:
    """Tensorized spot-scanning beam collection for one sample."""

    spot_positions_mm: torch.Tensor
    spot_weights: torch.Tensor
    spot_layer_index: torch.Tensor
    spot_mask: torch.Tensor
    layer_energies_mev: torch.Tensor
    layer_sigmas_mm: torch.Tensor
    layer_mask: torch.Tensor
    gantry_angles: torch.Tensor
    field_size: tuple[int, int]
    iso_center: tuple[float, float, float] | torch.Tensor | None
    sad_mm: torch.Tensor | float = 10_000.0
    iso_centers: torch.Tensor = field(init=False, repr=False)
    sad_values_mm: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.spot_positions_mm.dim() != 3 or self.spot_positions_mm.shape[-1] != 2:
            raise ValueError(
                f"spot_positions_mm must be [G, S, 2], got {tuple(self.spot_positions_mm.shape)}"
            )
        g, s, _ = self.spot_positions_mm.shape
        if self.spot_weights.shape != (g, s):
            raise ValueError(
                f"spot_weights must be [{g}, {s}], got {tuple(self.spot_weights.shape)}"
            )
        if self.spot_layer_index.shape != (g, s):
            raise ValueError(
                f"spot_layer_index must be [{g}, {s}], got {tuple(self.spot_layer_index.shape)}"
            )
        if self.spot_mask.shape != (g, s):
            raise ValueError(f"spot_mask must be [{g}, {s}], got {tuple(self.spot_mask.shape)}")
        if self.layer_energies_mev.dim() != 2 or self.layer_energies_mev.shape[0] != g:
            raise ValueError(
                f"layer_energies_mev must be [G, L], got {tuple(self.layer_energies_mev.shape)}"
            )
        self.layer_sigmas_mm = _normalize_layer_sigmas(self.layer_energies_mev, self.layer_sigmas_mm)
        if self.layer_mask.shape != self.layer_energies_mev.shape:
            raise ValueError("layer_mask must match layer_energies_mev")
        if self.gantry_angles.shape != (g,):
            raise ValueError(f"gantry_angles must be [{g}], got {tuple(self.gantry_angles.shape)}")

        self.iso_centers = _normalize_iso_centers(self.iso_center, g, self.device, self.dtype)
        sad_tensor = torch.as_tensor(self.sad_mm, device=self.device, dtype=self.dtype)
        if sad_tensor.shape == ():
            self.sad_values_mm = sad_tensor.expand(g).clone()
            self.sad_mm = float(sad_tensor.item())
        elif sad_tensor.shape == (g,):
            self.sad_values_mm = sad_tensor.clone()
            if torch.allclose(self.sad_values_mm, self.sad_values_mm[:1].expand_as(self.sad_values_mm)):
                self.sad_mm = float(self.sad_values_mm[0].item())
            else:
                self.sad_mm = self.sad_values_mm
        else:
            raise ValueError(f"sad_mm must be scalar or [{g}], got {tuple(sad_tensor.shape)}")
        if torch.allclose(self.iso_centers, self.iso_centers[:1].expand_as(self.iso_centers)):
            self.iso_center = tuple(float(v) for v in self.iso_centers[0].detach().cpu().tolist())
        else:
            self.iso_center = None

    @classmethod
    def from_beams(cls, beams: list[IonSpotBeam]) -> "IonSpotBeamSequence":
        if not beams:
            raise ValueError("Cannot create IonSpotBeamSequence from an empty list")

        first = beams[0]
        if any(beam.field_size != first.field_size for beam in beams):
            raise ValueError("All spot beams must share the same field_size")

        device = first.device
        dtype = first.dtype
        num_beams = len(beams)
        max_spots = max(beam.num_spots for beam in beams)
        max_layers = max(beam.num_layers for beam in beams)

        spot_positions = torch.zeros(num_beams, max_spots, 2, device=device, dtype=dtype)
        spot_weights = torch.zeros(num_beams, max_spots, device=device, dtype=dtype)
        spot_layer_index = torch.zeros(num_beams, max_spots, device=device, dtype=torch.long)
        spot_mask = torch.zeros(num_beams, max_spots, device=device, dtype=torch.bool)

        layer_energies = torch.zeros(num_beams, max_layers, device=device, dtype=dtype)
        layer_sigmas = torch.zeros(num_beams, max_layers, 2, device=device, dtype=dtype)
        layer_mask = torch.zeros(num_beams, max_layers, device=device, dtype=torch.bool)
        gantry_angles = torch.zeros(num_beams, device=device, dtype=dtype)
        iso_centers = torch.zeros(num_beams, 3, device=device, dtype=dtype)
        sad_values = torch.zeros(num_beams, device=device, dtype=dtype)

        for idx, beam in enumerate(beams):
            s = beam.num_spots
            l = beam.num_layers
            gantry_angles[idx] = beam.gantry_angle
            iso_centers[idx] = torch.tensor(beam.iso_center, device=device, dtype=dtype)
            sad_values[idx] = float(beam.sad_mm)
            if s > 0:
                spot_positions[idx, :s] = beam.spot_positions_mm
                spot_weights[idx, :s] = beam.spot_weights
                spot_layer_index[idx, :s] = beam.spot_layer_index
                spot_mask[idx, :s] = True
            layer_energies[idx, :l] = beam.layer_energies_mev
            layer_sigmas[idx, :l] = beam.layer_sigmas_mm
            layer_mask[idx, :l] = True

        return cls(
            spot_positions_mm=spot_positions,
            spot_weights=spot_weights,
            spot_layer_index=spot_layer_index,
            spot_mask=spot_mask,
            layer_energies_mev=layer_energies,
            layer_sigmas_mm=layer_sigmas,
            layer_mask=layer_mask,
            gantry_angles=gantry_angles,
            field_size=first.field_size,
            iso_center=iso_centers,
            sad_mm=sad_values,
        )

    @staticmethod
    def stack(
        sequences: list["IonSpotBeamSequence"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("Cannot stack an empty list of IonSpotBeamSequence")

        first = sequences[0]
        if any(len(seq) != len(first) for seq in sequences):
            raise ValueError("All IonSpotBeamSequence objects must have the same beam count")
        if any(seq.field_size != first.field_size for seq in sequences):
            raise ValueError("All IonSpotBeamSequence objects must share the same field_size")

        batch_size = len(sequences)
        num_beams = len(first)
        max_spots = max(seq.spot_positions_mm.shape[1] for seq in sequences)
        max_layers = max(seq.layer_energies_mev.shape[1] for seq in sequences)
        device = first.device
        dtype = first.dtype

        spot_positions = torch.zeros(batch_size, num_beams, max_spots, 2, device=device, dtype=dtype)
        spot_weights = torch.zeros(batch_size, num_beams, max_spots, device=device, dtype=dtype)
        spot_layer_index = torch.zeros(batch_size, num_beams, max_spots, device=device, dtype=torch.long)
        spot_mask = torch.zeros(batch_size, num_beams, max_spots, device=device, dtype=torch.bool)

        layer_energies = torch.zeros(batch_size, num_beams, max_layers, device=device, dtype=dtype)
        layer_sigmas = torch.zeros(batch_size, num_beams, max_layers, 2, device=device, dtype=dtype)
        layer_mask = torch.zeros(batch_size, num_beams, max_layers, device=device, dtype=torch.bool)

        for idx, seq in enumerate(sequences):
            s = seq.spot_positions_mm.shape[1]
            l = seq.layer_energies_mev.shape[1]
            spot_positions[idx, :, :s] = seq.spot_positions_mm
            spot_weights[idx, :, :s] = seq.spot_weights
            spot_layer_index[idx, :, :s] = seq.spot_layer_index
            spot_mask[idx, :, :s] = seq.spot_mask
            layer_energies[idx, :, :l] = seq.layer_energies_mev
            layer_sigmas[idx, :, :l] = seq.layer_sigmas_mm
            layer_mask[idx, :, :l] = seq.layer_mask

        return (
            spot_positions,
            spot_weights,
            spot_layer_index,
            spot_mask,
            layer_energies,
            layer_sigmas,
            layer_mask,
        )

    def __len__(self) -> int:
        return self.spot_positions_mm.shape[0]

    def __getitem__(self, idx: int) -> IonSpotBeam:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Beam index {idx} out of range")

        spot_mask = self.spot_mask[idx]
        layer_mask = self.layer_mask[idx]
        return IonSpotBeam(
            gantry_angle=float(self.gantry_angles[idx]),
            spot_positions_mm=self.spot_positions_mm[idx, spot_mask],
            spot_weights=self.spot_weights[idx, spot_mask],
            spot_layer_index=self.spot_layer_index[idx, spot_mask],
            layer_energies_mev=self.layer_energies_mev[idx, layer_mask],
            layer_sigmas_mm=self.layer_sigmas_mm[idx, layer_mask],
            field_size=self.field_size,
            iso_center=tuple(float(v) for v in self.iso_centers[idx].detach().cpu().tolist()),
            sad_mm=float(self.sad_values_mm[idx].detach().cpu().item()),
        )

    def __iter__(self) -> Iterator[IonSpotBeam]:
        for idx in range(len(self)):
            yield self[idx]

    @property
    def device(self) -> torch.device:
        return self.spot_positions_mm.device

    @property
    def dtype(self) -> torch.dtype:
        return self.spot_positions_mm.dtype

    @property
    def layer_sigmas_x_mm(self) -> torch.Tensor:
        return self.layer_sigmas_mm[..., 0]

    @property
    def layer_sigmas_y_mm(self) -> torch.Tensor:
        return self.layer_sigmas_mm[..., 1]

    @property
    def requires_grad(self) -> bool:
        return (
            self.spot_positions_mm.requires_grad
            or self.spot_weights.requires_grad
            or self.layer_energies_mev.requires_grad
            or self.layer_sigmas_mm.requires_grad
        )

    def detach(self) -> "IonSpotBeamSequence":
        return IonSpotBeamSequence(
            spot_positions_mm=self.spot_positions_mm.detach(),
            spot_weights=self.spot_weights.detach(),
            spot_layer_index=self.spot_layer_index.detach(),
            spot_mask=self.spot_mask.detach(),
            layer_energies_mev=self.layer_energies_mev.detach(),
            layer_sigmas_mm=self.layer_sigmas_mm.detach(),
            layer_mask=self.layer_mask.detach(),
            gantry_angles=self.gantry_angles.detach(),
            field_size=self.field_size,
            iso_center=self.iso_centers.detach(),
            sad_mm=self.sad_values_mm.detach(),
        )

    def clone(self) -> "IonSpotBeamSequence":
        return IonSpotBeamSequence(
            spot_positions_mm=self.spot_positions_mm.clone(),
            spot_weights=self.spot_weights.clone(),
            spot_layer_index=self.spot_layer_index.clone(),
            spot_mask=self.spot_mask.clone(),
            layer_energies_mev=self.layer_energies_mev.clone(),
            layer_sigmas_mm=self.layer_sigmas_mm.clone(),
            layer_mask=self.layer_mask.clone(),
            gantry_angles=self.gantry_angles.clone(),
            field_size=self.field_size,
            iso_center=self.iso_centers.clone(),
            sad_mm=self.sad_values_mm.clone(),
        )

    def to(self, target: torch.device | str | torch.dtype) -> "IonSpotBeamSequence":
        if isinstance(target, torch.dtype):
            spot_layer_index = self.spot_layer_index.clone()
            spot_mask = self.spot_mask.clone()
            layer_mask = self.layer_mask.clone()
        else:
            spot_layer_index = self.spot_layer_index.to(target)
            spot_mask = self.spot_mask.to(target)
            layer_mask = self.layer_mask.to(target)

        return IonSpotBeamSequence(
            spot_positions_mm=self.spot_positions_mm.to(target),
            spot_weights=self.spot_weights.to(target),
            spot_layer_index=spot_layer_index,
            spot_mask=spot_mask,
            layer_energies_mev=self.layer_energies_mev.to(target),
            layer_sigmas_mm=self.layer_sigmas_mm.to(target),
            layer_mask=layer_mask,
            gantry_angles=self.gantry_angles.to(target),
            field_size=self.field_size,
            iso_center=self.iso_centers.to(target) if not isinstance(target, torch.dtype) else self.iso_centers.clone(),
            sad_mm=self.sad_values_mm.to(target) if not isinstance(target, torch.dtype) else self.sad_values_mm.clone(),
        )
