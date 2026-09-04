"""Backward-compatible re-export.

The DoseRAD Geant4 phantom material model is core physics and now lives in
:mod:`pydose_rt.physics.materials`. This shim keeps the historical
``training.common.materials`` import path working.
"""

from __future__ import annotations

from pydose_rt.physics.materials import (
    GEANT4_DENSITY_BIN_G_CM3,
    GEANT4_HU_BOUNDS,
    GEANT4_NUM_MATERIALS,
    geant4_density_bin_from_density,
    geant4_density_bounds_from_hu_lut,
    geant4_material_id_from_hu,
)

__all__ = [
    "GEANT4_DENSITY_BIN_G_CM3",
    "GEANT4_HU_BOUNDS",
    "GEANT4_NUM_MATERIALS",
    "geant4_density_bin_from_density",
    "geant4_density_bounds_from_hu_lut",
    "geant4_material_id_from_hu",
]
