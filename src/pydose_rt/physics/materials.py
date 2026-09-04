from __future__ import annotations

import numpy as np


# Geant4 DoseRAD material bins from DICOMphantom.cc. The simulator creates
# density-specific material variants with densityDiff=0.001 g/cm3, but the
# coarse composition family is selected from these HU intervals first.
GEANT4_HU_BOUNDS = np.asarray(
    [
        -1024,
        -950,
        -90,
        -64,
        -38,
        -24,
        -10,
        4,
        18,
        70,
        *range(120, 1621, 20),
        4000,
    ],
    dtype=np.float32,
)

GEANT4_NUM_MATERIALS = int(GEANT4_HU_BOUNDS.size - 1)
GEANT4_DENSITY_BIN_G_CM3 = 0.001

# Mass radiation length rho*X0 [g/cm^2] of the DoseRAD Geant4 materials, computed
# from their elemental compositions in DICOMphantom.cc (PDG element X0 + compound
# rule 1/X0 = sum w_i/X0_i), keyed by the material's physical density [g/cm^3].
# Used by the heterogeneity-aware lateral scattering (Fuchs/Kanematsu dH) to replace
# the water radiation length (36.08) with the true material value. Note it is NOT
# monotonic in density (air/lung ~37, soft tissue ~40-41 since H/C-rich, bone ~26)
# but single-valued, so np.interp/torch interp over the (monotone) density grid works.
GEANT4_DENSITY_GRID = [0.0012, 0.26, 0.9528, 0.9787, 0.9926, 1.001, 1.01, 1.024, 1.076, 1.108, 1.11, 1.121, 1.135, 1.148, 1.162, 1.175, 1.188, 1.202, 1.215, 1.229, 1.242, 1.255, 1.269, 1.282, 1.296, 1.309, 1.323, 1.336, 1.349, 1.363, 1.376, 1.39, 1.403, 1.417, 1.43, 1.443, 1.457, 1.47, 1.484, 1.497, 1.511, 1.524, 1.537, 1.551, 1.564, 1.578, 1.591, 1.604, 1.618, 1.631, 1.645, 1.658, 1.672, 1.685, 1.698, 1.712, 1.725, 1.739, 1.752, 1.766, 1.779, 1.792, 1.806, 1.819, 1.833, 1.846, 1.86, 1.873, 1.886, 1.9, 1.913, 1.927, 1.94, 1.953, 1.967, 1.98, 1.994, 2.007, 2.021, 2.034, 2.047, 2.061, 2.074, 2.088, 2.101, 3.708]
GEANT4_RHOX0_GRID = [37.037, 36.529, 41.757, 40.625, 41.262, 40.326, 39.453, 38.635, 36.727, 37.054, 37.013, 36.667, 36.298, 35.943, 35.604, 35.278, 34.966, 34.695, 34.405, 34.126, 33.858, 33.599, 33.35, 33.11, 32.878, 32.655, 32.438, 32.219, 32.017, 31.822, 31.632, 31.449, 31.271, 31.098, 30.93, 30.767, 30.609, 30.456, 30.306, 30.162, 30.02, 29.883, 29.749, 29.619, 29.492, 29.369, 29.248, 29.131, 29.017, 28.904, 28.795, 28.689, 28.586, 28.483, 28.384, 28.287, 28.193, 28.099, 28.009, 27.921, 27.833, 27.748, 27.665, 27.583, 27.504, 27.426, 27.349, 27.274, 27.2, 27.128, 27.057, 26.989, 26.921, 26.854, 26.788, 26.724, 26.661, 26.599, 26.538, 26.478, 26.419, 26.362, 26.305, 26.249, 26.195, 26.195]


def geant4_material_id_from_hu(hu: np.ndarray) -> np.ndarray:
    """Map HU values to the DoseRAD Geant4 coarse material index."""
    hu_arr = np.asarray(hu, dtype=np.float32)
    clipped = np.clip(hu_arr, GEANT4_HU_BOUNDS[0], GEANT4_HU_BOUNDS[-1])
    material_id = np.searchsorted(GEANT4_HU_BOUNDS[1:], clipped, side="right")
    return np.clip(material_id, 0, GEANT4_NUM_MATERIALS - 1).astype(np.int64, copy=False)


def geant4_density_bounds_from_hu_lut(hu_to_density_entries: list[dict[str, float]]) -> np.ndarray:
    """Convert Geant4 HU material boundaries to density boundaries with the DoseRAD LUT."""
    hu = np.asarray([float(entry["hu"]) for entry in hu_to_density_entries], dtype=np.float32)
    density = np.asarray([float(entry["density_g_cm3"]) for entry in hu_to_density_entries], dtype=np.float32)
    order = np.argsort(hu)
    return np.interp(
        GEANT4_HU_BOUNDS,
        hu[order],
        density[order],
        left=float(density[order][0]),
        right=float(density[order][-1]),
    ).astype(np.float32, copy=False)


def geant4_density_bin_from_density(density_g_cm3: np.ndarray) -> np.ndarray:
    """Return the density-bin midpoint used when Geant4 clones material variants."""
    density = np.asarray(density_g_cm3, dtype=np.float32)
    step = np.float32(GEANT4_DENSITY_BIN_G_CM3)
    return (step * (density / step).astype(np.int32) + 0.5 * step).astype(np.float32, copy=False)
