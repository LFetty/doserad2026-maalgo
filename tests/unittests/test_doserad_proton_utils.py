from pathlib import Path
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import doserad_proton_utils as proton_utils


def test_masked_metrics_reports_high_dose_mae_separately() -> None:
    reference = np.asarray([1.0, 0.01], dtype=np.float32)
    prediction = np.zeros_like(reference)

    metrics = proton_utils._masked_metrics(prediction, reference, mask_fraction=0.0)

    assert np.isclose(metrics["mae_pct_max"], 50.5)
    assert np.isclose(metrics["mae_high10_pct_max"], 100.0)


def test_make_beamlet_batch_sequence_honors_ray_and_beamlet_selection(monkeypatch):
    plan = {
        "SAD": 1200.0,
        "beams": [
            {
                "gantry_angle": 0.0,
                "rays": [
                    {
                        "ray_target": [0.0, 0.0, 0.0],
                        "beamlets": [{"energy": 70.0}, {"energy": 71.0}],
                    },
                    {
                        "ray_target": [0.0, 0.0, 0.0],
                        "beamlets": [{"energy": 80.0}, {"energy": 81.0}],
                    },
                ],
            }
        ],
    }

    monkeypatch.setattr(
        proton_utils,
        "_match_ray_layer_sigmas_mm",
        lambda *, beamlets, **_: ([[1.0, 1.0] for _ in beamlets], 1000.0),
    )

    sequence, ssd_values_mm = proton_utils._make_beamlet_batch_sequence(
        plan=plan,
        beam_parameters={},
        ct_hu=np.zeros((2, 2, 2), dtype=np.float32),
        origin_zyx=np.zeros(3, dtype=np.float32),
        resolution_zyx=(1.0, 1.0, 1.0),
        beam_index=0,
        particles_per_beamlet=1_000_000.0,
        gantry_offset_deg=0.0,
        skin_hu_threshold=-500.0,
        sigma_mode="beam_params",
        bams_to_iso_dist_mm=1000.0,
        lut=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        ray_indices=[1],
        beamlet_index=1,
    )

    assert len(sequence) == 1
    assert sequence[0].layer_energies_mev.tolist() == [81.0]
    assert ssd_values_mm == [1000.0]
