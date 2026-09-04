from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from container.inference import (
    _clip_sct_hu,
    _densify,
    _output_slot_counts,
    _write_streaming_mha,
)


def _beamlet(output_file_idx: int) -> dict:
    return {
        "output_info": {
            "output_file_idx": output_file_idx,
            "idx_in_output": 0,
            "minimum_cutoff": 0.0,
        }
    }


def test_output_slot_counts_track_shared_slots() -> None:
    metadata = [
        {
            "beams": [{"rays": [{"beamlets": [_beamlet(0), _beamlet(1)]}]}],
        },
        {
            "beams": [{"rays": [{"beamlets": [_beamlet(1), _beamlet(2)]}]}],
        },
    ]

    assert _output_slot_counts(metadata) == {0: 1, 1: 2, 2: 1}


def test_clip_sct_hu_uses_supported_ct_range() -> None:
    source = np.array([-2000.0, -1024.0, 0.25, 3071.0, 5000.0], dtype=np.float64)

    actual = _clip_sct_hu(source, -1024.0, 3071.0)

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(
        actual,
        np.array([-1024.0, -1024.0, 0.25, 3071.0, 3071.0], dtype=np.float32),
    )


def test_streaming_mha_matches_join_series(tmp_path) -> None:
    ref = sitk.Image(7, 6, 5, sitk.sitkFloat32)
    ref.SetSpacing((1.25, 2.5, 3.75))
    ref.SetOrigin((-4.5, 6.25, 8.0))
    ref.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    frames = {
        0: {"dose": np.arange(12, dtype=np.float32).reshape(2, 2, 3), "offset": (1, 2, 3)},
        1: None,
        2: {"dose": np.full((1, 3, 2), 0.125, dtype=np.float32), "offset": (4, 0, 1)},
    }

    output = tmp_path / "streamed.mha"
    _write_streaming_mha(output, frames, ref, compression_level=1)
    actual = sitk.ReadImage(str(output))

    expected_frames = []
    full_shape = tuple(reversed(ref.GetSize()))
    for frame_idx in range(3):
        image = sitk.GetImageFromArray(_densify(frames[frame_idx], full_shape))
        image.CopyInformation(ref)
        expected_frames.append(image)
    expected = sitk.JoinSeries(expected_frames)

    assert actual.GetSize() == expected.GetSize()
    assert actual.GetSpacing() == expected.GetSpacing()
    assert actual.GetOrigin() == expected.GetOrigin()
    assert actual.GetDirection() == expected.GetDirection()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(expected))
