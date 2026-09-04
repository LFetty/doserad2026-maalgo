"""Container inference for the DoseRAD2026 challenge — Proton dose on CT.

Deliberately minimal: it only does what the Grand Challenge algorithm container
needs — read the mounted input contract, run the *validated* PB + dense-BEV
correction compute path, and write the output contract. All physics/model code
is reused verbatim from ``doserad_proton_utils`` / ``pydose_rt`` so this stays a
thin adapter, not a second implementation.

Split into two phases so it maps onto the GC invoke API:
  * ``DoseModel.load()``  -> do this during ``GET /health`` (unmetered time):
                             load LUT, machine model, checkpoint, fuse RepVGG.
  * ``DoseModel.run(in, out)`` -> do this on ``POST /invoke``: read /input,
                             compute, write /output.

Local test (no server):
    uv run python container/inference.py --input <dir> --output <dir>
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
import sys
import time
import zlib
from pathlib import Path
import types

import numpy as np
import torch


def _install_pathlib_pickle_compat() -> None:
    """Allow Python 3.12 to read checkpoints serialized by Python 3.13.

    Python 3.13 moved the concrete path classes into the private
    ``pathlib._local`` module. A checkpoint containing an argparse namespace or
    config with a ``Path`` therefore records that module name. The official
    PyTorch 2.10 runtime image currently uses Python 3.12, where the same
    classes still live directly in ``pathlib``. Registering this import alias
    restores pickle compatibility without touching tensors or model state.
    """

    if "pathlib._local" in sys.modules:
        return
    import pathlib

    local = types.ModuleType("pathlib._local")
    for name in (
        "Path",
        "PosixPath",
        "WindowsPath",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
    ):
        setattr(local, name, getattr(pathlib, name))
    sys.modules["pathlib._local"] = local


_install_pathlib_pickle_compat()

# --- make the repo packages + shared script helpers importable --------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))            # for `training.*`, `src` package (pydose_rt)
sys.path.insert(0, str(_REPO / "scripts"))  # for `doserad_proton_utils`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `standalone_regression_inference`

from doserad_proton_utils import (  # noqa: E402
    _make_beamlet_batch_sequence,
    _origin_zyx,
    _resolution_zyx,
)
from pydose_rt.data.machine_config import MachineConfig  # noqa: E402
from pydose_rt.engine.ion_dose_engine import IonDoseEngine  # noqa: E402
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT  # noqa: E402
from pydose_rt.physics.spr import patient_dose_mask, spr_and_mass_density  # noqa: E402
from pydose_rt.sparse.ions import IonSparseHooks  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed recipe — must match the shipped checkpoint.
# ---------------------------------------------------------------------------
# Weights live here. In the container this is set to the mounted model dir
# (e.g. /opt/ml/model) via the MODEL_DIR env var; locally it defaults to ./model.
DEFAULT_MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(_REPO / "model")))
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "latest_ema.pt"
DEFAULT_MACHINE_MAT = DEFAULT_MODEL_DIR / "lut_fast_3d_1e8_opt.mat"
DEFAULT_BEAM_PARAMS = DEFAULT_MODEL_DIR / "beam_parameters.json"
# MR->sCT converter bundle (only used for the proton-MRI task); shipped in the
# same model tarball under mrtoct_bundle/.
DEFAULT_SCT_BUNDLE = DEFAULT_MODEL_DIR / "mrtoct_bundle"

# Task/modality selector. "proton_ct" (default) reads the CT directly; "proton_mri"
# reads the MR and converts it to a synthetic CT before the (identical) dose compute.
# Grand Challenge runs one task per algorithm, so this is baked into the image via ENV.
GC_TASK = os.environ.get("GC_TASK", "proton_ct").strip().lower()
MRI_MODE = GC_TASK.endswith("mri")
SOURCE_IMAGE_BASE = (
    "radiation-dose-calculation-source-mri-image" if MRI_MODE
    else "radiation-dose-calculation-source-ct-image"
)

# Grand Challenge invoke mounts (overridable for local testing).
GC_INPUT_DIR = Path(os.environ.get("GC_INPUT_DIR", "/input"))
GC_OUTPUT_DIR = Path(os.environ.get("GC_OUTPUT_DIR", "/output"))

PARTICLES_PER_BEAMLET = 1_000_000.0
LATERAL_MODEL = "gauss_double"
SIGMA_MODE = "beam_params"
SPLIT_MODE = "split"
HETEROGENEOUS_MCS = True
BAMS_TO_ISO_DIST_MM = 1000.0
SKIN_HU_THRESHOLD = -500.0
N_OUTPUT_SLOTS = 10
# Real plans have 2000-3000 beamlets/beam. Running them all through one engine call
# keeps the whole (1, G, D, cH, cW) edep tensor + G per-beamlet crops on the GPU at
# once (OOM on a 24 GB A10G). Cap beamlets per engine call and move each result to
# CPU between chunks so GPU memory stays bounded regardless of plan size.
# Measured on the fixed BEV path with a 244-beamlet, two-image request (2026-08-29):
#   dense=6/chunk=32  -> 15.01 GiB max reserved
#   dense=2/chunk=32  ->  6.46 GiB max reserved (but 2.63 s slower)
#   dense=2/chunk=128 ->  8.90 GiB max reserved (but 3.49 s slower)
#   dense=4/chunk=128 -> 13.29 GiB max reserved (same compute speed as dense=6)
# On OOM we shrink the dense forward first, then split rays (see compute_group_adaptive),
# so an unexpected plan can self-correct.
MAX_BEAMLETS_PER_CHUNK = int(os.environ.get("MAX_BEAMLETS_PER_CHUNK", "32"))
DENSE_HOOK_BATCH_ITEMS = int(os.environ.get("DENSE_HOOK_BATCH_ITEMS", "2"))
DENSE_HOOK_AMP = os.environ.get("DENSE_HOOK_AMP", "").lower() in {
    "1", "true", "yes", "on",
}
# -1 asks ITK to use the image-IO default. Lower non-negative levels trade file size for
# compression speed while remaining lossless.
OUTPUT_COMPRESSION_LEVEL = int(os.environ.get("OUTPUT_COMPRESSION_LEVEL", "-1"))
OUTPUT_WRITE_WORKERS = int(os.environ.get("OUTPUT_WRITE_WORKERS", "1"))
PIPELINE_OUTPUT_WRITES = os.environ.get("PIPELINE_OUTPUT_WRITES", "").lower() in {
    "1", "true", "yes", "on",
}
STREAMING_MHA_WRITES = os.environ.get("STREAMING_MHA_WRITES", "").lower() in {
    "1", "true", "yes", "on",
}
# MRI-only synthetic-CT execution. Network convolutions can use FP16 while the standalone
# converter retains float32 Gaussian accumulation and HU denormalization. Clip to the
# actual CT range recorded in the exported Dataset038 bundle metadata.
SCT_INFERENCE_AMP = os.environ.get("SCT_INFERENCE_AMP", "").lower() in {
    "1", "true", "yes", "on",
}
SCT_PATCH_BATCH_SIZE = int(os.environ.get("SCT_PATCH_BATCH_SIZE", "1"))
SCT_HU_MIN = float(os.environ.get("SCT_HU_MIN", "-1024"))
SCT_HU_MAX = float(os.environ.get("SCT_HU_MAX", "3071"))


# Settings that may be overridden at runtime from the model tarball's config.json.
# Everything here is module-level above; the tarball is cheap to re-upload (12 MB) while
# the image is ~15 GB, so tuning happens through the config, not a rebuild.
CONFIGURABLE = (
    "MAX_BEAMLETS_PER_CHUNK",
    "DENSE_HOOK_BATCH_ITEMS",
    "DENSE_HOOK_AMP",
    "OUTPUT_COMPRESSION_LEVEL",
    "OUTPUT_WRITE_WORKERS",
    "PIPELINE_OUTPUT_WRITES",
    "STREAMING_MHA_WRITES",
    "SCT_INFERENCE_AMP",
    "SCT_PATCH_BATCH_SIZE",
    "SCT_HU_MIN",
    "SCT_HU_MAX",
    "PARTICLES_PER_BEAMLET",
    "LATERAL_MODEL",
    "SIGMA_MODE",
    "SPLIT_MODE",
    "HETEROGENEOUS_MCS",
    "BAMS_TO_ISO_DIST_MM",
    "SKIN_HU_THRESHOLD",
    "N_OUTPUT_SLOTS",
    "COMPILE_CORRECTION_MODEL",
    "COMPILE_DYNAMIC_SHAPES",
    "COMPILE_MODE",
    "COMPILE_CACHE_FILE",
    "SAVE_COMPILE_CACHE",
    "WARMUP_ON_LOAD",
    "PAD_INFERENCE_BATCH",
)
CONFIG_FILENAME = "config.json"

# --- torch.compile (opt-in) -------------------------------------------------
# Off by default and never fatal: compile *cache artifacts are GPU-arch specific*
# (5080 sm_120 / A100 sm_80 / A10G sm_86 all mismatch), so a cache built elsewhere is
# useless on the target and the kernels must be compiled on the machine that runs.
# torch.compile() itself returns immediately — compilation happens on the FIRST forward.
COMPILE_CORRECTION_MODEL = os.environ.get("COMPILE_CORRECTION_MODEL", "").lower() in {"1", "true", "yes", "on"}
# Dynamic shapes matter: patient grids differ, and a static-shape compile re-compiles for
# every new shape *inside the metered invoke*.
COMPILE_DYNAMIC_SHAPES = os.environ.get("COMPILE_DYNAMIC_SHAPES", "1").lower() in {"1", "true", "yes", "on"}
# `max-autotune` spends more unmetered startup time benchmarking kernels but reduces the
# fixed model's steady-state forward time. Keep configurable because platform startup
# budgets differ even when invoke time is the scored quantity.
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")
# Loaded from MODEL_DIR when present (ship it in the model tarball once we have a valid
# cache for the target arch); written to the output dir when SAVE_COMPILE_CACHE is set,
# which is how a cache can be harvested from a platform try-out run.
COMPILE_CACHE_FILE = os.environ.get("COMPILE_CACHE_FILE", "compile_cache.bin")
SAVE_COMPILE_CACHE = os.environ.get("SAVE_COMPILE_CACHE", "").lower() in {"1", "true", "yes", "on"}
# Force compilation during the unmetered health phase by running one real beamlet at load.
# Without this, torch.compile's laziness puts the whole compile inside the metered /invoke.
WARMUP_ON_LOAD = os.environ.get("WARMUP_ON_LOAD", "1").lower() in {"1", "true", "yes", "on"}
# Pad the ragged tail chunk up to DENSE_HOOK_BATCH_ITEMS so the correction model only ever
# sees one batch size. Without it the last chunk of every beam is a fresh shape and
# torch.compile recompiles inside the metered /invoke. Numerically inert (GroupNorm is
# per-sample and padded rows are never scattered back), so it is safe to leave on even when
# compile is off; it only costs the wasted rows of the final chunk.
PAD_INFERENCE_BATCH = os.environ.get("PAD_INFERENCE_BATCH", "1").lower() in {"1", "true", "yes", "on"}


def _clip_sct_hu(array: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return writable float32 sCT data clipped to the supported CT HU interval."""
    if lower >= upper:
        raise ValueError(f"invalid sCT HU range [{lower}, {upper}]")
    result = np.asarray(array, dtype=np.float32)
    if not result.flags.writeable:
        result = result.copy()
    np.clip(result, lower, upper, out=result)
    return result


def _apply_runtime_config(model_dir: Path) -> dict:
    """Apply `<model_dir>/config.json` over the module defaults.

    Precedence: built-in default < config.json < environment variable, so an env var
    set for a one-off debug run still wins. Unknown keys are reported and ignored
    rather than failing the run — a typo in the config must not cost a submission.
    Values are coerced to the type of the existing default.
    """
    cfg_path = Path(model_dir) / CONFIG_FILENAME
    applied: dict = {}
    if not cfg_path.is_file():
        print(f"[config] no {CONFIG_FILENAME} in {model_dir}; using built-in defaults", flush=True)
        return applied
    try:
        raw = json.loads(cfg_path.read_text())
    except Exception as exc:  # noqa: BLE001 - never let a bad config kill the run
        print(f"[config] WARNING: could not parse {cfg_path}: {exc}; using defaults", flush=True)
        return applied
    if not isinstance(raw, dict):
        print(f"[config] WARNING: {cfg_path} is not a JSON object; using defaults", flush=True)
        return applied

    g = globals()
    for key, value in raw.items():
        if key.startswith("_"):  # `_comment` etc. — documentation, not settings
            continue
        name = key.upper()
        if name not in CONFIGURABLE:
            print(f"[config] WARNING: ignoring unknown key {key!r}", flush=True)
            continue
        if name in os.environ:  # explicit env override wins
            print(f"[config] {name}: config value {value!r} overridden by env", flush=True)
            continue
        current = g[name]
        try:
            coerced = type(current)(value) if not isinstance(current, bool) else bool(value)
        except Exception as exc:  # noqa: BLE001
            print(f"[config] WARNING: bad value for {name}: {value!r} ({exc}); keeping {current!r}", flush=True)
            continue
        g[name] = coerced
        applied[name] = coerced
    print(f"[config] loaded {cfg_path}: {applied or 'nothing applied'}", flush=True)
    return applied


def _effective_settings() -> dict:
    g = globals()
    return {name: g[name] for name in CONFIGURABLE}


def _is_cuda_oom(exc: BaseException) -> bool:
    """True for CUDA OOM in any of the shapes torch raises it (typed error, or a
    RuntimeError/AcceleratorError whose message says out of memory)."""
    if isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return "out of memory" in str(exc).lower()


class DoseModel:
    """Loads once (health phase), runs many times (invoke phase)."""

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        machine_mat: Path = DEFAULT_MACHINE_MAT,
        beam_params: Path = DEFAULT_BEAM_PARAMS,
        sct_bundle: Path = DEFAULT_SCT_BUNDLE,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.machine_mat = Path(machine_mat)
        self.beam_params_path = Path(beam_params)
        self.sct_bundle = Path(sct_bundle)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.lut: PyRadPlanIonLUT | None = None
        self.machine_config: MachineConfig | None = None
        self.correction_hook = None
        self.sparse_hooks: IonSparseHooks | None = None
        self.beam_parameters: dict = {}
        self.sct_converter = None  # MR->sCT (proton-MRI task only)

    # -- health phase --------------------------------------------------------
    def load(self) -> "DoseModel":
        # Runtime overrides come from the model tarball (cheap to re-upload) and must be
        # applied before anything reads a setting — the hook below captures the batch size.
        _apply_runtime_config(self.checkpoint.parent)
        print(f"[config] effective: {_effective_settings()}", flush=True)

        torch.set_float32_matmul_precision("high")
        # Match the eval split mode used during validation.
        from pydose_rt.engine import ion_dose_engine
        ion_dose_engine.SPLITTING_MODE = SPLIT_MODE

        self.beam_parameters = json.loads(self.beam_params_path.read_text())
        self.lut = PyRadPlanIonLUT(self.machine_mat)
        self.machine_config = MachineConfig(
            tpr_20_10=0.7,
            number_of_leaf_pairs=40,
            fit_air_offset_mm=0.0,
            bams_to_iso_dist_mm=BAMS_TO_ISO_DIST_MM,
        )

        from training.proton.hooks import ProtonDenseBevCorrectionHook

        hook = ProtonDenseBevCorrectionHook.from_checkpoint(
            self.checkpoint,
            device=self.device,
            dtype=self.dtype,
            available_energies=self.lut.available_energies,
            bev_crop_hw=None,
            max_inference_batch_items=DENSE_HOOK_BATCH_ITEMS,
            inference_amp=DENSE_HOOK_AMP,
            pad_inference_batch=PAD_INFERENCE_BATCH,
        )
        # RepVGG fold (arch-independent): collapse train-time branches to plain convs.
        if hasattr(hook.model, "fuse_repvgg"):
            hook.model.fuse_repvgg()
        if COMPILE_CORRECTION_MODEL:
            self._compile_correction_model(hook)

        self.correction_hook = hook
        self.sparse_hooks = IonSparseHooks(dense_bev=hook)

        # torch.compile is LAZY: the call above only wraps the model, and the kernel
        # compilation would otherwise happen on the first forward — i.e. inside the metered
        # /invoke. Force it here instead; /health time is unmetered, so this makes it free.
        # Synthetic shapes first: compilation depends on shapes, not values, so this needs
        # no /input at all and removes the old "is /input mounted at load?" failure mode.
        # The real-data path stays only as a fallback if the synthetic call ever mismatches.
        if COMPILE_CORRECTION_MODEL and WARMUP_ON_LOAD:
            if not self._warmup_synthetic(hook):
                self._warmup(GC_INPUT_DIR)

        # proton-MRI task: load the MR->sCT converter now (health phase, unmetered).
        if MRI_MODE:
            from standalone_regression_inference import StandaloneRegressionInference

            if not self.sct_bundle.exists():
                raise FileNotFoundError(
                    f"proton-MRI task requires the sCT bundle at {self.sct_bundle} "
                    "(ship mrtoct_bundle/ inside the model tarball)"
                )
            self.sct_converter = StandaloneRegressionInference(
                str(self.sct_bundle),
                device=str(self.device),
                inference_amp=SCT_INFERENCE_AMP,
                patch_batch_size=SCT_PATCH_BATCH_SIZE,
            )
        return self

    # -- MR -> synthetic CT (proton-MRI task); identity for the CT task ---------
    def _to_ct(self, src_image):
        """Return an HU image for the dose engine. CT task: pass through. MRI task:
        convert the MR to a synthetic CT (same registered grid) via the nnUNet model."""
        if not MRI_MODE:
            return src_image
        import SimpleITK as sitk

        assert self.sct_converter is not None
        mr_arr = sitk.GetArrayFromImage(src_image).astype(np.float32, copy=False)  # (Z,Y,X)
        sct_arr = _clip_sct_hu(
            np.asarray(self.sct_converter.predict(mr_arr)).squeeze(),
            SCT_HU_MIN,
            SCT_HU_MAX,
        )
        sct = sitk.GetImageFromArray(sct_arr)
        sct.CopyInformation(src_image)  # sCT lives on the (co-registered) MR grid
        return sct

    # -- invoke phase --------------------------------------------------------
    def run(self, input_dir: Path, output_dir: Path) -> None:
        import SimpleITK as sitk

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = _read_metadata(input_dir)
        _log_metadata_schema(metadata)
        # output_file_idx -> {idx_in_output: cropped-dose payload}
        slots: dict[int, dict[int, dict | None]] = {}
        # output_file_idx -> reference sitk image (grid/geometry for that slot)
        slot_refs: dict[int, "sitk.Image"] = {}

        # Phase 1: read every source image; for the MRI task convert MR->sCT up front
        # so the (large) converter can be evicted from the GPU before dose compute.
        ct_images = []
        for entry in metadata:
            src_image = _read_source_image(input_dir, int(entry["image_file_idx"]))
            ct_images.append(self._to_ct(src_image))  # identity (CT) or MR->sCT (MRI)
        # Free the sCT model from the GPU — dose compute needs the full VRAM budget.
        self._release_sct()

        # Phase 2: dose compute per patient. A slot can be written as soon as the last
        # metadata entry that contributes to it is complete. This overlaps CPU-only MHA
        # assembly/compression for patient N with GPU dose compute for patient N+1 while
        # preserving the exact output data and ordering.
        if PIPELINE_OUTPUT_WRITES:
            remaining_slot_items = _output_slot_counts(metadata)
            slot_workers = min(N_OUTPUT_SLOTS, max(1, OUTPUT_WRITE_WORKERS))
            inner_workers = max(1, (os.cpu_count() or 4) // slot_workers)
            futures = []
            submitted: set[int] = set()
            _t_pipeline = time.perf_counter()
            with ThreadPoolExecutor(max_workers=slot_workers) as pool:
                def _submit_finalized_slot(ofidx: int) -> None:
                    if ofidx in submitted:
                        return
                    futures.append(pool.submit(
                        _write_sitk_slot,
                        output_dir,
                        ofidx + 1,
                        slots.get(ofidx),
                        slot_refs.get(ofidx),
                        inner_workers,
                    ))
                    submitted.add(ofidx)

                for entry, ct_image in zip(metadata, ct_images):
                    self._process_patient(
                        ct_image,
                        entry,
                        slots,
                        slot_refs,
                        remaining_slot_items=remaining_slot_items,
                        on_slot_ready=_submit_finalized_slot,
                    )
                    self._free_cuda()
                # Complete the ten-slot contract, including unused placeholders.
                for ofidx in range(N_OUTPUT_SLOTS):
                    if ofidx not in submitted:
                        _submit_finalized_slot(ofidx)
                for future in futures:
                    future.result()
            print(f"[write] pipelined output span {time.perf_counter() - _t_pipeline:.1f}s", flush=True)
        else:
            for entry, ct_image in zip(metadata, ct_images):
                self._process_patient(ct_image, entry, slots, slot_refs)
                self._free_cuda()
            _t_write = time.perf_counter()
            _write_output(output_dir, slots, slot_refs)
            print(f"[write] output stage {time.perf_counter() - _t_write:.1f}s", flush=True)
        self._save_compile_cache(output_dir)

    def _warmup_synthetic(self, hook) -> bool:
        """Compile the correction model on synthetic tensors, with no `/input` dependency.

        Only `hook.model` is wrapped by torch.compile, and compilation keys on shapes and
        dtypes, not values -- so feeding real patient data was never necessary. Doing it
        this way removes the open risk that `/input` is not mounted when the container
        loads: previously that made `_warmup` bail, and the whole compile slid into the
        first metered `/invoke`.

        Shapes mirror `_build_bev_features`:
            features   [N, C, D, cH, cW]     dose_pb  [N, 1, D, cH, cW]
            valid_mask [N, 1, D, cH, cW]     fan_mask [N, 1, 1, cH, cW]  (depth-invariant)
            material   [N, 1, D, cH, cW]     sigma_mm [N, 2]
        Two different depths are compiled on purpose. PyTorch only marks a dimension
        symbolic once it has seen it vary; a single depth would bake D in as a constant and
        the first real beamlet of another depth would recompile inside the metered path --
        the same class of bug as warming with a batch of 1.
        """
        import torch as _torch

        t0 = time.perf_counter()
        try:
            model = hook.model
            dev, dt = self.device, self.dtype
            n = max(1, int(DENSE_HOOK_BATCH_ITEMS))
            from training.proton.hooks import BEV_FEATURE_CHANNELS
            c = int(BEV_FEATURE_CHANNELS[hook.bev_feature_set])
            # bev_crop_h/w are HALF-widths: _crop_slices() returns
            # slice(centre - half, centre + half), so the tensor the model sees is twice
            # each of them (13,37 -> 26x74). Using them directly compiles a quarter of the
            # real voxel count, the warmup then "succeeds", and the first real forward
            # recompiles inside the metered /invoke -- the same trap as the shipped
            # 44x128-vs-26x74 BEV-crop bug.
            ch, cw = 2 * int(hook.bev_crop_h), 2 * int(hook.bev_crop_w)

            energies = getattr(model, "available_energies", None)
            # Inference tensors carry a distinct dispatch key. Creating these outside the
            # context compiled a normal-tensor graph, so real inference tensors missed its
            # guards and triggered a full recompile inside the metered invoke.
            with _torch.inference_mode():
                for depth in (320, 480):  # two depths => D stays symbolic
                    feats = _torch.zeros((n, c, depth, ch, cw), device=dev, dtype=dt)
                    dose = _torch.zeros((n, 1, depth, ch, cw), device=dev, dtype=dt)
                    valid = _torch.ones((n, 1, depth, ch, cw), device=dev, dtype=_torch.bool)
                    fan = _torch.ones((n, 1, 1, ch, cw), device=dev, dtype=_torch.bool)
                    mat = _torch.zeros((n, 1, depth, ch, cw), device=dev, dtype=_torch.long)
                    sigma = _torch.full((n, 2), 5.0, device=dev, dtype=dt)
                    energy = None
                    if energies is not None and len(energies) > 0:
                        val = float(energies[len(energies) // 2])
                        energy = _torch.full((n,), val, device=dev, dtype=dt)
                    # Match the real hook's autocast context. Compiling a FP32 warmup graph
                    # while DENSE_HOOK_AMP is enabled leaves the first FP16 invoke to compile
                    # a second graph in the metered path.
                    amp_enabled = bool(hook.inference_amp) and dev.type == "cuda" and dt == _torch.float32
                    with _torch.autocast(
                        device_type="cuda",
                        dtype=_torch.float16,
                        enabled=amp_enabled,
                    ):
                        hook._model_forward_tta(feats, dose, valid, fan, mat, energy, sigma)
            print(f"[warmup] synthetic compile in {time.perf_counter() - t0:.1f}s "
                  f"(N={n} C={c} HxW={ch}x{cw} depths=320,480)", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001 - best effort, falls back to real data
            print(f"[warmup] synthetic failed after {time.perf_counter() - t0:.1f}s "
                  f"({type(exc).__name__}: {exc}); falling back to real input", flush=True)
            return False
        finally:
            self._free_cuda()

    def _warmup(self, input_dir: Path) -> None:
        """Run ONE real beamlet through the full path so torch.compile emits its kernels
        during the unmetered health phase.

        Uses real data because only the tensor *shapes* drive compilation — which means we
        can skip the MR->sCT step on the MRI task and feed the MR straight in (same grid,
        so identical shapes). Never fatal: if /input is not readable yet, or anything else
        goes wrong, we simply pay the compile on the first real forward instead."""
        t0 = time.perf_counter()
        try:
            metadata = _read_metadata(Path(input_dir))
            if not metadata:
                print("[warmup] no metadata; skipping", flush=True)
                return
            entry = metadata[0]
            src = _read_source_image(Path(input_dir), int(entry["image_file_idx"]))
            beams = entry.get("beams") or []
            if not beams or not beams[0].get("rays"):
                print("[warmup] no beams; skipping", flush=True)
                return
            # Warm the shapes the METERED path actually uses, not just any shape. The dense
            # hook batches beamlets DENSE_HOOK_BATCH_ITEMS at a time, so a single-beamlet
            # warmup compiles a graph the real run never calls: the first real batch then
            # recompiles *inside* /invoke, which measured +10.4 s against eager and made
            # compile a net loss. Take one full batch plus a ragged tail so both the full
            # and partial shapes are compiled here, and dynamic=True generalises over the
            # batch dimension in the unmetered phase rather than on the first metered call.
            n_warm = DENSE_HOOK_BATCH_ITEMS + 2
            ray0 = dict(beams[0]["rays"][0])
            warm_beamlets = list(ray0["beamlets"][:n_warm])
            # Rays carry few beamlets each; pull from later rays until we have enough.
            extra_rays = []
            for ray in beams[0]["rays"][1:]:
                if len(warm_beamlets) >= n_warm:
                    break
                take = list(ray["beamlets"])[: n_warm - len(warm_beamlets)]
                if take:
                    extra_rays.append({**ray, "beamlets": take})
                    warm_beamlets.extend(take)
            ray0["beamlets"] = list(ray0["beamlets"][:n_warm])
            beam0 = dict(beams[0])
            beam0["rays"] = [ray0, *extra_rays]
            tiny = {"image_file_idx": entry["image_file_idx"], "beams": [beam0]}
            self._process_patient(src, tiny, {}, {})  # results discarded
            print(f"[warmup] compiled on real shapes in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort
            print(f"[warmup] skipped after {time.perf_counter() - t0:.1f}s ({exc})", flush=True)
        finally:
            self._free_cuda()

    def _compile_correction_model(self, hook) -> None:
        """Wrap the correction model in torch.compile, loading cache artifacts if we have
        any for this machine. Any failure degrades to eager — a compile problem must never
        cost a submission."""
        try:
            cache_path = self.checkpoint.parent / COMPILE_CACHE_FILE if COMPILE_CACHE_FILE else None
            if cache_path is not None and cache_path.is_file() and hasattr(torch.compiler, "load_cache_artifacts"):
                torch.compiler.load_cache_artifacts(cache_path.read_bytes())
                print(f"[compile] loaded cache artifacts from {cache_path}", flush=True)
            elif cache_path is not None:
                print(f"[compile] no cache at {cache_path}; kernels compile on first forward", flush=True)
            compile_kwargs: dict[str, object] = {"dynamic": COMPILE_DYNAMIC_SHAPES}
            if COMPILE_MODE != "default":
                compile_kwargs["mode"] = COMPILE_MODE
            hook.model = torch.compile(hook.model, **compile_kwargs)
            print(
                f"[compile] enabled (dynamic={COMPILE_DYNAMIC_SHAPES}, mode={COMPILE_MODE})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to eager, never fail
            print(f"[compile] WARNING: disabled ({exc}); running eager", flush=True)

    def _save_compile_cache(self, output_dir: Path) -> None:
        """Write compile cache artifacts next to the outputs. Used to harvest a cache for
        the *target* architecture from a try-out run — never inside a slot directory, since
        each slot must contain exactly one output.mha."""
        if not (SAVE_COMPILE_CACHE and COMPILE_CORRECTION_MODEL):
            return
        try:
            if not hasattr(torch.compiler, "save_cache_artifacts"):
                print("[compile] save_cache_artifacts unavailable in this torch build", flush=True)
                return
            artifacts = torch.compiler.save_cache_artifacts()
            blob = artifacts[0] if isinstance(artifacts, tuple) else artifacts
            if not blob:
                print("[compile] nothing to save (no compiled artifacts)", flush=True)
                return
            dest = Path(output_dir) / COMPILE_CACHE_FILE
            dest.write_bytes(blob)
            print(f"[compile] saved cache artifacts -> {dest} ({len(blob) / 1e6:.1f} MB)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[compile] WARNING: could not save cache ({exc})", flush=True)

    def _shrink_dense_batch(self) -> bool:
        """Halve the dense-correction forward batch (the dominant memory term) for the
        rest of the run. Returns False when already at 1 item. The hook reads
        ``max_inference_batch_items`` inside forward, so this takes effect immediately."""
        global DENSE_HOOK_BATCH_ITEMS
        if DENSE_HOOK_BATCH_ITEMS <= 1:
            return False
        DENSE_HOOK_BATCH_ITEMS = max(1, DENSE_HOOK_BATCH_ITEMS // 2)
        if self.correction_hook is not None:
            self.correction_hook.max_inference_batch_items = DENSE_HOOK_BATCH_ITEMS
        return True

    def _free_cuda(self) -> None:
        """Drop cached blocks so the next chunk (or an OOM retry) starts clean."""
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def _log_gpu_peak(self, tag: str) -> None:
        """Report peak GPU allocation — the number we need from the platform's logs
        when tuning MAX_BEAMLETS_PER_CHUNK / DENSE_HOOK_BATCH_ITEMS."""
        if self.device.type != "cuda":
            return
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        print(f"[gpu] {tag}: peak allocated {peak:.2f} GiB, reserved {reserved:.2f} GiB", flush=True)
        torch.cuda.reset_peak_memory_stats()

    def _release_sct(self) -> None:
        """Evict the MR->sCT converter from the GPU (no longer needed after phase 1)."""
        if self.sct_converter is None:
            return
        try:
            self.sct_converter.model.to("cpu")
        except Exception:
            pass
        self.sct_converter = None
        self._free_cuda()

    # -- one patient (one metadata entry / one CT) ---------------------------
    def _process_patient(
        self,
        ct_image,
        entry: dict,
        slots: dict,
        slot_refs: dict,
        *,
        remaining_slot_items: dict[int, int] | None = None,
        on_slot_ready: Callable[[int], None] | None = None,
    ) -> None:
        import SimpleITK as sitk

        assert self.lut is not None and self.machine_config is not None
        assert self.correction_hook is not None and self.sparse_hooks is not None

        ct_hu = sitk.GetArrayFromImage(ct_image).astype(np.float32, copy=False)
        origin_zyx = _origin_zyx(ct_image)
        resolution_zyx = _resolution_zyx(ct_image)
        ct_hu_t = torch.from_numpy(ct_hu).to(device=self.device, dtype=self.dtype)

        hu_to_density = self.beam_parameters.get("hu_to_density", {}).get("entries", None)

        # Dense-BEV crop setup. Use the PER-AXIS crops `from_checkpoint` already read out
        # of the training args (bev_crop_h/bev_crop_w). `bev_crop_hw` is only a legacy
        # scalar fallback for checkpoints predating the per-axis fields: on this model it
        # is 64 while the trained crops are (13, 37), so deriving the window from it fed
        # the correction net a 44x128 BEV instead of the 26x74 it was trained on — a 2.9x
        # larger, out-of-distribution input. Do NOT call set_bev_crop_half_widths() here;
        # that overwrote the correct checkpoint values.
        self.correction_hook.set_hu_volume(ct_hu_t)
        crop_h = int(self.correction_hook.bev_crop_h)
        crop_w = int(self.correction_hook.bev_crop_w)
        dense_field_size = (crop_h * 2, crop_w * 2)
        res_h, _, res_w = resolution_zyx
        trained_aspect = (crop_w * 1.0) / (crop_h * 3.0)  # training grid was 1x1x3 mm
        if abs((crop_w * res_w) / max(crop_h * res_h, 1e-8) - trained_aspect) > 0.05:
            print(f"[warn] grid spacing {resolution_zyx} differs from the 1x1x3 mm the crop "
                  f"({crop_h},{crop_w}) was trained on; BEV window aspect is off", flush=True)

        plan = {"beams": entry["beams"]}
        # Dose-scoring region: everything but air open to the outside. A plain density
        # threshold also zeroes internal cavities (trachea, bowel gas), where the MC
        # reference carries real dose. Density-only, so beam-independent -> compute once.
        _, _mass_for_mask = spr_and_mass_density(ct_hu_t, 150.0, hu_to_density)
        dose_mask = patient_dose_mask(_mass_for_mask)
        print(f"[dose_mask] external air {int((~dose_mask).sum()):,} voxels; internal "
              f"cavities kept {int((_mass_for_mask <= 0.03).sum()) - int((~dose_mask).sum()):,}",
              flush=True)
        del _mass_for_mask

        with torch.inference_mode():
            for beam_index, beam_json in enumerate(plan["beams"]):
                rays = beam_json["rays"]
                # SPR / mass density for this beam's mean energy (as in eval). Computed
                # once over ALL beamlets so chunking is bit-identical to a single call.
                e_ref = float(np.mean([
                    float(bl["energy"])
                    for ray in rays
                    for bl in ray["beamlets"]
                ]) or 150.0)
                beam_spr, beam_mass = spr_and_mass_density(ct_hu_t, e_ref, hu_to_density)
                beam_input = beam_spr.unsqueeze(0)
                beam_mass_input = beam_mass.unsqueeze(0)

                # One engine per beam, reused across chunks: compute_dose_* calls
                # _initialize_layers(sequence) itself, and that re-initializes whenever
                # beam count / angles / iso centers / SAD differ — so reuse is safe and
                # skips redundant setup when a chunk's geometry happens to match.
                engine = IonDoseEngine(
                    machine_config=self.machine_config,
                    lut=self.lut,
                    dose_grid_spacing=resolution_zyx,
                    dose_grid_shape=ct_hu.shape,
                    beam_template=None,
                    device=self.device,
                    dtype=self.dtype,
                    lateral_model=LATERAL_MODEL,
                    transport_step_mm=None,
                    sparse_hooks=self.sparse_hooks,
                    field_size=dense_field_size,
                    heterogeneous_mcs=HETEROGENEOUS_MCS,
                    material_radiation_length=False,
                )
                engine.set_patient_dose_mask(dose_mask)

                def compute_group(ray_group: list[int]) -> None:
                    """Run one bounded engine call over ``ray_group`` and stash its
                    per-beamlet doses on the CPU."""
                    sequence, ssd_values_mm = _make_beamlet_batch_sequence(
                        plan=plan,
                        beam_parameters=self.beam_parameters,
                        ct_hu=ct_hu,
                        origin_zyx=origin_zyx,
                        resolution_zyx=resolution_zyx,
                        beam_index=beam_index,
                        ray_indices=ray_group,
                        particles_per_beamlet=PARTICLES_PER_BEAMLET,
                        gantry_offset_deg=0.0,
                        skin_hu_threshold=SKIN_HU_THRESHOLD,
                        sigma_mode=SIGMA_MODE,
                        bams_to_iso_dist_mm=BAMS_TO_ISO_DIST_MM,
                        lut=self.lut,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    per_beamlet = engine.compute_dose_bev_lattice_sparse_batch(
                        sequence,
                        beam_input,
                        mass_density_image=beam_mass_input,
                        overwrite=False,
                        ssd_mm=ssd_values_mm,
                        finalize_chunk_size=DENSE_HOOK_BATCH_ITEMS,
                        return_per_beamlet=True,
                    )
                    del sequence

                    # per_beamlet is in ray->beamlet order over this ray_group; zip
                    # against output_info flattened in the same order.
                    flat_output_info = [
                        bl["output_info"]
                        for ri in ray_group
                        for bl in rays[ri]["beamlets"]
                    ]
                    if len(per_beamlet) != len(flat_output_info):
                        raise RuntimeError(
                            f"beamlet/output_info mismatch: {len(per_beamlet)} vs {len(flat_output_info)}"
                        )

                    for pb, oinfo in zip(per_beamlet, flat_output_info):
                        # Store the CROPPED dose (+offset), not a full-grid frame:
                        # densification to the full grid happens once, at write time.
                        ofidx = int(oinfo["output_file_idx"])
                        iidx = int(oinfo["idx_in_output"])
                        cutoff = float(oinfo["minimum_cutoff"])
                        slot_refs.setdefault(ofidx, ct_image)
                        slots.setdefault(ofidx, {})[iidx] = _beamlet_crop_cpu(pb, cutoff)
                        if remaining_slot_items is not None:
                            remaining_slot_items[ofidx] -= 1
                            if remaining_slot_items[ofidx] < 0:
                                raise RuntimeError(f"output slot {ofidx} completion count underflow")
                            if remaining_slot_items[ofidx] == 0 and on_slot_ready is not None:
                                # This slot is immutable from this point onward. Submit it
                                # immediately so CPU output work overlaps later engine chunks,
                                # even when the standardized runtime case contains one image.
                                on_slot_ready(ofidx)

                    del per_beamlet
                    # NOTE: deliberately NOT calling empty_cache()/gc.collect() here.
                    # It does not lower peak *allocated* (it only returns cached blocks to
                    # the driver, which torch must then re-request), and measured at ~4 s
                    # of overhead per engine call. Freeing happens on OOM retry and at
                    # beam/patient boundaries instead.

                def compute_group_adaptive(ray_group: list[int]) -> None:
                    """As ``compute_group``, but recover from CUDA OOM instead of failing
                    the submission. The dense forward dominates the peak (~5 GiB/item), so
                    shrink that first; only once it is at 1 item do we split the ray group.
                    Both are safe: shrinking the forward is a batching change, and beamlets
                    are independent and keyed by idx_in_output, so a retried group
                    recomputes the same frames."""
                    while True:
                        try:
                            compute_group(ray_group)
                            return
                        except Exception as exc:  # noqa: BLE001 - re-raised unless it's OOM
                            if not _is_cuda_oom(exc):
                                raise
                        self._free_cuda()
                        n_bl = sum(len(rays[ri]["beamlets"]) for ri in ray_group)
                        if self._shrink_dense_batch():
                            print(
                                f"[oom] {len(ray_group)} rays / {n_bl} beamlets did not fit; "
                                f"retrying with dense batch {DENSE_HOOK_BATCH_ITEMS}",
                                flush=True,
                            )
                            continue
                        if len(ray_group) <= 1:
                            raise RuntimeError(
                                f"out of memory on a single ray ({n_bl} beamlets) with the "
                                "dense batch already at 1 — cannot reduce further"
                            )
                        print(
                            f"[oom] {len(ray_group)} rays / {n_bl} beamlets did not fit; "
                            "splitting the ray group",
                            flush=True,
                        )
                        break
                    mid = len(ray_group) // 2
                    compute_group_adaptive(ray_group[:mid])
                    compute_group_adaptive(ray_group[mid:])

                # Chunk beamlets (by whole rays) so each engine call — and the GPU
                # tensors it holds — is bounded, then evict to CPU between chunks.
                groups = _group_ray_indices(rays, MAX_BEAMLETS_PER_CHUNK)
                n_beamlets = sum(len(r["beamlets"]) for r in rays)
                print(
                    f"[beam {beam_index}] {len(rays)} rays / {n_beamlets} beamlets "
                    f"-> {len(groups)} chunks (cap {MAX_BEAMLETS_PER_CHUNK})",
                    flush=True,
                )
                for group in groups:
                    compute_group_adaptive(group)

                del beam_spr, beam_mass, beam_input, beam_mass_input
                self._free_cuda()
                self._log_gpu_peak(f"beam {beam_index}")

        del ct_hu_t


# ---------------------------------------------------------------------------
# I/O helpers (Grand Challenge input/output contract)
# ---------------------------------------------------------------------------
def _log_metadata_schema(metadata: list[dict]) -> None:
    """Report every key present at each level of the real challenge metadata.

    We have never seen the real file -- the example submission ships no test data --
    so a field we silently ignore (a weight, an MU, a normalisation) would be invisible.
    This prints the observed schema once per run; anything here we do not consume is a
    lead. Keys only, never values, so it cannot leak challenge data into the log.
    """
    levels: dict[str, set] = {"entry": set(), "beam": set(), "ray": set(),
                              "beamlet": set(), "output_info": set()}
    for entry in metadata:
        levels["entry"].update(entry.keys())
        for beam in entry.get("beams", []):
            levels["beam"].update(beam.keys())
            for ray in beam.get("rays", []):
                levels["ray"].update(ray.keys())
                for bl in ray.get("beamlets", []):
                    levels["beamlet"].update(bl.keys())
                    levels["output_info"].update((bl.get("output_info") or {}).keys())
    for name, keys in levels.items():
        print(f"[schema] {name}: {sorted(keys)}", flush=True)


def _read_metadata(input_dir: Path) -> list[dict]:
    matches = list(input_dir.glob("stacked-*-beam-level-metadata.json"))
    if not matches:
        raise FileNotFoundError(f"no beam-level metadata json under {input_dir}")
    return json.loads(matches[0].read_text())


def _read_source_image(input_dir: Path, image_file_idx: int):
    """Read the task's source image (CT or MR) for a given metadata entry."""
    import SimpleITK as sitk

    slot = image_file_idx + 1  # metadata is 0-indexed, slots are 1-indexed
    slot_dir = input_dir / "images" / f"{SOURCE_IMAGE_BASE}-{slot}"
    mhas = list(slot_dir.glob("*.mha"))
    if len(mhas) != 1:
        raise FileNotFoundError(f"expected exactly one .mha in {slot_dir}, found {len(mhas)}")
    return sitk.ReadImage(str(mhas[0]))


def _group_ray_indices(rays: list[dict], max_beamlets: int) -> list[list[int]]:
    """Group ray indices so each group's total beamlet count stays <= max_beamlets
    (a single ray heavier than the cap forms its own group). Preserves ray order so
    the per-beamlet output stays aligned with the metadata."""
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_n = 0
    for ri, ray in enumerate(rays):
        nb = len(ray["beamlets"])
        if cur and cur_n + nb > max_beamlets:
            groups.append(cur)
            cur, cur_n = [], 0
        cur.append(ri)
        cur_n += nb
    if cur:
        groups.append(cur)
    return groups


def _beamlet_crop_cpu(pb: dict | None, cutoff: float) -> dict | None:
    """Move a per-beamlet dose crop to CPU (thresholded), keeping it small — a few
    hundred KB vs a ~160 MB full-grid frame. Densified to the full grid at write time.
    ``pb`` is None when the engine emitted no dose for this beamlet."""
    if pb is None:
        return None
    crop = pb["dose"][0].detach().to("cpu", torch.float32).numpy()
    if cutoff > 0.0:
        crop = crop.copy()
        crop[crop <= cutoff] = 0.0
    pz, py, px = (int(o) for o in pb["offset"])
    return {"dose": crop, "offset": (pz, py, px)}


def _densify(payload: dict | None, full_shape: tuple[int, int, int]) -> np.ndarray:
    """Paste a cropped beamlet dose into a full-grid zero frame (None -> all zeros)."""
    frame = np.zeros(full_shape, dtype=np.float32)
    if payload is not None:
        crop = payload["dose"]
        pz, py, px = payload["offset"]
        frame[pz:pz + crop.shape[0], py:py + crop.shape[1], px:px + crop.shape[2]] = crop
    return frame


def _write_compressed(image, output_path: Path) -> None:
    import SimpleITK as sitk

    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(output_path))
    writer.UseCompressionOn()
    writer.SetCompressionLevel(OUTPUT_COMPRESSION_LEVEL)
    writer.Execute(image)


def _format_mha_values(values) -> str:
    """Round-trip-safe MetaImage number formatting for spacing/origin/direction."""
    return " ".join(format(float(value), ".17g") for value in values)


def _write_streaming_mha(
    output_path: Path,
    frames_by_idx: dict[int, dict | None],
    ref,
    compression_level: int,
) -> tuple[float, float]:
    """Write a compressed 4D float MHA one dense frame at a time.

    SimpleITK's conventional path materializes every 3D frame, then ``JoinSeries``
    allocates another full 4D copy before compression.  MetaImage stores a 4D image as
    contiguous frame-major float data followed by ordinary zlib compression, so we can
    produce the identical voxel stream with one reusable frame-sized allocation.

    ``CompressedDataSize`` occurs before the local payload in an MHA header.  Reserve a
    fixed-width decimal field, stream the compressed bytes, then seek back to fill its
    final value without buffering the compressed file in RAM or a temporary file.
    """
    n = max(frames_by_idx) + 1
    if sorted(frames_by_idx) != list(range(n)):
        raise RuntimeError(f"idx_in_output not contiguous: {sorted(frames_by_idx)}")

    size_xyz = tuple(int(value) for value in ref.GetSize())
    spacing_xyz = tuple(float(value) for value in ref.GetSpacing())
    origin_xyz = tuple(float(value) for value in ref.GetOrigin())
    direction_3d = tuple(float(value) for value in ref.GetDirection())
    if len(size_xyz) != 3 or len(direction_3d) != 9:
        raise ValueError("streaming MHA output requires a 3D reference image")

    direction_4d = (
        # MetaIO serializes this matrix column-major relative to SimpleITK's
        # row-major GetDirection()/SetDirection() tuple.
        direction_3d[0], direction_3d[3], direction_3d[6], 0.0,
        direction_3d[1], direction_3d[4], direction_3d[7], 0.0,
        direction_3d[2], direction_3d[5], direction_3d[8], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    size_placeholder = "0" * 20
    header = (
        "ObjectType = Image\n"
        "NDims = 4\n"
        "BinaryData = True\n"
        "BinaryDataByteOrderMSB = False\n"
        "CompressedData = True\n"
        f"CompressedDataSize = {size_placeholder}\n"
        f"TransformMatrix = {_format_mha_values(direction_4d)}\n"
        f"Offset = {_format_mha_values((*origin_xyz, 0.0))}\n"
        "CenterOfRotation = 0 0 0 0\n"
        f"ElementSpacing = {_format_mha_values((*spacing_xyz, 1.0))}\n"
        f"DimSize = {size_xyz[0]} {size_xyz[1]} {size_xyz[2]} {n}\n"
        "AnatomicalOrientation = ????\n"
        "ElementType = MET_FLOAT\n"
        "ElementDataFile = LOCAL\n"
    ).encode("ascii")
    size_offset = header.index(size_placeholder.encode("ascii"))
    full_shape = tuple(reversed(size_xyz))
    compressor = zlib.compressobj(level=compression_level)
    compressed_size = 0
    densify_seconds = 0.0
    compression_started = time.perf_counter()
    with output_path.open("w+b") as output:
        output.write(header)
        for frame_idx in range(n):
            started = time.perf_counter()
            frame = _densify(frames_by_idx[frame_idx], full_shape)
            densify_seconds += time.perf_counter() - started
            block = compressor.compress(memoryview(frame).cast("B"))
            output.write(block)
            compressed_size += len(block)
        block = compressor.flush()
        output.write(block)
        compressed_size += len(block)
        if compressed_size >= 10**20:
            raise RuntimeError("compressed MHA payload exceeds reserved size field")
        output.seek(size_offset)
        output.write(f"{compressed_size:020d}".encode("ascii"))
    return densify_seconds, time.perf_counter() - compression_started - densify_seconds


def _output_slot_counts(metadata: list[dict]) -> dict[int, int]:
    """Count contributions so each slot can be written at its exact completion point."""
    counts: dict[int, int] = {}
    for entry in metadata:
        for beam in entry.get("beams", []):
            for ray in beam.get("rays", []):
                for beamlet in ray.get("beamlets", []):
                    output_info = beamlet.get("output_info") or {}
                    if "output_file_idx" in output_info:
                        ofidx = int(output_info["output_file_idx"])
                        counts[ofidx] = counts.get(ofidx, 0) + 1
    return counts


def _write_sitk_slot(
    output_dir: Path,
    slot: int,
    frames_by_idx: dict[int, dict | None] | None,
    ref,
    inner_workers: int,
) -> None:
    import SimpleITK as sitk

    # GC output contract (per official example-submission): outputs live under
    # /output/images/stacked-radiation-dose-map-{N}/output.mha
    slot_dir = output_dir / "images" / f"stacked-radiation-dose-map-{slot}"
    slot_dir.mkdir(parents=True, exist_ok=True)
    if not frames_by_idx or ref is None:
        placeholder = sitk.Image(1, 1, sitk.sitkFloat32)  # unused slot (matches example)
        _write_compressed(placeholder, slot_dir / "output.mha")
        return
    # contiguous 0..N-1 order; genuine 4D via JoinSeries (never GetImageFromArray on 4D).
    n = max(frames_by_idx) + 1
    if sorted(frames_by_idx) != list(range(n)):
        raise RuntimeError(f"slot {slot}: idx_in_output not contiguous: {sorted(frames_by_idx)}")
    if STREAMING_MHA_WRITES:
        _t_stream = time.perf_counter()
        densify_seconds, compression_seconds = _write_streaming_mha(
            slot_dir / "output.mha",
            frames_by_idx,
            ref,
            OUTPUT_COMPRESSION_LEVEL,
        )
        print(
            f"[write] slot {slot}: n={n} stream_densify={densify_seconds:.2f}s "
            f"stream_compress={compression_seconds:.2f}s "
            f"total={time.perf_counter() - _t_stream:.2f}s",
            flush=True,
        )
        return
    full_shape = tuple(reversed(ref.GetSize()))  # (z, y, x)
    _t_den = time.perf_counter()
    # Densify dominates the write stage: measured 2026-08-07 at 3.87 s of a 5.3 s write
    # (~42% of the whole metered invoke) on a 498x493x164 grid. Each frame is a 161 MB
    # full-grid volume that GetImageFromArray then copies, so 30 frames move ~4.8 GB
    # twice. Reusing one buffer removed only the allocation (-17%); the rest is the copy
    # itself, which is per-frame independent -- so run the frames on a thread pool.
    # numpy's paste and SimpleITK's buffer copy both drop the GIL, so this scales.
    def _build(i: int):
        img = sitk.GetImageFromArray(_densify(frames_by_idx[i], full_shape))
        img.CopyInformation(ref)  # exact grid match to the source image
        return img

    workers = min(n, inner_workers)
    if workers > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            ordered = list(pool.map(_build, range(n)))  # map preserves order
    else:
        ordered = [_build(i) for i in range(n)]
    _d_den = time.perf_counter() - _t_den
    _t_join = time.perf_counter()
    stacked = sitk.JoinSeries(ordered)
    _d_join = time.perf_counter() - _t_join
    # JoinSeries owns a full 4D copy. Release the N input volumes before compression,
    # which halves live host memory and makes bounded parallel slot writes practical.
    del ordered
    _t_w = time.perf_counter()
    _write_compressed(stacked, slot_dir / "output.mha")
    print(f"[write] slot {slot}: n={n} densify={_d_den:.2f}s join={_d_join:.2f}s "
          f"write={time.perf_counter() - _t_w:.2f}s", flush=True)
    del stacked


def _write_output(
    output_dir: Path,
    slots: dict[int, dict[int, dict | None]],
    slot_refs: dict[int, "object"],
) -> None:
    slot_workers = min(N_OUTPUT_SLOTS, max(1, OUTPUT_WRITE_WORKERS))
    inner_workers = max(1, (os.cpu_count() or 4) // slot_workers)

    if slot_workers > 1:
        with ThreadPoolExecutor(max_workers=slot_workers) as pool:
            list(pool.map(
                lambda slot: _write_sitk_slot(
                    output_dir,
                    slot,
                    slots.get(slot - 1),
                    slot_refs.get(slot - 1),
                    inner_workers,
                ),
                range(1, N_OUTPUT_SLOTS + 1),
            ))
    else:
        for slot in range(1, N_OUTPUT_SLOTS + 1):
            _write_sitk_slot(
                output_dir,
                slot,
                slots.get(slot - 1),
                slot_refs.get(slot - 1),
                inner_workers,
            )


# ---------------------------------------------------------------------------
# Grand Challenge invoke entrypoints (used by app.py).
#   init_model() -> called once at startup (/health-time, unmetered).
#   run(model)   -> called per POST /invoke; reads /input, writes /output.
# ---------------------------------------------------------------------------
def init_model() -> "DoseModel":
    return DoseModel().load()


def run(model: "DoseModel") -> None:
    model.run(GC_INPUT_DIR, GC_OUTPUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("/input"))
    ap.add_argument("--output", type=Path, default=Path("/output"))
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--machine-mat", type=Path, default=DEFAULT_MACHINE_MAT)
    ap.add_argument("--beam-params", type=Path, default=DEFAULT_BEAM_PARAMS)
    args = ap.parse_args()

    # The load-time warmup reads GC_INPUT_DIR (which is /input in the container). When
    # driven from the CLI the input lives elsewhere, so point it at the same directory or
    # the warmup silently skips.
    global GC_INPUT_DIR
    GC_INPUT_DIR = Path(args.input)

    t0 = time.perf_counter()
    model = DoseModel(
        checkpoint=args.checkpoint,
        machine_mat=args.machine_mat,
        beam_params=args.beam_params,
    ).load()
    print(f"[load] {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    model.run(args.input, args.output)
    print(f"[run]  {time.perf_counter() - t1:.1f}s", flush=True)


if __name__ == "__main__":
    main()
