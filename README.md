# DoseRAD2026 — proton dose engine + learned correction (MAALGO)

Our entry to the [DoseRAD2026 Grand Challenge](https://doserad2026.grand-challenge.org/),
tasks **Proton dose on CT** and **Proton dose on MRI**.

The method is a **differentiable analytic proton pencil-beam engine** whose residual against
Monte-Carlo is removed by a **learned correction network operating in beam's-eye view**. Both
halves are PyTorch and differentiable end to end, which is what makes the machine LUT itself
trainable against water-phantom MC.

The engine is built on [PyDoseRT](https://github.com/UMU-DDI/PyDoseRT), a differentiable dose-calculation
framework; `src/pydose_rt/` is that codebase, extended here with the ion/proton path
(`ion_dose_engine`, `ion_lut`, `spr`, `materials`, the DoseRAD data loaders). The photon
engine is kept intact as the framework it came from. No photon *experiments* are part of
this work.

> **On the proton engine.** The proton/ion dose engine in `src/pydose_rt/` is being upstreamed:
> it will become part of the official [PyDoseRT](https://github.com/UMU-DDI/PyDoseRT)
> repository. This repository is a **snapshot of the code as submitted to DoseRAD2026**, kept
> so the challenge results stay reproducible against the exact version that produced them.
> It is not the place to track the engine's ongoing development. Once the upstream release
> lands, use PyDoseRT for the engine and treat this repository as the challenge record.

---

## Pipeline

```
   Geant4 material table            water-phantom MC (1e8 / 1e9 histories)
            │                                    │
            ▼                                    ▼
   deterministic SPR / WEQ          scripts/export_proton_lut_fast.py  stage 1: fit
   (physics/spr.py)                 scripts/optimize_lut_water.py      stage 2: backprop
            │                                    │                      through the engine
            │                                    ▼
            │                        lut_fast_3d_1e8_opt.mat  ── the machine model
            │                                    │
            └────────────┬───────────────────────┘
                         ▼
          IonDoseEngine  (engine/ion_dose_engine.py)
          double-Gaussian core+halo, heterogeneous MCS, sparse spot hooks
                         │
                         │  per-beamlet BEV volume (26 x 74 field, 13 x 37 crop)
                         ▼
          RepVGG U-Net correction  (training/common/repvgg_unet_corrector.py)
          additive residual, energy + sigma + material conditioning
                         │
                         ▼
              corrected per-beamlet dose  ──►  container/ (Grand Challenge invoke API)
```

## What is in here

| Path | What it is |
|---|---|
| `src/pydose_rt/` | The dose engine. Proton path: `engine/ion_dose_engine.py`, `physics/kernels/ion_lut.py`, `physics/spr.py`, `physics/materials.py`, `data/ion_beam.py`, `data/doserad.py`. Photon path retained from upstream PyDoseRT. |
| `training/` | The correction-network trainer (`proton/train_dense_correction.py`), the models (`common/repvgg_unet_corrector.py` and the earlier fan/lattice correctors), runtime hooks. |
| `scripts/` | LUT fitting and differentiable calibration, per-case evaluation against the official metrics, container fixtures and data preparation. |
| `container/` | The Grand Challenge algorithm container (invoke API, per-beamlet 4-D output) for both the CT and MRI variants. |
| `model_ct/`, `model_mri/` | The shipped model bundles: EMA checkpoint, machine LUT, beam parameters, runtime config. |
| `docs/` | Method notes: the pencil-beam kernel model, WEQ computation, heterogeneity-aware MCS, and the split-kernel water calibration. |
| `example_data/` | The pyRadPlan base machine model and the fitted LUTs. |

## Shipped model

`model_ct/latest_ema.pt` and `model_mri/latest_ema.pt` are the **same** dose-correction
network — the MRI variant differs only in its MR→synthetic-CT front end. It is **epoch 27,
step 193 888**, EMA weights: 30 epochs of full-beamlet training on one A100, run in two
stints because the first hit a wall-clock limit at epoch 10 and was resumed from `latest.pt`
with the LR schedule, EMA and optimizer state intact.

### The settings that produced it

Read back out of the checkpoint's own `args`, so this is what actually ran — not what we
meant to run:

```bash
python training/proton/train_dense_correction.py \
  `# data` \
  --case-list training/splits/train_cases.txt \
  --val-case-list training/splits/val_cases.txt \
  --beam-params-path <DoseRAD2026>/proton/training/beam_parameters.json \
  --machine-mat example_data/mc_fit_smooth/lut_fast_3d_1e8_opt.mat \
  `# analytic baseline` \
  --mode gauss_double --heterogeneous-mcs --sigma-mode beam_params \
  --particles-per-beamlet 1e6 --bams-to-iso-dist-mm 1000.0 --skin-hu-threshold -500.0 \
  --field-size 26 74 --bev-crop-h 13 --bev-crop-w 37 \
  `# model: RepVGG U-Net, 887k params` \
  --model-kind repvgg_unet --use-repvgg --model-depth 0 \
  --unet-native-dim 64 --unet-stage-dims 96 128 192 --unet-stage-blocks 2 2 2 4 2 \
  --unet-extra-stage-dim 256 --unet-extra-stage-blocks 4 \
  --unet-equalize-axis w --unet-equalize-factor 3 \
  --unet-norm group --unet-energy-conditioning embedding \
  --unet-conditioning-injection entrance --use-sigma-conditioning \
  --material-embedding-dim 4 --depth-kernel-size 11 --mix-ratio 0.5 --dropout 0.0 \
  --bev-feature-set v2 --residual-mode additive --additive-scale-frac 0.25 \
  `# objective` \
  --w-dose 1.0 --w-energy 0.0 --w-idd-z 0.0 --w-halo-int 0.0 \
  --loss-mask additive --loss-high10-frac 0.7 --loss-high10-weight 0.15 \
  --bev-deep-supervision-weight 0.05 \
  `# optimisation` \
  --epochs 30 --beam-sampling full --max-beamlets-per-beam 10 --beamlet-sampling random \
  --no-augmentation --lr 5e-4 --optimizer adamw --poly-power 0.9 \
  --weight-decay 1e-5 --grad-clip 1.0 --ema-decay 0.999 \
  --amp --amp-dtype bfloat16 --dtype float32 \
  `# selection, determinism, checkpointing` \
  --best-metric mae_high10_pct --validate-every-epochs 1 \
  --seed 12345 --val-seed 12345 \
  --compile-model --compile-mode default --checkpoint-every-steps 2000 --resume
```

Notes on the parts that are easy to get wrong:

- **`--bev-feature-set v2`** is 9 input channels: the v1 eight plus range-relative depth
  `(weq − R_peak(E)) / depth_scale`. It has to be trained from scratch — warm-starting adds
  the new channel as a zero column that never leaves zero.
- **`--beam-sampling full`** makes `--steps-per-epoch` irrelevant: an epoch is one pass over
  every training beamlet, ≈ 7 180 steps here.
- **`--amp-dtype bfloat16`** (the default). fp16 autocast overflows in the decoder on models
  this size and shows up as a loss explosion around step 1e6.
- **`--compile-model` and `--resume`** entered on the second stint, not the first;
  loading an eager checkpoint into a compiled model is handled by `load_model_state_dict`.
- The **topological dose mask** (`pydose_rt.physics.spr.patient_dose_mask`) is automatic, not
  a flag: internal air cavities are scored, instead of being thresholded away by density.

## Data and weights

**The dose-correction weights are in this repository.** `model_ct/` and `model_mri/` each
hold a complete runtime bundle — the 3.5 MB EMA checkpoint, the machine LUT, the beam
parameters and the runtime config — so the CT container can be built and run straight from
a clone, with nothing to download.

Two artifacts are too large for git and live on Hugging Face:

| Artifact | Size | Location |
|---|---|---|
| Water-phantom Monte-Carlo simulations (the LUT fit's input) | ~240 GB | [`datasets/zimmeryWo/MC_proton_simulation_DoseRAD2026`](https://huggingface.co/datasets/zimmeryWo/MC_proton_simulation_DoseRAD2026) |
| MR→synthetic-CT nnU-Net bundle (MRI task only) | 409 MB | [`zimmeryWo/MRI-sCT_converter-DoseRAD2026`](https://huggingface.co/zimmeryWo/MRI-sCT_converter-DoseRAD2026) |

Download the sCT bundle to `model_mri/mrtoct_bundle/` before building the MRI image; the CT
task does not need it.

```bash
hf download zimmeryWo/MRI-sCT_converter-DoseRAD2026 \
    --include "mrtoct_bundle/*" --local-dir model_mri/
```

**The MR→synthetic-CT network is not trained by the code in this repository.** It is an
nnU-Net regression model (deep-supervision MAE with a feature-space regularization term),
trained with a separate fork: **https://github.com/LFetty/nnUNet**
(branch `feature/image-to-image-translation`). Only *inference* lives here —
`container/standalone_regression_inference.py` runs the exported TorchScript bundle with no
nnU-Net dependency, and the model card at the link above documents the normalization,
the required output clipping to `[-1024, 3071]`, and the settings we submitted.

The challenge data itself is not redistributed. Paths are always command-line arguments
(`--dose-root`, `--beam-params-path`, `--mc-dir`, `--edep-dir`); nothing is hard-coded to a
local mount.

## Reproducing

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

**1 — Fit the machine LUT** from water-phantom Monte Carlo. Two stages: an analytic
per-energy fit of the double-Gaussian lateral shape and the depth dose, then a calibration
that *backpropagates through the dose engine* against the same MC.

```bash
# stage 1 — fit all energies (the shipped recipe: direct double-Gaussian fit, 74 mm
# lateral integration window matching the pencil-beam kernel support)
uv run python scripts/export_proton_lut_fast.py \
    --edep-dir <water-phantom MC root> \
    --double-fit-mode direct --kernel-width-mm 74 \
    --output-mat-path example_data/mc_fit_smooth/lut_fast_3d_1e9.mat

# stage 2 — differentiable calibration of the fitted curves against the MC
uv run python scripts/optimize_lut_water.py --all \
    --mc-dir <water-phantom MC root> \
    --base-lut example_data/mc_fit_smooth/lut_fast_3d_1e9.mat \
    --out-lut  example_data/mc_fit_smooth/lut_fast_3d_1e9_opt.mat
```

`scripts/fit_proton_lut.py` is the single-energy diagnostic behind stage 1;
`scripts/compare_proton_luts.py` diffs two LUTs; `scripts/benchmark_pb_vs_mc_water.py`
scores the analytic engine against water MC.

**2 — Train the correction network** with the settings under
[The settings that produced it](#the-settings-that-produced-it) above. It is a single
`python training/proton/train_dense_correction.py` invocation — scheduling it is up to you;
ours took two A100 stints of ~5 days. `training/README.md` covers the argument surface and
the decisions behind the defaults, and the root `Dockerfile` is the CUDA/PyTorch image it
trained in.

**3 — Evaluate** against the challenge MC with the official per-beamlet metrics (masked MAE
and z-axis IDD distance, one value per beamlet, `nanmean` over all of them):

```bash
# one case, with the shipped correction model
uv run python scripts/evaluate_doserad_proton_case.py \
    --case-dir <DoseRAD2026>/proton/training/1ABB011 \
    --correction-checkpoint model_ct/latest_ema.pt

# the whole set; omit --correction-checkpoint to score the uncorrected analytic baseline
uv run python scripts/evaluate_doserad_proton_case.py \
    --cases-dir <DoseRAD2026>/proton/training \
    --correction-checkpoint model_ct/latest_ema.pt
```

`scripts/compare_container_vs_eval.py` scores the *container* path against the same ground
truth, which is what catches an error that exists only in the deployed path.

For the MRI task, `container/gen_sct_cases.py` writes case directories in which the true CT
is replaced by the model's synthetic CT, with the plan and reference dose carried over
unchanged. Running the evaluation above on those directories isolates the dosimetric cost of
the MR→sCT step from the dose engine itself.

**4 — Build the submission container:**

```bash
docker build -f container/Dockerfile.lean     -t doserad-proton-ct:lean .
docker build -f container/Dockerfile.mri.lean -t doserad-proton-mri:lean .
```

The container reads its bundle from `$MODEL_DIR` (`/opt/ml/model` in the image). To exercise
the full invoke path locally against a training case — it builds a Grand Challenge request,
runs inference, and verifies that the per-beamlet reassembly equals the engine's summed path:

```bash
MODEL_DIR=$PWD/model_ct uv run python container/selftest.py --case <DoseRAD2026>/proton/training/1ABB006
```

The input/output contract is the Grand Challenge algorithm interface for the DoseRAD2026
proton tasks; `container/app.py` and `container/inference.py` implement it, and
`container/selftest.py` exercises it end to end.
`container/SHA256SUMS-lean-runtime-20260830.txt` records the digests of the images we submitted.

## Tests

```bash
uv run python -m pytest tests/unittests
```

265 tests, no external data required. `tests/smoketests/` and `tests/benchmarks/` need
clinical DICOM and are excluded from that run.

## License

MIT — see `LICENSE`.
