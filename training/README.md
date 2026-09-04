# Training the dose-correction network

The analytic engine (`pydose_rt.engine.ion_dose_engine.IonDoseEngine`) computes a per-beamlet
dose in beam's-eye view; a convolutional network predicts the residual against the challenge
Monte-Carlo reference for that beamlet. Engine and network run inside the same graph, so the
correction is trained against MC end to end.

## Layout

- `proton/train_dense_correction.py` — **the maintained trainer.** Dense per-beamlet BEV
  correction; this is what produced the shipped model.
- `proton/train_dota_direct.py` — the direct-prediction (DoTA-style) arm. Kept as the
  negative result it is: ~33x worse than correcting the analytic baseline.
- `proton/hooks.py` — runtime hooks that inject the trained corrector into the engine's
  sparse ion path. The same hooks are used by the inference container.
- `common/repvgg_unet_corrector.py` — the shipped model: a RepVGG U-Net over the BEV volume,
  conditioned on energy, spot sigma, and material, emitting an additive residual. RepVGG
  branches are folded at load time when `trainable=False`.
- `common/ray_sequence_corrector.py`, `common/separable_fan_grid_corrector.py`,
  `common/dota_corrector.py` — earlier correction architectures (fan-conv, separable
  lattice, transformer). Superseded, retained for the ablation record.
- `common/checkpoints.py` — portable state-dict save/load shared with the container.
- `common/masked_norm.py`, `common/materials.py` — normalisation over the valid-dose mask,
  and the Geant4 material table bridge.

## Running it

One process, one invocation:

```bash
PYTHONPATH=src:.:scripts uv run python -u training/proton/train_dense_correction.py ...
```

The complete argument list that produced the shipped checkpoint — read back out of the
checkpoint itself — is in the top-level [`README.md`](../README.md#the-settings-that-produced-it).
`--help` lists the full surface. `PYTHONPATH` must include `scripts/`: the trainer shares
`doserad_proton_utils.py` with the evaluation and container code.

The shipped model is 30 epochs of full-beamlet training on one A100, run in two stints
because the first hit a wall-clock limit at epoch 10. `--resume` restores model, EMA,
optimizer, the polynomial LR schedule, the grad scaler, step and history from `latest.pt`
and skips the epochs already done, so a split run is equivalent to an uninterrupted one —
which is why the shipped checkpoint carries `resume=True`. The continuation also switched
`torch.compile` on (`--compile-model --compile-mode default`, ~34% off epoch wall time);
`load_model_state_dict` handles the `_orig_mod.` prefix in both directions, so an eager
checkpoint loads into a compiled model and back.

Use `--checkpoint-every-steps` on any long run: at 0 (the default) a kill costs the whole
in-flight epoch, which at full beam sampling is 8–12 hours.

## Things worth knowing before changing anything

- **bfloat16 autocast, not fp16.** The recurring "loss explodes around step 1e6" on 887k+
  models was fp16 autocast overflow in the decoder, not instability. bf16 is the default.
- **EMA weights (`--ema-decay 0.999`) are what we ship.** EMA wins by ~5.7% late in a long
  run and *loses* on short screens, where it has not converged — do not judge it on a screen.
- **The dose mask is topological** (`pydose_rt.physics.spr.patient_dose_mask`): only external
  air is excluded. Thresholding on density instead blinds the model in trachea and bowel gas,
  where the MC has real dose.
- **Steps and objective terms are orthogonal levers.** More steps and full-beamlet exposure
  buy voxel MAE; objective terms (`--w-halo-int`, `--w-idd-z`, `--w-energy`) buy integral/IDD
  fidelity. Neither moves the other.
- **Warm-starting cannot screen new input channels.** A zero-initialised new column stays
  zero, because the existing channels already explain the residual. Feature-set arms have to
  be trained from scratch.

## Data

`--case-list` and `--val-case-list` take one case directory per line; `splits/train_cases.txt`
and `splits/val_cases.txt` are the 67/8 split the shipped model was trained on. Those lines are
written as `/data/proton/training/<case>`, which is where the training image mounts the
challenge data — rewrite the prefix for your own layout, or pass `--case-dir` / `--val-case-dir`
directly. Relative paths resolve against the working directory. Nothing else is hard-coded.

Reference doses are read per beamlet. The loader prefers a `.b2nd` next to each `.mha` and
falls back to the `.mha` itself, so training runs straight off the challenge data — but
reading a full-grid MetaImage per beamlet dominates the input pipeline.
`scripts/convert_ref_doses_b2nd.py` writes the `.b2nd` form once (nonzero bounding box,
Blosc2-compressed, the original `.mha` left in place); we ran it over the training set
before the shipped run.
