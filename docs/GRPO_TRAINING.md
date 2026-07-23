# S2 Pro Korean LoRA-GRPO training and verification manual

## 1. Scope and verified status

This manual records the Fish-local S2 Pro broad ordinary-LoRA GRPO mechanics
pipeline that was implemented and run on 2026-07-22. It is not the Fish S2
trainer, a Fish S2 reproduction, rsLoRA training, a quality result, or a
full-data training run.

The fixed model is Fish Audio S2 Pro.

- Hugging Face repository: <https://huggingface.co/fishaudio/s2-pro>
- Base revision: `1de9996b6be38b745688de084d87a5633f714e4e`
- Fixed revision URL: <https://huggingface.co/fishaudio/s2-pro/tree/1de9996b6be38b745688de084d87a5633f714e4e>
- Model type: `fish_qwen3_omni`
- Public metadata BF16 parameter count: `4,561,852,416`

| Item | Status | Verified scope |
|---|---|---|
| S2 Pro checkpoint loading | Tested | The fixed revision loaded from both safetensor shards with all keys matched. |
| S2 codec semantic extraction | Tested | `modded_dac_vq_s2` processed three staged train files and one staged dev file. |
| Broad ordinary-LoRA SFT | Tested | `r=8`, `alpha=16`, dropout `0.01`, one optimizer step on one RTX 4060 Ti 16 GB. |
| `text2semantic_grpo` config and trainer | Tested | The Fish-local GRPO config composed and a full S2 Pro training step ran. |
| DualAR rollout and behavior log-probabilities | Tested | `G=2` grouped completions stored generated Slow q0 and Fast q1-q9 behavior log-probabilities at rollout. |
| Frozen KL reference | Tested | The frozen SFT adapter snapshot was restored only to score KL reference log-probabilities. |
| Current-policy scoring and GRPO update | Tested | The live policy re-scored stored actions, used behavior as the PPO denominator, and updated LoRA only. |
| Adapter-only checkpoint and resume | Tested | Step 1 checkpoint restored model, optimizer, scheduler, and GRPO metadata, then reached optimizer step 2. |
| Korean production reward | Not available | The production gate fails closed until a Korean human-validated scorer exists. |
| MLP-only rsLoRA `r=16`, `alpha=64` | Not implemented | No `alpha/sqrt(r)` scaling or compatible rsLoRA warm start exists. |
| Quality, CER, speaker similarity, and full-data training | Not established | No validated scorer, evaluation, rights clearance, or full-data result exists. |

The verified mechanics output is:

```text
/home/ms-ms-home2/fish-s2pro-real4-smoke/results/s2pro_real4_r8a16_grpo_final
```

Its two verified checkpoints are `step_000000001.ckpt` and
`step_000000002.ckpt`.

## 2. Non-negotiable gates

### 2.1 Run project code only in the project Docker container

Run tests, dependency use, checkpoint loading, preprocessing, SFT, and GRPO
inside the project Docker container. The host is for read-only inspection, Git,
Docker orchestration, and approved staging copies only. Keep Fish and
`archive/GLM-TTS` dependencies separate.

### 2.2 Training rights are a hard stop gate

File access does not grant training, voice-cloning, redistribution, or
evaluation rights. Confirm the following before any run outside the documented
mechanics smoke:

- Service and voice training permissions.
- Recorded-person training and voice-cloning consent.
- S2 Pro weight and codec license terms.
- Data retention, export, and deletion requirements.
- Reference-audio and human-label evaluation permissions.

If any item is unclear, stop. Never infer consent from file availability.

### 2.3 Preserve the source tree

The intended source is:

```text
/home/ms-ms-home2/dataset/tts/250623_tts_jiwoo_whole_services/tts
```

Mount it read-only. Never rename, rewrite, move, or add `.lab`, `.npy`,
manifests, or temporary files beside source MP3 files. The semantic extractor
writes `.npy` beside its input audio, so it must receive staged copies only.

A manifest must retain source relative path, raw filename stem, normalized
transcript, voice and service identity, include or exclude reason, source and
audio hashes, snapshot hash, and split. Group duplicate normalized text,
session, and duplicate audio across split boundaries. Use text or
session-disjoint and voice or service-held-out evaluation, not an utterance-only
random split.

Normalize a filename stem as follows:

1. Apply strict `urllib.parse.unquote_plus()` exactly once.
2. Remove BOM, collapse whitespace, trim, and apply NFC while preserving `?`
   and `!` order.
3. If `%xx` remains, apply strict `urllib.parse.unquote()` at most once and
   only for non-URL, non-email text.
4. Never decode repeatedly to a fixed point.
5. Convert numeric `%` to `퍼센트` and non-URL literal `+` to `플러스`.
6. Exclude ambiguous percent, URL or email residual encoding, unresolved nested
   escapes, DOS 8.3 aliases, empty or non-text stems, and decode or control
   failures with explicit reason codes.
7. Accept only NFC transcripts with no `%`, literal `+`, `%xx` remainder, BOM,
   or control character. Preserve digit sequences and `?` and `!` order.

The verified read-only snapshot `71214cfc...e6508` contained 793,184 MP3s,
792,779 accepted transcripts, and 405 exclusions. Re-freeze and recount before
any new work because the source tree can change. The selected four-file smoke
snapshot hash was
`ec1d3f719cf55cbbe912f05e97c5ed38aa8d318a0b4ec739f032c7da1684135e`.

## 3. S2 Pro preprocessing and SFT smoke facts

Use `modded_dac_vq_s2` for this S2 Pro revision. Its codec context buffers are
16384 and 4096. The older `modded_dac_vq` configuration uses 8192 and 2048 and
fails checkpoint loading with shape mismatches.

The broad ordinary-LoRA SFT smoke used three staged train MP3 files and one
staged dev MP3 file, batch size 1, one dataloader worker, one visible RTX 4060
Ti 16 GB, and `max_steps=1`. It reached loss 30.50, wrote
`step_000000001.ckpt`, and peaked at 10,319 MiB. This proves preprocessing,
checkpoint load, forward, backward, and optimizer plumbing only.

The SFT adapter used by the verified GRPO run is:

```text
/work/results/s2pro_real4_r8a16/checkpoints/step_000000001.ckpt
```

Its SHA-256 is
`431bfb6c7953833702d718c7bc99403cf2bcbff9fc5b1d5a6ef53f48c46a4035`.

It is a broad ordinary-LoRA adapter with `r=8`, `alpha=16`, and targets
`attention`, `mlp`, `embeddings`, and `output`. SFT used dropout `0.01`.
GRPO continues the same topology with dropout `0`. This broad baseline is not
MLP-only rsLoRA and not a Fish S2 reproduction.

For S2 semantic extraction, run only on staging directories:

```bash
python tools/vqgan/extract_vq.py data/staging/train \
  --num-workers 1 \
  --batch-size 16 \
  --config-name modded_dac_vq_s2 \
  --checkpoint-path checkpoints/s2-pro/codec.pth

python tools/vqgan/extract_vq.py data/staging/dev \
  --num-workers 1 \
  --batch-size 16 \
  --config-name modded_dac_vq_s2 \
  --checkpoint-path checkpoints/s2-pro/codec.pth
```

Build separate protobuf directories for each split. Never use the test split
for training or model selection.

```bash
python tools/llama/build_dataset.py \
  --input data/staging/train \
  --output data/protos/train \
  --text-extension .lab \
  --num-workers 16

python tools/llama/build_dataset.py \
  --input data/staging/dev \
  --output data/protos/dev \
  --text-extension .lab \
  --num-workers 16
```

The historical SFT smoke image was `fishaudio/fish-speech:nightly-dev`, image
ID `sha256:37a7d6feec604a2df9f949a2820b3d1e057550dd2e3776ea5c2419543bf48853`,
and repo digest
`fishaudio/fish-speech@sha256:14a36bad3678e61dd97f510a0e6b824f5350a4d2642fe6eebbe9f7d35f1ed282`.
Its bundled protobuf 3.19.6 is too old for generated dataset code. Align it
inside the container with the project's `protobuf>=3.20,<6` override before
protobuf packing.

## 4. Implemented GRPO mechanics

The entry config is `fish_speech/configs/text2semantic_grpo.yaml`. It starts
from the fixed S2 Pro base revision and loads the broad SFT adapter into the
same ordinary-LoRA topology. `fish_speech/configs/lora/r_8_alpha_16_grpo.yaml`
sets `r=8`, `lora_alpha=16`, and `lora_dropout=0`.

The verified smoke used these parameters:

| Parameter | Value |
|---|---|
| Config | `text2semantic_grpo` |
| Base revision | `1de9996b6be38b745688de084d87a5633f714e4e` |
| SFT warm start | `/work/results/s2pro_real4_r8a16/checkpoints/step_000000001.ckpt` |
| SFT adapter checkpoint SHA-256 | `431bfb6c7953833702d718c7bc99403cf2bcbff9fc5b1d5a6ef53f48c46a4035` |
| LoRA topology | Broad `attention`, `mlp`, `embeddings`, `output` |
| LoRA rank, alpha, dropout | `8`, `16`, `0` |
| Group size | `G=2` |
| Generated-frame cap | 16 |
| Rollout seed | 1235 |
| Actions | Slow q0 and Fast q1 through q9 only |
| PPO clip | 0.2 |
| KL weight | 0.1 |
| Reward | `[0, 1]`, `TEST_ONLY_NOT_A_QUALITY_REWARD` |
| Advantage | Centered group reward, no group-standard-deviation normalization |
| Scheduler override | `model.lr_scheduler.lr_lambda.num_warmup_steps=0` |

`grpo_rollout.py` samples `G` DualAR completions, stores generated actions and
the behavior log-probabilities at rollout, and writes EOS, truncation, length,
and generated-only masks. The Slow policy action is q0. Fast q0 is the same
semantic action in codec form, so the Fast policy branch uses q1 through q9
only. Prompt, reference, and padding tokens are excluded from both action masks.

`grpo_module.py` keeps three distinct policy roles:

1. The behavior policy snapshot generates a rollout and supplies the PPO
   denominator log-probabilities.
2. The frozen SFT adapter snapshot is restored only while scoring the KL
   reference, then the current adapter is restored.
3. The current policy re-scores the exact stored actions before the update.

The frozen SFT adapter hash is:

```text
7086e88223b9b19ac8f883d9275f645b200824928b209cce6e73ea1263c161c1
```

It is KL-only. It is not the PPO denominator and must not replace stored
behavior log-probabilities.

`grpo.py` computes centered, no-std advantages. It uses
`exp(current - behavior)` for PPO ratios, clips with 0.2, and uses the frozen
reference only in the sampled K3 KL term. It selects active entries before
exponentiation, rejects broadcastable shape mismatches, and averages Slow and
Fast action losses with equal branch weight.

`grpo_state.py` snapshots and restores unmerged LoRA adapters. `grpo_checkpoint.py`
handles adapter-only state and metadata. `grpo_module.py` saves base revision,
reference adapter hash, adapter config, reward mode, rollout configuration, EOS,
truncation, length, and final step metrics in `grpo_metadata`. Lightning saves
optimizer and scheduler state alongside that adapter-only model state.

### Implementation file map

| File | Responsibility |
|---|---|
| `fish_speech/models/text2semantic/grpo.py` | Exact-shape GRPO loss, generated-action masks, centered advantage, PPO clip, and K3 KL. |
| `fish_speech/models/text2semantic/grpo_state.py` | LoRA snapshot, restore, and frozen-reference context. |
| `fish_speech/models/text2semantic/grpo_rollout.py` | DualAR rollout, stored behavior log-probabilities, action masks, EOS, and truncation. |
| `fish_speech/models/text2semantic/grpo_module.py` | S2 Pro training module, fail-closed reward gate, policy roles, logging, and checkpoint metadata. |
| `fish_speech/models/text2semantic/grpo_checkpoint.py` | Adapter-only checkpoint state, restore checks, and scalar metadata handling. |
| `fish_speech/configs/text2semantic_grpo.yaml` | GRPO train entry configuration. |
| `fish_speech/configs/lora/r_8_alpha_16_grpo.yaml` | Broad ordinary-LoRA GRPO adapter configuration. |
| `tests/test_grpo_loss.py` | Loss, clipping, masks, K3 KL, nonfinite masked values, and shape checks. |
| `tests/test_grpo_state.py` | Snapshot and frozen-reference restoration checks. |
| `tests/test_grpo_rollout.py` | Grouped rollout, stored behavior scores, q0 exclusion, EOS, truncation, and empty action checks. |
| `tests/test_grpo_module.py` | Reward gate, SFT load, adapter-only checkpoint, and resume contracts. |
| `tests/test_grpo_config.py` | Hydra composition for the GRPO entry config and LoRA config. |
| `tests/test_grpo_sample_training.py` | Tiny shifted DualAR ordinary-LoRA update plumbing. |

## 5. Exact verified Docker commands

Run these commands only after the rights and staging gates above are met. They
use the verified smoke paths and must run in the project container. The source
dataset is deliberately not mounted because the run consumes staged protobufs
under `/work/data/protos`. The repository mount is read-only and the work mount
is writable.

```bash
export REPO=/home/ms-ms-home2/workspace_research/fish-speech
export WORK=/home/ms-ms-home2/fish-s2pro-real4-smoke

docker run --rm \
  --gpus '"device=0"' \
  --mount type=bind,src="$REPO",dst=/repo,readonly \
  --mount type=bind,src="$WORK",dst=/work \
  --workdir /repo \
  fishaudio/fish-speech@sha256:14a36bad3678e61dd97f510a0e6b824f5350a4d2642fe6eebbe9f7d35f1ed282 \
  bash -lc 'python fish_speech/train.py --config-name text2semantic_grpo \
    project=s2pro_real4_r8a16_grpo_final \
    paths.run_dir=/work/results/s2pro_real4_r8a16_grpo_final \
    pretrained_ckpt_path=/work/checkpoints/s2-pro \
    tokenizer.model_path=/work/checkpoints/s2-pro \
    train_dataset.proto_files="[/work/data/protos/train]" \
    val_dataset.proto_files="[/work/data/protos/dev]" \
    sft_adapter_ckpt_path=/work/results/s2pro_real4_r8a16/checkpoints/step_000000001.ckpt \
    sft_adapter_ckpt_sha256=431bfb6c7953833702d718c7bc99403cf2bcbff9fc5b1d5a6ef53f48c46a4035 \
    provenance_manifest_sha256=638a7c476a91a79cb2033870a642f06dddc5695ebd2e7daf689fbbd7cae3d2a3 \
    +lora@model.model.lora_config=r_8_alpha_16_grpo \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.max_steps=1 \
    trainer.limit_val_batches=0 \
    data.batch_size=1 \
    data.num_workers=1 \
    callbacks.model_checkpoint.every_n_train_steps=1 \
    model.max_new_tokens=16 \
    model.rollout_seed=1235 \
    model.mechanics_smoke=true \
    model.allow_unvalidated_reward=true \
    model.lr_scheduler.lr_lambda.num_warmup_steps=0'
```

The explicit warmup override is required. An earlier warmup attempt whose first
step had an effective zero learning rate is rejected as mechanics evidence and
is not the result reported here.

The following command is the successful actual resume to optimizer step 2. It
loads `step_000000001.ckpt` explicitly and changes only `trainer.max_steps` and
`ckpt_path` relative to the verified step-1 recipe.

```bash
docker run --rm \
  --gpus '"device=0"' \
  --mount type=bind,src="$REPO",dst=/repo,readonly \
  --mount type=bind,src="$WORK",dst=/work \
  --workdir /repo \
  fishaudio/fish-speech@sha256:14a36bad3678e61dd97f510a0e6b824f5350a4d2642fe6eebbe9f7d35f1ed282 \
  bash -lc 'python fish_speech/train.py --config-name text2semantic_grpo \
    project=s2pro_real4_r8a16_grpo_final \
    paths.run_dir=/work/results/s2pro_real4_r8a16_grpo_final \
    pretrained_ckpt_path=/work/checkpoints/s2-pro \
    tokenizer.model_path=/work/checkpoints/s2-pro \
    train_dataset.proto_files="[/work/data/protos/train]" \
    val_dataset.proto_files="[/work/data/protos/dev]" \
    sft_adapter_ckpt_path=/work/results/s2pro_real4_r8a16/checkpoints/step_000000001.ckpt \
    sft_adapter_ckpt_sha256=431bfb6c7953833702d718c7bc99403cf2bcbff9fc5b1d5a6ef53f48c46a4035 \
    provenance_manifest_sha256=638a7c476a91a79cb2033870a642f06dddc5695ebd2e7daf689fbbd7cae3d2a3 \
    +lora@model.model.lora_config=r_8_alpha_16_grpo \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.max_steps=2 \
    trainer.limit_val_batches=0 \
    data.batch_size=1 \
    data.num_workers=1 \
    callbacks.model_checkpoint.every_n_train_steps=1 \
    model.max_new_tokens=16 \
    model.rollout_seed=1235 \
    model.mechanics_smoke=true \
    model.allow_unvalidated_reward=true \
    model.lr_scheduler.lr_lambda.num_warmup_steps=0 \
    +ckpt_path=/work/results/s2pro_real4_r8a16_grpo_final/checkpoints/step_000000001.ckpt'
```

The verified resume log reports `Restored all states from the checkpoint` and
stops at `max_steps=2`. Do not use an older checkpoint loader that expects base
tensors in an adapter-only GRPO state dict.

## 6. Evidence from the verified run

Evidence is under `/home/ms-ms-home2/fish-s2pro-real4-smoke/evidence`. Resolved
run configuration is in TensorBoard `version_0` for step 1 and `version_2` for
the successful step-2 resume.

The canonical provenance manifest is `grpo-final-provenance.json`, SHA-256
`638a7c476a91a79cb2033870a642f06dddc5695ebd2e7daf689fbbd7cae3d2a3`.
It records source and image provenance, model/tokenizer/codec/shard hashes,
source snapshot and split/protobuf hashes, adapter scaling, optimizer/scheduler,
RNG and rollout settings, behavior log-probability format, reward revision, and
the mechanics-only evaluation-manifest status. Both checkpoints embed this hash
and reject a resume configured with a different manifest.

| Metric | Step 1 | Step 2 after resume |
|---|---:|---:|
| Optimizer step | 1 | 2 |
| Loss | 0.0039461 | 0.004097639117389917 |
| Reward vector | `[0, 1]` | `[0, 1]` |
| Reward mean, variance | 0.5, 0.25 | 0.5, 0.25 |
| Slow active actions | 32 | 32 |
| Fast active actions | 288 | 288 |
| Slow PPO ratio | 0.9921875 | 0.98828125 |
| Fast PPO ratio | 1.0 | 1.0 |
| Slow KL delta | 0.0 | 0.01116943359375 |
| Fast KL delta | 0.0 | 0.0152587890625 |
| Gradient norm | 15.9375 | 10 |
| EOS rate | 0 | 0 |
| Truncation rate | 1 | 1 |

The final step-2 checkpoint has `global_step=2` and optimizer state step 2.
Peak GPU memory was 10,338 MiB on the RTX 4060 Ti 16 GB. Compared with the SFT
adapter, step 1 changed 204 LoRA tensors and step 2 changed 404. Both saved GRPO
checkpoints contain no base tensors. The adapter-only `state_dict` contains 408
LoRA state tensors and the Lightning checkpoint retains optimizer state for
resume.

Every sampled completion reached the 16-frame cap, emitted no EOS, and was
marked truncated. This is direct mechanics evidence for rollout metadata and
generated-action masks. It is not evidence of usable generation length, audio
quality, or reward quality.

The final GRPO test log at `/work/evidence/grpo-tests-final.log` passed 23/23
tests in the project container. The actual S2 Pro run then completed the first
update and an actual checkpoint restore to the second update.

## 7. Reward and production gate

The smoke reward is the fixed vector `[0, 1]`. Its only purpose is to create a
non-equal group advantage and exercise the mechanics. It is explicitly
`TEST_ONLY_NOT_A_QUALITY_REWARD`.

The module permits that reward only when both mechanics smoke flags are set:

```text
model.mechanics_smoke=true
model.allow_unvalidated_reward=true
```

Without those flags, production GRPO fails closed with no validated Korean GRPO
reward scorer. Do not attach a production quality claim to this run.

Before production GRPO, validate every enabled scorer on Korean human-labelled
holdout data. The validation must cover Korean CER normalization, deletion and
insertion behavior, digits, English code switching, speaker same and different
pairs, calibration, and correlation with human judgement. Pin scorer revision,
preprocessing, and calibration. If no scorer passes, stop at SFT LoRA.

## 8. Evaluation, provenance, and stop conditions

Keep base, broad ordinary-LoRA SFT, compatible SFT, LoRA-GRPO, and reward
ablations on identical prompts, references, decoding settings, seeds, and
failure accounting. Report Korean CER and WER rules, validated speaker
similarity, blind human preference, failure and truncation rate, RTF, peak
memory, KL, entropy, and reward-hacking audits. Never rank author-reported and
local metrics from different datasets, normalizers, scorers, or judges.

Each SFT or GRPO artifact must retain:

- Fish Speech source revision and Docker image provenance.
- S2 Pro base revision, tokenizer, codec, shard, index, and config hashes.
- Source snapshot, split manifest, protobuf hashes, and evaluation manifest.
- Adapter targets, rank, alpha, dropout, scaling, and SFT adapter hash.
- Trainable adapter state, optimizer state, scheduler state, global step, RNG,
  and rollout settings.
- Frozen reference hash, behavior log-probability format, reward scorer
  revision, and EOS and truncation metadata.

Stop immediately if rights or consent are unclear, the source snapshot or split
audit cannot be reproduced, a checkpoint cannot exactly restore its required
metadata, masks include prompt, reference, or padding actions, behavior and
reference roles are mixed, loss or gradient is nonfinite, or production reward
validation is absent.

## 9. Strict limitations

This run proves only the local broad ordinary-LoRA GRPO mechanics and an actual
resume on a tiny staged real-data subset. It does not prove any of the following:

- Audio quality improvement.
- Korean CER, WER, or speaker-similarity improvement.
- A production reward or a Korean human-validated scorer.
- Data rights, voice-cloning consent, or full-data training permission.
- Full-data stability, long-horizon rollout behavior, or a usable EOS rate.
- MLP-only rsLoRA `r=16`, `alpha=64`, or `alpha/sqrt(r)` scaling.
- Fish S2 trainer behavior or Fish S2 reproduction.

The durable training direction remains:

```text
base -> compatible LoRA SFT -> LoRA-GRPO
```

Never run another SFT stage after GRPO. Preserve one adapter topology across a
given SFT warm start and GRPO continuation. A future Fish-S2-aligned experiment
needs a separate MLP-only rsLoRA SFT warm start and a real rsLoRA scaling
implementation. Changing config numbers alone is insufficient.
