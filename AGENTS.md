# Project Context for Agents

## Research direction

- The detailed TTS/GRPO survey and citations are in `docs/PAPER.md`.
- The current reproducible baseline includes ordinary-LoRA SFT and an
  implemented S2 Pro broad ordinary-LoRA GRPO mechanics pipeline. It is Fish
  local work, not the Fish S2 trainer or a Fish S2 reproduction.
- The target direction is `base -> compatible LoRA SFT -> LoRA-GRPO`. Never run
  another SFT stage after GRPO.
- `archive/GLM-TTS` is a read-only reference for rollout, reward serving, and
  generated-token policy loss. Its shipped launcher is full-parameter
  `PRETRAIN` GRPO; LoRA is not wired into that path.
- Keep Fish and archived GLM-TTS dependencies in separate environments.

## Execution environment

- Perform implementation, testing, preprocessing, and training inside the
  project Docker container. Do not install or run project dependencies directly
  on the host; use the host only for read-only inspection, Git operations, and
  Docker orchestration.

## LoRA-GRPO consensus

- Keep one adapter topology across the SFT warm-start and GRPO continuation.
- The public Fish `r_8_alpha_16` adapter is a broad ordinary-LoRA baseline, not
  a reproduction of Fish S2.
- A Fish-S2-aligned experiment needs a separate MLP-only rsLoRA warm-start with
  rank 16 and alpha 64. Implement rsLoRA scaling; changing config numbers alone
  is insufficient.
- Optimize generated Slow/Fast audio tokens only. Mask prompt, reference, and
  padding tokens.
- Keep rollout behavior log-probabilities separate from the frozen KL reference
  policy. Record EOS and truncation explicitly.
- Prefer centered no-std group advantages. Treat GLM's two-stage reward
  z-normalization and dynamic clipping as ablations, not defaults.
- Enable only reward scorers validated against Korean human-labelled data. If
  none pass validation, production GRPO must fail closed and stop at SFT LoRA.
  The explicit mechanics-only test reward is not a quality reward.

## Dataset constraints

- Intended source: `/home/ms-ms-home2/dataset/tts/250623_tts_jiwoo_whole_services/tts`.
- A tiny copied subset may be used in the Fish project Docker environment only
  for pipeline-mechanics smoke tests. Mount the source read-only and keep all
  labels, semantic codes, protobufs, checkpoints, and logs outside the source.
- Read-only inventory on 2026-07-22: about 11 GB, 128 top-level voice/service
  directories, and about 793K MP3 files. There are no `.lab`/`.npy` sidecars;
  the URL-encoded filename stem is the transcript.
- Never rename, rewrite, move, or add sidecars to source MP3 files. Build a
  separate manifest containing the original path/stem, normalized transcript,
  snapshot hash, and exclusion reason.
- Decode with strict `urllib.parse.unquote_plus()` exactly once. This converts
  filename `+` separators to spaces and `%3F`/`%21` to `?`/`!`. Remove BOM,
  collapse whitespace, trim, and apply NFC while preserving `?` and `!` order.
- If `%xx` remains, apply at most one strict `urllib.parse.unquote()` pass only
  for non-URL/non-email text. Never decode repeatedly to a fixed point.
- Convert numeric `%` to the spoken token `퍼센트` and non-URL literal `+` to
  `플러스`. Exclude ambiguous percent, URL/email residual encoding, unresolved
  nested escapes, DOS 8.3 aliases, empty/non-text stems, and decode/control
  failures with explicit reason codes.
- Accepted transcripts must contain zero `%`, literal `+`, `%xx` remnants, BOM,
  control characters, or non-NFC text. Digit sequences and `?`/`!` order must
  have zero drift; URLs must have zero normalization drift.
- The verified read-only snapshot `71214cfc...e6508` had 793,184 MP3s, 792,779
  accepted transcripts, and 405 exclusions. Re-freeze and recount because the
  source tree may change.
- Group duplicate normalized text across splits. That snapshot had 705,134
  unique texts and 87,645 duplicate utterances.
- Training rights remain a blocker. Do not infer consent or permission from
  file availability.
- Never run semantic extraction against the source tree: it writes `.npy` next
  to audio. Freeze leakage-safe manifests first, then work in a staged copy.
- Use text/session-disjoint and voice/service-held-out evaluation. The dataset
  is highly imbalanced, so do not rely on an utterance-only random split.

## Verified real-data S2 Pro smoke

- On 2026-07-22, an ordinary-LoRA SFT smoke test completed one optimizer step
  on one RTX 4060 Ti 16 GB using three staged train MP3s and one staged dev MP3.
- The exact S2 Pro revision was
  `1de9996b6be38b745688de084d87a5633f714e4e`; all checkpoint keys matched.
- Use `modded_dac_vq_s2` for S2 Pro semantic extraction. The public codec has
  context buffers 16384/4096; the older `modded_dac_vq` config uses 8192/2048
  and fails checkpoint loading with shape mismatches.
- The smoke artifacts are under `/home/ms-ms-home2/fish-s2pro-real4-smoke`.
  Selected-source snapshot SHA-256 is
  `ec1d3f719cf55cbbe912f05e97c5ed38aa8d318a0b4ec739f032c7da1684135e`.
- The successful run used broad ordinary LoRA `r=8`, `alpha=16`, batch size 1,
  one dataloader worker, one visible GPU, and `max_steps=1`. It reached loss
  30.50, wrote `step_000000001.ckpt`, and peaked at 10,319 MiB GPU memory.
- The historical SFT smoke image `fishaudio/fish-speech:nightly-dev` ships
  protobuf 3.19.6, which is too old for the generated dataset code. Align it
  with the project's `protobuf>=3.20,<6` override inside the container before
  protobuf packing.
- The image used had ID
  `sha256:37a7d6feec604a2df9f949a2820b3d1e057550dd2e3776ea5c2419543bf48853`
  and repo digest `fishaudio/fish-speech@sha256:14a36bad3678e61dd97f510a0e6b824f5350a4d2642fe6eebbe9f7d35f1ed282`.
- This run proves preprocessing, checkpoint load, forward, backward, and
  optimizer plumbing only. It does not establish data rights or model quality.

## Verified S2 Pro broad ordinary-LoRA GRPO mechanics smoke

- On 2026-07-22, the implemented `text2semantic_grpo` pipeline completed one
  S2 Pro GRPO step and an actual resume to step 2 on one RTX 4060 Ti 16 GB.
  The verified output is
  `/home/ms-ms-home2/fish-s2pro-real4-smoke/results/s2pro_real4_r8a16_grpo_final`.
- It starts from the broad ordinary-LoRA SFT adapter
  `/work/results/s2pro_real4_r8a16/checkpoints/step_000000001.ckpt` on base
  revision `1de9996b6be38b745688de084d87a5633f714e4e`. GRPO uses the same broad
  targets, `r=8`, `alpha=16`, dropout `0`; this is not rsLoRA.
- The SFT adapter checkpoint SHA-256 is
  `431bfb6c7953833702d718c7bc99403cf2bcbff9fc5b1d5a6ef53f48c46a4035`.
- The canonical provenance manifest is
  `/home/ms-ms-home2/fish-s2pro-real4-smoke/evidence/grpo-final-provenance.json`,
  SHA-256 `638a7c476a91a79cb2033870a642f06dddc5695ebd2e7daf689fbbd7cae3d2a3`.
  Both GRPO checkpoints embed and resume-validate this hash.
- The rollout uses `G=2`, 16 generated frames, seed 1235, Slow q0 plus Fast
  q1-q9 actions, generated-only masks, explicit EOS and truncation metadata,
  and behavior log-probabilities stored at rollout. Advantages are centered
  without group-std normalization. PPO clip is 0.2 and `kl_weight=0.1`.
- The frozen SFT adapter hash
  `7086e88223b9b19ac8f883d9275f645b200824928b209cce6e73ea1263c161c1` is used
  only as the KL reference. The current policy re-scores stored actions.
  Behavior log-probabilities remain the PPO denominator.
- The mechanics-only reward is fixed at `[0, 1]` and is explicitly
  `TEST_ONLY_NOT_A_QUALITY_REWARD`. It is enabled only by the explicit smoke
  flags. Production reward remains fail-closed until a Korean
  human-validated scorer exists.
- Step 1 loss was 0.0039461, with Slow/Fast active actions 32/288, Slow/Fast
  ratios 0.9921875/1.0, grad norm 15.9375, 204 changed LoRA tensors, and no
  base tensors saved. Step 2 resumed successfully with `global_step=2`,
  optimizer state step 2, loss 0.004097639117389917, reward mean/variance
  0.5/0.25, Slow/Fast active actions 32/288, Slow/Fast ratios 0.98828125/1.0,
  Slow/Fast KL deltas 0.01116943359375/0.0152587890625, EOS/truncation rates
  0/1, and peak memory 10,338 MiB. Checkpoints exist for both steps.
- The final GRPO test log passed 23/23 tests.
- Every sampled completion hit the 16-frame cap, had no EOS, and was marked
  truncated. These runs prove mechanics and resume only, not audio quality,
  Korean CER or speaker similarity, data rights, full-data training, rsLoRA,
  or Fish S2 reproduction. See `docs/GRPO_TRAINING.md` for commands and
  evidence.

## Evidence and evaluation

- Research PDFs, provenance, and checksums are under ignored
  `archive/papers/`; do not commit them unless explicitly requested.
- Pin paper versions, repository/model-card revisions, checkpoints, tokenizer,
  codec, reward models, and evaluation manifests.
- As of 2026-07-22, the comparison set is Fish Audio S2 Pro, Higgs TTS 3,
  GLM-TTS Base, OmniVoice, and the public Qwen3-TTS 12Hz family. Older models
  may be used only when they contain the relevant RL method.
- Separate author-reported metrics from local reproductions. Do not rank values
  produced by different datasets, normalization, ASR/SIM models, or judges.
- Compare base, SFT LoRA, compatible SFT, LoRA-GRPO, and reward ablations with
  identical prompts, references, decoding, seeds, and failure accounting.
- Report Korean CER/WER rules, validated speaker similarity, blind human
  preference, failure/truncation rate, RTF, peak memory, KL, entropy, and reward
  hacking audits.
- Respect model-weight licenses and voice-cloning consent restrictions.
