# Fish LoRA-GRPO Korean Waveform Reward Integration Plan

## 목표

현재 `TextToSemanticGRPO._rewards()`의 mechanics-only 고정 보상 `[0, 1]`을
GLM-TTS식 `rollout -> waveform -> reward service -> group advantage` 경로로
교체한다.

첫 production 후보는 **한국어 CER reward만 활성화**한다. 화자 유사도는 응답
schema와 config에 포함하되 weight `0.0`으로 유지한다. emotion, laughter, MOS,
pitch, energy reward는 이번 범위에서 구현하거나 활성화하지 않는다.

데이터는 사용자 소유이므로 권리 확인을 blocker로 두지 않는다. rsLoRA 전환이나
추가 SFT도 이번 작업에 포함하지 않는다.

## 고정 설계

1. GLM-TTS의 서비스 경계만 재사용한다. Chinese Paraformer, `zhconv`, 중국어
   laughter 문자, 사설 IP, rank별 port, shared temporary WAV path, exception-to-zero,
   group z-normalization은 가져오지 않는다.
2. Fish rollout의 `generated_frames: [B,G,T,10]`만 waveform으로 복원한다.
   `[B*G,10,T]`로 변환한 뒤 S2 codec의 `DAC.from_indices()`를 호출한다.
3. Codec은 Fish trainer process에서 CPU로 실행한다. 정확한 loader는
   `fish_speech.models.dac.inference.load_model("modded_dac_vq_s2", codec.pth,
   device="cpu")`이다. GPU 0은 4.6B policy 전용으로 유지한다.
4. Reward service는 별도 Docker 환경과 process로 실행한다. 기본은 CPU-only다.
   GPU reward inference는 물리 GPU 1 이상이 있을 때만 명시적으로 허용한다.
5. Trainer는 candidate waveform을 mono PCM16 base64로 전송한다. Reward service는
   caller filesystem path를 받거나 candidate WAV를 디스크에 쓰지 않는다.
6. ASR은 `openai/whisper-large-v3` revision
   `06f233fe06e710322aca913c1bc4249a0d71fce1`로 고정한다. Transformers
   `4.57.3`에서 `language="korean"`, `task="transcribe"`를 강제한다.
7. Korean CER normalization은 NFC 후 Unicode separator, punctuation, control
   category를 제거하고 Latin은 case-fold한다. Hangul, digit, 나머지 letter/number는
   보존한다.
8. Scalar CER reward는 `clipped_linear_cer_v1`로 고정한다:
   `max(0.0, 1.0 - min(cer, 1.0))`.
9. Total reward는 최초 pilot에서 `1.0 * cer_reward + 0.0 * speaker_similarity`다.
   기존 centered no-std group advantage를 그대로 사용하며 std normalization을
   추가하지 않는다.
10. Timeout, ASR failure, schema mismatch, scorer/calibration mismatch, missing result,
    duplicate index, non-finite/out-of-range score는 모두 training step을 중단한다.
    실패를 reward `0`으로 바꾸지 않는다.
11. Mechanics reward는 기존 두 smoke flag에서만 남긴다. Production reward는 승인된
    calibration artifact가 없으면 fail closed한다.

## 변경 파일

| 작업 | 경로 | 책임 |
|---|---|---|
| 추가 | `reward_service/pyproject.toml` | 독립 reward 환경과 정확한 dependency pin |
| 추가 | `reward_service/uv.lock` | 재현 가능한 service lockfile |
| 추가 | `reward_service/Dockerfile` | CPU-only 기본 image |
| 추가 | `reward_service/src/fish_grpo_reward/contracts.py` | Pydantic request/response/error schema |
| 추가 | `reward_service/src/fish_grpo_reward/cer.py` | Korean normalization, Whisper 호출, CER transform |
| 추가 | `reward_service/src/fish_grpo_reward/app.py` | `/healthz`, `/v1/korean-cer` FastAPI endpoint |
| 추가 | `reward_service/src/fish_grpo_reward/validate.py` | Offline validation/calibration CLI |
| 추가 | `reward_service/tests/test_contracts.py` | Protocol validation tests |
| 추가 | `reward_service/tests/test_cer.py` | Korean normalization/CER tests |
| 추가 | `reward_service/tests/test_app.py` | Health, mismatch, ASR failure tests |
| 추가 | `fish_speech/models/text2semantic/grpo_waveform_reward.py` | Codec decode, strict HTTP client, `[B,G]` reward assembly |
| 변경 | `fish_speech/models/text2semantic/grpo_module.py` | Mechanics/validated reward mode 선택 |
| 변경 | `fish_speech/models/text2semantic/grpo_metrics.py` | CER, reward, latency, zero-variance group logging |
| 변경 | `fish_speech/models/text2semantic/grpo_checkpoint.py` | Reward identity와 calibration resume validation |
| 변경 | `fish_speech/datasets/semantic.py` | GRPO batch에 정확한 target transcript 전달 |
| 변경 | `fish_speech/configs/text2semantic_grpo.yaml` | Disabled-by-default reward block |
| 추가 | `fish_speech/configs/reward/korean_cer.yaml` | Production Korean CER override |
| 추가 | `tests/test_grpo_waveform_reward.py` | Codec layout, ordering, client fail-closed tests |
| 추가 | `tests/test_semantic_reward_text.py` | Batch transcript contract tests |
| 변경 | `tests/test_grpo_module.py` | Production gate와 reward propagation tests |
| 변경 | `tests/test_grpo_checkpoint.py` | Reward metadata mismatch matrix |
| 변경 | `tests/test_grpo_config.py` | Default fail-closed와 override composition |
| 변경 | `docs/GRPO_TRAINING.md` | Service, validation, smoke/resume 명령과 evidence |

## Reward protocol

### Request

`KoreanCerRequest`:

- `schema_version = "fish-grpo-reward/v1"`
- `request_id: UUID`
- `scorer_id = "openai/whisper-large-v3"`
- `scorer_revision = "06f233fe06e710322aca913c1bc4249a0d71fce1"`
- `calibration_sha256`
- `candidates`: rollout 순서대로 정확히 `B*G`개

각 candidate:

- `candidate_index: int`
- `pcm16_base64: str`
- `sample_rate_hz: int`
- `reference_text: str`

### Response

`KoreanCerResponse`는 request identity를 그대로 반환하고 candidate마다 다음을
반환한다.

- `candidate_index`
- `normalized_reference`
- `normalized_hypothesis`
- `cer`
- `cer_reward`
- `speaker_similarity: null`
- `service_latency_ms`

HTTP status:

- `400`: malformed payload/audio/text
- `409`: scorer revision 또는 calibration mismatch
- `500`: ASR inference failure

Fish client는 non-200, timeout, invalid JSON/schema, response count 불일치,
missing/duplicate index, identity mismatch, non-finite score를
`WaveformRewardError`로 변환해 step을 중단한다.

## Dependency와 runtime 경계

Reward service dependency는 Fish 환경과 분리한다.

- `transformers==4.57.3`
- `openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1`
- CPU build의 `torch`, `torchaudio`
- `accelerate`, `fastapi`, `uvicorn`, `pydantic`, `jiwer`, `numpy`
- raw PCM array를 직접 넘기므로 runtime `ffmpeg`는 필수 dependency로 추가하지
  않는다.

Whisper large-v3는 1.55B model이므로 CPU에서 동작하지만 느릴 수 있다. 따라서
validation과 one-step smoke에서 p50/p95 latency를 측정하고 GRPO step wall time에
포함한다. 처리량이 부족해도 GPU 0에 reward model을 올리는 fallback은 금지한다.

Model license provenance에는 HF artifact metadata의 Apache-2.0 표기와 OpenAI
upstream의 MIT weight/code 표기를 모두 기록한다. Model cache/image를 외부 배포할
경우 별도 license 확인을 거친다.

## 실행 순서

### 1. Reward service TDD

- [ ] Protocol test를 먼저 작성하고 malformed base64, empty text, revision mismatch,
  duplicate candidate index가 실패하는 red를 확인한다.
- [ ] Korean normalizer와 CER transform test를 작성한다. Hangul, whitespace,
  punctuation, digit, English code-switch, insertion/deletion case를 포함한다.
- [ ] Whisper를 fake scorer로 대체한 app test를 작성한다. ASR exception이 500이고
  fabricated zero reward가 반환되지 않음을 확인한다.
- [ ] Service implementation과 CPU Docker image를 만든다.

Acceptance:

- `/healthz`가 schema/scorer/revision/normalizer/transform identity를 반환한다.
- Request에 path field가 없고 service가 candidate audio file을 만들지 않는다.
- Broad `except` 후 zero reward 반환이 없다.

### 2. Dataset transcript contract

- [ ] `semantic.py`와 collator에서 첫 generated semantic target에 대응하는 normalized
  transcript를 `reward_text`로 보존한다.
- [ ] GRPO config에서 transcript가 누락될 수 있는 text-skip augmentation을 끈다.
- [ ] 기존 SFT/non-GRPO batch contract가 바뀌지 않는 regression test를 추가한다.

Acceptance:

- Batch size 1에서 `reward_text`가 학습 target text와 byte-for-byte 일치한다.
- Empty/missing transcript는 rollout 전에 실패한다.

### 3. Fish codec와 strict client

- [ ] `grpo_waveform_reward.py`의 layout test를 먼저 작성한다.
- [ ] `[B,G,T,10] -> [B*G,10,T]` 변환과 `DAC.from_indices()` decode를 구현한다.
- [ ] EOS 이후 frame은 `record.lengths`로 잘라 scorer에 보내며 prompt/reference token은
  decode하지 않는다.
- [ ] PCM16 conversion, request ordering, response reordering, timeout/error contract를
  구현한다.

Acceptance:

- Candidate 순서가 group order와 정확히 일치한다.
- Codec decode는 `torch.no_grad()`이고 policy graph와 분리된다.
- Candidate 하나의 decode/score 실패도 전체 step을 중단한다.

### 4. GRPO module 연결

- [ ] `reward.mode`를 `fail_closed | mechanics_smoke | korean_cer`로 제한한다.
- [ ] `mechanics_smoke`는 기존 두 explicit flag에서만 `[0,1]`을 허용한다.
- [ ] `korean_cer`는 health identity와 calibration SHA를 startup에서 확인한다.
- [ ] `_rewards(record, reward_text)`가 service 결과를 `[B,G]` tensor로 반환하도록
  연결한다.
- [ ] Rollout, behavior logprob, frozen KL reference, current re-score, generated-only
  masks, PPO loss, LoRA optimizer selection은 수정하지 않는다.
- [ ] Raw CER/reward, service latency, zero-variance group rate를 기록한다.

Acceptance:

- Production config에서 mechanics flags는 false다.
- Service unavailable/mismatch면 loss 계산과 optimizer step이 발생하지 않는다.
- Equal reward group은 gradient 0으로 남기며 GLM식 reroll이나 z-normalization을
  추가하지 않는다.

### 5. Checkpoint와 provenance

- [ ] Resume metadata에 reward schema, scorer ID/revision/content SHA, calibration SHA,
  codec SHA, normalizer/transform version, CER/SIM weight, endpoint identity를 추가한다.
- [ ] 위 값 하나라도 바뀐 checkpoint resume가 training 전에 실패하는 test를
  추가한다.
- [ ] 기존 adapter-only state, optimizer/scheduler/global step 저장을 유지한다.
- [ ] Canonical provenance manifest와 checkpoint SHA를 갱신한다.

Acceptance:

- 두 checkpoint 모두 base tensor 0개다.
- Reward identity mismatch matrix 전체가 fail closed한다.
- Resume 후 optimizer state가 다음 step으로 진행한다.

### 6. Korean scorer validation gate

- [ ] Policy train/eval split과 분리된 immutable validation manifest를 만든다.
- [ ] clean, deletion, insertion, digit, English code-switch 5개 stratum을 각각 최소
  100개로 구성한다.
- [ ] Candidate마다 blinded human transcription-accuracy rating 2개 이상을 기록한다.
- [ ] CER reward와 human score Spearman `>= 0.60`, bootstrap 95% lower bound `> 0`,
  correct/error reward separation `>= 0.10`, 모든 stratum 방향성이 양수여야 승인한다.
- [ ] 실패하면 `approved=false`를 기록하고 GRPO는 시작하지 않는다.

외부 evidence:

```text
$WORK/evidence/korean-cer-v1/
  validation-manifest.jsonl
  validation-report.json
  calibration.json
  checksums.sha256
```

`calibration.json`에는 manifest SHA, scorer ID/revision/content SHA, service image
digest, normalizer/transform version, thresholds, statistics, `approved`를 포함한다.

### 7. 실제 one-step와 resume

- [ ] Reward service를 CPU-only로 실행하고 `/healthz` identity를 evidence로 저장한다.
- [ ] 기존 broad ordinary-LoRA `r=8`, `alpha=16`, dropout `0` SFT adapter에서 `G=2`,
  16 generated frames로 GRPO 1 step을 실행한다.
- [ ] 동일 scorer/calibration/provenance로 checkpoint를 step 2까지 실제 resume한다.
- [ ] GPU process inventory에서 reward service가 GPU 0을 사용하지 않음을 확인한다.

Service launch 형태:

```bash
docker run --rm -d --name fish-grpo-korean-cer \
  --network host --cpus=12 \
  -e REWARD_DEVICE=cpu \
  -e REWARD_ASR_ID=openai/whisper-large-v3 \
  -e REWARD_ASR_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1 \
  --mount type=bind,src="$ASR_CACHE",dst=/models/asr,readonly \
  fish-grpo-korean-cer:dev
```

Fish GRPO override 형태:

```text
model.mechanics_smoke=false
model.allow_unvalidated_reward=false
+reward@model.reward=korean_cer
model.reward.endpoint=http://127.0.0.1:8090/v1/korean-cer
model.reward.codec_checkpoint_path=/work/checkpoints/s2-pro/codec.pth
model.reward.codec_device=cpu
model.reward.calibration_path=/work/evidence/korean-cer-v1/calibration.json
model.reward.calibration_sha256=<approved artifact SHA>
```

## 검증 명령

모든 project test와 실행은 Docker 안에서 수행한다.

Reward service:

```bash
docker run --rm \
  --mount type=bind,src="$REPO",dst=/repo,readonly \
  -e PYTHONPATH=/repo/reward_service/src \
  -w /repo/reward_service \
  fish-grpo-korean-cer:dev \
  python -m unittest discover -s tests -v
```

Fish:

```bash
docker run --rm --gpus '"device=0"' --network host \
  --mount type=bind,src="$REPO",dst=/repo,readonly \
  --mount type=bind,src="$WORK",dst=/work \
  --workdir /repo \
  fishaudio/fish-speech@sha256:14a36bad3678e61dd97f510a0e6b824f5350a4d2642fe6eebbe9f7d35f1ed282 \
  bash -lc 'PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v'
```

추가 gate:

- Ruff/LSP/no-excuse audit clean.
- Reward service image와 ASR cache/model hash 고정.
- One-step와 resume log에 finite CER/reward, latency, KL, behavior/current ratio,
  EOS/truncation 기록.
- Step 2 checkpoint의 `global_step=2`, optimizer state step 2 확인.
- Source MP3에는 write가 없고 모든 artifact는 `$WORK` 아래에만 생성.

## 완료 조건

- 한국어 CER reward가 실제 Fish-generated waveform에서 계산된다.
- Mechanics `[0,1]` reward 없이 one-step와 실제 resume가 성공한다.
- Service/ASR/calibration/codec identity가 checkpoint와 provenance에 고정된다.
- Reward 장애가 silent zero reward가 아니라 fail-closed training stop으로 나타난다.
- Existing generated-only PPO mechanics와 LoRA-only checkpoint 계약이 유지된다.
- SIM weight는 `0.0`, emotion/laughter/MOS/pitch/energy는 disabled 상태다.
- 전체 test, lint, diagnostics, real artifact QA가 통과한다.

## 명시적 제외

- Chinese GLM reward model의 직접 사용
- rsLoRA 또는 LoRA topology 변경
- 추가 SFT stage
- GPU 0에서 reward model 실행
- Kubernetes, queue, retry farm, speculative distributed orchestration
- Unvalidated SIM/emotion/laughter/quality reward 활성화
- Reward server failure를 zero reward로 숨기는 fallback
- 사용자의 명시적 요청 없는 commit 또는 push
