# Fish S2 Pro LoRA/GRPO 한국어 학습 매뉴얼

기존 upstream README는 [README.en.md](README.en.md)에 보존했다. 이 문서는 이
작업공간에서 검증한 Fish S2 Pro ordinary-LoRA SFT와 LoRA-GRPO mechanics smoke를
재현하기 위한 운영 매뉴얼이다.

## 현재 상태

- 기준 모델: Fish Audio S2 Pro revision `1de9996b6be38b745688de084d87a5633f714e4e`
- 검증된 경로: `base -> broad ordinary-LoRA SFT -> LoRA-GRPO mechanics`
- SFT LoRA: broad ordinary LoRA `r=8`, `alpha=16`, dropout `0`
- GRPO LoRA: 동일 adapter topology 유지
- 검증 범위: preprocessing, checkpoint load, forward/backward, PPO-style GRPO loss,
  LoRA-only checkpoint, actual resume
- 미검증 범위: 음질, 한국어 CER/WER, speaker similarity, full-data training, rsLoRA,
  Fish S2 reproduction

자세한 실행 기록과 evidence는 [docs/GRPO_TRAINING.md](docs/GRPO_TRAINING.md)를
참조한다. 다음 production reward 연결 계획은 [PLAN.md](PLAN.md)를 참조한다.

## 절대 규칙

1. 구현, 테스트, 전처리, 학습은 프로젝트 Docker container 안에서만 실행한다.
2. 원본 데이터셋은 read-only로만 mount한다.
3. 원본 MP3 파일명, 위치, sidecar를 수정하지 않는다.
4. semantic extraction은 원본 source tree에 대해 실행하지 않는다.
5. GRPO 뒤에 SFT를 다시 실행하지 않는다.
6. rollout behavior log-probability와 frozen KL reference policy를 분리한다.
7. prompt, reference, padding token은 loss에서 제외하고 generated Slow/Fast audio
   token만 최적화한다.
8. mechanics-only reward `[0, 1]`는 smoke 전용이다. production reward가 아니다.
9. 검증되지 않은 reward scorer는 fail closed한다.

## 데이터 준비

의도한 원본 데이터 위치:

```text
/home/ms-ms-home2/dataset/tts/250623_tts_jiwoo_whole_services/tts
```

처리 원칙:

- 파일명 stem을 transcript source로 사용한다.
- `urllib.parse.unquote_plus()`를 정확히 한 번 적용한다.
- BOM 제거, whitespace collapse, trim, NFC 정규화를 수행한다.
- 숫자 `%`는 `퍼센트`, non-URL literal `+`는 `플러스`로 처리한다.
- URL/email residual encoding, nested escape, control character, empty text,
  unresolved `%xx`는 exclusion reason과 함께 제외한다.
- split은 text/session-disjoint와 voice/service-held-out을 우선한다.

원본 tree에는 아무것도 쓰지 말고 별도 manifest와 staged copy를 만든다.

## S2 Pro ordinary-LoRA SFT smoke

검증된 smoke는 실제 MP3 4개(train 3/dev 1)로 1 optimizer step을 완료했다.

핵심 설정:

- Codec config: `fish_speech/configs/modded_dac_vq_s2.yaml`
- Codec ID: `modded_dac_vq_s2`
- LoRA config: `fish_speech/configs/lora/r_8_alpha_16_grpo.yaml`와 같은 broad target
- Batch size: `1`
- GPU: RTX 4060 Ti 16 GB 1장

성공 기준:

- S2 Pro checkpoint key가 모두 일치한다.
- optimizer step 1개가 완료된다.
- LoRA checkpoint가 생성된다.
- source data tree에 write가 없다.

검증된 SFT adapter:

```text
/home/ms-ms-home2/fish-s2pro-real4-smoke/results/s2pro_real4_r8a16/checkpoints/step_000000001.ckpt
```

SHA-256:

```text
431bfb6c7953833702d718c7bc99403cf2bcbff9fc5b1d5a6ef53f48c46a4035
```

## LoRA-GRPO mechanics smoke

GRPO는 위 SFT adapter에서 바로 시작한다. 추가 SFT stage를 넣지 않는다.

핵심 구현 파일:

- `fish_speech/models/text2semantic/grpo.py`
- `fish_speech/models/text2semantic/grpo_rollout.py`
- `fish_speech/models/text2semantic/grpo_module.py`
- `fish_speech/models/text2semantic/grpo_state.py`
- `fish_speech/models/text2semantic/grpo_checkpoint.py`
- `fish_speech/models/text2semantic/grpo_metrics.py`

핵심 config:

- `fish_speech/configs/text2semantic_grpo.yaml`
- `fish_speech/configs/lora/r_8_alpha_16_grpo.yaml`

검증된 mechanics:

- `G=2`
- 16 generated frames
- Slow q0와 Fast q1-q9 action만 active
- behavior log-probabilities는 rollout 시점에 저장
- frozen SFT adapter는 KL reference 전용
- current policy는 stored action을 다시 score
- group advantage는 centered no-std
- PPO clip `0.2`
- `kl_weight=0.1`
- LoRA tensor만 checkpoint에 저장

Resume override는 반드시 `+ckpt_path=...`를 사용한다.

```text
+ckpt_path=/work/results/s2pro_real4_r8a16_grpo_final/checkpoints/step_000000001.ckpt
```

`ckpt_path=...`만 쓰면 Hydra struct error가 난다.

## 검증된 최종 GRPO 산출물

```text
/home/ms-ms-home2/fish-s2pro-real4-smoke/results/s2pro_real4_r8a16_grpo_final
```

Provenance manifest:

```text
/home/ms-ms-home2/fish-s2pro-real4-smoke/evidence/grpo-final-provenance.json
```

Provenance SHA-256:

```text
638a7c476a91a79cb2033870a642f06dddc5695ebd2e7daf689fbbd7cae3d2a3
```

성공 기준:

- step 1 checkpoint 생성
- step 1에서 LoRA tensor만 변경
- step 2 실제 resume 성공
- optimizer state가 step 2로 진행
- checkpoint에 base tensor 저장 0개
- provenance hash mismatch resume 실패
- final GRPO tests 23/23 pass

## Production reward 다음 단계

현재 reward `[0, 1]`은 test-only다. production 후보는 한국어 CER reward부터
연결한다.

계획 요약:

- Fish generated frames `[B,G,T,10]`를 S2 codec으로 waveform 복원
- 별도 CPU reward service로 waveform 전송
- `openai/whisper-large-v3` revision
  `06f233fe06e710322aca913c1bc4249a0d71fce1` 사용
- Transformers `4.57.3`
- `language="korean"`, `task="transcribe"` 강제
- CER reward는 `max(0.0, 1.0 - min(cer, 1.0))`
- speaker similarity는 schema/config 자리만 두고 weight `0.0`
- timeout, ASR failure, schema mismatch, calibration mismatch는 training abort

구현 전에 [PLAN.md](PLAN.md)의 validation gate와 checkpoint provenance 조건을 먼저
만족해야 한다.

## 빠른 점검표

학습 또는 smoke 전 확인:

- [ ] Docker container 안에서 실행 중인가?
- [ ] 원본 dataset mount가 read-only인가?
- [ ] staged copy와 manifest가 원본 tree 밖에 있는가?
- [ ] `modded_dac_vq_s2` codec config를 쓰는가?
- [ ] SFT adapter SHA를 실행 전에 검증하는가?
- [ ] GRPO resume에 `+ckpt_path`를 쓰는가?
- [ ] production reward 없이 mechanics smoke임을 로그와 config에 명시했는가?
- [ ] checkpoint provenance SHA가 고정되어 있는가?

## 참고 문서

- Upstream README: [README.en.md](README.en.md)
- GRPO 실행 기록: [docs/GRPO_TRAINING.md](docs/GRPO_TRAINING.md)
- TTS/GRPO survey: [docs/PAPER.md](docs/PAPER.md)
- Korean waveform reward plan: [PLAN.md](PLAN.md)
- Agent constraints: [AGENTS.md](AGENTS.md)
