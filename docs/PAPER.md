# Fish Speech LoRA-GRPO 조사 노트

최종 갱신: 2026-07-22 (Asia/Seoul)

## 1. 범위와 근거

이 문서는 현재 저장소의 Fish Speech/Fish Audio S2 학습 코드에 LoRA 기반
GRPO(Group Relative Policy Optimization)를 추가하기 전에 필요한 근거를 정리한다.
비교 대상은 2026-07-22 기준 각 계열의 최신 공개 모델이다. 최신 모델 자체에 RL
구현이 없으면, 같은 계열의 이전 RL 모델은 방법론 참고로만 분리한다.

근거 표시는 다음처럼 구분한다.

- **[Local]** 현재 체크아웃한 소스와 데이터에서 직접 확인한 사실
- **[Paper]** 논문 저자가 보고한 사실 또는 실험 결과
- **[Official]** 공식 저장소, 릴리스 또는 모델 카드의 주장
- **[Proposal]** 이 프로젝트에서 검증해야 할 설계 제안
- **[Unresolved]** 공개 근거만으로 결정할 수 없는 항목

로컬 기준 revision은 Fish Speech `e5e292632cb11e7a27b2b7487f58f612bc101e13`,
`archive/GLM-TTS`는 `4b944f4be7b6c55454751715081f4dc83992897a`다.
논문 PDF 13편은 `archive/papers/`에 버전 고정 파일명으로 보관했으며 출처와
체크섬은 각각 `SOURCES.tsv`, `SHA256SUMS`에 있다. Higgs TTS 3는 전용 기술
논문이 없으므로 공식 모델 카드를 1차 근거로 사용하며, 다른 논문을 대신 넣지 않았다.

## 2. 요약 결론

1. **[Local]** 현재 Fish 학습기는 GRPO가 아니라 supervised cross-entropy SFT다.
   LoRA는 이미 지원하지만 기본 설정은 `lora_config: null`이고, 문서의 Hydra
   override가 rank 8/alpha 16 LoRA를 켠다.
2. **[Local]** 공개 GLM-TTS는 rollout, reward server, group normalization,
   generated-token mask, clipped policy loss를 가진 가장 가까운 TTS GRPO 참고 코드다.
   그러나 제공 launcher는 `PRETRAIN` full-parameter 경로이며 LoRA가 실제 GRPO
   경로에 연결되어 있지 않다.
3. **[Paper]** Fish Audio S2는 MLP-only rsLoRA(rank 16, alpha 64), Slow/Fast AR
   공동 최적화, group mean만 뺀 advantage, reference policy KL을 설명하지만
   trainer, reward model, group size, reward weight는 공개하지 않았다
   ([Fish S2 §4.3](https://arxiv.org/html/2603.08823v2#S4.SS3)).
4. **[Proposal]** 따라서 첫 구현은 GLM 코드를 복사하는 방식이 아니라 Fish의
   generation/LoRA 경로에 최소 GRPO loop를 붙여야 한다. rollout policy와 KL
   reference를 분리하고, 생성된 audio token만 loss에 포함하며, 한국어에서
   검증된 reward만 단계적으로 켠다.
5. **[Local]** 지정 데이터는 별도 `.lab` 대신 MP3 파일 stem에 URL-encoded
   transcript를 넣었다. stem을 decode해 manifest를 만들고, DOS 8.3 단축명이나
   text가 아닌 이름만 제외하면 된다. 이용 권리 확인은 별도 선행 조건이다.

## 3. 현재 저장소 기준선

### 3.1 SFT와 LoRA

**[Local]** `fish_speech/train.py`는 Hydra로 Lightning datamodule/model/trainer를
구성하고 `trainer.fit()`을 호출한다
([source](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/fish_speech/train.py#L35-L121)).
`TextToSemantic._step()`은 text token CE와 semantic codebook CE를 더한다. reward,
rollout, reference policy, policy ratio는 없다
([source](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/fish_speech/models/text2semantic/lit_module.py#L109-L191)).

**[Local]** 공식 LoRA 명령은 다음 override를 사용한다
([finetune guide](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/docs/en/finetune.md#L90-L125)).

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
  project="$project" \
  +lora@model.model.lora_config=r_8_alpha_16
```

기본 adapter는 `r=8`, `alpha=16`, dropout `0.01`이다. target 생략 시 Slow와
Fast AR의 attention, MLP, embedding, output을 모두 교체하고 LoRA 파라미터만
trainable로 만든다
([config](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/fish_speech/configs/lora/r_8_alpha_16.yaml),
[implementation](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/fish_speech/models/text2semantic/lora.py#L32-L105)).
Lightning checkpoint는 LoRA 사용 시 adapter 파라미터만 남긴다
([source](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/fish_speech/models/text2semantic/lit_module.py#L32-L42)).

**[Local]** 기본 `text2semantic_finetune.yaml`은 train과 validation이 모두
`data/protos`를 가리킨다. 이는 동작 확인용일 뿐 모델 비교용 held-out 평가가 아니다
([config](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/fish_speech/configs/text2semantic_finetune.yaml#L28-L51)).

### 3.2 데이터 전처리의 변경 범위

**[Local]** `tools/vqgan/extract_vq.py`는 각 audio 옆에 같은 stem의 `.npy`를
쓴다
([source](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/tools/vqgan/extract_vq.py#L128-L139)).
`tools/llama/build_dataset.py`는 `.npy`와 같은 stem의 transcript를 읽어 protobuf
shard를 만든다
([source](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/tools/llama/build_dataset.py#L23-L52)).
따라서 semantic 추출은 원본 경로에서 실행하지 말고 split manifest에 따라 만든
staging tree에서 실행해야 한다.

### 3.3 RL 이후 SFT 금지

**[Official]** Fish 문서는 RL-aligned 모델 위에 다시 SFT하면 분포가 이동하여 성능이
저하될 수 있다고 경고한다
([finetune guide](https://github.com/fishaudio/fish-speech/blob/e5e292632cb11e7a27b2b7487f58f612bc101e13/docs/en/finetune.md#L1-L6)).
실험 순서는 `base -> compatible SFT -> GRPO`로 단방향이어야 한다.

## 4. 데이터셋 점검

대상 경로:

```text
/home/ms-ms-home2/dataset/tts/250623_tts_jiwoo_whole_services/tts
```

**[Local, 2026-07-22]** 읽기 전용 inventory 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| 크기 | 약 11 GB |
| 최상위 voice/service 디렉터리 | 128 |
| MP3 (`.mp3` + `.MP3`) | 약 793K (동시 갱신 중이라 scan마다 소폭 변동) |
| `.wav`, `.flac`, `.lab`, `.npy` | 0 |
| 기타 파일 | `desktop.ini` 1개 |
| transcript 저장 방식 | MP3 파일 stem의 URL-encoded text |

### 4.1 읽기 전용 normalization snapshot

2026-07-22 02:58:40 UTC에 sorted relative path와 raw stem을 메모리에 고정한
snapshot은 MP3 793,184개였고 SHA-256은
`71214cfc91dbd1d23d26fb24825a779a97d0c2844d8f66ceae01e2ccc67e6508`이었다.
검사는 `os.walk()`로 이름만 읽었으며 audio, filename, sidecar에 대한 write/rename은
0회였다. source tree가 갱신 중이므로 실제 manifest 생성 때 count와 fingerprint를
다시 고정한다.

첫 `unquote_plus()`와 whitespace/NFC 정리 뒤에도 `%`가 남은 파일이 449개,
literal `+`가 남은 파일이 659개, `%xx` 형태가 남은 파일이 8개였다. 이는 decode가
실패한 것이 아니라 `%25`가 실제 percent를, `%2B`가 실제 plus를 나타내거나 URL이
한 번 더 encode된 경우다. 따라서 `%`와 `+`를 일괄 삭제하지 않고 의미에 따라
처리했다.

| 구분 | 규칙 | 성격 |
|---|---|---|
| 첫 decode | strict `unquote_plus()` 정확히 1회: raw `+`는 공백, `%xx`는 UTF-8 decode | 근거 기반 |
| 기본 정리 | BOM 제거, whitespace run을 공백 하나로 축약, trim, Unicode NFC | 근거 기반 |
| nested escape | URL/email이 아니고 `%xx`가 남을 때 strict `unquote()`를 최대 1회 | 근거 기반 |
| 반복 decode | fixed-point/무제한 decode 금지 | 안전 규칙 |
| numeric percent | 숫자 바로 뒤 `%`를 `퍼센트`로 치환 | 프로젝트 정책 |
| 기타 percent | 의미를 추측하지 않고 `ambiguous_percent`로 제외 | 프로젝트 정책 |
| literal plus | URL/email이 아니면 `플러스`로 치환 | 프로젝트 정책 |
| URL/email residual | URL 내부 `%xx`, `%`, `+`를 재작성하지 않고 제외 | 안전 규칙 |
| `?`, `!` | decode하되 순서와 개수를 보존 | 근거 기반 |

정확한 bounded 변환식은 다음과 같다. `exclude()`는 원본을 지우는 동작이 아니라
manifest에서 제외 사유를 기록하는 동작이다.

```python
import re
import unicodedata
from urllib.parse import unquote, unquote_plus

DOS_ALIAS = re.compile(r"^[A-Za-z0-9_]{1,6}~[A-Za-z0-9]$")
BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
URL_OR_EMAIL = re.compile(r"https?://\S+|\S+@\S+\.\S+", re.I)

def clean(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    return unicodedata.normalize("NFC", text)

def filename_to_transcript(stem: str) -> str:
    if not stem or DOS_ALIAS.fullmatch(stem) or BAD_PERCENT.search(stem):
        exclude("invalid_stem")

    text = clean(unquote_plus(stem, encoding="utf-8", errors="strict"))
    if ESCAPE.search(text):
        if URL_OR_EMAIL.search(text):
            exclude("url_encoded_residual")
        text = clean(unquote(text, encoding="utf-8", errors="strict"))
        if ESCAPE.search(text):
            exclude("nested_escape_unresolved")

    if URL_OR_EMAIL.search(text) and ("%" in text or "+" in text):
        exclude("url_symbol_residual")
    text = re.sub(r"(?<=\d)\s*%", " 퍼센트 ", text)
    if "%" in text:
        exclude("ambiguous_percent")
    text = re.sub(r"\s*\+\s*", " 플러스 ", text)
    text = clean(text)

    if not text or not any(ch.isalnum() for ch in text):
        exclude("non_text")
    assert "%" not in text and "+" not in text and not ESCAPE.search(text)
    return text
```

### 4.2 최종 stop gate

위 규칙으로 15개 edge-case fixture와 전체 snapshot을 반복 검사한 최종 결과는
다음과 같다.

| 항목 | 결과 |
|---|---:|
| 허용 transcript | 792,779 |
| 제외 합계 | 405 |
| 고유 normalized text | 705,134 |
| duplicate utterance | 87,645 |
| 숫자 `%` → `퍼센트` 치환 | 438회 |
| literal `+` → `플러스` 치환 | 788회 |
| 허용 text의 `%` / `+` / `%xx` | **0 / 0 / 0** |
| BOM / control / non-NFC | **0 / 0 / 0** |
| 숫자 sequence drift | **0** |
| `?`/`!` sequence drift | **0** |
| URL mutation | **0** |

제외 사유는 `non_text=271`, `dos_8_3_alias=89`, `ambiguous_percent=22`,
`empty_stem=16`, `url_encoded_residual=3`, `nested_decode_error=3`,
`nested_escape_unresolved=1`이었다. 최종 변환이 새로 합친 text는 세 그룹
(`플러스`, `-2 플러스 2`, `3 플러스 1`)뿐이며, 이 항목과 기존 duplicate는 같은
normalized-text group으로 묶어 split 경계를 넘지 않게 한다.

분포는 매우 불균형하다. `kyuri`가 369,666개로 약 46.6%, `seoyeon`이
66,714개, `dsayuri`가 27,165개다. 단순 utterance random split은 거의 같은
voice/service와 유사 문장이 모든 split에 섞일 가능성이 높다.

### 데이터 사용 전 필수 조건

1. **[Proposal] filename manifest 고정:** 원본 audio와 filename은 절대 바꾸지 않고,
   raw stem, normalized transcript, voice/service, exclusion reason, snapshot hash를
   별도 manifest에 기록한다.
2. **[Blocker] 권리 확인:** 디렉터리명에 상용 TTS service/voice로 보이는 항목이
   포함되어 있다. 각 서비스 약관이 출력물을 경쟁 모델 학습에 허용하는지, 실인물
   음성이 있다면 동의 범위가 무엇인지 확인해야 한다.
3. **[Proposal] manifest-first split:** 원본은 수정하지 않고
   `train/dev/test` manifest를 먼저 고정한다. 같은 문장, 같은 원본 session,
   duplicate audio hash는 split을 넘지 않게 한다.
4. **[Proposal] 두 평가 축:** 같은 voice에서 새 문장을 평가하는
   `text-held-out`과, voice/service 전체를 제외하는 `voice-held-out`을 둘 다 둔다.
5. **[Proposal] sampling cap:** dominant voice가 gradient를 지배하지 않도록
   voice-balanced sampler 또는 명시적 per-voice cap을 사용한다. 전체 79만 건을
   바로 처리하기 전에 작고 균형 잡힌 pilot manifest로 파이프라인을 검증한다.

## 5. 최신 모델 계열 비교

아래의 수치는 서로 다른 protocol에서 나온 경우 직접 순위로 비교하지 않는다.
“한국어 지원”과 “한국어 평가 공개”도 별도 항목이다.

| 계열의 최신 공개 모델 | 구조 / tokenizer | 한국어 근거 | 공개 adaptation / RL | 재현성 및 라이선스 |
|---|---|---|---|---|
| **Fish Audio S2 Pro** (`fishaudio/s2-pro`) | Qwen3 기반 Slow AR 4B + Fast AR 약 400M, Modified DAC 10 codebook 약 21 Hz | Tier 2 지원. MiniMax multilingual: WER 1.180, SIM 0.817; CV3: WER 2.76 ([S2 Table 2-3](https://arxiv.org/html/2603.08823v2#S6.T2)) | 공개 LoRA SFT, GRPO-aligned weight. GRPO trainer/reward는 미공개 | 코드/weight 모두 Fish Audio Research License, 비상업 연구 중심 |
| **Higgs TTS 3** (`bosonai/higgs-tts-3-4b`) | 약 4B AR, 8 codebook, 25 fps, 24 kHz | 한국어를 WER/CER `<5` 그룹에 포함하나 개별 값과 evaluator 미공개 ([model card](https://huggingface.co/bosonai/higgs-tts-3-4b)) | 공식 FT/LoRA/RL recipe 없음. 이전 Higgs TTS 2.5가 GRPO 방법론 참고 | 전용 논문 없음. Boson research/non-commercial license |
| **GLM-TTS Base** (`zai-org/GLM-TTS`) | 1.5B, Llama text-to-token AR + Flow Matching, 32K speech vocab 25 Hz | 중국어 중심 + 영중 혼용. 한국어 검증 없음 | SFT/LoRA mode와 GRPO source 공개. `GLM-TTS_RL` weight는 미공개 | 코드 Apache-2.0, HF weight MIT |
| **OmniVoice** (`k2-fsa/OmniVoice`, software 0.2.1) | 약 0.8B discrete NAR masked-diffusion LM, 8 codebook, 600+ 언어 | MiniMax: WER 2.651/SIM-o 0.828; FLEURS: 3.78 ([paper Table 3/10](https://arxiv.org/html/2604.00688v3#S4.T3)) | full training/fine-tuning 공개, 공식 LoRA/RL 없음 | 코드 Apache-2.0, weight는 모델 카드상 CC-BY-NC(버전 미명시) |
| **Qwen3-TTS 12Hz** 공개군 | 0.6B/1.7B, dual-track AR, 16 codebook 12.5 Hz, 24 kHz | Base: 0.6B WER 1.741/SIM 0.812, 1.7B WER 1.755/SIM 0.799 ([paper Table 6](https://arxiv.org/html/2601.15621v1#S4.T6)) | Base single-speaker SFT 공개. 내부 DPO→GSPO post-training은 논문에만 기술 | 코드/weight Apache-2.0 |

### 비교 해석

- **Fish S2 Pro**는 이 저장소와 구조적으로 가장 직접 연결되며 실제 GRPO-aligned
  weight가 있다. 단, 공개 SFT 코드가 논문의 RL trainer인 것은 아니다.
- **GLM-TTS**는 공개 코드 관점의 주 참고 대상이다. 그러나 중국어 reward stack과
  full-parameter launch를 Fish에 그대로 이식할 수 없다.
- **Higgs TTS 3**는 2026-06 공개된 최신 Higgs 모델이다. GRPO가 명시된 모델은
  이전 세대 Higgs TTS 2.5이므로 최신 성능 비교와 RL 방법론 근거를 분리한다
  ([v2.5 announcement](https://www.boson.ai/blog/higgs-audio-v2.5)).
- **OmniVoice**는 광범위한 한국어/다국어 평가와 공개 fine-tuning pipeline이
  장점이지만 NAR diffusion 구조라 Fish의 AR-GRPO 구현 template은 아니다.
- **Qwen3-TTS**는 공개 Base SFT와 permissive license가 장점이다. 논문 post-training은
  GRPO가 아니라 DPO/GSPO이므로 용어를 섞지 않는다
  ([Qwen3-TTS §3.2](https://arxiv.org/html/2601.15621v1#S3.SS2)).

## 6. GLM-TTS GRPO 구현 분석

### 6.1 실제 경로

**[Local]** 처리 흐름은 다음과 같다.

1. prompt 하나당 8개 speech-token completion을 생성한다
   ([config](https://github.com/zai-org/GLM-TTS/blob/4b944f4be7b6c55454751715081f4dc83992897a/grpo/config/lm_llama_casual_glm_32k_GRPO_cer.yaml#L15-L32)).
2. Flow decoder로 waveform을 만들고 reward server에 임시 파일 경로, 정답 text,
   reference audio 정보를 보낸다
   ([rollout](https://github.com/zai-org/GLM-TTS/blob/4b944f4be7b6c55454751715081f4dc83992897a/grpo/grpo_utils.py#L195-L265)).
3. Chinese ASR CER, WavLM-ECAPA SIM, emotion2vec, laughter detector를 계산한다
   ([reward server](https://github.com/zai-org/GLM-TTS/blob/4b944f4be7b6c55454751715081f4dc83992897a/grpo/reward_server.py#L114-L213)).
4. 각 reward를 group 안에서 z-normalize하고 가중합을 다시 normalize한다.
   기본 weight는 CER 1, SIM 0.1이며 emotion/pitch/energy는 0이다
   ([normalization](https://github.com/zai-org/GLM-TTS/blob/4b944f4be7b6c55454751715081f4dc83992897a/grpo/grpo_utils.py#L296-L350)).
5. prompt와 padding을 mask하고 generated token에만 clipped surrogate와 KL을
   계산한다
   ([loss](https://github.com/zai-org/GLM-TTS/blob/4b944f4be7b6c55454751715081f4dc83992897a/cosyvoice/utils/train_utils_grpo.py#L248-L359)).
6. 모든 reward가 같은 group은 최대 3번 다시 rollout한다
   ([executor](https://github.com/zai-org/GLM-TTS/blob/4b944f4be7b6c55454751715081f4dc83992897a/cosyvoice/utils/executor_grpo.py#L102-L128)).

### 6.2 그대로 복사하면 안 되는 이유

- **LoRA 미연결:** launcher는 `--mode=PRETRAIN`이고 `apply_lora()` 정의는 실제
  training path에서 호출되지 않는다
  ([launcher](https://github.com/zai-org/GLM-TTS/blob/4b944f4be7b6c55454751715081f4dc83992897a/grpo/pretrain_GRPO_single.sh#L23-L41)).
- **policy 역할 혼합:** 학습 전 한 번 복제한 정적 `ref_model`을 PPO ratio 분모와
  KL reference 양쪽에 쓴다. rollout behavior policy와 KL reference는 별도로
  관리해야 한다.
- **중국어 고정 reward:** `zhconv`, Chinese Paraformer, Chinese laughter 문자를
  사용하므로 한국어 reward가 아니다.
- **설정 drift:** YAML의 `micro_batch_size`는 실제 slicing에 연결되지 않고,
  RAS sampling path는 전달된 temperature/top-p를 그대로 사용하지 않는다.
- **종료 정보 손실:** max token에 걸린 completion과 정상 EOS를 Episode에 기록하지
  않아 truncation penalty와 실패율을 계산할 수 없다.
- **인프라 고정:** 사설 IP, `808{LOCAL_RANK}` port, 공유 filesystem, NVIDIA/NCCL,
  외부 reward checkpoint를 가정한다.

GLM 코드의 가치가 낮다는 뜻은 아니다. **rollout → waveform → reward → group
advantage → generated-token loss**라는 모듈 경계는 참고하되, policy state와
reward는 Fish에 맞게 다시 연결해야 한다.

## 7. GRPO 계열 방법론에서 채택할 것

| 근거 | 핵심 | Fish TTS에 대한 해석 |
|---|---|---|
| DeepSeekMath GRPO ([v3](https://arxiv.org/abs/2402.03300v3)) | critic 없이 group-relative advantage | prompt별 여러 speech completion을 비교하는 출발점 |
| Dr.GRPO ([v2](https://arxiv.org/abs/2503.20783v2)) | group std와 response length normalization의 bias 분석 | reward 저분산 group에서 잡음을 증폭하지 않도록 std division을 피하고 길이 정의를 명시 |
| DAPO ([v2](https://arxiv.org/abs/2503.14476v2)) | asymmetric clip, dynamic sampling, token-level loss, overlong shaping | zero-variance group 처리와 truncation 기록을 차용하되 무제한 filtering은 금지 |
| Fish Audio S2 ([v2](https://arxiv.org/abs/2603.08823v2)) | centered no-std advantage, Dual-AR 공동 RL, MLP-only rsLoRA, reference KL | 목표 설계의 가장 가까운 논문 근거 |
| GLM-TTS ([v1](https://arxiv.org/abs/2512.14291v1)) | 실제 TTS waveform reward service와 공개 loss scaffold | 시스템 분해 참고, 알고리즘/중국어 reward는 그대로 복제하지 않음 |

## 8. LoRA 기반 GRPO 권고안

### 8.1 두 실험 track을 분리한다

**Track A: 최소 통합 검증**

- 현재 공개 `r_8_alpha_16` topology로 SFT LoRA baseline을 만든다.
- 같은 topology와 adapter를 GRPO로 이어간다.
- 결과 이름은 `Fish local LoRA-GRPO`로 하고 Fish S2 재현이라고 부르지 않는다.
- 목적은 rollout, mask, checkpoint, reward plumbing의 정확성 검증이다.

**Track B: Fish S2 논문 정합 실험**

- 처음부터 별도 MLP-only **rsLoRA** `r=16`, `alpha=64` SFT warm-start를 만든다.
- 동일 adapter topology를 GRPO까지 유지하고 base/codec은 고정한다.
- 현재 `loralib` 구현은 일반 `alpha/r` scaling이므로 rsLoRA의
  `alpha/sqrt(r)` scaling을 실제로 구현·검증해야 한다. config 숫자만 바꾸면
  rsLoRA가 아니다.
- 공개 논문이 정확한 Slow/Fast target name을 제공하지 않으므로 target list는
  별도 ablation 대상으로 기록한다.

현재 broad rank-8 SFT adapter를 중간에 MLP-only rank-16으로 바꾸어 이어 학습하면
SFT와 GRPO 효과, topology 효과가 섞인다. Track B는 compatible SFT부터 새로 시작한다.

### 8.2 정책과 loss

**[Proposal] 최소 구현 contract:**

1. policy는 SFT adapter에서 시작한다.
2. KL reference는 같은 SFT checkpoint의 동결 snapshot이다.
3. rollout 시점의 `old_logprob`를 completion token별로 저장한다. 이것은 KL
   reference와 다른 역할이다.
4. prompt당 우선 `G=8` completion을 사용한다. 이는 GLM의 공개 default이지
   Fish S2의 공개 hyperparameter는 아니다.
5. advantage는 Fish S2/Dr.GRPO 방향에 맞춰 `A_i = R_i - mean(R_group)`으로 하고
   group std로 나누지 않는다.
6. prompt/reference/padding은 0, 생성된 Slow/Fast audio token만 1인 loss mask를
   만든다.
7. 첫 pilot은 rollout group당 1 update epoch로 시작한다. 여러 epoch로 재사용할
   때만 old-policy ratio와 clip의 영향이 커지므로 ratio/clip fraction을 함께 본다.
8. clip을 쓴다면 공개 GLM 시작값 `low=0.2`, `high=0.3`은 baseline일 뿐이다.
   dynamic widening은 ablation 전에는 끈다.
9. learned TTS reward는 hacking 위험이 있으므로 KL을 0으로 고정하지 않는다.
   작은 non-zero beta를 sweep하고 KL/token, entropy, reward 상승과 held-out 품질을
   동시에 본다. 정확한 beta는 공개 Fish 근거가 없어 pilot으로 결정한다.
10. EOS, truncation, 생성 token 수, waveform decode 실패를 Episode에 저장한다.

### 8.3 Reward gate

GLM의 Chinese reward model은 사용하지 않는다. 후보 scorer는 학습 전에 독립된
한국어 human-labelled holdout에서 검증한다.

| reward | 활성화 조건 | 첫 pilot |
|---|---|---|
| 내용 정확도 | 한국어 CER/어절 WER와 deletion/insertion, 숫자·영문 code-switch subgroup에서 검증 | 통과 시 weight 1.0 |
| 화자 유사도 | same/different speaker ROC/EER와 human similarity 상관 검증 | 처음에는 0; 통과 후 0.1 ablation |
| 음질/자연스러움 | 한국어 MOS/CMOS와 상관, 잡음·속도별 오류 분석 | 0 |
| emotion/tag | 각 한국어 tag의 human activation label과 FPR/FNR 검증 | 0 |
| laughter/비언어 | 한국어 표기와 실제 acoustic event detector 검증 | 0 |

검증된 scorer가 하나도 없으면 GRPO를 돌리지 않고 SFT LoRA에서 중단한다. reward
모델의 training/evaluation split도 policy 평가 split과 분리해야 한다.

### 8.4 Checkpoint contract

adapter weight만 저장하면 재현 가능한 GRPO resume가 아니다. 다음을 함께 남긴다.

- adapter, optimizer, scheduler, global step, RNG state
- base/SFT/reference adapter revision 또는 SHA-256
- tokenizer/codec revision
- old-policy semantics와 update epoch 수
- reward model revision, normalization, weight, calibration version
- decoding 설정, split/evaluation manifest hash
- EOS/truncation 정책과 invalid rollout 처리 버전

## 9. 학습 계획

### Phase 0: 데이터와 권리

1. section 4의 bounded 규칙으로 filename stem을 정규화하되 원본 파일과 이름은
   변경하지 않고, snapshot hash와 제외 사유를 포함한 transcript manifest를 만든다.
2. 서비스별 출력물 학습 권리, 실인물 동의, voice cloning 제한을 확인한다.
3. duplicate/text/session/voice leakage를 제거한 manifest를 고정한다.
4. 원본을 건드리지 않는 staging tree에서만 semantic token을 추출한다.

### Phase 1: SFT 기준선

1. **B0:** 변경하지 않은 base checkpoint.
2. **B1:** 공개 broad `r8/alpha16` LoRA SFT.
3. **B2:** MLP-only rsLoRA `r16/alpha64` compatible SFT.
4. 같은 data, step budget, seed, validation manifest로 B1/B2를 비교한다.
5. B2가 SFT에서 불안정하면 GRPO로 넘어가지 않는다.

### Phase 2: Reward validation

1. policy 학습과 무관한 한국어 human-labelled subset을 만든다.
2. ASR/SIM/quality/tag scorer를 각각 검증하고 calibration curve를 남긴다.
3. reward 간 scale과 correlation을 확인한다.
4. prompt별 모든 reward가 같은 zero-variance 비율을 측정한다.

### Phase 3: GRPO pilot

1. 균형 잡힌 작은 manifest, 한 seed, `G=8`, CER-only로 end-to-end를 검증한다.
2. generated-token mask, old/reference logprob, KL, truncation을 unit test한다.
3. zero-variance group은 최대 3회만 재시도하고 retry/discard 비율을 기록한다.
4. held-out CER가 좋아져도 MOS/SIM/실패율이 악화되면 중단한다.

### Phase 4: ablation과 최종 학습

1. CER-only vs CER+검증된 SIM.
2. KL beta sweep.
3. B1 topology vs B2 topology는 별도 track으로 비교.
4. group size와 clip은 한 번에 하나만 바꾼다.
5. 최종 비교는 최소 3 seed와 confidence interval을 보고한다.

## 10. 평가 계획

### 10.1 비교군

- `B0`: base Fish checkpoint
- `B1`: 현재 public-style broad rank-8 LoRA SFT
- `B2`: GRPO-compatible MLP-only rsLoRA SFT
- `G1`: B1에서 시작한 local plumbing pilot
- `G2`: B2에서 시작한 S2-aligned topology LoRA-GRPO
- ablation: CER-only, CER+SIM, KL beta, group size, topology

외부 최신 모델은 reference generation으로 포함할 수 있지만 weight license와 serving
조건을 기록하고, 동일 checkpoint revision을 pin한다.

### 10.2 고정 protocol

- 동일 target text, reference audio/transcript, random seed set, sampling budget
- 동일 max token, temperature/top-p/top-k, loudness와 sample-rate 후처리
- best-of-N이나 재생성 없이 최초 결과와 실패를 모두 포함
- checkpoint, tokenizer/codec, ASR/SIM/judge revision 기록
- author-reported metric과 local reproduction을 같은 표의 별도 열로 표시

### 10.3 한국어 지표

1. **CER:** Unicode NFC 후 Hangul syllable 단위. 공백/문장부호/숫자/영문을 어떻게
   처리했는지 규칙과 함께 보고한다.
2. **어절 WER:** 공백 기반만 쓰지 말고 숫자 읽기, 조사 결합, code-switch 오류를
   별도 subgroup으로 본다.
3. **speaker SIM:** 한국어 pair에서 검증된 encoder만 사용하고 EER/calibration을
   같이 제공한다.
4. **human MOS/CMOS:** 자연스러움, 발음 정확도, 음색 유사도, instruction/tag
   준수를 분리하고 blind randomized paired test로 수행한다.
5. **안정성:** empty audio, decode error, EOS 실패, truncation, 비정상 반복,
   평균/상위 95% audio-token 길이.
6. **효율:** RTF, time-to-first-audio, peak GPU memory, rollout/reward/update wall time.
7. **학습 건전성:** KL/token, entropy, clip fraction, zero-variance group,
   resampling/discard율, train reward와 held-out human score의 상관.

### 10.4 평가 세트

- 내부 `text-held-out`: 같은 voice의 새 문장
- 내부 `voice/service-held-out`: 학습에서 제외한 voice 또는 service
- 공개 한국어: 라이선스와 protocol을 고정한 FLEURS Korean 등
- multilingual regression: base의 일반 능력 손실을 확인하는 별도 set
- expressive set: 감정/비언어 tag를 검증할 human-labelled set

Seed-TTS Eval, FLEURS, EmergentTTS-Eval은 목적과 judge가 다르다. 논문 수치를 하나의
leaderboard로 합치지 말고, 가능한 경우 동일 manifest/evaluator로 직접 재실행한다
([Seed-TTS](https://arxiv.org/abs/2406.02430v1),
[FLEURS](https://arxiv.org/abs/2205.12446v1),
[EmergentTTS-Eval](https://arxiv.org/abs/2505.23009v1)).

## 11. 위험과 미해결 항목

- **[Risk]** transcript가 filename에만 있으므로 bounded decode와 spoken-symbol
  policy를 manifest 생성 시 재현 가능하게 고정해야 한다. 무제한 decode, 원본
  rename, decoded text의 path 사용은 금지한다.
- **[Blocker]** 서비스 생성 음성/실인물 음성의 학습 및 voice cloning 권리가 확인되지 않았다.
- **[Unresolved]** Fish S2의 exact group size, reward weights, KL beta, optimizer/LR,
  target module name은 공개되지 않았다.
- **[Unresolved]** 공개 Fish LoRA는 일반 LoRA이며 rsLoRA가 아니다.
- **[Unresolved]** 한국어 ASR/SIM/quality reward 후보의 calibration 데이터가 없다.
- **[Risk]** reward를 evaluation에도 그대로 쓰면 reward hacking을 탐지할 수 없다.
- **[Risk]** `kyuri` 중심의 심한 데이터 불균형은 speaker/style collapse를 만들 수 있다.
- **[Risk]** Fish와 GLM-TTS는 dependency와 model stack이 다르다. GLM environment를
  Fish environment에 섞지 말고 archive reference로 격리한다.
- **[Risk]** GRPO 뒤 SFT를 수행하면 alignment가 사라질 수 있다.

## 12. 참고문헌과 공식 자료

로컬 PDF 경로는 `archive/papers/` 기준이다.

1. Hu et al., [LoRA](https://arxiv.org/abs/2106.09685v2), `2106.09685v2-lora.pdf`.
2. Kalajdzievski, [rsLoRA](https://arxiv.org/abs/2312.03732v1), `2312.03732v1-rslora.pdf`.
3. Shao et al., [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300v3), `2402.03300v3-deepseekmath-grpo.pdf`.
4. Liu et al., [Understanding R1-Zero-Like Training / Dr.GRPO](https://arxiv.org/abs/2503.20783v2), `2503.20783v2-dr-grpo.pdf`.
5. Yu et al., [DAPO](https://arxiv.org/abs/2503.14476v2), `2503.14476v2-dapo.pdf`.
6. Liao et al., [Fish-Speech](https://arxiv.org/abs/2411.01156v2), `2411.01156v2-fish-speech.pdf`.
7. Liao et al., [Fish Audio S2](https://arxiv.org/abs/2603.08823v2), `2603.08823v2-fish-audio-s2.pdf`.
8. Cui et al., [GLM-TTS](https://arxiv.org/abs/2512.14291v1), `2512.14291v1-glm-tts.pdf`.
9. Hu et al., [Qwen3-TTS](https://arxiv.org/abs/2601.15621v1), `2601.15621v1-qwen3-tts.pdf`.
10. Zhu et al., [OmniVoice](https://arxiv.org/abs/2604.00688v3), `2604.00688v3-omnivoice.pdf`.
11. Anastassiou et al., [Seed-TTS](https://arxiv.org/abs/2406.02430v1), `2406.02430v1-seed-tts.pdf`.
12. Conneau et al., [FLEURS](https://arxiv.org/abs/2205.12446v1), `2205.12446v1-fleurs.pdf`.
13. Boson AI et al., [EmergentTTS-Eval](https://arxiv.org/abs/2505.23009v1), `2505.23009v1-emergenttts-eval.pdf`.
14. Boson AI, [Higgs TTS 3 official model card](https://huggingface.co/bosonai/higgs-tts-3-4b).
15. Fish Audio, [S2 Pro model card](https://huggingface.co/fishaudio/s2-pro).
16. Z.ai, [GLM-TTS repository](https://github.com/zai-org/GLM-TTS/tree/4b944f4be7b6c55454751715081f4dc83992897a).
17. k2-fsa, [OmniVoice 0.2.1](https://github.com/k2-fsa/OmniVoice/releases/tag/0.2.1).
18. Qwen, [Qwen3-TTS official collection](https://huggingface.co/collections/Qwen/qwen3-tts).
