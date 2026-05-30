# 학습 실행 가이드

Gemma 도메인 LoRA 학습과 ACT Co-training 실행 절차입니다.
설계·정책은 [`README.md`](../../README.md) "학습 파이프라인 — Gemma 도메인 LoRA"
및 "학습 파이프라인 — ACT Co-training" 절 참조.

---

## Gemma 도메인 LoRA

Q&A 데이터 생성(`generate_function_call_qa.py` 의 Cat-B, `generate_verify_qa.py`
의 Cat-C)과 70:30 비율 혼합(`merge_qa_dataset.py`)을 거쳐 단일 JSONL 을 만든
후, 아래 두 단계로 학습·변환합니다.

### 학습 환경 준비 (외부 PC)

```bash
pip install -r training/scripts/requirements.txt
export OPENAI_API_KEY=sk-...   # gpt-4o-mini 호출용 (Cat-B/C 합성기 공통)
```

### Cat-B 합성 (함수 호출 시나리오)

```bash
# 도메인별로 1회씩 (recycling, tableware)
python training/scripts/generate_function_call_qa.py \
  --domain         recycling \
  --label-vocab    deploy/config/label_vocab.yaml \
  --output         training/data/qa_recycling_catB.jsonl \
  --external-llm   gpt-4o-mini \
  --num-samples    500
```

> 비용 추정: 500 샘플 ≈ $0.02–0.05 (gpt-4o-mini). 진행률·누적 토큰·비용이
> stderr 로 50샘플마다 출력된다.

**먼저 dry run (10샘플)** 으로 출력 JSONL 한두 줄 직접 검사 권장:
```bash
python training/scripts/generate_function_call_qa.py \
  --domain recycling --output /tmp/catB_dry.jsonl --num-samples 10
head -n 2 /tmp/catB_dry.jsonl | python -m json.tool
```

### Cat-C 합성 (검증 시나리오) — Sim 자산 준비 필요

(`generate_verify_qa.py` 본체 구현 후 작성)

### Q&A 병합 (70:30, oversample)

```bash
python training/scripts/merge_qa_dataset.py \
  --cat-b   training/data/qa_recycling_catB.jsonl \
  --cat-c   training/data/qa_recycling_catC.jsonl \
  --output  training/data/qa_recycling_merged.jsonl
```

Cat-C 가 비어 있어도 동작 (Cat-B 만 사용 + 경고). 1차 학습 시 Cat-B 만으로
시험하려면 빈 파일을 `--cat-c` 로 넘기면 됨:
```bash
: > /tmp/catC_empty.jsonl
python training/scripts/merge_qa_dataset.py \
  --cat-b training/data/qa_recycling_catB.jsonl \
  --cat-c /tmp/catC_empty.jsonl \
  --output training/data/qa_recycling_merged.jsonl
```

### Unsloth QLoRA 파인튜닝

```bash
python training/scripts/train_gemma_qlora.py \
  --base-model google/gemma-4-e2b \
  --dataset training/data/qa_recycling_merged.jsonl \
  --output-dir training/outputs/lora_recycling \
  --rank 16 --alpha 32 \
  --epochs 3
```

### GGUF 변환

```bash
bash training/scripts/convert_lora_to_gguf.sh \
  training/outputs/lora_recycling \
  models/gemma/adapters/adapter_recycling.gguf
```

---

## ACT Co-training

LeRobot 0.4.4 멀티 데이터셋 가중치 학습으로 Sim + Real 혼합.

```bash
bash training/scripts/train_act_cotraining.sh
```

스크립트 내부의 가중치·하이퍼파라미터·환경 변수는 `train_act_cotraining.sh` 헤더
주석 참조.
