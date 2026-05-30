# 학습 실행 가이드

Gemma 도메인 LoRA 학습과 ACT Co-training 실행 절차입니다.
설계·정책은 [`README.md`](../../README.md) "학습 파이프라인 — Gemma 도메인 LoRA"
및 "학습 파이프라인 — ACT Co-training" 절 참조.

---

## Gemma 도메인 LoRA

Q&A 데이터 생성(`generate_function_call_qa.py` 의 Cat-B, `generate_verify_qa.py`
의 Cat-C)과 70:30 비율 혼합(`merge_qa_dataset.py`)을 거쳐 단일 JSONL 을 만든
후, 아래 두 단계로 학습·변환합니다.

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
