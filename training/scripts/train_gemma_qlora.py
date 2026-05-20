"""
train_gemma_qlora.py

Gemma 4 E2B base 모델에 도메인 LoRA를 Unsloth QLoRA로 파인튜닝한다.

CLI 사용 예 (README L646-654):
  python train_gemma_qlora.py \\
    --base-model  google/gemma-4-e2b \\
    --dataset     training/data/qa_recycling_merged.jsonl \\
    --output-dir  training/outputs/lora_recycling \\
    --rank        16 \\
    --alpha       32 \\
    --epochs      3

학습 데이터:
  merge_qa_dataset.py가 만든 단일 JSONL. 각 행은 {"messages": [...]} 형식이며,
  Cat-A(텍스트만) / Cat-B(텍스트만) / Cat-C(이미지+텍스트)가 섞여 있다.
  Unsloth가 multi-modal 입력을 어떻게 받는지에 따라 Cat-C 행 처리 방식이 결정.

선행 조건:
  - GPU VRAM 16GB 이상 (RTX 5070급, README L129)
  - Unsloth 설치
  - HuggingFace gemma-4-e2b 액세스 권한

산출물:
  ${output_dir}/adapter_model.safetensors 등 — convert_lora_to_gguf.sh가 GGUF로 변환

미구현 / 결정 필요 항목:
  - Unsloth 버전과 Gemma 4 E2B 호환성 확인 (운영 시점에 확정)
  - LoRA 적용 모듈 (target_modules) — 모든 attention proj / MLP 일부 등
  - Cat-C 멀티모달 행의 처리 — 별도 학습 분리 vs 통합
  - 학습률 스케줄, weight decay, gradient clip
  - validation split 전략
"""

from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Gemma 도메인 LoRA QLoRA 학습')
    p.add_argument('--base-model', required=True,  help='HF base model id 또는 로컬 경로')
    p.add_argument('--dataset',    required=True,  help='merge_qa_dataset.py 출력 JSONL')
    p.add_argument('--output-dir', required=True,  help='LoRA 저장 디렉터리')
    p.add_argument('--rank',       type=int, default=16, help='LoRA rank')
    p.add_argument('--alpha',      type=int, default=32, help='LoRA alpha')
    p.add_argument('--epochs',     type=int, default=3,  help='에폭 수')
    p.add_argument('--lr',         type=float, default=2e-4, help='학습률')
    p.add_argument('--batch-size', type=int, default=4, help='per-device batch size')
    return p.parse_args()


def build_model_and_tokenizer(args):
    """Unsloth FastLanguageModel.from_pretrained로 4bit base 로드 + LoRA 부착.

    TODO: Unsloth 버전 확정 후 구현.
      from unsloth import FastLanguageModel
      model, tokenizer = FastLanguageModel.from_pretrained(
          model_name=args.base_model, load_in_4bit=True, ...
      )
      model = FastLanguageModel.get_peft_model(
          model, r=args.rank, lora_alpha=args.alpha, ...
      )
    """
    raise NotImplementedError("[train_gemma_qlora] build_model_and_tokenizer() 미구현")


def build_dataset(args, tokenizer):
    """JSONL → HF Dataset (messages 형식 chat template 적용).

    TODO: chat template / Cat-C 멀티모달 처리 정책 확정 후 구현.
    """
    raise NotImplementedError("[train_gemma_qlora] build_dataset() 미구현")


def train(model, tokenizer, dataset, args):
    """SFTTrainer 또는 unsloth 학습 루프.

    TODO: 학습률 스케줄 / 로깅 / 체크포인트 정책 확정 후 구현.
    """
    raise NotImplementedError("[train_gemma_qlora] train() 미구현")


def main():
    args = parse_args()
    model, tok = build_model_and_tokenizer(args)
    ds = build_dataset(args, tok)
    train(model, tok, ds, args)
    # model.save_pretrained(args.output_dir) 등은 train() 내부 또는 여기서.


if __name__ == '__main__':
    main()
