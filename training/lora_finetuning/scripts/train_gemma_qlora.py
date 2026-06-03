"""
train_gemma_qlora.py

Gemma 4 E2B 비전 모델에 도메인 LoRA 를 Unsloth FastVisionModel + QLoRA 로 학습.

데이터:
  merge_qa_dataset.py 출력 단일 JSONL (function_call text-only + verify image+text 통합,
  Unsloth typed-array content 형식). 각 행 {"messages": [...]}.

이미지 처리 (lazy):
  verify user content 의 image 필드는 image_dir 기준 상대 path. 학습 시점에
  custom collator 가 PIL.Image.open(image_dir/path) 로 변환 — 데이터를 메모리에
  전부 올리지 않음.

text-only function_call 호환성 (per_device_batch_size=1 가정):
  Custom collator 가 sample 의 image 유무를 판단해 processor 에
  images=None / PIL_list 로 호출 (unslothai/unsloth#1590 패턴).
  batch 안에 image 있는/없는 sample 이 섞이지 않도록 batch_size=1 강제.
  effective batch 는 gradient_accumulation_steps 로 확보.

모델 / LoRA:
  unsloth/gemma-4-E2B-it (4-bit) → FastVisionModel.get_peft_model
  finetune_vision_layers=False (language layers 만 학습 — Unsloth 권장),
  rank/alpha 는 CLI 기본 16/32.

CLI 예:
  python training/lora_finetuning/scripts/train_gemma_qlora.py \\
    --base-model    unsloth/gemma-4-E2B-it \\
    --dataset       training/lora_finetuning/data/recycling/qa_merged.jsonl \\
    --image-dir     training/lora_finetuning/data/recycling/verify_images \\
    --output-dir    training/lora_finetuning/outputs/lora_recycling \\
    --rank 16 --alpha 32 --epochs 3

산출물:
  ${output_dir}/  — LoRA adapter (adapter_model.safetensors) + processor /
                    tokenizer config. convert_lora_to_gguf.sh 가 GGUF 로 변환.

선행 조건:
  pip install -r training/lora_finetuning/requirements.txt
  Unsloth 호환 GPU + CUDA (≥16 GB VRAM 권장)
  HuggingFace 'unsloth/gemma-4-E2B-it' 액세스
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional


# ──────────────────────────────────────────────
# CLI / args
# ──────────────────────────────────────────────

@dataclass
class TrainArgs:
    base_model:             str
    dataset:                str
    image_dir:              str
    output_dir:             str
    rank:                   int
    alpha:                  int
    epochs:                 float
    lr:                     float
    batch_size:             int
    grad_accum:             int
    max_seq_len:            int
    max_steps:              int
    warmup_steps:           int
    logging_steps:          int
    save_steps:             int
    seed:                   int
    finetune_vision_layers: bool


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(
        description='Gemma 4 E2B LoRA fine-tune (FastVisionModel + QLoRA)')
    p.add_argument('--base-model', default='unsloth/gemma-4-E2B-it',
                   help='HF model id 또는 로컬 경로')
    p.add_argument('--dataset', required=True,
                   help='merge_qa_dataset.py 출력 JSONL')
    p.add_argument('--image-dir', required=True,
                   help='verify sample 의 image path 베이스 (image 필드와 join)')
    p.add_argument('--output-dir', required=True,
                   help='LoRA adapter 저장 디렉터리')
    p.add_argument('--rank', type=int, default=16, help='LoRA rank')
    p.add_argument('--alpha', type=int, default=32, help='LoRA alpha')
    p.add_argument('--epochs', type=float, default=3.0)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch-size', type=int, default=1,
                   help='per-device batch (text-only/image+text 혼합 안전성 위해 1 강제)')
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--max-seq-len', type=int, default=2048)
    p.add_argument('--max-steps', type=int, default=-1,
                   help='-1=epochs 기반, >0=steps 강제 (epochs 무시)')
    p.add_argument('--warmup-steps',  type=int, default=10)
    p.add_argument('--logging-steps', type=int, default=10)
    p.add_argument('--save-steps',    type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--finetune-vision-layers', action='store_true',
                   help='vision encoder 도 fine-tune (기본은 language only — Unsloth 권장)')
    a = p.parse_args()
    return TrainArgs(
        base_model = a.base_model,
        dataset    = os.path.expanduser(a.dataset),
        image_dir  = os.path.expanduser(a.image_dir),
        output_dir = os.path.expanduser(a.output_dir),
        rank=a.rank, alpha=a.alpha, epochs=a.epochs, lr=a.lr,
        batch_size=a.batch_size, grad_accum=a.grad_accum,
        max_seq_len=a.max_seq_len, max_steps=a.max_steps,
        warmup_steps=a.warmup_steps, logging_steps=a.logging_steps,
        save_steps=a.save_steps, seed=a.seed,
        finetune_vision_layers=a.finetune_vision_layers,
    )


# ──────────────────────────────────────────────
# JSONL 로드 (image 는 path 그대로 — collator 가 PIL 로 변환)
# ──────────────────────────────────────────────

def load_samples(jsonl_path: str) -> List[dict]:
    """JSONL → list of {"messages": [...]}. 형식 검증 + 결손 sample 거부."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f'dataset 파일 없음: {jsonl_path}')
    samples: List[dict] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for ln_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'{jsonl_path}:{ln_no} JSON 파싱 실패: {e}') from e
            if not isinstance(obj.get('messages'), list):
                raise ValueError(f'{jsonl_path}:{ln_no} "messages" 키 없거나 list 아님')
            samples.append(obj)
    if not samples:
        raise ValueError(f'dataset 이 비어 있음: {jsonl_path}')
    return samples


def _count_image_items(sample: dict) -> int:
    n = 0
    for m in sample.get('messages', []):
        content = m.get('content', [])
        if isinstance(content, list):
            for it in content:
                if isinstance(it, dict) and it.get('type') == 'image':
                    n += 1
    return n


def _resolve_images_in_sample(sample: dict, image_dir: str):
    """sample 의 messages 를 순회하며 image string path 를 PIL.Image 로 교체.

    원본을 변경하지 않고 새 dict 반환. images 리스트는 processor 호출용으로 함께.
    image_dir 가 절대경로 또는 상대경로, image 필드도 절대/상대 모두 허용.
    """
    from PIL import Image  # noqa: PLC0415

    images: List[Any] = []
    new_messages: List[dict] = []
    for msg in sample.get('messages', []):
        role = msg.get('role')
        content = msg.get('content', [])
        if not isinstance(content, list):
            new_messages.append({'role': role, 'content': content})
            continue
        new_content: List[dict] = []
        for item in content:
            if not (isinstance(item, dict) and item.get('type') == 'image'):
                new_content.append(item)
                continue
            img_ref = item.get('image', '')
            if not img_ref:
                raise ValueError(f"image item 의 'image' 필드 비어 있음: {item!r}")
            abs_path = img_ref if os.path.isabs(img_ref) \
                else os.path.join(image_dir, img_ref)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(
                    f'image 파일 없음: {abs_path} '
                    f'(sample 의 image="{img_ref}", image_dir="{image_dir}")'
                )
            img = Image.open(abs_path).convert('RGB')
            images.append(img)
            new_content.append({'type': 'image', 'image': img})
        new_messages.append({'role': role, 'content': new_content})
    return {'messages': new_messages}, images


# ──────────────────────────────────────────────
# 모델 + LoRA + collator
# ──────────────────────────────────────────────

def build_model_and_processor(args: TrainArgs):
    """FastVisionModel.from_pretrained + LoRA PEFT 부착."""
    from unsloth import FastVisionModel  # noqa: PLC0415

    model, processor = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit              = True,
        use_gradient_checkpointing = 'unsloth',
    )
    model = FastVisionModel.get_peft_model(
        model,
        r                          = args.rank,
        lora_alpha                 = args.alpha,
        lora_dropout               = 0,
        bias                       = 'none',
        finetune_vision_layers     = args.finetune_vision_layers,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        random_state               = args.seed,
        use_rslora                 = False,
    )
    FastVisionModel.for_training(model)
    return model, processor


def _find_subseq(haystack: List[int], needle: List[int], start: int = 0) -> int:
    """haystack 안에서 needle 토큰 시퀀스의 첫 인덱스 (없으면 -1)."""
    n, m = len(haystack), len(needle)
    if m == 0 or n - start < m:
        return -1
    for i in range(start, n - m + 1):
        ok = True
        for k in range(m):
            if haystack[i + k] != needle[k]:
                ok = False
                break
        if ok:
            return i
    return -1


def _probe_assistant_delimiters(processor) -> tuple:
    """chat template 을 실측해 assistant turn 의 (prefix_ids, suffix_ids) 추출.

    Gemma 4 E2B 처럼 모델별로 chat template 마커가 다르므로 (
    `<|turn>role\\n ... <|end|>` vs `<start_of_turn>role\\n ... <end_of_turn>`
    등) probe 대화 1개를 렌더해 user/assistant 마커 사이/직후의 토큰을
    실제로 잘라온다.

    반환:
      prefix_ids: assistant content 직전 토큰 시퀀스 (role tag + 개행 등 포함)
      suffix_ids: assistant content 직후 토큰 시퀀스 (turn 종료 토큰 등, 5개 cap)
                  loss masking 시 suffix 도 포함해 종료 토큰 생성 학습 보장.
    """
    USER_M = 'PROBEUSERMARKERXYZ'
    ASST_M = 'PROBEASSTMARKERXYZ'
    probe_plain = [
        {'role': 'user',      'content': USER_M},
        {'role': 'assistant', 'content': ASST_M},
    ]
    probe_typed = [
        {'role': 'user',      'content': [{'type': 'text', 'text': USER_M}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': ASST_M}]},
    ]
    text = None
    for probe in (probe_plain, probe_typed):
        try:
            text = processor.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=False)
            break
        except Exception:  # noqa: BLE001
            continue
    if text is None:
        raise RuntimeError('probe 대화 chat template 렌더 실패 (plain/typed 둘 다)')

    u_char = text.find(USER_M)
    a_char = text.find(ASST_M)
    if u_char < 0 or a_char < 0 or a_char <= u_char:
        raise RuntimeError(
            f'probe 마커 미발견 — chat template 렌더 결과:\n{text[:500]!r}')

    tok = processor.tokenizer
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(enc['input_ids'])
    offsets = list(enc['offset_mapping'])

    def first_tok_at_or_after(char_pos: int) -> int:
        for i, (s, _e) in enumerate(offsets):
            if s >= char_pos:
                return i
        return len(offsets)

    def first_tok_starting_after(char_pos: int) -> int:
        for i, (s, _e) in enumerate(offsets):
            if s >= char_pos:
                return i
        return len(offsets)

    u_end_char = u_char + len(USER_M)
    a_start_char = a_char
    a_end_char = a_char + len(ASST_M)

    prefix_start_tok = first_tok_at_or_after(u_end_char)
    prefix_end_tok   = first_tok_at_or_after(a_start_char)
    suffix_start_tok = first_tok_starting_after(a_end_char)

    prefix_ids = input_ids[prefix_start_tok:prefix_end_tok]
    suffix_ids = input_ids[suffix_start_tok:suffix_start_tok + 5]

    if not prefix_ids:
        raise RuntimeError(
            f'prefix 토큰 비어 있음 — chat template head:\n{text[:300]!r}')

    if _find_subseq(input_ids, prefix_ids) < 0:
        raise RuntimeError(
            f'prefix 자가 검증 실패. prefix_ids={prefix_ids}, '
            f'text head:\n{text[:300]!r}')
    print(f'[probe] prefix_ids = {prefix_ids}  ({tok.decode(prefix_ids)!r})')
    print(f'[probe] suffix_ids = {suffix_ids}  ({tok.decode(suffix_ids)!r})')
    return prefix_ids, suffix_ids


def _find_assistant_token_spans(input_ids: List[int],
                                prefix_ids: List[int],
                                suffix_ids: List[int]) -> List[tuple]:
    """input_ids 안에서 assistant turn 의 (start, end) 토큰 인덱스를 찾는다.

    탐색 방식: prefix_ids 시퀀스 위치를 찾아 그 직후를 content_start 로,
    이어 suffix_ids 시퀀스를 찾아 그 끝을 content_end (exclusive) 로 둔다.
    suffix 가 안 나오면 시퀀스 끝까지 사용.
    """
    spans: List[tuple] = []
    n = len(input_ids)
    pos = 0
    while True:
        p = _find_subseq(input_ids, prefix_ids, pos)
        if p < 0:
            break
        content_start = p + len(prefix_ids)
        if suffix_ids:
            s = _find_subseq(input_ids, suffix_ids, content_start)
            content_end = (s + len(suffix_ids)) if s >= 0 else n
        else:
            content_end = n
        spans.append((content_start, content_end))
        pos = content_end
    return spans


def make_data_collator(processor, image_dir: str):
    """function_call (text-only) + verify (image+text) 혼합 collator (batch_size=1 가정).

    각 sample 마다 image 유무 판단:
      - image 있음: processor(text=..., images=PIL_list, ...)
      - image 없음: processor(text=..., images=None, ...)

    Loss masking (2026-06-02 추가):
      Causal LM SFT 의 labels 를 assistant turn 영역만 유효(=input_ids),
      나머지는 -100 으로 설정. 1차 학습에서 모델이 [tool_result] 직후
      'SUCCESS' 같은 user 컨텐츠 자체를 모방하던 회귀를 차단.
      _probe_assistant_delimiters 가 실측 chat template 으로 assistant
      turn 의 prefix/suffix 토큰 시퀀스를 추출 (모델별 마커 차이 대응),
      _find_assistant_token_spans 가 input_ids 안에서 해당 시퀀스로 span 검출.
    """
    prefix_ids, suffix_ids = _probe_assistant_delimiters(processor)

    def collate(batch: List[dict]) -> dict:
        if len(batch) != 1:
            raise RuntimeError(
                f'본 collator 는 per_device_batch_size=1 만 지원 '
                f'(현재 batch={len(batch)}). '
                'effective batch 는 gradient_accumulation_steps 로 확보.'
            )
        sample = batch[0]
        resolved, images = _resolve_images_in_sample(sample, image_dir)
        text = processor.apply_chat_template(
            resolved['messages'],
            tokenize             = False,
            add_generation_prompt = False,
        )
        kwargs = {
            'text':          [text],
            'return_tensors': 'pt',
            'padding':       True,
        }
        if images:
            kwargs['images'] = images
        inputs = processor(**kwargs)

        import torch  # noqa: PLC0415
        ids_2d = inputs['input_ids']
        labels = torch.full_like(ids_2d, fill_value=-100)
        for b in range(ids_2d.shape[0]):
            id_list = ids_2d[b].tolist()
            spans = _find_assistant_token_spans(
                id_list, prefix_ids, suffix_ids)
            if not spans:
                raise RuntimeError(
                    'assistant turn span 미검출 — probe prefix_ids 가 '
                    'collate 시점 input_ids 에 없음. '
                    f'prefix_ids={prefix_ids}, text head=\n{text[:300]!r}'
                )
            for (s, e) in spans:
                labels[b, s:e] = ids_2d[b, s:e]
        inputs['labels'] = labels
        return inputs
    return collate


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────

def train(model, processor, samples: List[dict], args: TrainArgs) -> None:
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    try:
        from unsloth import is_bfloat16_supported  # noqa: PLC0415
        bf16 = bool(is_bfloat16_supported())
    except ImportError:
        bf16 = False

    cfg = SFTConfig(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        warmup_steps                = args.warmup_steps,
        num_train_epochs            = args.epochs if args.max_steps <= 0 else 1,
        max_steps                   = args.max_steps,
        learning_rate               = args.lr,
        bf16                        = bf16,
        fp16                        = not bf16,
        logging_steps               = args.logging_steps,
        save_steps                  = args.save_steps,
        output_dir                  = args.output_dir,
        report_to                   = 'none',
        remove_unused_columns       = False,
        dataset_text_field          = '',
        dataset_kwargs              = {'skip_prepare_dataset': True},
        max_length                  = args.max_seq_len,
        seed                        = args.seed,
        save_total_limit            = 3,
    )
    collator = make_data_collator(processor, args.image_dir)
    trainer = SFTTrainer(
        model            = model,
        train_dataset    = samples,
        processing_class = processor.tokenizer,
        data_collator    = collator,
        args             = cfg,
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(
        f'[train_gemma_qlora] saved LoRA adapter + processor → {args.output_dir}',
        file=sys.stderr,
    )


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(
        f'[train_gemma_qlora] start — base={args.base_model} '
        f'dataset={args.dataset} image_dir={args.image_dir} '
        f'output={args.output_dir}',
        file=sys.stderr,
    )
    samples = load_samples(args.dataset)
    n_text_only = sum(1 for s in samples if _count_image_items(s) == 0)
    n_vision    = len(samples) - n_text_only
    print(
        f'[train_gemma_qlora] loaded {len(samples)} samples '
        f'(text-only={n_text_only}, image+text={n_vision})',
        file=sys.stderr,
    )
    model, processor = build_model_and_processor(args)
    train(model, processor, samples, args)


if __name__ == '__main__':
    main()
