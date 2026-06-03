"""
smoke_test_lora.py

학습된 도메인 LoRA adapter 가 freeze 매트릭스 분포대로 응답하는지 확인하는
sanity check 스크립트. GGUF 변환 / 배포 전에 외부 PC 에서 1회 실행.

검증 항목:
  function_call 샘플 (text-only):
    - 응답이 단일 JSON object 로 파싱되는가
    - {"function": str, "args": dict} 키가 있는가
    - function ∈ {look_scene, execute_subtask, verify_result}
  verify 샘플 (image+text):
    - 응답이 단일 JSON object 로 파싱되는가
    - {"status": str, "reasoning": str} 키가 있는가
    - status ∈ {SUCCESS, FAIL, PARTIAL, UNCLEAR}

샘플 출처:
  --dataset 의 merged.jsonl 에서 image 유무로 두 그룹 분리, 각 그룹에서
  random.sample(N) 픽. 학습 데이터와 겹칠 수 있으나 본 스크립트는 형식
  정합성 확인이 목적이므로 held-out 분리 없이 진행한다 (정량 평가는 별도).

CLI 예:
  python training/lora_finetuning/scripts/smoke_test_lora.py \\
    --base-model   unsloth/gemma-4-E2B-it \\
    --adapter-dir  training/lora_finetuning/outputs/lora_recycling \\
    --dataset      training/lora_finetuning/data/recycling/recycling_merged.jsonl \\
    --image-dir    training/lora_finetuning/data/recycling/verify_images \\
    --n-function-call 8 \\
    --n-verify        8

선행 조건:
  외부 학습 PC (Ubuntu + CUDA, VRAM ≥ 16 GB), train_gemma_qlora.py 와
  동일한 unsloth / transformers 환경.

출력:
  stdout 에 sample 별 (expected vs actual) + 종합 PASS/FAIL 카운트.
  종료 코드: 모든 sample 형식 검증 통과 = 0, 하나라도 실패 = 1.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


FUNCTION_NAMES = {'look_scene', 'execute_subtask', 'verify_result'}
VERIFY_STATUSES = {'SUCCESS', 'FAIL', 'PARTIAL', 'UNCLEAR'}


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@dataclass
class Args:
    base_model:       str
    adapter_dir:      str
    dataset:          str
    image_dir:        str
    n_function_call:  int
    n_verify:         int
    max_new_tokens:   int
    seed:             int


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description='Gemma LoRA adapter smoke test (format sanity)')
    p.add_argument('--base-model',  default='unsloth/gemma-4-E2B-it')
    p.add_argument('--adapter-dir', required=True,
                   help='train_gemma_qlora.py 가 저장한 LoRA 디렉터리')
    p.add_argument('--dataset',     required=True,
                   help='merged.jsonl (image 유무로 두 그룹 분리)')
    p.add_argument('--image-dir',   required=True,
                   help='verify sample image path base')
    p.add_argument('--n-function-call', type=int, default=8)
    p.add_argument('--n-verify',        type=int, default=8)
    p.add_argument('--max-new-tokens',  type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    return Args(
        base_model      = a.base_model,
        adapter_dir     = os.path.expanduser(a.adapter_dir),
        dataset         = os.path.expanduser(a.dataset),
        image_dir       = os.path.expanduser(a.image_dir),
        n_function_call = a.n_function_call,
        n_verify        = a.n_verify,
        max_new_tokens  = a.max_new_tokens,
        seed            = a.seed,
    )


# ──────────────────────────────────────────────
# 데이터 로드 & 분리
# ──────────────────────────────────────────────

def _has_image(sample: dict) -> bool:
    for m in sample.get('messages', []):
        content = m.get('content', [])
        if not isinstance(content, list):
            continue
        for it in content:
            if isinstance(it, dict) and it.get('type') == 'image':
                return True
    return False


def load_split(jsonl_path: str) -> Tuple[List[dict], List[dict]]:
    """JSONL → (function_call_samples, verify_samples)."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f'dataset 파일 없음: {jsonl_path}')
    fc: List[dict] = []
    vf: List[dict] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            (vf if _has_image(obj) else fc).append(obj)
    return fc, vf


def _strip_last_assistant(sample: dict) -> Tuple[dict, str]:
    """마지막 assistant 메시지를 떼어 (prompt_sample, expected_text) 반환."""
    msgs = sample['messages']
    if not msgs or msgs[-1].get('role') != 'assistant':
        raise ValueError(f'마지막 메시지가 assistant 가 아님: {msgs[-1] if msgs else None}')
    last = msgs[-1]
    content = last.get('content', [])
    if isinstance(content, list):
        expected = ''.join(
            it.get('text', '') for it in content
            if isinstance(it, dict) and it.get('type') == 'text'
        )
    else:
        expected = str(content)
    prompt_sample = {'messages': msgs[:-1]}
    return prompt_sample, expected


# ──────────────────────────────────────────────
# 모델 로드 (base + LoRA)
# ──────────────────────────────────────────────

def load_model_with_adapter(base_model: str, adapter_dir: str):
    """FastVisionModel base → PeftModel.load_adapter."""
    from unsloth import FastVisionModel  # noqa: PLC0415

    model, processor = FastVisionModel.from_pretrained(
        base_model,
        load_in_4bit              = True,
        use_gradient_checkpointing = False,
    )
    # PEFT adapter 부착
    from peft import PeftModel  # noqa: PLC0415
    model = PeftModel.from_pretrained(model, adapter_dir)
    FastVisionModel.for_inference(model)
    return model, processor


# ──────────────────────────────────────────────
# 추론
# ──────────────────────────────────────────────

def _resolve_images(sample: dict, image_dir: str):
    from PIL import Image  # noqa: PLC0415
    images = []
    new_msgs = []
    for msg in sample['messages']:
        content = msg.get('content', [])
        if not isinstance(content, list):
            new_msgs.append(msg)
            continue
        new_content = []
        for it in content:
            if isinstance(it, dict) and it.get('type') == 'image':
                ref = it.get('image', '')
                path = ref if os.path.isabs(ref) else os.path.join(image_dir, ref)
                img = Image.open(path).convert('RGB')
                images.append(img)
                new_content.append({'type': 'image', 'image': img})
            else:
                new_content.append(it)
        new_msgs.append({'role': msg['role'], 'content': new_content})
    return {'messages': new_msgs}, images


def generate(model, processor, sample: dict, image_dir: str,
             max_new_tokens: int) -> str:
    resolved, images = _resolve_images(sample, image_dir)
    text = processor.apply_chat_template(
        resolved['messages'],
        tokenize             = False,
        add_generation_prompt = True,
    )
    kwargs = {'text': [text], 'return_tensors': 'pt', 'padding': True}
    if images:
        kwargs['images'] = images
    inputs = processor(**kwargs).to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        do_sample      = False,
        temperature    = 1.0,
    )
    # prompt 부분 제거
    prompt_len = inputs['input_ids'].shape[1]
    gen_ids = out_ids[0, prompt_len:]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────
# 검증
# ──────────────────────────────────────────────

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    """응답 텍스트에서 첫 JSON object 추출 (코드 블록 / 주변 텍스트 허용)."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def check_function_call(text: str) -> Tuple[bool, str]:
    obj = _extract_json(text)
    if obj is None:
        return False, 'JSON 파싱 실패'
    if 'function' not in obj or 'args' not in obj:
        return False, f"키 누락 (got keys={list(obj.keys())})"
    if not isinstance(obj['args'], dict):
        return False, f"'args' 가 dict 아님 (type={type(obj['args']).__name__})"
    fn = obj['function']
    if fn not in FUNCTION_NAMES:
        return False, f"function='{fn}' ∉ {sorted(FUNCTION_NAMES)}"
    return True, f'OK (function={fn})'


def check_verify(text: str) -> Tuple[bool, str]:
    obj = _extract_json(text)
    if obj is None:
        return False, 'JSON 파싱 실패'
    if 'status' not in obj or 'reasoning' not in obj:
        return False, f"키 누락 (got keys={list(obj.keys())})"
    st = obj['status']
    if st not in VERIFY_STATUSES:
        return False, f"status='{st}' ∉ {sorted(VERIFY_STATUSES)}"
    if not isinstance(obj['reasoning'], str) or not obj['reasoning'].strip():
        return False, 'reasoning 비어 있음'
    return True, f'OK (status={st})'


# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────

def _run_group(name: str, samples: List[dict], n: int, model, processor,
               image_dir: str, max_new_tokens: int,
               checker) -> Tuple[int, int]:
    if not samples:
        print(f'\n=== {name} ===\n  (해당 카테고리 sample 없음 — skip)')
        return 0, 0
    n = min(n, len(samples))
    picks = random.sample(samples, n)
    print(f'\n=== {name} (N={n}) ===')
    n_pass = 0
    for i, smp in enumerate(picks, 1):
        prompt_sample, expected = _strip_last_assistant(smp)
        try:
            actual = generate(model, processor, prompt_sample, image_dir,
                              max_new_tokens)
        except Exception as e:  # noqa: BLE001
            print(f'  [{i}/{n}] 추론 실패: {e}')
            continue
        ok, detail = checker(actual)
        flag = 'PASS' if ok else 'FAIL'
        if ok:
            n_pass += 1
        print(f'  [{i}/{n}] {flag}  {detail}')
        print(f'        expected: {expected[:160]}')
        print(f'        actual  : {actual[:160]}')
    print(f'  → {name}: {n_pass}/{n} 통과')
    return n_pass, n


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    print(f'[smoke] base_model = {args.base_model}')
    print(f'[smoke] adapter    = {args.adapter_dir}')
    print(f'[smoke] dataset    = {args.dataset}')

    if not os.path.isdir(args.adapter_dir):
        print(f'[smoke] ERROR: adapter-dir 없음: {args.adapter_dir}',
              file=sys.stderr)
        return 1
    expected_files = ['adapter_config.json', 'adapter_model.safetensors']
    missing = [f for f in expected_files
               if not os.path.exists(os.path.join(args.adapter_dir, f))]
    if missing:
        print(f'[smoke] ERROR: adapter-dir 필수 파일 누락: {missing}',
              file=sys.stderr)
        return 1

    fc_samples, vf_samples = load_split(args.dataset)
    print(f'[smoke] function_call sample = {len(fc_samples)}, '
          f'verify sample = {len(vf_samples)}')

    model, processor = load_model_with_adapter(
        args.base_model, args.adapter_dir)

    fc_pass, fc_n = _run_group(
        'function_call', fc_samples, args.n_function_call,
        model, processor, args.image_dir, args.max_new_tokens,
        check_function_call)
    vf_pass, vf_n = _run_group(
        'verify', vf_samples, args.n_verify,
        model, processor, args.image_dir, args.max_new_tokens,
        check_verify)

    total_pass = fc_pass + vf_pass
    total_n    = fc_n + vf_n
    print(f'\n=== 종합 ===')
    print(f'  function_call : {fc_pass}/{fc_n}')
    print(f'  verify        : {vf_pass}/{vf_n}')
    print(f'  total         : {total_pass}/{total_n}')

    return 0 if (total_pass == total_n and total_n > 0) else 1


if __name__ == '__main__':
    sys.exit(main())
