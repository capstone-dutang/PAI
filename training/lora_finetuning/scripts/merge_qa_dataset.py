"""
merge_qa_dataset.py

Q&A 데이터 생성 마지막 단계 — function_call(함수 호출)와 verify(검증) 두 JSONL 을
고정 비율 70 : 30 으로 병합하여 단일 학습용 JSONL 을 만든다.

freeze 결정 (2026-05-25):
  도메인이 분리수거(3종)와 식기 정리로 좁아 domain_knowledge (도메인 지식 QA)는
  파인튜닝 효용이 거의 없어 제거. LoRA 의 실제 작업 표면인 함수 호출(function_call)과
  시각 검증(verify)만 학습한다. 비율은 70 : 30 단일 옵션으로 freeze.

부족분 처리 정책 (oversample):
  비율 대비 더 큰 쪽을 전부 사용하고, 더 작은 쪽을 중복 허용으로 oversample
  하여 70:30 을 맞춘다. 예: function_call=500, verify=100 → function_call 전부 사용,
  verify 를 약 214 개로 oversample (≈2.14x 중복). 데이터 손실 없이 비율 보장.

CLI 사용 예:
  python training/lora_finetuning/scripts/merge_qa_dataset.py \\
    --function_call   training/lora_finetuning/data/recycling/recycling_function_call_qa.jsonl \\
    --verify          training/lora_finetuning/data/recycling/recycling_verify_qa.jsonl \\
    --output          training/lora_finetuning/data/recycling/recycling_qa_merged.jsonl

엣지 케이스:
  - function_call 또는 verify 한 쪽이 비어 있으면 다른 쪽만 셔플해서 반환 + 경고.
    1차 학습(verify 미준비) 시 verify 자리에 빈 파일 전달 가능.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List


# freeze 비율 — 변경 시 LoRA 재학습 필요
RATIO_B = 0.70
RATIO_C = 0.30


def parse_args():
    p = argparse.ArgumentParser(description='function_call / verify JSONL 70:30 병합')
    p.add_argument('--function_call',  required=True, help='function_call 함수 호출 JSONL')
    p.add_argument('--verify',  required=True, help='verify 검증 시나리오 JSONL')
    p.add_argument('--output', required=True, help='병합 JSONL 출력 경로')
    p.add_argument('--seed',   type=int, default=0, help='셔플 시드')
    return p.parse_args()


def load_jsonl(path: str) -> list:
    """JSONL 을 메모리로 로드. 파일이 없거나 비어 있으면 [] 반환."""
    if not os.path.exists(path):
        print(f'[merge_qa_dataset] 경고 — {path} 가 존재하지 않음. 빈 리스트로 처리.',
              file=sys.stderr)
        return []
    items: list = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f'{path}:{lineno} JSON 파싱 실패: {e}'
                ) from e
    return items


def _resample(pool: list, target: int, rng: random.Random) -> list:
    """target 개수만큼 pool 에서 추출.

    target <= len(pool): rng.sample (중복 없이 부분 선택)
    target >  len(pool): 전체 셔플 사본을 반복 + 잔여분만 sample (oversample)
    """
    if not pool or target <= 0:
        return []
    n = len(pool)
    if target <= n:
        return rng.sample(pool, target)
    full_cycles = target // n
    remainder   = target - full_cycles * n
    out: list = []
    for _ in range(full_cycles):
        idxs = list(range(n))
        rng.shuffle(idxs)
        out.extend(pool[i] for i in idxs)
    if remainder:
        out.extend(rng.sample(pool, remainder))
    return out


def merge_with_ratio(b: list, c: list, seed: int) -> list:
    """RATIO_B : RATIO_C 비율로 b/c 를 oversample 병합 후 전체 셔플.

    비율 대비 더 큰 쪽을 전부 사용, 작은 쪽을 oversample 한다.
    한 쪽이 비어 있으면 다른 쪽만 셔플해서 반환 + 경고.
    """
    rng = random.Random(seed)
    if not b and not c:
        return []
    if not c:
        print('[merge_qa_dataset] 경고 — verify 가 비어 있음. '
              'function_call 만 사용 (70:30 비율 깨짐, 1차 학습 등 일시 허용).',
              file=sys.stderr)
        out = list(b)
        rng.shuffle(out)
        return out
    if not b:
        print('[merge_qa_dataset] 경고 — function_call 가 비어 있음. '
              'verify 만 사용 (70:30 비율 깨짐).',
              file=sys.stderr)
        out = list(c)
        rng.shuffle(out)
        return out

    # 비율 대비 어느 쪽이 더 큰지 판정:
    #   len(b)/RATIO_B 와 len(c)/RATIO_C 중 큰 쪽이 비율 기준 풍부한 측.
    #   비교를 정수 산술로 안전화: len(b)*RATIO_C  vs  len(c)*RATIO_B.
    if len(b) * RATIO_C >= len(c) * RATIO_B:
        # B 가 비율 대비 풍부 → B 전부 사용, C 를 oversample
        n_b = len(b)
        n_c = max(1, round(n_b * RATIO_C / RATIO_B))
    else:
        # C 가 비율 대비 풍부 → C 전부 사용, B 를 oversample
        n_c = len(c)
        n_b = max(1, round(n_c * RATIO_B / RATIO_C))

    selected_b = _resample(b, n_b, rng)
    selected_c = _resample(c, n_c, rng)
    merged = selected_b + selected_c
    rng.shuffle(merged)

    print(
        f'[merge_qa_dataset] mix — '
        f'input(B={len(b)}, C={len(c)}) → '
        f'output(B={n_b}, C={n_c}, total={len(merged)}) '
        f'(target ratio B:C = {RATIO_B}:{RATIO_C})',
        file=sys.stderr,
    )
    return merged


def save_jsonl(items: list, path: str) -> None:
    """JSONL 저장. 출력 디렉터리는 자동 생성."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    b = load_jsonl(os.path.expanduser(args.function_call))
    c = load_jsonl(os.path.expanduser(args.verify))
    merged = merge_with_ratio(b, c, args.seed)
    save_jsonl(merged, os.path.expanduser(args.output))
    print(
        f'[merge_qa_dataset] done — wrote {len(merged)} lines → {args.output}',
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
