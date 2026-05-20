"""
merge_qa_dataset.py

Q&A 데이터 생성 마지막 단계 — Cat-A/B/C로 만든 3개 JSONL을 비율 옵션에 따라
하나의 학습용 JSONL로 병합한다.

비율 옵션 (README L603-609):
  | 옵션          | Cat-A | Cat-B | Cat-C | 특징                       |
  |---------------|-------|-------|-------|----------------------------|
  | balanced      | 50%   | 30%   | 20%   | 표준, 도메인 지식 우선     |
  | function_heavy| 40%   | 40%   | 20%   | 함수 호출 정확도 우선      |
  | verify_heavy  | 40%   | 30%   | 30%   | grasp/verify 안정성 우선   |

비율은 출력 데이터셋 내 카테고리별 행 수의 목표 비율이다. 입력 풀이
부족하면 부족분을 oversampling하거나 cap하는 정책 결정이 필요.

CLI 사용 예 (예정):
  python merge_qa_dataset.py \\
    --cat-a   training/data/qa_recycling_catA.jsonl \\
    --cat-b   training/data/qa_recycling_catB.jsonl \\
    --cat-c   training/data/qa_recycling_catC.jsonl \\
    --ratio   balanced \\
    --output  training/data/qa_recycling_merged.jsonl

미구현 / 결정 필요 항목:
  - 어떤 옵션을 default로 둘지 (README #2 미확정)
  - 부족분 처리: oversample (중복 허용) / cap (목표 비율 깨짐)
  - 셔플 시드
  - 동일 도메인 vs 도메인 혼합 학습 — 본 스크립트는 단일 도메인 전제
"""

from __future__ import annotations

import argparse


RATIO_TABLE = {
    'balanced':       {'a': 0.50, 'b': 0.30, 'c': 0.20},
    'function_heavy': {'a': 0.40, 'b': 0.40, 'c': 0.20},
    'verify_heavy':   {'a': 0.40, 'b': 0.30, 'c': 0.30},
}


def parse_args():
    p = argparse.ArgumentParser(description='Cat-A/B/C JSONL 비율 혼합')
    p.add_argument('--cat-a', required=True, help='Cat-A 도메인 지식 JSONL')
    p.add_argument('--cat-b', required=True, help='Cat-B 함수 호출 JSONL')
    p.add_argument('--cat-c', required=True, help='Cat-C 검증 시나리오 JSONL')
    p.add_argument('--ratio', choices=list(RATIO_TABLE.keys()), required=True,
                   help='비율 옵션 (README 미확정 #2)')
    p.add_argument('--output', required=True, help='병합 JSONL 출력 경로')
    p.add_argument('--seed',   type=int, default=0, help='셔플 시드')
    return p.parse_args()


def load_jsonl(path: str) -> list:
    """JSONL을 메모리로 로드.

    TODO: 파일 단순 줄 단위 json.loads — 구현 단순. 큰 파일 처리 시 lazy 도입 검토.
    """
    raise NotImplementedError("[merge_qa_dataset] load_jsonl() 미구현")


def merge_with_ratio(a: list, b: list, c: list, ratio: dict, seed: int) -> list:
    """비율에 맞춰 a/b/c를 샘플링하여 단일 리스트로 결합.

    TODO: 부족분 처리 정책(oversample/cap) 확정 후 구현.
    """
    raise NotImplementedError("[merge_qa_dataset] merge_with_ratio() 미구현")


def save_jsonl(items: list, path: str) -> None:
    """JSONL 저장.

    TODO: open + json.dumps + ensure_ascii=False 단순 구현.
    """
    raise NotImplementedError("[merge_qa_dataset] save_jsonl() 미구현")


def main():
    args = parse_args()
    a = load_jsonl(args.cat_a)
    b = load_jsonl(args.cat_b)
    c = load_jsonl(args.cat_c)
    merged = merge_with_ratio(a, b, c, RATIO_TABLE[args.ratio], args.seed)
    save_jsonl(merged, args.output)


if __name__ == '__main__':
    main()
