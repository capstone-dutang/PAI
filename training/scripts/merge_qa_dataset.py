"""
merge_qa_dataset.py

Q&A 데이터 생성 마지막 단계 — Cat-B(함수 호출)와 Cat-C(검증) 두 JSONL 을
고정 비율 70 : 30 으로 병합하여 단일 학습용 JSONL 을 만든다.

freeze 결정 (2026-05-25):
  도메인이 분리수거(3종)와 식기 정리로 좁아 Cat-A(도메인 지식 QA)는
  파인튜닝 효용이 거의 없어 제거. LoRA 의 실제 작업 표면인 함수 호출(Cat-B)과
  시각 검증(Cat-C)만 학습한다. 비율은 70 : 30 단일 옵션으로 freeze.

CLI 사용 예:
  python merge_qa_dataset.py \\
    --cat-b   training/data/qa_recycling_catB.jsonl \\
    --cat-c   training/data/qa_recycling_catC.jsonl \\
    --output  training/data/qa_recycling_merged.jsonl

비율은 출력 데이터셋 내 카테고리별 행 수의 목표 비율이다. 입력 풀이 부족하면
부족분 처리 정책 결정이 필요 (TODO 참조).

미구현 / 결정 필요 항목:
  - 부족분 처리: oversample (중복 허용) / cap (목표 비율 깨짐)
  - 셔플 시드 정책
  - 동일 도메인 vs 도메인 혼합 학습 — 본 스크립트는 단일 도메인 전제
"""

from __future__ import annotations

import argparse


# freeze 비율 — 변경 시 LoRA 재학습 필요
RATIO_B = 0.70
RATIO_C = 0.30


def parse_args():
    p = argparse.ArgumentParser(description='Cat-B / Cat-C JSONL 70:30 병합')
    p.add_argument('--cat-b',  required=True, help='Cat-B 함수 호출 JSONL')
    p.add_argument('--cat-c',  required=True, help='Cat-C 검증 시나리오 JSONL')
    p.add_argument('--output', required=True, help='병합 JSONL 출력 경로')
    p.add_argument('--seed',   type=int, default=0, help='셔플 시드')
    return p.parse_args()


def load_jsonl(path: str) -> list:
    """JSONL 을 메모리로 로드.

    TODO: 파일 단순 줄 단위 json.loads — 구현 단순. 큰 파일 처리 시 lazy 도입 검토.
    """
    raise NotImplementedError("[merge_qa_dataset] load_jsonl() 미구현")


def merge_with_ratio(b: list, c: list, seed: int) -> list:
    """RATIO_B : RATIO_C 비율로 b/c 를 샘플링하여 단일 리스트로 결합.

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
    b = load_jsonl(args.cat_b)
    c = load_jsonl(args.cat_c)
    merged = merge_with_ratio(b, c, args.seed)
    save_jsonl(merged, args.output)


if __name__ == '__main__':
    main()
