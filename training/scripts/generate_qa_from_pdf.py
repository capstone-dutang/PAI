"""
generate_qa_from_pdf.py

Part C-1 / Cat-A — 도메인 PDF 문서에서 자연어 Q&A 쌍을 자동 생성한다.
Gemma 도메인 LoRA 학습 데이터의 첫 번째 카테고리(도메인 지식)를 만든다.

흐름:
  PDF → 텍스트 추출 → 청크 분할 → 외부 LLM에 Q&A 생성 요청 → 필터링 → JSONL

출력 포맷 (Cat-A 예시 — README L611-619):
  {
    "messages": [
      {"role": "user",      "content": "유리병은 어떻게 분리수거하나요?"},
      {"role": "assistant", "content": "유리병은 내용물을 비우고 ..."}
    ]
  }

CLI 사용 예 (예정):
  python generate_qa_from_pdf.py \\
    --pdf-dir         data/recycling_pdfs/ \\
    --output          training/data/qa_recycling_catA.jsonl \\
    --external-llm    gpt-4 \\
    --num-qa-per-chunk 5

미구현 항목 (구현 시 결정 필요):
  - 외부 LLM 클라이언트 (OpenAI / Anthropic / 로컬)
  - PDF 추출 라이브러리 (pdfminer.six / pypdf / unstructured)
  - 청크 분할 정책 (토큰 수 / 의미 단위)
  - 필터링 정책 (할루시네이션 제거, 중복 제거, 길이 컷오프)
"""

from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Cat-A — PDF에서 도메인 지식 Q&A 생성')
    p.add_argument('--pdf-dir',          required=True, help='도메인 PDF 디렉터리')
    p.add_argument('--output',           required=True, help='JSONL 출력 경로')
    p.add_argument('--external-llm',     default='gpt-4', help='Q&A 생성용 외부 LLM')
    p.add_argument('--num-qa-per-chunk', type=int, default=5, help='청크당 Q&A 개수')
    return p.parse_args()


def extract_text_from_pdfs(pdf_dir: str) -> list:
    """PDF → 텍스트 chunk 리스트.

    TODO: 라이브러리 결정 후 구현.
    """
    raise NotImplementedError("[generate_qa_from_pdf] extract_text_from_pdfs() 미구현")


def generate_qa(chunk: str, llm: str, n: int) -> list:
    """청크 1개 → Q&A n개. 외부 LLM 호출.

    TODO: LLM 클라이언트 / prompt 템플릿 확정 후 구현.
    """
    raise NotImplementedError("[generate_qa_from_pdf] generate_qa() 미구현")


def filter_qa(qa_list: list) -> list:
    """할루시네이션·중복 필터링.

    TODO: 정책 확정 후 구현.
    """
    raise NotImplementedError("[generate_qa_from_pdf] filter_qa() 미구현")


def main():
    args = parse_args()
    chunks = extract_text_from_pdfs(args.pdf_dir)
    all_qa = []
    for c in chunks:
        all_qa.extend(generate_qa(c, args.external_llm, args.num_qa_per_chunk))
    all_qa = filter_qa(all_qa)
    # JSONL 저장은 출력 포맷이 messages 형식으로 고정 — 구현 시 단순 jsonl.write
    raise NotImplementedError(
        "[generate_qa_from_pdf] 저장 단계 미구현 — extract/generate/filter 완료 후 작성"
    )


if __name__ == '__main__':
    main()
