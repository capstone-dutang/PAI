"""
generate_verify_qa.py

Part C-1 / Cat-C — verify_result 함수의 시각 검증 정확도를 끌어올리기 위한
이미지 + 명령 → enum 결과 Q&A를 합성한다.

이미지는 Sim 렌더링으로 생성하고, ground truth 라벨은 Sim 내부 상태로부터
자동 부여한다 (예: 객체가 목표 통 안에 들어갔는지 충돌 검사).

출력 포맷 (Cat-C 예시 — README L635-643):
  {
    "messages": [
      {"role": "user",
       "content": "[image] 사용자 명령: '캔을 캔류 통에 넣어줘'. 작업이 완료되었나요?"},
      {"role": "assistant",
       "content": "{\\"status\\": \\"SUCCESS\\", \\"reasoning\\": \\"파란색 캔류 수거함 안에 알루미늄 캔이 들어 있음\\"}"}
    ]
  }

CLI 사용 예 (예정):
  python generate_verify_qa.py \\
    --domain      recycling \\
    --output      training/data/qa_recycling_catC.jsonl \\
    --num-scenes  500 \\
    --so101-usd   path/to/so101.usd \\
    --object-usd  path/to/objects/

처리 흐름:
  1) Sim에서 SUCCESS / FAIL / PARTIAL / UNCLEAR 시나리오를 의도적으로 구성
       SUCCESS  : 객체가 목표 통 내부 + 안정 상태
       FAIL     : 객체가 통 밖 / 떨어진 위치 / 잘못된 통
       PARTIAL  : 일부만 들어감 / 가장자리 걸침
       UNCLEAR  : 가림 / 시야 가장자리 / 모호한 상태
  2) 각 시나리오를 RGB로 렌더링
  3) Ground truth 라벨러가 status + reasoning 자동 생성
       status는 결정론적, reasoning 텍스트는 외부 LLM으로 생성하거나
       템플릿 채움
  4) JSONL로 저장 (이미지는 dataset 디렉터리에 별도 저장 + path를 messages에 임베드)

미구현 항목 (구현 시 결정 필요):
  - Sim 시나리오 합성 정책 (각 enum별 비율 / 다양성)
  - reasoning 텍스트 생성: 템플릿 vs 외부 LLM
  - 이미지 저장 형식: messages에 file path / base64 / dataset card 분리
  - LoRA 학습기가 multi-modal 입력을 받는 방식 (이미지를 base64로 messages에
    임베드하는 방식이 일반적이지만, Unsloth 측 지원 형식 확인 필요)
"""

from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Cat-C — 시각 검증 시나리오 Q&A 합성')
    p.add_argument('--domain',     required=True, help='대상 도메인')
    p.add_argument('--output',     required=True, help='JSONL 출력 경로')
    p.add_argument('--num-scenes', type=int, default=500, help='생성 시나리오 수')
    p.add_argument('--so101-usd',  required=True, help='SO-101 USD 경로')
    p.add_argument('--object-usd', required=True, help='도메인 객체 USD 디렉터리')
    return p.parse_args()


def synthesize_scenario(domain: str, target_status: str):
    """status enum에 부합하는 Sim 시나리오 한 건 구성.

    TODO: Sim 환경 + 객체 배치 정책 확정 후 구현.
    """
    raise NotImplementedError("[generate_verify_qa] synthesize_scenario() 미구현")


def render_rgb(scenario) -> bytes:
    """시나리오를 RGB 이미지로 렌더링.

    TODO: Isaac Sim 카메라 캡처 — generate_act_dataset.py와 동일 카메라 설정 사용.
    """
    raise NotImplementedError("[generate_verify_qa] render_rgb() 미구현")


def label_ground_truth(scenario) -> dict:
    """시나리오의 ground truth status + reasoning 산출.

    Returns:
      {"status": "SUCCESS" | "FAIL" | "PARTIAL" | "UNCLEAR", "reasoning": "..."}

    TODO: status는 결정론적(객체-통 충돌 검사), reasoning 생성 방식 결정 필요.
    """
    raise NotImplementedError("[generate_verify_qa] label_ground_truth() 미구현")


def main():
    args = parse_args()
    raise NotImplementedError(
        "[generate_verify_qa] main 합성 루프 미구현. "
        "synthesize_scenario / render_rgb / label_ground_truth 구현 후 작성."
    )


if __name__ == '__main__':
    main()
