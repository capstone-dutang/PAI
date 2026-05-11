"""
generate_function_call_qa.py

Part C-1 / Cat-B — agent_node가 사용하는 3개 함수
(look_scene / execute_subtask / verify_result)에 대한 함수 호출 시나리오 Q&A를
외부 LLM으로 합성한다. Gemma 도메인 LoRA가 배포 시 함수 호출 응답을
정확히 산출하도록 학습시키는 데이터.

출력 포맷 (Cat-B 예시 — README L622-632):
  {
    "messages": [
      {"role": "system",
       "content": "사용 가능 함수: look_scene(prompt), execute_subtask(target_description), verify_result(original_command)"},
      {"role": "user", "content": "쓰레기 좀 정리해줘"},
      {"role": "assistant",
       "content": "{\\"function\\": \\"look_scene\\", \\"args\\": {\\"prompt\\": \\"분리수거할 물체를 찾아주세요\\"}}"}
    ]
  }

CLI 사용 예 (예정):
  python generate_function_call_qa.py \\
    --domain         recycling \\
    --output         training/data/qa_recycling_catB.jsonl \\
    --external-llm   gpt-4 \\
    --num-samples    500

미구현 항목 (구현 시 결정 필요):
  - 함수 시그니처 — agent_node 구현 시점에 확정 (현재는 README 예시 그대로)
    실제 prompt 시스템 메시지에 들어갈 시그니처 텍스트는 agent_node가 런타임에
    Gemma에 보내는 것과 정확히 일치해야 한다 (LoRA가 학습한 분포 그대로 받기 위함).
  - 외부 LLM 클라이언트
  - 사용자 명령 다양성 확보 — 도메인별 명령 풀 + paraphrase
  - 단일 함수 호출 / 다단계 시나리오 비율
"""

from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Cat-B — 함수 호출 시나리오 Q&A 합성')
    p.add_argument('--domain',       required=True, help='대상 도메인 (recycling/library/...)')
    p.add_argument('--output',       required=True, help='JSONL 출력 경로')
    p.add_argument('--external-llm', default='gpt-4', help='Q&A 합성용 외부 LLM')
    p.add_argument('--num-samples',  type=int, default=500, help='생성할 샘플 수')
    return p.parse_args()


def build_system_prompt() -> str:
    """
    Gemma에 들어가는 시스템 프롬프트 (사용 가능 함수 시그니처 선언).

    TODO: agent_node가 런타임에 보내는 시스템 프롬프트와 정확히 일치시켜야 한다.
          agent_node._call_look_scene/execute_subtask/verify_result 구현 시점에
          확정.
    """
    raise NotImplementedError(
        "[generate_function_call_qa] build_system_prompt() 미구현 "
        "(agent_node 함수 시그니처 확정 필요)"
    )


def synthesize_user_command(domain: str, llm: str) -> str:
    """도메인 맥락에 맞는 사용자 명령 1건 합성.

    TODO: prompt 풀 / paraphrase 정책 확정 후 구현.
    """
    raise NotImplementedError("[generate_function_call_qa] synthesize_user_command() 미구현")


def synthesize_function_call_response(user_command: str, llm: str) -> str:
    """사용자 명령 → 적절한 함수 호출 JSON 문자열.

    TODO: 단계별(look_scene → execute_subtask → verify_result) 정책 확정 후 구현.
    """
    raise NotImplementedError(
        "[generate_function_call_qa] synthesize_function_call_response() 미구현"
    )


def main():
    args = parse_args()
    raise NotImplementedError(
        "[generate_function_call_qa] main 합성 루프 미구현. "
        "build_system_prompt / synthesize_* 구현 후 단순 루프 + JSONL 저장."
    )


if __name__ == '__main__':
    main()
