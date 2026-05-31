"""
prompts.py

Gemma agent 의 시스템 프롬프트와 상태 enum 의 단일 출처(SSoT).

배포 측 agent_node 와 학습 측 generate_function_call_qa(Cat-B 합성)·
generate_verify_qa(Cat-C) 가 모두 본 모듈을 import 한다. 본 모듈을 수정하면
LoRA 학습 데이터와 배포 함수 호출 분포가 동시에 영향을 받으므로 변경 시
LoRA 재학습이 필요하다.

import 경로:
  배포 (ROS2 패키지 내부 — colcon build 후 자동으로 sys.path 에 들어감):
      from robot_arm_pkg.prompts import SYSTEM_PROMPT, VerifyStatus, ExecStatus

  학습 (저장소 루트의 training/lora_finetuning 에서 실행):
      import os, sys
      sys.path.insert(0, os.path.abspath(os.path.join(
          os.path.dirname(__file__),
          '..', '..', 'deploy', 'ros2_ws', 'src', 'robot_arm_pkg')))
      from robot_arm_pkg.prompts import SYSTEM_PROMPT, VerifyStatus, ExecStatus

freeze 결정 (2026-05-25):
  - SSoT 위치       : robot_arm_pkg 내부 (ROS2 패키지가 소유)
  - 호출 형식       : 자연어 system prompt + JSON 응답 (OpenAI tool calling 미사용)
  - verify_result   : stateless — original_command 인자로 받음
  - enum 상수       : 본 모듈에 합쳐 관리. ExecStatus / VerifyStatus 모두 대문자
                       (ROS ExecutionStatus.status 필드도 대문자로 통일)
"""

from __future__ import annotations


# ──────────────────────────────────────────────
# 시스템 프롬프트
# ──────────────────────────────────────────────
#
# Gemma agent 는 다음 세 함수만 호출하며, 응답은 JSON 한 객체로만 제한된다.
# Cat-B 학습 데이터의 assistant 응답 분포가 본 프롬프트와 정확히 일치해야 한다.
# target_description 의 vocabulary 는 deploy/config/label_vocab.yaml 참조.

SYSTEM_PROMPT: str = (
    '사용 가능 함수:\n'
    '  look_scene(prompt: str) -> str\n'
    '      RGB 장면을 보고 잡을 객체의 자연어 묘사를 반환한다.\n'
    '      반환 형식 (엄격): "the {attribute} {object} {spatial}"\n'
    '      - 영어 고정, 5~15 단어\n'
    '      - attribute: 색/재질/크기 중 변별 가능한 것 1개\n'
    '      - object:    can / plastic / paper / plate / bowl / cup / fork / spoon 등 단일 명사\n'
    '      - spatial:   위치/관계 표현 (예: "on the right side of the table")\n'
    '      예: "the blue can on the right side of the table"\n'
    '\n'
    '  execute_subtask(target_description: str) -> str\n'
    '      look_scene 의 반환 형식 그대로 받아 ACT 에 전달.\n'
    '      SUCCESS / FAIL / TIMEOUT 중 하나 반환.\n'
    '\n'
    '  verify_result(original_command: str) -> str\n'
    '      직전 subtask 완료 여부 시각 검증.\n'
    '      SUCCESS / FAIL / PARTIAL / UNCLEAR 중 하나 반환.\n'
    '\n'
    '응답 규칙:\n'
    '  - 함수 호출은 JSON 한 객체로만 응답: '
    '{"function": "<name>", "args": {...}}\n'
    '  - 추가 자연어 설명/추론 텍스트는 응답에 포함하지 않는다.\n'
)


# ──────────────────────────────────────────────
# 상태 enum
# ──────────────────────────────────────────────
#
# 문자열 값 자체가 SYSTEM_PROMPT 안에 박혀 있으므로, 본 클래스의 값을 바꾸려면
# SYSTEM_PROMPT 도 같이 수정하고 LoRA 재학습이 필요하다.

class VerifyStatus:
    """verify_result 함수의 반환 enum.

    Gemma 응답(JSON {"function":"verify_result","args":{...}} 의 tool_result)과
    agent_node 내부 분기 모두 본 문자열을 사용한다.
    """
    SUCCESS = 'SUCCESS'
    FAIL    = 'FAIL'
    PARTIAL = 'PARTIAL'
    UNCLEAR = 'UNCLEAR'


class ExecStatus:
    """execute_subtask 함수의 반환 enum.

    act_node 가 ExecutionStatus.status (ROS 메시지의 string 필드) 에 본 값을
    그대로 채워 발행하고, agent_node 가 그 문자열을 Gemma 의 tool_result 로
    전달한다. ROS 와 Gemma 양쪽에서 같은 대문자 값을 쓰므로 case 변환이 없다.
    """
    SUCCESS = 'SUCCESS'
    FAIL    = 'FAIL'
    TIMEOUT = 'TIMEOUT'


# ──────────────────────────────────────────────
# Verify (시각 검증) 시스템 프롬프트
# ──────────────────────────────────────────────
#
# verify_result 의 실제 시각 검증 단계는 함수 호출 대화와 별개의 Gemma 호출 (RGB
# 이미지 + original_command → status JSON) 로 처리된다. 본 프롬프트는 그 별도
# 호출에서 사용한다 (agent_node._call_verify_result 의 Gemma 시각 검증 step).
#
# 학습 측 generate_verify_qa(Cat-C) 가 본 프롬프트를 system 메시지로 박은
# JSONL 을 생성한다. 변경 시 LoRA 재학습 필요.

VERIFY_SYSTEM_PROMPT: str = (
    'You evaluate whether a robot grasp-and-place subtask achieved the user\n'
    'request. You receive an RGB image of the current scene and the user\'s\n'
    'original command.\n'
    '\n'
    'Output STRICT JSON, a single object, with no surrounding text:\n'
    '  {"status": <enum>, "reasoning": <english string>}\n'
    '\n'
    'status enum:\n'
    '  SUCCESS  — target object is in the correct destination and stable\n'
    '  FAIL     — target object is in the wrong place, on the floor, or the\n'
    '             task was not performed\n'
    '  PARTIAL  — target is at the correct destination but unstable, partly\n'
    '             outside, or fell over\n'
    '  UNCLEAR  — target is occluded, off-camera, or the scene is ambiguous\n'
    '\n'
    'reasoning rules:\n'
    '  - English only, 5-25 words\n'
    '  - Refer to objects using the template "the {attribute} {object} {spatial}"\n'
    '    (e.g., "the blue can in the can bin"). vocabulary follows\n'
    '    deploy/config/label_vocab.yaml.\n'
    '  - State the observed outcome. Do NOT explain the criteria or process.\n'
)
