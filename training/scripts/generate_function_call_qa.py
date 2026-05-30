"""
generate_function_call_qa.py

Cat-B — agent_node가 사용하는 3개 함수
(look_scene / execute_subtask / verify_result)에 대한 함수 호출 시나리오 Q&A를
외부 LLM으로 합성한다. Gemma 도메인 LoRA가 배포 시 함수 호출 응답을
정확히 산출하도록 학습시키는 데이터.

target_description 형식 — 확정 사항:
  영어 반구조화 템플릿  "the {attribute} {object} {spatial}"
  예: "the blue can on the right side of the table"
  길이: 5~15 단어
  vocabulary: deploy/config/label_vocab.yaml과 일치

look_scene의 반환 description, execute_subtask의 target_description 인자,
verify_result reasoning 내 객체 참조 표현 모두 위 템플릿 분포를 따라야 한다.
Sim 라벨 / Real 라벨 / 배포 시 Gemma 출력이 모두 같은 분포가 되어야 ACT가
학습한 임베딩 공간 안에서 동작.

출력 포맷 (JSONL 한 줄당 하나의 sample):
  {
    "messages": [
      {"role": "system",  "content": <SYSTEM_PROMPT>},   # robot_arm_pkg.prompts
      {"role": "user",    "content": "쓰레기 좀 정리해줘"},
      {"role": "assistant","content": "{\\"function\\": \\"look_scene\\",
                                       \\"args\\": {\\"prompt\\": \\"...\\"}}"}
    ]
  }

다단계 시나리오 예 (한 sample 내에 assistant 발화 여러 개):
  user        → "캔을 캔류 통에 넣어줘"
  assistant   → look_scene(prompt="describe the can to grasp")
  tool_result → "the red can on the left side of the table"   # description 형식
  assistant   → execute_subtask(target_description="the red can on the left side of the table")
  tool_result → "SUCCESS"
  assistant   → verify_result(original_command="캔을 캔류 통에 넣어줘")

CLI 사용 예 (예정):
  python generate_function_call_qa.py \\
    --domain         recycling \\
    --label-vocab    deploy/config/label_vocab.yaml \\
    --output         training/data/qa_recycling_catB.jsonl \\
    --external-llm   gpt-4 \\
    --num-samples    500

미구현 항목 (구현 시 결정 필요):
  - 외부 LLM 클라이언트 (OpenAI / Anthropic / 로컬)
  - 사용자 명령 다양성 확보 — 도메인별 명령 풀 + paraphrase
  - 단일 함수 호출 / 다단계 시나리오 비율
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


# ──────────────────────────────────────────────
# SSoT import — 시스템 프롬프트와 상태 enum 은 robot_arm_pkg.prompts 에 있다.
# training/scripts/ 는 패키지가 아니므로 sys.path 에 ROS2 패키지 루트를 추가해
# robot_arm_pkg 를 import 한다.
# ──────────────────────────────────────────────

_ROBOT_ARM_PKG_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'deploy', 'ros2_ws', 'src', 'robot_arm_pkg',
))
if _ROBOT_ARM_PKG_ROOT not in sys.path:
    sys.path.insert(0, _ROBOT_ARM_PKG_ROOT)

from robot_arm_pkg.prompts import SYSTEM_PROMPT, VerifyStatus, ExecStatus  # noqa: E402,F401


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class QAArgs:
    domain:       str
    label_vocab:  str
    output:       str
    external_llm: str
    num_samples:  int
    seed:         int


def parse_args() -> QAArgs:
    p = argparse.ArgumentParser(description='Cat-B — 함수 호출 시나리오 Q&A 합성')
    p.add_argument('--domain',       required=True, help='대상 도메인 (recycling/tableware/...)')
    p.add_argument('--label-vocab',  default='deploy/config/label_vocab.yaml',
                   help='target_description 어휘 사전')
    p.add_argument('--output',       required=True, help='JSONL 출력 경로')
    p.add_argument('--external-llm', default='gpt-4', help='Q&A 합성용 외부 LLM')
    p.add_argument('--num-samples',  type=int, default=500, help='생성할 샘플 수')
    p.add_argument('--seed',         type=int, default=0)
    a = p.parse_args()
    return QAArgs(
        domain       = a.domain,
        label_vocab  = os.path.expanduser(a.label_vocab),
        output       = os.path.expanduser(a.output),
        external_llm = a.external_llm,
        num_samples  = a.num_samples,
        seed         = a.seed,
    )


# ──────────────────────────────────────────────
# vocabulary 로드 (record_real_teleop / sim과 공유)
# ──────────────────────────────────────────────

def load_vocab(path: str, domain: str) -> dict:
    """label_vocab.yaml 로드 + 도메인 추출.

    spatials 는 (전역 spatial_qualifiers + 도메인별 spatial_qualifiers) 병합본.
    record_real_teleop.py / record_sim_teleop.py 의 load_vocab 과 동일 규약.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    domains = raw.get('domains') or {}
    if domain not in domains:
        raise ValueError(f'도메인 {domain}이 label_vocab에 없음.')
    domain_block = domains[domain] or {}
    global_spatials = list(raw.get('spatial_qualifiers') or [])
    domain_spatials = list(domain_block.get('spatial_qualifiers') or [])
    return {
        'template': raw['template'],
        'spatials': global_spatials + domain_spatials,
        'objects':  domain_block['objects'],
    }


def sample_description(vocab: dict, rng: random.Random) -> str:
    obj_name = rng.choice(list(vocab['objects'].keys()))
    attr     = rng.choice(vocab['objects'][obj_name]['attributes']
                          if isinstance(vocab['objects'][obj_name], dict)
                          else vocab['objects'][obj_name])
    spat     = rng.choice(vocab['spatials'])
    return vocab['template'].format(attribute=attr, object=obj_name, spatial=spat)


# ──────────────────────────────────────────────
# 시스템 프롬프트
# ──────────────────────────────────────────────

def build_system_prompt() -> str:
    """robot_arm_pkg.prompts.SYSTEM_PROMPT 를 그대로 반환.

    agent_node 와 본 합성기는 같은 SSoT 상수를 import 하므로 두 곳의 문자열은
    자동으로 일치한다. 변경 시 LoRA 재학습 필요.
    """
    return SYSTEM_PROMPT


# ──────────────────────────────────────────────
# Q&A 합성 (외부 LLM 클라이언트 미구현)
# ──────────────────────────────────────────────

def synthesize_user_command(domain: str, llm: str) -> str:
    """도메인 맥락에 맞는 사용자 명령(한국어/영어 혼합) 1건 합성.

    TODO: 외부 LLM 호출 또는 도메인 명령 풀 + paraphrase 정책 확정 후 구현.
    """
    raise NotImplementedError(
        '[generate_function_call_qa] synthesize_user_command() 미구현'
    )


def synthesize_function_call_response(
    user_command: str,
    vocab:        dict,
    rng:          random.Random,
    llm:          str,
) -> List[dict]:
    """사용자 명령 → 다단계 함수 호출 시나리오 메시지 리스트.

    반환 예 (단일 step):
      [{"role": "assistant",
        "content": "{\"function\": \"look_scene\",
                     \"args\": {\"prompt\": \"describe the object to grasp\"}}"}]

    다단계는 tool/role 시퀀스로 확장. look_scene 결과는 본 vocab에서 샘플링한
    description으로 채워 학습 분포를 보장 (외부 LLM이 임의 형식 생성하는 것 차단).

    TODO: 외부 LLM 호출부 및 단계별 비율 정책 확정 후 구현.
          현재는 다음 규칙만 명시:
            - look_scene tool_result는 sample_description(vocab, rng)로 강제 주입
            - execute_subtask args.target_description은 look_scene 결과를 그대로 전달
    """
    raise NotImplementedError(
        '[generate_function_call_qa] synthesize_function_call_response() 미구현'
    )


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    vocab = load_vocab(args.label_vocab, args.domain)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    system_prompt = build_system_prompt()

    raise NotImplementedError(
        '[generate_function_call_qa] main 합성 루프 미구현. '
        'synthesize_* 구현 후 단순 루프 + JSONL 저장. '
        f'system_prompt / vocab 준비 완료 (domain={args.domain}, '
        f'objects={list(vocab["objects"].keys())})'
    )


if __name__ == '__main__':
    main()
