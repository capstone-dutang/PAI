"""
generate_function_call_qa.py

Part C-1 / Cat-B — agent_node가 사용하는 3개 함수
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
      {"role": "system",  "content": <FUNCTION_CALL_SYSTEM_PROMPT>},
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
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


# ──────────────────────────────────────────────
# 함수 시그니처 — agent_node와 정확히 일치해야 함
# (agent_node._run_function_loop 구현 시점에 동기화)
# ──────────────────────────────────────────────

FUNCTION_CALL_SYSTEM_PROMPT = (
    '사용 가능 함수:\n'
    '  look_scene(prompt: str) -> str\n'
    '      RGB 장면을 보고 잡을 객체의 자연어 묘사를 반환한다.\n'
    '      반환 형식 (엄격): "the {attribute} {object} {spatial}"\n'
    '      - 영어 고정, 5~15 단어\n'
    '      - attribute: 색/재질/크기 중 변별 가능한 것 1개\n'
    '      - object:    can / bottle / paper / book 등 단일 명사\n'
    '      - spatial:   위치/관계 표현 (예: "on the right side of the table")\n'
    '      예: "the blue can on the right side of the table"\n'
    '\n'
    '  execute_subtask(target_description: str) -> str\n'
    '      look_scene의 반환 형식 그대로 받아 ACT에 전달. SUCCESS/FAIL/TIMEOUT 중 하나 반환.\n'
    '\n'
    '  verify_result(original_command: str) -> str\n'
    '      직전 subtask 완료 여부 시각 검증. SUCCESS/FAIL/PARTIAL/UNCLEAR 중 하나 반환.\n'
    '\n'
    '응답 규칙:\n'
    '  - 함수 호출은 JSON 한 객체로만 응답: '
    '{"function": "<name>", "args": {...}}\n'
    '  - 추가 자연어 설명/추론 텍스트는 응답에 포함하지 않는다.\n'
)


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
    p.add_argument('--domain',       required=True, help='대상 도메인 (recycling/library/...)')
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
# vocabulary 로드 (record_teleop / sim과 공유)
# ──────────────────────────────────────────────

def load_vocab(path: str, domain: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    if domain not in (raw.get('domains') or {}):
        raise ValueError(f'도메인 {domain}이 label_vocab에 없음.')
    return {
        'template': raw['template'],
        'spatials': raw['spatial_qualifiers'],
        'objects':  raw['domains'][domain]['objects'],
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
    """agent_node가 런타임에 보내는 시스템 프롬프트와 정확히 일치시켜야 한다.

    동기화 책임:
      - agent_node._run_function_loop 구현 시 본 상수와 동일 문자열 사용
      - 변경 시 LoRA 재학습 필요
    """
    return FUNCTION_CALL_SYSTEM_PROMPT


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
