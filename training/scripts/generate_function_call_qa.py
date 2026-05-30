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

CLI 사용 예:
  export OPENAI_API_KEY=...
  python training/scripts/generate_function_call_qa.py \\
    --domain         recycling \\
    --label-vocab    deploy/config/label_vocab.yaml \\
    --output         training/data/qa_recycling_catB.jsonl \\
    --external-llm   gpt-4o-mini \\
    --num-samples    500

구현 정책 (2026-05-25 freeze):
  - LLM (gpt-4o-mini) 는 사용자 명령 1건 합성에만 사용 (1 호출/샘플).
  - 함수 호출 시퀀스(assistant turns)는 결정론적으로 합성:
      assistant   → look_scene(prompt=LOOK_SCENE_PROMPTS 중 랜덤)
      tool_result → sample_description(vocab) — 형식·vocabulary 강제
      assistant   → execute_subtask(target_description=look_scene 결과)
      tool_result → ExecStatus 가중 샘플링 (SUCCESS 80% / FAIL 15% / TIMEOUT 5%)
      assistant   → verify_result(original_command=user_command)
    LLM 으로 JSON 생성을 맡기지 않는 이유는 형식 신뢰성 확보 + 분포 보장.
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

# llm_client.py 는 본 파일과 같은 디렉터리. 스크립트로 실행 시 자동 sys.path 에
# 포함되지만 명시적으로 추가해 안전성 확보.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from llm_client import LLMClient  # noqa: E402


# ──────────────────────────────────────────────
# 도메인별 컨텍스트 / 합성 보조 상수
# ──────────────────────────────────────────────

# synthesize_user_command 가 LLM 에 전달하는 도메인 자연어 설명.
# 새 도메인 추가 시 본 dict 갱신 필요.
DOMAIN_DESCRIPTIONS: Dict[str, str] = {
    'recycling': (
        '재활용 분리수거 — 캔/플라스틱(페트병)/종이를 종류별 수거함에 분리해서 넣기. '
        '테이블 위의 객체를 집어 알맞은 통으로 이동시키는 작업.'
    ),
    'tableware': (
        '식기 정리 — 접시/그릇/컵/포크/숟가락을 정해진 위치(선반·정리함)로 옮기기. '
        '테이블 위의 식기를 집어 정리하는 작업.'
    ),
}

USER_CMD_SYSTEM_PROMPT = (
    'You generate ONE realistic short voice command that a user would say to a '
    'home assistant robot.\n'
    'Output rules:\n'
    '  - Korean OR English (vary freely across samples — both are acceptable).\n'
    '  - 3 to 20 words/characters total. Conversational, natural.\n'
    '  - Vary tone: polite formal ("정리해주세요"), casual ("정리 좀"),\n'
    '    brief imperative, or descriptive request.\n'
    '  - Sometimes specify a target object (e.g., "캔만 정리"), sometimes generic.\n'
    '  - No quotes, no explanation, no list — output ONLY the command text.\n'
)

# look_scene 의 prompt 인자 풀 (assistant turn 의 다양성 확보).
LOOK_SCENE_PROMPTS: List[str] = [
    'describe the object to grasp',
    'what should I pick up from this scene',
    'find the next target object',
    'identify the object I should handle',
    'tell me what to grasp next',
    'describe the next target to pick up',
]

# execute_subtask tool_result 분포 (실제 운영의 대략 분포 모사).
EXEC_STATUS_WEIGHTS = [
    (ExecStatus.SUCCESS, 0.80),
    (ExecStatus.FAIL,    0.15),
    (ExecStatus.TIMEOUT, 0.05),
]


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
    p.add_argument('--external-llm', default='gpt-4o-mini',
                   help='Q&A 합성용 외부 LLM (OpenAI 모델 ID)')
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

def synthesize_user_command(domain: str, llm: LLMClient) -> str:
    """도메인 맥락에 맞는 사용자 명령(한국어/영어 혼합) 1건 합성.

    LLM 한 번 호출. 응답 텍스트의 앞뒤 따옴표/공백을 제거하고 반환.
    """
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, domain)
    messages = [
        {'role': 'system', 'content': USER_CMD_SYSTEM_PROMPT},
        {'role': 'user',
         'content': f'Domain: {domain_desc}\n'
                    f'Generate ONE user voice command for this domain.'},
    ]
    text = llm.chat(messages, temperature=0.9, max_tokens=80)
    # 앞뒤 따옴표·공백 제거 (LLM 이 종종 인용부호로 감쌈)
    return text.strip().strip('"\'').strip()


def _weighted_choice(choices, rng: random.Random):
    items, weights = zip(*choices)
    return rng.choices(items, weights=weights, k=1)[0]


def _assistant_call(function_name: str, args: dict) -> dict:
    """assistant 의 함수 호출 메시지 — JSON 한 객체 문자열."""
    payload = {'function': function_name, 'args': args}
    return {
        'role': 'assistant',
        'content': json.dumps(payload, ensure_ascii=False),
    }


def synthesize_function_call_response(
    user_command: str,
    vocab:        dict,
    rng:          random.Random,
    llm:          Optional[LLMClient] = None,   # 현재 미사용 — API 일관성 유지용
) -> List[dict]:
    """사용자 명령 → 다단계 함수 호출 시나리오 메시지 리스트.

    결정론적 합성 (LLM 미사용):
      assistant   → look_scene(prompt=LOOK_SCENE_PROMPTS 랜덤)
      tool_result → sample_description(vocab, rng)        # 형식·vocabulary 강제
      assistant   → execute_subtask(target_description=look_scene 결과)
      tool_result → ExecStatus 가중 샘플링
      assistant   → verify_result(original_command=user_command)
    """
    del llm  # 호출 일관성용 파라미터, 현재 사용 안 함

    look_prompt = rng.choice(LOOK_SCENE_PROMPTS)
    target_desc = sample_description(vocab, rng)
    exec_result = _weighted_choice(EXEC_STATUS_WEIGHTS, rng)

    return [
        _assistant_call('look_scene', {'prompt': look_prompt}),
        {'role': 'tool', 'content': target_desc},
        _assistant_call('execute_subtask', {'target_description': target_desc}),
        {'role': 'tool', 'content': exec_result},
        _assistant_call('verify_result', {'original_command': user_command}),
    ]


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
    llm = LLMClient(model=args.external_llm)

    print(
        f'[generate_function_call_qa] start — domain={args.domain} '
        f'objects={list(vocab["objects"].keys())} '
        f'spatials={len(vocab["spatials"])} num_samples={args.num_samples} '
        f'model={args.external_llm} output={args.output}',
        file=sys.stderr,
    )

    n_written = 0
    n_failed  = 0
    with open(args.output, 'w', encoding='utf-8') as f:
        for i in range(args.num_samples):
            try:
                user_cmd = synthesize_user_command(args.domain, llm)
                if not user_cmd:
                    raise ValueError('빈 user_command')
                turns = synthesize_function_call_response(user_cmd, vocab, rng)
            except Exception as e:  # noqa: BLE001
                n_failed += 1
                print(
                    f'[generate_function_call_qa] sample {i+1} 실패: '
                    f'{type(e).__name__}: {e}',
                    file=sys.stderr,
                )
                continue

            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_cmd},
                *turns,
            ]
            f.write(json.dumps({'messages': messages}, ensure_ascii=False) + '\n')
            n_written += 1

            if (i + 1) % 50 == 0 or (i + 1) == args.num_samples:
                print(
                    f'[generate_function_call_qa] progress {i+1}/{args.num_samples} '
                    f'written={n_written} failed={n_failed} | {llm.format_usage()}',
                    file=sys.stderr,
                )

    print(
        f'[generate_function_call_qa] done — written={n_written}/{args.num_samples} '
        f'failed={n_failed} | {llm.format_usage()} | output={args.output}',
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
