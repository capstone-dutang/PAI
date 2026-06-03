"""
generate_function_call_qa.py

function_call — agent_node가 사용하는 3개 함수
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

출력 포맷 (JSONL — Unsloth FastVisionModel 통합 학습용 typed-array content):
  {
    "messages": [
      {"role": "system",
       "content": [{"type": "text", "text": "<SYSTEM_PROMPT>"}]},
      {"role": "user",
       "content": [{"type": "text", "text": "쓰레기 좀 정리해줘"}]},
      {"role": "assistant",
       "content": [{"type": "text",
                    "text": "{\\"function\\":\\"look_scene\\",\\"args\\":{...}}"}]},
      {"role": "user",
       "content": [{"type": "text",
                    "text": "[tool_result] the red can on the left side of the table"}]},
      ... (execute_subtask, tool_result, verify_result 3턴 추가)
    ]
  }

다단계 시나리오 (한 sample = 7 messages, user/assistant 교대):
  system    → <SYSTEM_PROMPT>
  user      → "캔을 캔류 통에 넣어줘"
  assistant → look_scene(prompt="describe the can to grasp")
  user      → "[tool_result] the red can on the left side of the table"
  assistant → execute_subtask(target_description="the red can on the left side of the table")
  user      → "[tool_result] SUCCESS"
  assistant → verify_result(original_command="캔을 캔류 통에 넣어줘")

  tool role 미사용 — Unsloth FastVisionModel + UnslothVisionDataCollator 는
  user/assistant 외 role 을 지원하지 않으므로 '[tool_result] ' prefix 가 붙은
  user 메시지로 치환한다. 런타임 agent_node 의 OpenAI-style tool feed 와도 일치.

CLI 사용 예:
  export OPENAI_API_KEY=...
  python training/lora_finetuning/scripts/generate_function_call_qa.py \\
    --domain         recycling \\
    --label-vocab    deploy/config/label_vocab.yaml \\
    --output         training/lora_finetuning/data/recycling/recycling_function_call_qa.jsonl \\
    --external-llm   gpt-4o-mini \\
    --num-samples    500

구현 정책 (2026-05-25 freeze, dry run 및 500샘플 분석 후 2회 보완):
  - LLM (gpt-4o-mini) 는 사용자 명령 1건 합성에만 사용 (1 호출/샘플).
    언어 per-sample 결정: 영어 ~30% / 한국어 ~70% (LANGUAGE_ENGLISH_RATIO).
    타겟 per-sample 결정: 도메인 객체 균등 순회 ~90% (OBJECT_SPECIFIC_RATIO)
      + collective term ~10%. 객체 분포 skew 와 표현 중복 동시 완화.
  - 함수 호출 시퀀스(assistant turns)는 결정론적으로 합성:
      assistant   → look_scene(prompt=...)
                    user_command 에서 vocab object 키워드 추출 성공 시
                    object-aware prompt (e.g., "describe the can to grasp"),
                    실패 시 generic prompt.
      tool_result → target_description
                    object 매칭 시 그 object 로 고정 + attribute/spatial 만 랜덤
                    (자기-참조 spatial 제외 — 예: paper 객체에 "next to the paper").
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
# training/lora_finetuning/scripts/ 는 패키지가 아니므로 sys.path 에 ROS2 패키지 루트를
# 추가해 robot_arm_pkg 를 import 한다.
# ──────────────────────────────────────────────

_ROBOT_ARM_PKG_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', 'deploy', 'ros2_ws', 'src', 'robot_arm_pkg',
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
    '  - Follow the Language and Guidance in the user message strictly\n'
    '    (Korean ONLY or English ONLY — never mix in one command).\n'
    '  - 3 to 20 words/characters total. Conversational, natural.\n'
    '  - Vary tone: polite formal, casual, brief imperative, or descriptive request.\n'
    '  - Vary phrasing — avoid repeating the same sentence structure used in '
    'previous samples.\n'
    '  - No quotes, no explanation, no list — output ONLY the command text.\n'
)

# 한국어 / 영어 명령 비율 — 영어 ~30% 강제 (rng 로 per-sample 결정).
LANGUAGE_ENGLISH_RATIO = 0.30

# 객체 균등 순회 — synthesize_user_command 에서 매 호출 객체 1개를 강제 지정
# (90%) 또는 collective 명령 (10%) 으로 분기. 객체 분포 skew 와 명령 중복을
# 동시에 완화. 500 샘플 기준:
#   recycling: 객체별 ~150개 + collective ~50
#   tableware: 객체별 ~90개  + collective ~50
OBJECT_SPECIFIC_RATIO = 0.90

# 객체 → 도메인·언어별 자연어 hint (LLM 에 "이 단어를 명령에 포함하라" 지시).
OBJECT_KEYWORD_HINTS: Dict[str, Dict[str, Dict[str, str]]] = {
    'recycling': {
        'can':     {'ko': '캔',          'en': 'can'},
        'plastic': {'ko': '페트병 또는 플라스틱', 'en': 'plastic bottle'},
        'paper':   {'ko': '종이',         'en': 'paper'},
    },
    'tableware': {
        'plate': {'ko': '접시',  'en': 'plate'},
        'fork':  {'ko': '포크',  'en': 'fork'},
        'spoon': {'ko': '숟가락', 'en': 'spoon'},
    },
}

# Collective term hint (10% 케이스) — generic fallback 학습 표본 확보용.
COLLECTIVE_HINTS: Dict[str, Dict[str, str]] = {
    'recycling': {
        'ko': "예: '쓰레기 좀 정리해줘', '분리수거 부탁해'",
        'en': "e.g., 'sort the trash', 'do the recycling please'",
    },
    'tableware': {
        'ko': "예: '식기 정리해줘', '그릇 좀 치워줘'",
        'en': "e.g., 'put the dishes away', 'clean up the tableware'",
    },
}

# look_scene 의 prompt 인자 풀 — 객체 미지정 (user_command 가 generic 일 때 사용).
LOOK_SCENE_PROMPTS_GENERIC: List[str] = [
    'describe the object to grasp',
    'what should I pick up from this scene',
    'find the next target object',
    'identify the object I should handle',
    'tell me what to grasp next',
    'describe the next target to pick up',
]

# look_scene prompt — 객체 지정형 (user_command 에서 추출된 object 를 substitute).
LOOK_SCENE_PROMPTS_OBJECT_AWARE: List[str] = [
    'describe the {object} to grasp',
    'find the {object} I should pick up',
    'identify the {object} to handle',
    'tell me which {object} to grasp',
    'describe the next {object} to pick up',
]

# 도메인별 user_command 키워드 → vocab object 매핑.
# 매칭은 substring (영어는 lowercase 비교, 한국어는 원문 비교).
# dict 삽입 순서 = 우선순위 (여러 키워드 매칭 시 앞쪽 우선).
KEYWORD_TO_OBJECT: Dict[str, Dict[str, str]] = {
    'recycling': {
        '캔':      'can',
        'can':     'can',
        '페트병':   'plastic',
        '플라스틱': 'plastic',
        'plastic': 'plastic',
        '종이':    'paper',
        'paper':   'paper',
    },
    'tableware': {
        '접시':    'plate',
        'plate':   'plate',
        '포크':    'fork',
        'fork':    'fork',
        '숟가락':   'spoon',
        'spoon':   'spoon',
    },
}

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
    p = argparse.ArgumentParser(description='function_call — 함수 호출 시나리오 Q&A 합성')
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


def _attrs_of(vocab: dict, obj_name: str) -> List[str]:
    """vocab.objects[obj_name].attributes 호환 추출 (dict / list 둘 다 허용)."""
    block = vocab['objects'][obj_name]
    return list(block['attributes'] if isinstance(block, dict) else block)


def sample_description_for_object(
    vocab:    dict,
    obj_name: str,
    rng:      random.Random,
) -> str:
    """주어진 object 로 description 생성.

    자기-참조 spatial 제외: object 명사가 spatial 문자열에 포함되면 후보에서 빠짐.
    예: object='paper' → 'next to the paper' 제외. 후보가 전부 빠지는 극단 케이스
    (도메인이 단일 객체이고 모든 spatial 이 그 객체 참조) 에서만 fallback 으로
    원본 spatials 에서 샘플링.
    """
    attr = rng.choice(_attrs_of(vocab, obj_name))
    candidates = [s for s in vocab['spatials'] if obj_name not in s]
    spat = rng.choice(candidates) if candidates else rng.choice(vocab['spatials'])
    return vocab['template'].format(attribute=attr, object=obj_name, spatial=spat)


def sample_description(vocab: dict, rng: random.Random) -> str:
    """object 를 랜덤 선택 후 sample_description_for_object 위임."""
    obj_name = rng.choice(list(vocab['objects'].keys()))
    return sample_description_for_object(vocab, obj_name, rng)


def extract_target_object(user_command: str, domain: str) -> Optional[str]:
    """user_command 에서 도메인 vocab object 키워드 추출. 못 찾으면 None.

    영어 키워드는 lowercase 비교, 한국어는 원문 비교. KEYWORD_TO_OBJECT 의 dict
    삽입 순서가 우선순위 (여러 키워드 동시 매칭 시 앞쪽 우선).
    """
    kw_map = KEYWORD_TO_OBJECT.get(domain) or {}
    cmd_lower = user_command.lower()
    for kw, obj in kw_map.items():
        haystack = cmd_lower if kw.isascii() else user_command
        needle   = kw.lower() if kw.isascii() else kw
        if needle in haystack:
            return obj
    return None


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
# Q&A 합성 (외부 LLM 클라이언트 = llm_client.LLMClient)
# ──────────────────────────────────────────────

def synthesize_user_command(
    domain:          str,
    vocab:           dict,
    llm:             LLMClient,
    rng:             random.Random,
    target_obj_hint: Optional[str] = None,
) -> str:
    """도메인 맥락에 맞는 사용자 명령 1건 합성.

    rng 로 per-sample 결정:
      - 언어: 영어 ~LANGUAGE_ENGLISH_RATIO 확률
      - 타겟: OBJECT_SPECIFIC_RATIO 확률로 도메인 객체 1개 (균등 순회 또는
              target_obj_hint), 나머지는 collective term hint.

    target_obj_hint (2026-05-25 추가):
      - None  (default, Cat-B 용): 객체를 본 함수가 random 선택 — 기존 동작 보존
      - <obj> (Cat-C 용): 호출자가 미리 결정한 target 객체를 hint 로 강제.
        Cat-C 의 synthesize_scenario 는 target_obj 를 먼저 정하므로 user_command
        도 같은 객체를 명시해야 verify 의 (target, command) 의미 정합이 유지됨.
        collective(10%) 케이스는 target 과 무관하므로 hint 무시.

    이로써 객체 분포 skew, 표현 중복, target-command 정합을 모두 처리한다.
    """
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, domain)
    use_english = rng.random() < LANGUAGE_ENGLISH_RATIO
    lang_hint = 'English ONLY' if use_english else 'Korean ONLY'
    lang_key  = 'en' if use_english else 'ko'

    use_specific = rng.random() < OBJECT_SPECIFIC_RATIO
    if use_specific and domain in OBJECT_KEYWORD_HINTS:
        # hint 가 주어지면 그대로 사용 (Cat-C), 없으면 균등 random (Cat-B)
        obj_name = target_obj_hint or rng.choice(list(vocab['objects'].keys()))
        kw_map   = OBJECT_KEYWORD_HINTS[domain].get(obj_name, {})
        kw       = kw_map.get(lang_key) or obj_name
        guidance = (
            f'The user wants to handle the {kw}. Explicitly mention this object '
            f'in the command. Vary the surrounding phrasing.'
        )
    else:
        coll = (COLLECTIVE_HINTS.get(domain, {}) or {}).get(lang_key, '')
        guidance = (
            'Use a generic/collective phrase that does NOT name any specific '
            f'object class. {coll}'
        )

    messages = [
        {'role': 'system', 'content': USER_CMD_SYSTEM_PROMPT},
        {'role': 'user',
         'content': (
             f'Domain: {domain_desc}\n'
             f'Language: {lang_hint}.\n'
             f'Guidance: {guidance}\n'
             f'Generate ONE user voice command for this domain.'
         )},
    ]
    text = llm.chat(messages, temperature=0.9, max_tokens=80)
    return text.strip().strip('"\'').strip()


def _weighted_choice(choices, rng: random.Random):
    items, weights = zip(*choices)
    return rng.choices(items, weights=weights, k=1)[0]


# verify (이미지) 통합 학습을 위해 Unsloth FastVisionModel 의 typed-array content
# 포맷을 사용한다. tool role 미지원이라 user role + '[tool_result] ' prefix 로
# 치환 (convert_function_call_to_vision_format.py 와 동일 규약).
TOOL_RESULT_PREFIX = '[tool_result] '


def _typed_text(text: str) -> List[Dict[str, str]]:
    """typed-array content 한 항목 (text only)."""
    return [{'type': 'text', 'text': text}]


def _msg_text(role: str, text: str) -> dict:
    """role 메시지 — content 는 typed-array text 1개."""
    return {'role': role, 'content': _typed_text(text)}


def _assistant_call(function_name: str, args: dict) -> dict:
    """assistant 의 함수 호출 메시지 — JSON 한 객체 문자열을 typed-array 로 wrap."""
    payload = {'function': function_name, 'args': args}
    return _msg_text('assistant', json.dumps(payload, ensure_ascii=False))


def _user_tool_result(value: str) -> dict:
    """tool_result 메시지 — user role + '[tool_result] ' prefix."""
    return _msg_text('user', TOOL_RESULT_PREFIX + str(value))


def synthesize_function_call_response(
    user_command: str,
    vocab:        dict,
    domain:       str,
    rng:          random.Random,
    llm:          Optional[LLMClient] = None,   # API 일관성용 — 현재 미사용
) -> List[dict]:
    """사용자 명령 → 다단계 함수 호출 시나리오 메시지 리스트.

    user_command 에서 vocab object 키워드를 추출한 경우:
      - look_scene prompt 에 해당 object 명시 (e.g., "describe the can to grasp")
      - target_description 의 object 도 그 키워드로 고정 (attribute/spatial 만 랜덤)
    추출 실패 시 (예: "쓰레기 좀 정리"):
      - generic prompt + 완전 랜덤 sampling

    반환 메시지는 user/assistant 교대 구조 (tool role 사용 안 함):
      assistant(look) → user(tool_result desc) → assistant(execute)
      → user(tool_result status) → assistant(verify)
    """
    del llm  # API 일관성용

    matched_obj = extract_target_object(user_command, domain)

    if matched_obj is not None:
        look_prompt = rng.choice(LOOK_SCENE_PROMPTS_OBJECT_AWARE).format(
            object=matched_obj)
        target_desc = sample_description_for_object(vocab, matched_obj, rng)
    else:
        look_prompt = rng.choice(LOOK_SCENE_PROMPTS_GENERIC)
        target_desc = sample_description(vocab, rng)

    exec_result = _weighted_choice(EXEC_STATUS_WEIGHTS, rng)

    return [
        _assistant_call('look_scene', {'prompt': look_prompt}),
        _user_tool_result(target_desc),
        _assistant_call('execute_subtask', {'target_description': target_desc}),
        _user_tool_result(exec_result),
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
                user_cmd = synthesize_user_command(args.domain, vocab, llm, rng)
                if not user_cmd:
                    raise ValueError('빈 user_command')
                turns = synthesize_function_call_response(
                    user_cmd, vocab, args.domain, rng)
            except Exception as e:  # noqa: BLE001
                n_failed += 1
                print(
                    f'[generate_function_call_qa] sample {i+1} 실패: '
                    f'{type(e).__name__}: {e}',
                    file=sys.stderr,
                )
                continue

            messages = [
                _msg_text('system', system_prompt),
                _msg_text('user',   user_cmd),
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
