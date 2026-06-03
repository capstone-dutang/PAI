"""
generate_verify_qa.py

verify — verify_result 시각 검증 시나리오 JSONL 합성 (2-pass 구조 중 1단계).

배포 시 agent_node._call_verify_result 는 Gemma 를 한 번 더 호출하여
RGB + original_command → status JSON 매핑을 얻는다. 본 스크립트는 그 호출의
training 분포를 만든다. function_call (함수 호출)과 같은 LoRA 어댑터를 공유하며,
Unsloth FastVisionModel typed-array content 형식으로 출력한다.

2-pass 구조 (verify 합성):
  본 스크립트(Step 1)는 LLM(gpt-4o-mini) 으로 시나리오 메타데이터 + JSONL 을
  만들고, [`render_verify_images.py`](render_verify_images.py)(Step 2) 가
  Isaac Sim 으로 PNG 를 렌더한다. 두 환경(openai vs Isaac Sim) 의존성을
  깨끗이 분리하기 위해서. 본 스크립트는 그 인터페이스로 scenarios sidecar
  (`{output}.scenarios.jsonl`) 도 함께 출력한다.

출력 (한 도메인 1회 실행 시):
  --output-jsonl 가 가리키는 파일               : 학습용 JSONL (vision typed-array)
  --output-jsonl 가 가리키는 파일 + '.scenarios.jsonl' : 시나리오 메타데이터 sidecar
                                                       (render_verify_images.py 의 입력)

JSONL 한 줄 (학습용):
  {
    "messages": [
      {"role": "system",
       "content": [{"type": "text", "text": "<VERIFY_SYSTEM_PROMPT>"}]},
      {"role": "user", "content": [
        {"type": "image", "image": "<image_dir 기준 상대 path, 학습 시 PIL.Image.open>"},
        {"type": "text",  "text":  "original_command: \"...\"\\nOutput the verification JSON."}
      ]},
      {"role": "assistant",
       "content": [{"type": "text",
                    "text": "{\\"status\\":\\"SUCCESS\\",\\"reasoning\\":\\"...\\"}"}]}
    ]
  }

scenarios sidecar 한 줄 (메타데이터):
  {scenario_id, domain, status, target_object, target_phrase, destination,
   original_command, image_relpath, image_abspath}

freeze 결정 (2026-05-25, C2 통합 vision 포맷 결정 후):
  값과 구현은 아래 캐노니컬 위치에만 박혀 있다. 본 docstring 은 navigation 만
  한다 — 다른 곳에서 값을 재기재하지 말 것 (drift 방지).

  · system prompt            : robot_arm_pkg.prompts.VERIFY_SYSTEM_PROMPT
                                function_call 의 SYSTEM_PROMPT 와 별개. agent_node 의
                                verify Gemma 별도 호출과 일치하는 단일 출처.
  · 메시지 구조 / image-first : build_sample()
                                3-msg single-turn (system + user + assistant),
                                Unsloth typed-array content, Gemma 4 vision 요구로
                                user content 에서 image 가 text 앞.
  · status 가중 분포          : STATUS_WEIGHTS 상수
  · target object 균등 순회   : synthesize_scenario()
  · target_phrase / cmd 합성   : generate_function_call_qa 의
                                sample_description_for_object /
                                synthesize_user_command 재사용 (분포 일관성).
  · reasoning 템플릿          : REASONING_TEMPLATES + generate_reasoning()
                                LLM 미사용 — JSON 형식·vocabulary 신뢰성 우선.
  · destination 매핑          : DESTINATIONS 상수
                                (Sim 자산 의 bin/holder 이름과 일치 시킬 것).
  · image path 스키마         : synthesize_scenario / build_sample
                                image_dir 기준 상대 path. 학습 시
                                PIL.Image.open(image_dir/path) 로 로드.
  · Sim 렌더링 (Step 2)       : render_verify_images.py
                                (별도 스크립트, ~/isaacsim/python.sh 로 실행).
                                본 파일은 PNG 를 만들지 않는다 — scenarios
                                sidecar 만 제공.

CLI (Step 1):
  python training/lora_finetuning/scripts/generate_verify_qa.py \\
    --domain         recycling \\
    --label-vocab    deploy/config/label_vocab.yaml \\
    --output-jsonl   training/lora_finetuning/data/recycling/recycling_verify_qa.jsonl \\
    --image-dir      training/lora_finetuning/data/recycling/verify_images/ \\
    --num-scenes     500 \\
    --external-llm   gpt-4o-mini

CLI (Step 2 — Sim 환경):
  ~/isaacsim/python.sh training/lora_finetuning/scripts/render_verify_images.py \\
    --scenarios       training/lora_finetuning/data/recycling/recycling_verify_qa.jsonl.scenarios.jsonl \\
    --scene-usd       sim/scenes/base_scene.usd \\
    --domain          recycling \\
    --image-dir       training/lora_finetuning/data/recycling/verify_images/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml  # noqa: F401  (load_vocab 가 사용)


# ──────────────────────────────────────────────
# SSoT / 공통 모듈 import
# ──────────────────────────────────────────────

_ROBOT_ARM_PKG_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', 'deploy', 'ros2_ws', 'src', 'robot_arm_pkg',
))
if _ROBOT_ARM_PKG_ROOT not in sys.path:
    sys.path.insert(0, _ROBOT_ARM_PKG_ROOT)
from robot_arm_pkg.prompts import VERIFY_SYSTEM_PROMPT, VerifyStatus  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from llm_client import LLMClient  # noqa: E402
from generate_function_call_qa import (  # noqa: E402
    load_vocab,
    sample_description_for_object,
    synthesize_user_command,
    _typed_text,
)


# ──────────────────────────────────────────────
# 상태 / 목적지 / reasoning 템플릿
# ──────────────────────────────────────────────

# 500 시나리오 기준 ≈ SUCCESS 250 / FAIL 125 / PARTIAL 75 / UNCLEAR 50.
# 변경 시 본 상수만 수정 (모듈 docstring 의 freeze 매트릭스는 본 상수를 가리킴).
STATUS_WEIGHTS: List[Tuple[str, float]] = [
    (VerifyStatus.SUCCESS, 0.50),
    (VerifyStatus.FAIL,    0.25),
    (VerifyStatus.PARTIAL, 0.15),
    (VerifyStatus.UNCLEAR, 0.10),
]

# 도메인 × 객체 → reasoning 에 박힐 destination 문구.
# 변경 시 Sim 자산의 시각 표식 이름과 일치 시킬 것.
#   - recycling: 객체별 분리 bin (3개 USD)
#   - tableware: 모든 식기 → 테이블 위 단일 고정 영역 (별도 객체 자산 없음,
#                Sim 상에서는 테이블 표면의 평면 prim / 매트로 시각 표시)
#                (2026-05-25 freeze)
DESTINATIONS: Dict[str, Dict[str, str]] = {
    'recycling': {
        'can':     'the can bin',
        'plastic': 'the plastic bin',
        'paper':   'the paper bin',
    },
    'tableware': {
        'plate': 'the target area on the table',
        'fork':  'the target area on the table',
        'spoon': 'the target area on the table',
    },
}

# status × 5 템플릿. {target} = target_phrase, {dest} = destination.
# 변동성은 target_phrase (attribute × spatial) 가 주로 담당 — 템플릿 수가 적어도
# 동일 status 내 표면 다양성은 충분. 추가 시 dict 항목만 늘리면 된다.
# 표현은 recycling(bin) / tableware(평면 영역) 양쪽에 자연스럽도록 일반화:
# "bin"·"rim" 같은 도메인 특정 어휘를 피하고 "inside / outside / edge" 위주 사용.
REASONING_TEMPLATES: Dict[str, List[str]] = {
    VerifyStatus.SUCCESS: [
        '{target} is now inside {dest}',
        '{target} has been placed in {dest}',
        '{target} sits stably in {dest}',
        '{target} is correctly placed in {dest}',
        'the task is done — {target} is in {dest}',
    ],
    VerifyStatus.FAIL: [
        '{target} is still on the table, not in {dest}',
        '{target} is on the floor next to {dest}',
        '{target} ended up outside {dest}',
        '{target} was left in the wrong place, not {dest}',
        '{target} fell off before reaching {dest}',
    ],
    VerifyStatus.PARTIAL: [
        '{target} is balancing on the edge of {dest}',
        '{target} is half inside {dest} and half outside',
        '{target} is tipped at the edge of {dest}',
        '{target} is tilted on top of {dest} and may slip',
        '{target} touched {dest} but is not fully inside',
    ],
    VerifyStatus.UNCLEAR: [
        '{target} is partially occluded by another object',
        '{target} is at the edge of the view, position uncertain',
        'the scene is ambiguous — {target} location is hard to confirm',
        '{target} appears blurred, outcome cannot be judged',
        'view is blocked, {target} status is unclear',
    ],
}


# ──────────────────────────────────────────────
# 데이터 클래스 / CLI
# ──────────────────────────────────────────────

@dataclass
class VerifyScenario:
    """1개 시나리오의 정적 메타데이터. Sim 배치/렌더링은 별도 함수가 소비."""
    scenario_id:      str            # e.g., 'recycling_00001'
    domain:           str
    status:           str            # VerifyStatus enum
    target_object:    str            # vocab object 명사 (can/plate/...)
    target_phrase:    str            # "the {attribute} {object} {spatial}"
    destination:      str            # reasoning 에 사용
    original_command: str            # user 메시지 안에 들어가는 user 명령
    image_relpath:    str            # JSONL 의 image 필드 (image_dir 기준 상대)
    image_abspath:    str            # render_scenario_image 가 저장할 절대 경로


@dataclass
class VerifyArgs:
    domain:       str
    label_vocab:  str
    output_jsonl: str
    image_dir:    str
    num_scenes:   int
    external_llm: str
    skip_render:  bool
    seed:         int


def parse_args() -> VerifyArgs:
    p = argparse.ArgumentParser(description='verify — 시각 검증 시나리오 JSONL 합성')
    p.add_argument('--domain',        required=True, help='대상 도메인 (recycling/tableware/...)')
    p.add_argument('--label-vocab',   default='deploy/config/label_vocab.yaml')
    p.add_argument('--output-jsonl',  required=True, help='JSONL 출력 경로')
    p.add_argument('--image-dir',     required=True,
                   help='렌더링된 PNG 저장 디렉터리. JSONL 의 image path 는 이 기준 상대.')
    p.add_argument('--num-scenes',    type=int, default=500)
    p.add_argument('--external-llm',  default='gpt-4o-mini',
                   help='original_command 합성용 OpenAI 모델 ID')
    p.add_argument('--skip-render',   action='store_true',
                   help='Isaac Sim 렌더링 건너뜀 (구조 검증·dry run 용). 학습 직전엔 false.')
    p.add_argument('--seed',          type=int, default=0)
    a = p.parse_args()
    return VerifyArgs(
        domain       = a.domain,
        label_vocab  = os.path.expanduser(a.label_vocab),
        output_jsonl = os.path.expanduser(a.output_jsonl),
        image_dir    = os.path.expanduser(a.image_dir),
        num_scenes   = a.num_scenes,
        external_llm = a.external_llm,
        skip_render  = a.skip_render,
        seed         = a.seed,
    )


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _weighted_choice(choices, rng: random.Random):
    items, weights = zip(*choices)
    return rng.choices(items, weights=weights, k=1)[0]


def _scenario_id(domain: str, index: int) -> str:
    return f'{domain}_{index:05d}'


def _destination_of(domain: str, obj_name: str) -> str:
    """DESTINATIONS 에서 obj_name 의 목적지 조회. 정의 없으면 'the destination' fallback."""
    dm = DESTINATIONS.get(domain) or {}
    return dm.get(obj_name) or 'the destination'


# ──────────────────────────────────────────────
# 시나리오 / reasoning / JSONL 샘플 합성
# ──────────────────────────────────────────────

def synthesize_scenario(
    args:  VerifyArgs,
    vocab: dict,
    llm:   LLMClient,
    rng:   random.Random,
    index: int,
) -> VerifyScenario:
    """1개 시나리오의 정적 데이터 합성 (Sim 비의존).

    - status: STATUS_WEIGHTS 로 가중 샘플
    - target_object: vocab.objects 중 균등 샘플 (per-call 1개)
    - target_phrase: sample_description_for_object (자기-참조 spatial 제외)
    - destination: _destination_of (DESTINATIONS 매핑)
    - original_command: function_call 합성기(synthesize_user_command) 재사용
      target_obj_hint=target_obj 로 강제 — verify 의 (target, command) 의미
      정합 유지 (Cat-C 만의 요구; Cat-B 는 hint=None 기본값 유지).
    - image path: image_dir 기준 '{scenario_id}.png'
    """
    status        = _weighted_choice(STATUS_WEIGHTS, rng)
    target_obj    = rng.choice(list(vocab['objects'].keys()))
    target_phrase = sample_description_for_object(vocab, target_obj, rng)
    destination   = _destination_of(args.domain, target_obj)

    original_command = synthesize_user_command(
        args.domain, vocab, llm, rng,
        target_obj_hint=target_obj,
    )
    if not original_command:
        raise ValueError('빈 original_command')

    sid           = _scenario_id(args.domain, index)
    image_relpath = f'{sid}.png'
    image_abspath = os.path.join(args.image_dir, image_relpath)

    return VerifyScenario(
        scenario_id      = sid,
        domain           = args.domain,
        status           = status,
        target_object    = target_obj,
        target_phrase    = target_phrase,
        destination      = destination,
        original_command = original_command,
        image_relpath    = image_relpath,
        image_abspath    = image_abspath,
    )


def generate_reasoning(scenario: VerifyScenario, rng: random.Random) -> str:
    """status 별 템플릿 1개 선택 → target/dest substitute. 결정론적 (LLM 미사용)."""
    templates = REASONING_TEMPLATES.get(scenario.status) or []
    if not templates:
        return f'{scenario.target_phrase} status unclear'
    tmpl = rng.choice(templates)
    return tmpl.format(target=scenario.target_phrase, dest=scenario.destination)


def _scenario_to_dict(s: VerifyScenario) -> dict:
    """VerifyScenario → sidecar JSONL 한 줄용 dict.

    render_verify_images.py 가 본 dict 를 직접 읽어 시나리오를 복원한다.
    """
    return {
        'scenario_id':      s.scenario_id,
        'domain':           s.domain,
        'status':           s.status,
        'target_object':    s.target_object,
        'target_phrase':    s.target_phrase,
        'destination':      s.destination,
        'original_command': s.original_command,
        'image_relpath':    s.image_relpath,
        'image_abspath':    s.image_abspath,
    }


def build_sample(scenario: VerifyScenario, reasoning: str) -> dict:
    """verify 형식 messages dict 1건 생성 (3-msg single-turn).

    user content 항목 순서 = image → text (Gemma 4 vision 요구).
    """
    asst_payload = {
        'status':    scenario.status,
        'reasoning': reasoning,
    }
    user_text = (
        f'original_command: "{scenario.original_command}"\n'
        f'Output the verification JSON.'
    )
    return {
        'messages': [
            {'role': 'system',
             'content': _typed_text(VERIFY_SYSTEM_PROMPT)},
            {'role': 'user', 'content': [
                # image-before-text 순서 강제 (Gemma 4 멀티모달 prompt 요구).
                {'type': 'image', 'image': scenario.image_relpath},
                {'type': 'text',  'text':  user_text},
            ]},
            {'role': 'assistant',
             'content': _typed_text(json.dumps(asst_payload, ensure_ascii=False))},
        ]
    }


# ──────────────────────────────────────────────
# Sim 렌더링 — 본 스크립트는 수행하지 않음 (Step 2 로 분리)
# ──────────────────────────────────────────────
#
# verify 의 PNG 렌더링은 본 스크립트가 직접 수행하지 않는다.
# 본 스크립트는 학습용 JSONL 과 scenarios sidecar 만 출력하며, PNG 생성은
# render_verify_images.py 가 Isaac Sim standalone 환경에서 처리한다.
# 두 환경(openai vs Isaac Sim) 의존성을 분리하기 위해서 2-pass 구조.
#
# scenarios sidecar 한 줄의 dict 스키마는 _scenario_to_dict() 참조.


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    vocab = load_vocab(args.label_vocab, args.domain)
    llm = LLMClient(model=args.external_llm)

    # 출력 디렉터리 사전 생성
    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)

    scenarios_path = args.output_jsonl + '.scenarios.jsonl'

    print(
        f'[generate_verify_qa] start — domain={args.domain} '
        f'objects={list(vocab["objects"].keys())} '
        f'num_scenes={args.num_scenes} image_dir={args.image_dir} '
        f'output={args.output_jsonl} scenarios={scenarios_path}',
        file=sys.stderr,
    )

    n_written = 0
    n_failed  = 0
    status_counter: Dict[str, int] = {}
    target_counter: Dict[str, int] = {}

    with open(args.output_jsonl, 'w', encoding='utf-8') as f_jsonl, \
         open(scenarios_path,    'w', encoding='utf-8') as f_scen:
        for i in range(args.num_scenes):
            try:
                scenario  = synthesize_scenario(args, vocab, llm, rng, i)
                reasoning = generate_reasoning(scenario, rng)
                sample    = build_sample(scenario, reasoning)
            except Exception as e:  # noqa: BLE001
                n_failed += 1
                print(
                    f'[generate_verify_qa] scenario {i+1} 실패: '
                    f'{type(e).__name__}: {e}',
                    file=sys.stderr,
                )
                continue

            # 학습 JSONL + scenarios sidecar 둘 다 기록 (render_verify_images.py 의 입력)
            f_jsonl.write(json.dumps(sample, ensure_ascii=False) + '\n')
            f_scen.write(json.dumps(_scenario_to_dict(scenario),
                                    ensure_ascii=False) + '\n')
            n_written += 1
            status_counter[scenario.status] = \
                status_counter.get(scenario.status, 0) + 1
            target_counter[scenario.target_object] = \
                target_counter.get(scenario.target_object, 0) + 1

            if (i + 1) % 50 == 0 or (i + 1) == args.num_scenes:
                print(
                    f'[generate_verify_qa] progress {i+1}/{args.num_scenes} '
                    f'written={n_written} failed={n_failed} | '
                    f'{llm.format_usage()}',
                    file=sys.stderr,
                )

    print(
        f'[generate_verify_qa] done — written={n_written}/{args.num_scenes} '
        f'failed={n_failed}\n'
        f'  status_dist = {status_counter}\n'
        f'  target_dist = {target_counter}\n'
        f'  {llm.format_usage()}\n'
        f'  jsonl       = {args.output_jsonl}\n'
        f'  scenarios   = {scenarios_path}',
        file=sys.stderr,
    )
    print(
        '[generate_verify_qa] 다음 단계 (PNG 렌더링):\n'
        '  ~/isaacsim/python.sh training/lora_finetuning/scripts/render_verify_images.py \\\n'
        f'    --scenarios {scenarios_path} \\\n'
        '    --scene-usd sim/scenes/base_scene.usd \\\n'
        f'    --domain    {args.domain} \\\n'
        f'    --image-dir {args.image_dir}',
        file=sys.stderr,
    )
    if args.skip_render:
        print(
            '[generate_verify_qa] note: --skip-render 는 deprecated — 본 스크립트는 '
            '이제 항상 JSONL+sidecar 만 출력하고 PNG 는 render_verify_images.py '
            '가 담당. 플래그 무시됨.',
            file=sys.stderr,
        )


if __name__ == '__main__':
    main()
