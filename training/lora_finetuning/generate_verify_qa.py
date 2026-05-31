"""
generate_verify_qa.py

Cat-C — verify_result 시각 검증 시나리오 JSONL 합성.

배포 시 agent_node._call_verify_result 는 Gemma 를 한 번 더 호출하여
RGB + original_command → status JSON 매핑을 얻는다. 본 스크립트는 그 호출의
training 분포를 만든다. Cat-B(함수 호출)와 같은 LoRA 어댑터를 공유하며,
Unsloth FastVisionModel typed-array content 형식으로 출력한다.

출력 포맷 (JSONL — 한 줄 = 한 샘플):
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

freeze 결정 (2026-05-25, C2 통합 vision 포맷 결정 후):
  값과 구현은 아래 캐노니컬 위치에만 박혀 있다. 본 docstring 은 navigation 만
  한다 — 다른 곳에서 값을 재기재하지 말 것 (drift 방지).

  · system prompt            : robot_arm_pkg.prompts.VERIFY_SYSTEM_PROMPT
                                Cat-B 의 SYSTEM_PROMPT 와 별개. agent_node 의
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
  · Sim 렌더링 (미구현 stub)  : render_scenario_image() docstring
                                --skip-render 로 우회 가능. 학습 직전 본체
                                구현 + 재실행 필수.

CLI:
  python training/lora_finetuning/generate_verify_qa.py \\
    --domain         recycling \\
    --label-vocab    deploy/config/label_vocab.yaml \\
    --output-jsonl   training/lora_finetuning/data/recycling/qa_cat_c.jsonl \\
    --image-dir      training/lora_finetuning/data/recycling/verify_images/ \\
    --num-scenes     500 \\
    --external-llm   gpt-4o-mini
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
    '..', '..', 'deploy', 'ros2_ws', 'src', 'robot_arm_pkg',
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
# 변경 시 Sim 자산 (sim/assets/objects/<domain>/) 의 bin/holder 이름과 일치 시킬 것.
DESTINATIONS: Dict[str, Dict[str, str]] = {
    'recycling': {
        'can':     'the can bin',
        'plastic': 'the plastic bin',
        'paper':   'the paper bin',
    },
    'tableware': {
        'plate': 'the dish rack',
        'bowl':  'the dish rack',
        'cup':   'the cup holder',
        'fork':  'the utensil holder',
        'spoon': 'the utensil holder',
    },
}

# status × 5 템플릿. {target} = target_phrase, {dest} = destination.
# 변동성은 target_phrase (attribute × spatial) 가 주로 담당 — 템플릿 수가 적어도
# 동일 status 내 표면 다양성은 충분. 추가 시 dict 항목만 늘리면 된다.
REASONING_TEMPLATES: Dict[str, List[str]] = {
    VerifyStatus.SUCCESS: [
        '{target} is now inside {dest}',
        '{target} has been placed in {dest}',
        '{target} sits stably in {dest}',
        '{target} is correctly stored in {dest}',
        'the task is done — {target} is in {dest}',
    ],
    VerifyStatus.FAIL: [
        '{target} is still on the table, not in {dest}',
        '{target} is on the floor next to {dest}',
        '{target} ended up outside {dest}',
        '{target} was placed in the wrong bin, not {dest}',
        '{target} fell off before reaching {dest}',
    ],
    VerifyStatus.PARTIAL: [
        '{target} is balancing on the edge of {dest}',
        '{target} is half inside {dest} and half outside',
        '{target} fell partly over the rim of {dest}',
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
    p = argparse.ArgumentParser(description='Cat-C — 시각 검증 시나리오 JSONL 합성')
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
    - original_command: Cat-B 합성기(synthesize_user_command) 재사용 (LLM 1회 호출)
    - image path: image_dir 기준 '{scenario_id}.png'
    """
    status        = _weighted_choice(STATUS_WEIGHTS, rng)
    target_obj    = rng.choice(list(vocab['objects'].keys()))
    target_phrase = sample_description_for_object(vocab, target_obj, rng)
    destination   = _destination_of(args.domain, target_obj)

    original_command = synthesize_user_command(args.domain, vocab, llm, rng)
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


def build_sample(scenario: VerifyScenario, reasoning: str) -> dict:
    """Cat-C 형식 messages dict 1건 생성 (3-msg single-turn).

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
# Sim 렌더링 (NotImplementedError 스텁)
# ──────────────────────────────────────────────

def render_scenario_image(scenario: VerifyScenario) -> None:
    """Isaac Sim 으로 시나리오의 RGB 이미지를 scenario.image_abspath 에 저장.

    구현 시 외부 의존 / 결정 필요:
      - Isaac Sim standalone app 컨텍스트 (record_sim_teleop 의 SimInterface 재사용 가능)
      - Sim 자산: sim/assets/objects/<domain>/ USD + base scene
      - 카메라 셋업: deploy/calibration 의 intrinsics/extrinsics 적용
        (setup_sim_camera.py 와 같은 로직)
      - 객체 배치 정책 (scenario.status 별):
          SUCCESS  → target_object 를 correct destination 내부에 안정 배치
          FAIL     → target_object 를 wrong bin / 바닥 / 테이블 잔존 위치에 배치
          PARTIAL  → target_object 를 destination 가장자리, 기울임/걸침 자세
          UNCLEAR  → target_object 를 다른 객체로 가림 / 시야 가장자리 / 흐림
      - distractor: 같은 도메인의 다른 객체 1-2 개 무작위 배치 (다중 객체 변별 학습)
      - 카메라 capture → PNG (RGB) 로 scenario.image_abspath 저장 (mkdir 포함)

    Sim 자산 준비 (사용자 측 작업) 완료 후 본 함수 구현. 그 전까지는
    NotImplementedError — main 루프가 catch 하여 경고 후 다음 시나리오로 진행.
    """
    raise NotImplementedError(
        '[generate_verify_qa] render_scenario_image() 미구현 — Sim 자산 준비 후 작성. '
        '구조 검증 시에는 --skip-render 사용.'
    )


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

    print(
        f'[generate_verify_qa] start — domain={args.domain} '
        f'objects={list(vocab["objects"].keys())} '
        f'num_scenes={args.num_scenes} skip_render={args.skip_render} '
        f'image_dir={args.image_dir} output={args.output_jsonl}',
        file=sys.stderr,
    )

    n_written         = 0
    n_failed          = 0
    n_render_failed   = 0
    status_counter:   Dict[str, int] = {}
    target_counter:   Dict[str, int] = {}

    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
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

            # JSONL 먼저 기록 — 렌더 실패해도 구조는 보존
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            n_written += 1
            status_counter[scenario.status]      = status_counter.get(scenario.status, 0) + 1
            target_counter[scenario.target_object] = target_counter.get(scenario.target_object, 0) + 1

            # Sim 렌더링
            if not args.skip_render:
                try:
                    render_scenario_image(scenario)
                except NotImplementedError as e:
                    n_render_failed += 1
                    if n_render_failed == 1:
                        print(
                            f'[generate_verify_qa] 경고: render_scenario_image '
                            f'미구현 — JSONL 만 출력. {e}',
                            file=sys.stderr,
                        )
                except Exception as e:  # noqa: BLE001
                    n_render_failed += 1
                    print(
                        f'[generate_verify_qa] render 실패 '
                        f'({scenario.scenario_id}): {type(e).__name__}: {e}',
                        file=sys.stderr,
                    )

            if (i + 1) % 50 == 0 or (i + 1) == args.num_scenes:
                print(
                    f'[generate_verify_qa] progress {i+1}/{args.num_scenes} '
                    f'written={n_written} failed={n_failed} '
                    f'render_failed={n_render_failed} | {llm.format_usage()}',
                    file=sys.stderr,
                )

    print(
        f'[generate_verify_qa] done — written={n_written}/{args.num_scenes} '
        f'failed={n_failed} render_failed={n_render_failed}\n'
        f'  status_dist = {status_counter}\n'
        f'  target_dist = {target_counter}\n'
        f'  {llm.format_usage()} | output={args.output_jsonl}',
        file=sys.stderr,
    )
    if args.skip_render or n_render_failed > 0:
        print(
            '[generate_verify_qa] 주의: PNG 가 없거나 미완성 — 본격 학습 전 '
            'Sim 렌더링 본체 구현 후 재실행 필수.',
            file=sys.stderr,
        )


if __name__ == '__main__':
    main()
