"""
convert_function_call_to_vision_format.py

기존 function_call JSONL (string content + tool role) 을 Unsloth FastVisionModel
학습용 typed-array content + tool→user 형식으로 변환한다.

freeze (2026-05-25): function_call / verify 통합 학습 (FastVisionModel) 결정에 따른
1회용 변환기. 신규 합성은 generate_function_call_qa.py 가 이미 신 형식으로
직접 생성하므로 추후 변환 불필요.

변환 규칙:
  1) 모든 role 의 content 가 string 이면 [{"type":"text","text": <원문>}] 로 wrap
  2) "tool" role 은 "user" role 로 변환, text 앞에 "[tool_result] " prefix
     (Unsloth FastVisionModel + UnslothVisionDataCollator 는 tool role 미지원 —
      user 로 변환 시 agent_node 의 런타임 OpenAI-style feed 와도 일치)
  3) 메시지 순서·개수 보존
  4) 이미 typed-array 인 content 는 그대로 유지 (멱등 변환)

CLI:
  python training/lora_finetuning/scripts/convert_function_call_to_vision_format.py \\
    --input  training/lora_finetuning/data/recycling/recycling_function_call_qa.jsonl \\
    --output training/lora_finetuning/data/recycling/recycling_function_call_qa.vision.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Union


TOOL_RESULT_PREFIX = '[tool_result] '


def _typed_text(text: str) -> List[Dict[str, str]]:
    return [{'type': 'text', 'text': text}]


def convert_message(msg: dict) -> dict:
    role    = msg.get('role')
    content: Union[str, list] = msg.get('content', '')

    # 이미 typed-array → 그대로 (멱등)
    if isinstance(content, list):
        out_role = 'user' if role == 'tool' else role
        if role == 'tool':
            # tool role 인데 content 가 typed-array 인 케이스: text 항목에 prefix 부여
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    new_content.append({
                        'type': 'text',
                        'text': TOOL_RESULT_PREFIX + item.get('text', ''),
                    })
                else:
                    new_content.append(item)
            return {'role': out_role, 'content': new_content}
        return {'role': out_role, 'content': content}

    # string content
    if role == 'tool':
        return {
            'role':    'user',
            'content': _typed_text(TOOL_RESULT_PREFIX + str(content)),
        }
    return {'role': role, 'content': _typed_text(str(content))}


def convert_sample(sample: dict) -> dict:
    return {'messages': [convert_message(m) for m in sample.get('messages', [])]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='function_call JSONL string→typed-array + tool→user 변환 (1회용)'
    )
    p.add_argument('--input',  required=True, help='원본 function_call JSONL')
    p.add_argument('--output', required=True, help='변환 결과 JSONL')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path  = os.path.expanduser(args.input)
    out_path = os.path.expanduser(args.output)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_in = n_out = n_tool_to_user = 0
    with open(in_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            sample = json.loads(line)
            n_tool_to_user += sum(
                1 for m in sample.get('messages', []) if m.get('role') == 'tool'
            )
            converted = convert_sample(sample)
            fout.write(json.dumps(converted, ensure_ascii=False) + '\n')
            n_out += 1

    print(
        f'[convert_function_call_to_vision_format] '
        f'{in_path} → {out_path}: samples={n_in} (tool→user 변환={n_tool_to_user})',
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
