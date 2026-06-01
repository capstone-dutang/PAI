"""
llm_client.py

OpenAI Chat Completions API 의 얇은 wrapper.
function_call (generate_function_call_qa) / verify (generate_verify_qa) /
sim/generate_language_labels 의 paraphrase 가 공유.

freeze (2026-05-25):
  - 기본 모델: gpt-4o-mini (학습 데이터 합성용)
  - API key: OPENAI_API_KEY 환경변수에서 읽음
  - 재시도: RateLimit / Connection / Timeout 에러에 한해 지수 백오프

전제:
  pip install "openai>=1.0"
  export OPENAI_API_KEY=...

비용 추정은 best-effort (PRICING_PER_M_TOKENS 표 기준). 운영 시점 가격 변동 가능 —
정확한 청구는 OpenAI dashboard 참조.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional


# USD per 1M tokens — 2025-Q1 기준. 운영 시점 변동 가능.
PRICING_PER_M_TOKENS: Dict[str, Dict[str, float]] = {
    'gpt-4o-mini':  {'input': 0.15, 'output': 0.60},
    'gpt-4o':       {'input': 2.50, 'output': 10.00},
    'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
    'o4-mini':      {'input': 1.10, 'output': 4.40},
}


class LLMClient:
    """OpenAI chat 호출 + 누적 토큰/비용 로깅."""

    def __init__(self, model: str = 'gpt-4o-mini', max_retries: int = 5):
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError(
                'OPENAI_API_KEY 환경변수가 설정되지 않음. '
                '`export OPENAI_API_KEY=...` 후 다시 실행하세요.'
            )
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                'openai SDK 미설치. `pip install "openai>=1.0"` 후 다시 실행하세요.'
            ) from e
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ── chat 호출 ─────────────────────────────────

    def chat(
        self,
        messages:    List[Dict[str, str]],
        json_mode:   bool             = False,
        temperature: float            = 0.7,
        max_tokens:  Optional[int]    = None,
        model:       Optional[str]    = None,
    ) -> str:
        """Chat completion 1회 호출 → 응답 본문 문자열 반환.

        json_mode=True 시 response_format={'type':'json_object'} 강제.
        호출 후 usage 토큰을 누적기에 합산한다.
        """
        from openai import (  # noqa: PLC0415
            APIConnectionError, APITimeoutError, RateLimitError,
        )
        kwargs: dict = {
            'model':       model or self.model,
            'messages':    messages,
            'temperature': temperature,
        }
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        if json_mode:
            kwargs['response_format'] = {'type': 'json_object'}

        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                if resp.usage is not None:
                    self.total_input_tokens  += resp.usage.prompt_tokens
                    self.total_output_tokens += resp.usage.completion_tokens
                return resp.choices[0].message.content or ''
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt + 1 == self.max_retries:
                    raise
                delay = min(60, 2 ** attempt)
                print(
                    f'[llm_client] {type(e).__name__} '
                    f'(attempt {attempt+1}/{self.max_retries}) — {delay}s 후 재시도',
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise RuntimeError('unreachable')

    def chat_json(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        """chat(json_mode=True) + json.loads 한 번에. 파싱 실패 시 ValueError."""
        text = self.chat(messages, json_mode=True, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f'LLM 응답이 유효 JSON 아님: {text[:200]!r}'
            ) from e

    # ── 사용량 / 비용 ────────────────────────────

    def total_usage(self) -> dict:
        """누적 토큰 사용량 + 추정 비용 (USD)."""
        pricing = PRICING_PER_M_TOKENS.get(self.model, {'input': 0.0, 'output': 0.0})
        cost_in  = self.total_input_tokens  * pricing['input']  / 1_000_000
        cost_out = self.total_output_tokens * pricing['output'] / 1_000_000
        return {
            'model':              self.model,
            'input_tokens':       self.total_input_tokens,
            'output_tokens':      self.total_output_tokens,
            'total_tokens':       self.total_input_tokens + self.total_output_tokens,
            'estimated_cost_usd': cost_in + cost_out,
        }

    def format_usage(self) -> str:
        u = self.total_usage()
        return (
            f"model={u['model']} "
            f"tokens={u['input_tokens']}+{u['output_tokens']}={u['total_tokens']} "
            f"cost≈${u['estimated_cost_usd']:.4f}"
        )
