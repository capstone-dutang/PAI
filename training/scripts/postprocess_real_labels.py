"""
postprocess_real_labels.py

실행 위치: Jetson 또는 외부 PC (ROS2 의존 없음, pure Python).

데이터 수집 후처리 — record_real_teleop.py가 수집한 LeRobotDataset 에피소드들에
외부 LLM으로 paraphrase를 추가하고, paraphrase 풀의 임베딩을 함께 계산하여
LeRobotDataset frame/episode metadata에 기록한다.

record_real_teleop.py는 실시간 수집 속도 유지를 위해 canonical 라벨 1개와 그
임베딩만 저장한다. 본 스크립트는 수집 종료 후 일괄 실행되어:
  1) labels_<repo>.jsonl 의 canonical 라벨을 외부 LLM에 입력
  2) paraphrase N개 생성
  3) paraphrase 전부에 대해 임베딩 계산 (Sim과 동일 모델)
  4) LeRobotDataset에 augmentation 풀로 반영
     → ACT 학습 시 무작위 추출하여 robustness 강화

CLI 사용 예:
  python postprocess_real_labels.py \\
    --dataset                ~/datasets/myname/dataset_real \\
    --labels-jsonl           labels_myname_dataset_real.jsonl \\
    --external-llm           gpt-4 \\
    --paraphrase-per-episode 3 \\
    --embed-model            sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

미구현 항목 (외부 LLM 클라이언트 / LeRobot 0.4.4 metadata API 확정 후):
  - synthesize_paraphrases() : 외부 LLM 호출부
  - apply_to_dataset()       : LeRobotDataset에 paraphrase 풀 + 임베딩 풀 반영

입출력 일관성 강제:
  - canonical은 영어 "the {attribute} {object} {spatial}" 형식
  - paraphrase는 의미는 보존하되 표현만 변형
    (예: "the blue can on the right" / "pick the blue can placed to the right" /
         "the right-side blue can")
  - paraphrase도 영어 고정 (다국어 임베딩 모델이지만 학습 분포 일치 위해)
  - 임베딩 D = 384 (paraphrase-multilingual-MiniLM-L12-v2)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


DEFAULT_EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EXPECTED_EMBED_DIM = 384


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@dataclass
class PostArgs:
    dataset:                str
    labels_jsonl:           str
    external_llm:           str
    paraphrase_per_episode: int
    embed_model:            str
    output_dir:             str


def parse_args() -> PostArgs:
    p = argparse.ArgumentParser(
        description='Real teleop 라벨 paraphrase + 임베딩 후처리'
    )
    p.add_argument('--dataset',      required=True,
                   help='record_real_teleop이 생성한 LeRobotDataset 경로')
    p.add_argument('--labels-jsonl', required=True,
                   help='record_real_teleop이 기록한 라벨 메타 JSONL')
    p.add_argument('--external-llm', default='gpt-4',
                   help='paraphrase 생성 외부 LLM 식별자')
    p.add_argument('--paraphrase-per-episode', type=int, default=3,
                   help='에피소드당 paraphrase 개수')
    p.add_argument('--embed-model', default=DEFAULT_EMBED_MODEL,
                   help='임베딩 모델 (Sim과 일치 필수)')
    p.add_argument('--output-dir',  default='',
                   help='paraphrase + 임베딩 결과를 별도 저장할 디렉터리 (선택)')
    a = p.parse_args()
    return PostArgs(
        dataset                = os.path.expanduser(a.dataset),
        labels_jsonl           = os.path.expanduser(a.labels_jsonl),
        external_llm           = a.external_llm,
        paraphrase_per_episode = a.paraphrase_per_episode,
        embed_model            = a.embed_model,
        output_dir             = os.path.expanduser(a.output_dir) if a.output_dir else '',
    )


# ──────────────────────────────────────────────
# 라벨 JSONL 로드
# ──────────────────────────────────────────────

def load_labels_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ──────────────────────────────────────────────
# 외부 LLM paraphrase (미구현)
# ──────────────────────────────────────────────

PARAPHRASE_SYSTEM_PROMPT = (
    'You generate paraphrases of a target description for a robot grasp task. '
    'Constraints: keep the SAME object, attribute, and spatial location; '
    'only vary the surface form. Output English only. '
    'Each paraphrase 5-15 words. No first-person, no action verbs other than '
    'optional "pick"/"grab" hints. One paraphrase per line, no numbering.'
)


def synthesize_paraphrases(
    canonical:    str,
    n:            int,
    external_llm: str,
) -> List[str]:
    """canonical 라벨 1개 → 의미 보존 paraphrase n개.

    TODO: 외부 LLM 클라이언트 구현 시
      - OpenAI / Anthropic / 로컬 vllm 중 하나 확정
      - API 키는 환경변수 (예: OPENAI_API_KEY)
      - PARAPHRASE_SYSTEM_PROMPT를 system role로 사용
      - 응답에서 빈 줄/번호 제거
      - 결과를 deduplicate
    """
    raise NotImplementedError(
        '[postprocess_real_labels] synthesize_paraphrases() 미구현 '
        '(외부 LLM 클라이언트 확정 필요)'
    )


# ──────────────────────────────────────────────
# 임베딩 (직접 로드, sim/generate_language_labels와 동일 전략)
# ──────────────────────────────────────────────

def load_embed_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    dim = int(model.get_sentence_embedding_dimension())
    if dim != EXPECTED_EMBED_DIM:
        raise RuntimeError(
            f'임베딩 차원 불일치 — model dim={dim}, 예상={EXPECTED_EMBED_DIM}. '
            f'모델이 paraphrase-multilingual-MiniLM-L12-v2인지 확인.'
        )
    return model


def embed_texts(model, texts: List[str]) -> List[List[float]]:
    arr = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return arr.astype('float32').tolist()


# ──────────────────────────────────────────────
# 데이터셋 반영 (미구현 — LeRobot 0.4.4 metadata API 확정 후)
# ──────────────────────────────────────────────

def apply_to_dataset(
    dataset_path:    str,
    augmentation:    Dict[int, dict],
) -> None:
    """
    augmentation: {episode_idx: {'labels': [str, ...], 'embeddings': [[float]...]}}

    TODO: LeRobotDataset 0.4.4의 episode metadata API 호출.
          학습 시 ACT Co-training 스크립트에서 random pick으로 사용 가능한 형식으로 저장.
    """
    raise NotImplementedError(
        '[postprocess_real_labels] apply_to_dataset() 미구현 '
        '(LeRobotDataset 0.4.4 metadata API 호출 작성 필요)'
    )


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    records = load_labels_jsonl(args.labels_jsonl)
    if not records:
        print(f'[postprocess] 라벨 레코드 없음: {args.labels_jsonl}')
        return

    model = load_embed_model(args.embed_model)

    augmentation: Dict[int, dict] = {}
    for rec in records:
        ep = int(rec['episode'])
        canonical = rec['canonical']

        paraphrases = synthesize_paraphrases(
            canonical=canonical,
            n=args.paraphrase_per_episode,
            external_llm=args.external_llm,
        )
        all_labels = [canonical] + paraphrases
        all_embeds = embed_texts(model, all_labels)
        augmentation[ep] = {
            'labels':     all_labels,
            'embeddings': all_embeds,
            'slots':      {k: rec[k] for k in ('object', 'attribute', 'spatial', 'domain')},
        }

    # 보조 산출물 — 별도 디렉터리에도 저장 (디버그/재현용)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(args.output_dir, 'paraphrase_pool.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for ep, payload in sorted(augmentation.items()):
                f.write(json.dumps(
                    {'episode': ep, **{k: v for k, v in payload.items() if k != 'embeddings'}},
                    ensure_ascii=False,
                ) + '\n')
        print(f'[postprocess] paraphrase 풀 저장: {out_path}')

    apply_to_dataset(args.dataset, augmentation)
    print(f'[postprocess] 완료 — episodes: {len(augmentation)}')


if __name__ == '__main__':
    main()
