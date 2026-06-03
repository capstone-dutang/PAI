"""
generate_language_labels.py

Sim 데이터 생성 후속 단계 — generate_act_dataset.py가 만든 Sim LeRobotDataset의 각
에피소드에 자연어 타겟 묘사를 부여하고, 임베딩 채널을 채운다.

두 가지 라벨 소스를 혼합한다 (README Sim 데이터 생성 참조):
  소스 1: Gemma 4 E2B 자체 라벨링
          - Sim에서 렌더링한 RGB 이미지를 Gemma에 입력
          - 배포 시 사용할 동일 모델로 라벨 생성 → 분포 일치
  소스 2: 외부 LLM Paraphrase
          - 소스 1의 라벨을 외부 LLM(GPT-4 등)에 입력하여 다양한 표현 생성
          - 표현 robustness 강화

비율: 에피소드당 Gemma 라벨 1개 + 외부 LLM paraphrase 2~3개. 학습 시 random
augmentation으로 무작위 추출.

라벨 형식 — 확정 사항:
  영어 반구조화 템플릿  "the {attribute} {object} {spatial}"
  예: "the blue can on the right side of the table"
  vocabulary: deploy/config/label_vocab.yaml과 일치
  (Sim/Real/배포 분포 일치 강제)

임베딩 모델 — 확정 사항:
  paraphrase-multilingual-MiniLM-L12-v2 (D=384)
  embedding_node / postprocess_real_labels.py와 동일 모델 (강제)

CLI 사용 예:
  python training/imitation_data/scripts/generate_language_labels.py \\
    --dataset                dataset_sim/ \\
    --gemma-server           http://localhost:8080 \\
    --label-vocab            deploy/config/label_vocab.yaml \\
    --domain                 recycling \\
    --external-llm           gpt-4 \\
    --paraphrase-per-episode 3 \\
    --embed-model            sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

미구현 항목 (구현 시 결정 필요):
  - Gemma /v1/chat/completions 페이로드 형식 (이미지 base64 + prompt)
  - 외부 LLM 클라이언트 인터페이스 (OpenAI / Anthropic / 로컬 LLM)
  - LeRobotDataset 0.4.4 features에 paraphrase 풀 + 임베딩 풀 반영하는 방법
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List


DEFAULT_EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EXPECTED_EMBED_DIM = 384


# ──────────────────────────────────────────────
# Gemma 자체 라벨링 system prompt — 템플릿 강제
# ──────────────────────────────────────────────

GEMMA_LABEL_SYSTEM_PROMPT = (
    'You are labeling a robot grasp scene. Output ONE English description of '
    'the object the robot will grasp, in this exact format:\n'
    '\n'
    '    "the {attribute} {object} {spatial}"\n'
    '\n'
    'Constraints:\n'
    '  - 5 to 15 words total.\n'
    '  - object: one of {can, plastic, paper, plate, fork, spoon} (use the actual class in the image).\n'
    '  - attribute: ONE discriminative attribute (color or material).\n'
    '  - spatial: a location/relation phrase, e.g.\n'
    '      "on the left side of the table",\n'
    '      "on the right side of the table",\n'
    '      "in the center of the table",\n'
    '      "next to the plastic".\n'
    '  - English only. No first-person, no action verbs.\n'
    '  - Output ONLY the description string. No quotes, no explanation.\n'
)


# ──────────────────────────────────────────────
# 외부 LLM paraphrase system prompt
# ──────────────────────────────────────────────

PARAPHRASE_SYSTEM_PROMPT = (
    'You generate paraphrases of a robot-grasp target description. '
    'Preserve the SAME object, attribute, and spatial location; vary only the surface form. '
    'English only. Each paraphrase 5-15 words. '
    'One paraphrase per line, no numbering, no quotes.'
)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@dataclass
class LabelArgs:
    dataset:                str
    gemma_server:           str
    label_vocab:            str
    domain:                 str
    external_llm:           str
    paraphrase_per_episode: int
    embed_model:            str


def parse_args() -> LabelArgs:
    p = argparse.ArgumentParser(description='Sim 데이터셋 자연어 라벨링')
    p.add_argument('--dataset',                required=True,
                   help='generate_act_dataset.py가 생성한 Sim LeRobotDataset 경로')
    p.add_argument('--gemma-server',           default='http://localhost:8080',
                   help='Gemma llama-server 주소 (자체 라벨링용)')
    p.add_argument('--label-vocab',            default='deploy/config/label_vocab.yaml',
                   help='vocabulary 검증용 (Sim/Real 분포 일치)')
    p.add_argument('--domain',                 required=True,
                   help='현재 라벨링 도메인')
    p.add_argument('--external-llm',           default='gpt-4',
                   help='paraphrase 생성에 사용할 외부 LLM 식별자')
    p.add_argument('--paraphrase-per-episode', type=int, default=3,
                   help='에피소드당 외부 LLM paraphrase 개수')
    p.add_argument('--embed-model',            default=DEFAULT_EMBED_MODEL,
                   help='임베딩 모델 (데이터 수집 / 배포와 동일 모델 강제)')
    a = p.parse_args()
    return LabelArgs(
        dataset                = os.path.expanduser(a.dataset),
        gemma_server           = a.gemma_server,
        label_vocab            = os.path.expanduser(a.label_vocab),
        domain                 = a.domain,
        external_llm           = a.external_llm,
        paraphrase_per_episode = a.paraphrase_per_episode,
        embed_model            = a.embed_model,
    )


# ──────────────────────────────────────────────
# Gemma 자체 라벨링
# ──────────────────────────────────────────────

def gemma_label_episode(rgb_frames, gemma_server: str) -> str:
    """
    에피소드를 대표하는 RGB 프레임을 Gemma에 입력하여 자연어 타겟 묘사 1건 산출.

    시스템 프롬프트는 GEMMA_LABEL_SYSTEM_PROMPT로 고정 — 출력 분포가
    record_real_teleop의 canonical 라벨과 일치하도록 보장.

    TODO: 구현 시 결정
      - 어떤 프레임을 Gemma에 줄지 (첫 프레임 / pre_grasp 시점 / 중간 K장 평균)
      - llama-server /v1/chat/completions 호출 형식 (base64 이미지 임베드)
      - 응답 검증: 템플릿 정규식과 매칭 안 되면 재시도 / vocabulary 외 단어
        포함 시 재시도 (label_vocab.yaml 사용)
    """
    raise NotImplementedError(
        '[generate_language_labels] gemma_label_episode() 미구현'
    )


# ──────────────────────────────────────────────
# 외부 LLM paraphrase
# ──────────────────────────────────────────────

def paraphrase_with_external_llm(
    base_label:   str,
    external_llm: str,
    n:            int,
) -> List[str]:
    """
    Gemma 라벨 1건을 받아 외부 LLM으로 paraphrase n개 생성.

    시스템 프롬프트는 PARAPHRASE_SYSTEM_PROMPT 고정. 의미는 보존 / 표현만 변형.

    TODO: 구현 시 결정
      - 외부 LLM 종류 확정 (OpenAI gpt-4 / Anthropic Claude / 로컬 vllm 등)
      - API 키 관리 (환경변수)
      - 안전장치: paraphrase가 의미를 바꾸지 않도록 검증
    """
    raise NotImplementedError(
        '[generate_language_labels] paraphrase_with_external_llm() 미구현'
    )


# ──────────────────────────────────────────────
# 임베딩 채널 채우기 (옵션 (a) 직접 로드 — README 권장)
# ──────────────────────────────────────────────

def embed_labels(labels: List[str], embed_model: str) -> List[List[float]]:
    """라벨 리스트 → 임베딩 벡터 리스트. embedding_node와 동일 모델."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embed_model)
    dim = int(model.get_sentence_embedding_dimension())
    if dim != EXPECTED_EMBED_DIM:
        raise RuntimeError(
            f'임베딩 차원 불일치 — model dim={dim}, 예상={EXPECTED_EMBED_DIM}. '
            f'모델이 paraphrase-multilingual-MiniLM-L12-v2인지 확인.'
        )
    arr = model.encode(labels, normalize_embeddings=True, convert_to_numpy=True)
    return arr.astype('float32').tolist()


# ──────────────────────────────────────────────
# 데이터셋 갱신
# ──────────────────────────────────────────────

def update_dataset_with_labels(args: LabelArgs) -> None:
    """
    LeRobotDataset의 각 에피소드에 (canonical, paraphrases, embeddings) 부여.

    TODO: 구현 시 결정
      - LeRobotDataset 0.4.4의 frame/episode metadata API 확인
      - paraphrase 다수 임베딩을 어떻게 저장할지 (학습 시 random pick 형태)
      - postprocess_real_labels.py와 동일한 저장 스키마를 사용해야 함
        (ACT Co-training 시 Sim/Real을 같은 방식으로 random pick)
    """
    raise NotImplementedError(
        '[generate_language_labels] update_dataset_with_labels() 미구현'
    )


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    update_dataset_with_labels(args)


if __name__ == '__main__':
    main()
