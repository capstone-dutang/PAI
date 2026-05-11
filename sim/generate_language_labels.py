"""
generate_language_labels.py

Part B 후속 단계 — generate_act_dataset.py가 만든 Sim LeRobotDataset의 각
에피소드에 자연어 타겟 묘사를 부여하고, 임베딩 채널을 채운다.

두 가지 라벨 소스를 혼합한다 (README Part B 참조):
  소스 1: Gemma 4 E2B 자체 라벨링
          - Sim에서 렌더링한 RGB 이미지를 Gemma에 입력
          - 배포 시 사용할 동일 모델로 라벨 생성 → 분포 일치
  소스 2: 외부 LLM Paraphrase
          - 소스 1의 라벨을 외부 LLM(GPT-4 등)에 입력하여 다양한 표현 생성
          - 표현 robustness 강화

비율: 에피소드당 Gemma 라벨 1개 + 외부 LLM paraphrase 2~3개. 학습 시 random
augmentation으로 무작위 추출.

CLI 사용 예 (README 명시):
  python generate_language_labels.py \\
    --dataset                dataset_sim/ \\
    --gemma-server           http://localhost:8080 \\
    --external-llm           gpt-4 \\
    --paraphrase-per-episode 3

미구현 항목 (구현 시 결정 필요):
  - Gemma /v1/chat/completions 페이로드 형식 (이미지 base64 + prompt)
  - 외부 LLM 클라이언트 인터페이스 (OpenAI / Anthropic / 로컬 LLM)
  - 임베딩 채널 채우기 방법:
      옵션 (a) embedding_node와 동일한 모델을 본 스크립트에서 직접 로드
              (모델 가중치 1회 로드 후 batch 처리 → 빠름)
      옵션 (b) ROS2 embedding_node에 의존 (동일 모델 보장은 자동, 실행 환경 복잡)
      → README는 (a)를 암시. 모델 확정 후 결정.
  - LeRobotDataset features에 다중 라벨/임베딩을 어떻게 저장할지
    (문자열 metadata vs frame당 임베딩 vs 에피소드당 다수 임베딩 + 학습 시 random pick)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@dataclass
class LabelArgs:
    dataset:                str
    gemma_server:           str
    external_llm:           str
    paraphrase_per_episode: int


def parse_args() -> LabelArgs:
    p = argparse.ArgumentParser(description='Part B — Sim 데이터셋 자연어 라벨링')
    p.add_argument('--dataset',                required=True,
                   help='generate_act_dataset.py가 생성한 Sim LeRobotDataset 경로')
    p.add_argument('--gemma-server',           default='http://localhost:8080',
                   help='Gemma llama-server 주소 (자체 라벨링용)')
    p.add_argument('--external-llm',           default='gpt-4',
                   help='paraphrase 생성에 사용할 외부 LLM 식별자')
    p.add_argument('--paraphrase-per-episode', type=int, default=3,
                   help='에피소드당 외부 LLM paraphrase 개수')
    a = p.parse_args()
    return LabelArgs(
        dataset                = a.dataset,
        gemma_server           = a.gemma_server,
        external_llm           = a.external_llm,
        paraphrase_per_episode = a.paraphrase_per_episode,
    )


# ──────────────────────────────────────────────
# Gemma 자체 라벨링
# ──────────────────────────────────────────────

def gemma_label_episode(rgb_frames, gemma_server: str) -> str:
    """
    에피소드를 대표하는 RGB 프레임을 Gemma에 입력하여 자연어 타겟 묘사 1건 산출.

    TODO: 구현 시 결정
      - 어떤 프레임을 Gemma에 줄지 (첫 프레임 / pre_grasp 시점 / 중간 K장 평균)
      - prompt 템플릿 — multi_target_scenes의 target_rotation과 일치하는 톤이
        되도록 가이드해야 학습 데이터 분포가 균일
      - llama-server /v1/chat/completions 호출 형식 (base64 이미지 임베드)
    """
    raise NotImplementedError(
        "[generate_language_labels] gemma_label_episode() 미구현"
    )


# ──────────────────────────────────────────────
# 외부 LLM paraphrase
# ──────────────────────────────────────────────

def paraphrase_with_external_llm(
    base_label: str,
    external_llm: str,
    n: int,
) -> List[str]:
    """
    Gemma 라벨 1건을 받아 외부 LLM으로 paraphrase n개 생성.

    TODO: 구현 시 결정
      - 외부 LLM 종류 확정 (OpenAI gpt-4 / Anthropic Claude / 로컬 vllm 등)
      - API 키 관리 (환경변수)
      - 톤 제어 prompt — 단순 표현/구어/정중 등 다양성 확보
      - 안전장치: paraphrase가 의미를 바꾸지 않도록 검증
    """
    raise NotImplementedError(
        "[generate_language_labels] paraphrase_with_external_llm() 미구현"
    )


# ──────────────────────────────────────────────
# 임베딩 채널 채우기
# ──────────────────────────────────────────────

def embed_labels(labels: List[str]) -> List[List[float]]:
    """
    라벨 리스트를 임베딩 벡터 리스트로 변환.

    TODO: 임베딩 모델 확정 후 구현 (README 미확정 #1)
      옵션 (a) 직접 로드:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(<path>)
        return model.encode(labels, normalize_embeddings=True).tolist()

      옵션 (b) ROS2 embedding_node 호출:
        rclpy 클라이언트로 /embed_text 서비스를 한 라벨씩 호출
        (ROS2 환경 의존 — 외부 PC에서 실행 시 부담)

      Part D/E/F와 동일한 모델·동일 D 사용이 강제 조건이다.
    """
    raise NotImplementedError(
        "[generate_language_labels] embed_labels() 미구현 (임베딩 모델 미확정)"
    )


# ──────────────────────────────────────────────
# 데이터셋 갱신
# ──────────────────────────────────────────────

def update_dataset_with_labels(dataset_path: str) -> None:
    """
    LeRobotDataset의 각 에피소드에 라벨 + 임베딩을 부여한다.

    TODO: 구현 시 결정
      - LeRobotDataset 0.4.4의 frame/episode metadata API 확인
      - language_embedding 채널을 frame 단위로 채울지 (에피소드 내 동일값)
        에피소드 metadata로만 저장할지
      - paraphrase 다수 임베딩을 어떻게 저장할지 (학습 시 random pick 가능 형태)
    """
    raise NotImplementedError(
        "[generate_language_labels] update_dataset_with_labels() 미구현"
    )


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    update_dataset_with_labels(args.dataset)
    # 실제 호출 흐름은 update_dataset_with_labels 구현 시점에 정의된다.


if __name__ == '__main__':
    main()
