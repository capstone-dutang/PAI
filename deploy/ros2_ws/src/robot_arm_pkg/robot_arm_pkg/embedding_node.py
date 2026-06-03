"""
embedding_node.py

자연어 타겟 묘사를 벡터 임베딩으로 변환하는 ROS2 서비스 노드.

agent_node와 record_real_teleop.py가 이 노드의 /embed_text 서비스를 호출하여
SubtaskCommand.language_embedding / observation.language_embedding 채널을
미리 채운다. ACT는 추론 시점에 임베딩 변환 지연이 없다.

환경:
  Python   3.10
  ROS2     Humble (Ubuntu 22.04 LTS)

서비스 서버:
  /embed_text (robot_arm_msgs/EmbedText)
    request : string text
    response: float32[] embedding, int32 dim

파라미터:
  model_name (str) — sentence-transformers 모델 식별자 또는 로컬 디렉터리 경로
                     기본: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                     (배포 시에는 ~/models/embedding/ 에 사전 다운로드 권장)
  device     (str) — 'cpu' | 'cuda' (기본: 자동 감지)
  normalize  (bool)— True면 L2 정규화 (코사인 유사도 학습 시 권장, 기본 True)

임베딩 모델 — 확정 사항:
  paraphrase-multilingual-MiniLM-L12-v2
    - 차원 D = 384
    - 크기 ~120MB (Jetson Orin Nano 8GB 메모리 표 내 ~50–300MB 슬롯)
    - 다국어 (50+ 언어) — 영어 description 고정 + 한국어 사용자 명령 둘 다 대응
    - paraphrase 학습되어 Sim 외부 LLM paraphrase / Real 후처리 paraphrase
      증강 전략과 시너지

  모델 변경 시:
    - 데이터 수집 데이터셋 재라벨링 필수
    - ACT 학습 obs 차원 변경 → 재학습 필수
    - SubtaskCommand.language_embedding 길이 변경
    - training/imitation_data/scripts/generate_language_labels.py /
      training/imitation_data/scripts/postprocess_real_labels.py도 동일 모델로 변경
"""

from __future__ import annotations

import os
from typing import List, Optional

import rclpy
from rclpy.node import Node

from robot_arm_msgs.srv import EmbedText


DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


class EmbeddingNode(Node):

    def __init__(self):
        super().__init__('embedding_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('model_name', DEFAULT_MODEL)
        self.declare_parameter('device', '')         # '' → 자동 감지
        self.declare_parameter('normalize', True)

        self.model_name: str = self.get_parameter('model_name').value
        self.device:     str = self.get_parameter('device').value
        self.normalize: bool = self.get_parameter('normalize').value

        # ── 모델 로드 ─────────────────────────────
        self._model = None
        self._dim: Optional[int] = None
        self._load_model()

        # ── 서비스 ────────────────────────────────
        self.create_service(EmbedText, '/embed_text', self._embed_callback)

        self.get_logger().info(
            f'embedding_node 시작 | model: {self.model_name} | '
            f'dim: {self._dim if self._dim is not None else "FAILED"} | '
            f'device: {self._resolved_device()}'
        )

    # ── 모델 로드 ─────────────────────────────────

    def _resolved_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            return 'cpu'

    def _load_model(self) -> None:
        """
        sentence-transformers 모델을 로드하고 self._dim을 결정한다.

        model_name이 로컬 디렉터리이면 그대로 로드, 아니면 HF Hub에서 다운로드.
        Jetson 배포 시에는 사전 다운로드를 권장 (오프라인 환경 대비).
        """
        import traceback

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            self.get_logger().error(
                f'[embedding_node] sentence-transformers import 실패 (ImportError): {e}. '
                'requirements.txt 확인 후 pip install -r requirements.txt 실행.'
            )
            return
        except Exception as e:
            # NumPy/scipy ABI 충돌 등 ImportError 외 import 실패도 진단 가능하게 노출.
            self.get_logger().error(
                f'[embedding_node] sentence-transformers import 중 예외 '
                f'({type(e).__name__}): {e}\n{traceback.format_exc()}'
            )
            return

        device = self._resolved_device()
        try:
            self._model = SentenceTransformer(self.model_name, device=device)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception as e:
            self.get_logger().error(
                f'[embedding_node] 모델 로드 실패 ({self.model_name}, '
                f'{type(e).__name__}): {e}\n{traceback.format_exc()}'
            )
            self._model = None
            self._dim = None

    # ── 추론 ──────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        """텍스트 → 임베딩 벡터(float32 list)."""
        vec = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vec.astype('float32').tolist()

    # ── 서비스 콜백 ───────────────────────────────

    def _embed_callback(
        self,
        request:  EmbedText.Request,
        response: EmbedText.Response,
    ) -> EmbedText.Response:
        """
        /embed_text 서비스 핸들러.

        모델 미로드 또는 빈 입력 시 빈 응답(dim=0)을 반환한다.
        호출 측은 dim==0을 오류 신호로 해석한다.
        """
        if self._model is None or self._dim is None:
            self.get_logger().warn(
                '[embedding_node] 모델 미로드 — 빈 응답 반환 (dim=0)',
                throttle_duration_sec=5.0,
            )
            response.embedding = []
            response.dim = 0
            return response

        text = (request.text or '').strip()
        if not text:
            self.get_logger().warn('[embedding_node] 빈 텍스트 요청 — dim=0 반환')
            response.embedding = []
            response.dim = 0
            return response

        try:
            vec = self._embed(text)
        except Exception as e:
            self.get_logger().error(f'[embedding_node] 추론 실패: {e}')
            response.embedding = []
            response.dim = 0
            return response

        response.embedding = vec
        response.dim = self._dim
        return response


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = EmbeddingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
