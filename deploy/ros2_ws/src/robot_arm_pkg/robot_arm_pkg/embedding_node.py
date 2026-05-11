"""
embedding_node.py

자연어 타겟 묘사를 벡터 임베딩으로 변환하는 ROS2 서비스 노드.

agent_node와 record_teleop.py가 이 노드의 /embed_text 서비스를 호출하여
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
  model_path (str) — 임베딩 모델 디렉터리 또는 파일 경로
                     기본: ~/models/embedding/

설계 메모 — 임베딩 차원 D:
  D는 이 노드가 로드한 모델에 의해 자동으로 결정되며 응답의 dim 필드로 반환된다.
  호출 측은 D를 사전에 알 필요가 없다. 단, Part D/E/F의 데이터 일관성을 위해
  수집/학습/배포 단계에서 동일한 모델 가중치를 사용해야 한다 (README 참조).

미구현 항목 (모델 확정 후):
  _load_model()  — 임베딩 모델 로드 및 self._dim 결정
  _embed(text)   — 추론
"""

from __future__ import annotations

import os
from typing import List, Optional

import rclpy
from rclpy.node import Node

from robot_arm_msgs.srv import EmbedText


class EmbeddingNode(Node):

    def __init__(self):
        super().__init__('embedding_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter(
            'model_path',
            os.path.expanduser('~/models/embedding/'),
        )
        self.model_path: str = self.get_parameter('model_path').value

        # ── 모델 로드 ─────────────────────────────
        # _model: 임베딩 모델 객체 (미구현 시 None)
        # _dim:   모델이 결정한 임베딩 차원 (미구현 시 None)
        self._model = None
        self._dim: Optional[int] = None
        self._load_model()

        # ── 서비스 ────────────────────────────────
        self.create_service(EmbedText, '/embed_text', self._embed_callback)

        self.get_logger().info(
            f'embedding_node 시작 | model_path: {self.model_path} | '
            f'dim: {self._dim if self._dim is not None else "TBD (모델 미확정)"}'
        )

    # ── 모델 로드 ─────────────────────────────────

    def _load_model(self) -> None:
        """
        임베딩 모델을 로드하고 self._model / self._dim 을 설정한다.

        TODO: 임베딩 모델 확정 후 구현
          후보: MiniLM-L6 / DistilBERT / CLIP text encoder / T5-small (README 미확정 #1)

          예시 (sentence-transformers MiniLM-L6 기준):
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_path)
            self._dim   = self._model.get_sentence_embedding_dimension()

          모델 변경 시:
            - Part D 수집 데이터셋 재라벨링 필수
            - Part E ACT 학습 obs 차원 변경 → 재학습 필수
            - SubtaskCommand.language_embedding 길이 변경
        """
        self.get_logger().warn(
            '[embedding_node] _load_model() 미구현. '
            '임베딩 모델 확정 후 구현하세요. (README 미확정 항목 #1)'
        )

    # ── 추론 ──────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        """
        텍스트를 임베딩 벡터로 변환.

        TODO: _load_model() 구현 후 구현
          예시:
            with torch.no_grad():
                vec = self._model.encode(text, normalize_embeddings=True)
            return vec.astype('float32').tolist()
        """
        raise NotImplementedError(
            "[embedding_node] _embed() 미구현. _load_model() 구현 후 활성화하세요."
        )

    # ── 서비스 콜백 ───────────────────────────────

    def _embed_callback(
        self,
        request:  EmbedText.Request,
        response: EmbedText.Response,
    ) -> EmbedText.Response:
        """
        /embed_text 서비스 핸들러.

        모델이 미로드 상태면 빈 응답(dim=0)을 반환한다. 호출 측이 dim==0을
        오류 신호로 해석하도록 README/관계자에게 약속해 둔다.
        """
        if self._model is None or self._dim is None:
            self.get_logger().warn(
                '[embedding_node] 모델 미로드 — 빈 응답 반환 (dim=0)',
                throttle_duration_sec=5.0,
            )
            response.embedding = []
            response.dim       = 0
            return response

        try:
            vec = self._embed(request.text)
        except NotImplementedError as e:
            self.get_logger().error(str(e))
            response.embedding = []
            response.dim       = 0
            return response

        response.embedding = list(vec)
        response.dim       = self._dim
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
