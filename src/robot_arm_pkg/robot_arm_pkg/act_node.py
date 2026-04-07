"""
act_node.py

ACT Policy (TensorRT) 로 관절 시퀀스를 추론하는 노드.
VLM이 발행한 object_pose를 캐싱하고, 30Hz로 joint_command를 발행.

환경:
  Python   3.10
  ROS2     Humble (Humboldt, 22.04 LTS)

구독 토픽:
  /camera/rgb    (sensor_msgs/Image)
  /camera/depth  (sensor_msgs/Image)
  /object_pose   (geometry_msgs/PoseStamped) — vlm_node에서 수신, 캐싱
  /joint_states  (sensor_msgs/JointState)    — robot_controller 피드백

발행 토픽:
  /joint_command (sensor_msgs/JointState)    — 30Hz

파라미터:
  model_path  (str)   — TensorRT 엔진 파일 경로 (.trt)
  chunk_size  (int)   — 한 번에 예측할 스텝 수 (기본 100)
  publish_hz  (float) — 발행 주기 (기본 30Hz)
"""

from __future__ import annotations

from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped


class ACTNode(Node):

    def __init__(self):
        super().__init__('act_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('chunk_size', 100)
        self.declare_parameter('publish_hz', 30.0)

        self.model_path = self.get_parameter('model_path').value
        self.chunk_size = self.get_parameter('chunk_size').value
        publish_hz      = self.get_parameter('publish_hz').value

        # ── 입력 캐시 ─────────────────────────────
        self._latest_rgb:    Optional[Image]       = None
        self._latest_depth:  Optional[Image]       = None
        self._latest_pose:   Optional[PoseStamped] = None  # VLM 결과 캐싱
        self._latest_joints: Optional[JointState]  = None

        # chunk 실행 상태
        # _chunk: 현재 실행 중인 관절 시퀀스 (list of JointState)
        # _chunk_idx: chunk 내 현재 스텝 인덱스
        self._chunk:     Optional[List[JointState]] = None
        self._chunk_idx: int                        = 0

        # ── 구독 ──────────────────────────────────
        self.create_subscription(Image,       '/camera/rgb',   self._rgb_callback,    10)
        self.create_subscription(Image,       '/camera/depth', self._depth_callback,  10)
        self.create_subscription(PoseStamped, '/object_pose',  self._pose_callback,   10)
        self.create_subscription(JointState,  '/joint_states', self._joints_callback, 10)

        # ── 발행 ──────────────────────────────────
        self.cmd_pub = self.create_publisher(JointState, '/joint_command', 10)

        # ── 타이머 ────────────────────────────────
        self.create_timer(1.0 / publish_hz, self._timer_callback)

        self.get_logger().info(
            f'act_node 시작 | 모델: {self.model_path} | chunk_size: {self.chunk_size} | {publish_hz}Hz'
        )

    # ── 구독 콜백 ─────────────────────────────────

    def _rgb_callback(self, msg: Image) -> None:
        self._latest_rgb = msg

    def _depth_callback(self, msg: Image) -> None:
        self._latest_depth = msg

    def _pose_callback(self, msg: PoseStamped) -> None:
        """
        VLM이 발행한 물체 위치 수신.
        새 pose 수신 시 chunk를 새로 예측.
        """
        self._latest_pose = msg
        self.get_logger().info('새 object_pose 수신 → 다음 주기에 chunk 재예측')
        # 새 pose가 오면 현재 chunk 리셋 → 다음 타이머에서 재예측
        self._chunk     = None
        self._chunk_idx = 0

    def _joints_callback(self, msg: JointState) -> None:
        self._latest_joints = msg

    # ── 타이머 콜백 (30Hz) ────────────────────────

    def _timer_callback(self) -> None:
        if self._latest_pose is None:
            self.get_logger().warn('/object_pose 미수신, 대기 중', throttle_duration_sec=5.0)
            return

        # chunk 소진 또는 미생성 시 새로 예측
        if self._chunk is None or self._chunk_idx >= len(self._chunk):
            self._chunk     = self._predict_chunk()
            self._chunk_idx = 0

        if self._chunk is None:
            return  # 예측 실패 시 스킵

        cmd = self._chunk[self._chunk_idx]
        self.cmd_pub.publish(cmd)
        self._chunk_idx += 1

    def _predict_chunk(self) -> Optional[List[JointState]]:
        """
        현재 입력으로 chunk_size 스텝의 관절 시퀀스 예측.

        TODO: TensorRT 엔진 확정 후 구현
          1. rgb, depth, pose, joints → 텐서 변환
          2. TensorRT 엔진으로 추론
          3. 출력 텐서 → JointState 리스트 변환 (chunk_size 개)

        Returns:
            List[JointState] 길이 chunk_size. 실패 시 None.
        """
        raise NotImplementedError(
            "[act_node] TensorRT 추론 미구현. "
            "모델 확정 후 _predict_chunk()를 구현하세요."
        )


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ACTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
