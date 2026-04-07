"""
vlm_node.py

Qwen3.5-4B GGUF 모델을 llama.cpp 서버로 실행하여
물체 위치(object_pose)를 추론하는 노드.

환경:
  Python   3.10
  ROS2     Humble (Humboldt, 22.04 LTS)

구독 토픽:
  /camera/rgb          (sensor_msgs/Image)
  /camera/depth        (sensor_msgs/Image)
  /task_command        (std_msgs/String)   — 터미널에서 발행
  /reflection_feedback (std_msgs/String)   — state_monitor로부터 재시도 요청

발행 토픽:
  /object_pose (geometry_msgs/PoseStamped) — 0.4Hz (2.5초마다)

파라미터:
  model_path   (str)   — GGUF 모델 파일 경로
  server_url   (str)   — llama.cpp 서버 주소
  infer_hz     (float) — 추론 주기 (기본 0.4Hz)
"""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


class VLMNode(Node):

    def __init__(self):
        super().__init__('vlm_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('server_url', '')
        self.declare_parameter('infer_hz',   0.4)

        self.model_path = self.get_parameter('model_path').value
        self.server_url = self.get_parameter('server_url').value
        infer_hz        = self.get_parameter('infer_hz').value

        # ── 최신 프레임 캐시 ──────────────────────
        self._latest_rgb:   Optional[Image] = None
        self._latest_depth: Optional[Image] = None
        self._task_command: Optional[str]   = None
        self._is_running = False   # 중복 추론 방지 플래그

        # ── 구독 ──────────────────────────────────
        self.create_subscription(Image,  '/camera/rgb',          self._rgb_callback,      10)
        self.create_subscription(Image,  '/camera/depth',        self._depth_callback,    10)
        self.create_subscription(String, '/task_command',        self._task_callback,     10)
        self.create_subscription(String, '/reflection_feedback', self._feedback_callback, 10)

        # ── 발행 ──────────────────────────────────
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)

        # ── 추론 타이머 ───────────────────────────
        self.create_timer(1.0 / infer_hz, self._infer_callback)

        self.get_logger().info(
            f'vlm_node 시작 | 모델: {self.model_path} | 서버: {self.server_url} | {infer_hz}Hz'
        )

    # ── 구독 콜백 ─────────────────────────────────

    def _rgb_callback(self, msg: Image) -> None:
        self._latest_rgb = msg

    def _depth_callback(self, msg: Image) -> None:
        self._latest_depth = msg

    def _task_callback(self, msg: String) -> None:
        self._task_command = msg.data
        self.get_logger().info(f'태스크 수신: {msg.data}')

    def _feedback_callback(self, msg: String) -> None:
        """state_monitor로부터 재시도 요청 수신."""
        self.get_logger().warn(f'재시도 요청 수신: {msg.data}')
        # TODO: 재시도 횟수 제한 로직 추가

    # ── 추론 타이머 콜백 ──────────────────────────

    def _infer_callback(self) -> None:
        if self._latest_rgb is None or self._latest_depth is None:
            self.get_logger().warn('카메라 프레임 미수신, 추론 스킵', throttle_duration_sec=5.0)
            return

        if self._task_command is None:
            self.get_logger().warn('/task_command 미수신, 추론 스킵', throttle_duration_sec=5.0)
            return

        if self._is_running:
            self.get_logger().warn('이전 추론 진행 중, 이번 주기 스킵')
            return

        self._is_running = True
        try:
            pose = self._run_inference(self._latest_rgb, self._latest_depth, self._task_command)
            if pose is not None:
                self.pose_pub.publish(pose)
        except Exception as e:
            self.get_logger().error(f'추론 실패: {e}')
        finally:
            self._is_running = False

    def _run_inference(
        self,
        rgb_msg:   Image,
        depth_msg: Image,
        task:      str,
    ) -> Optional[PoseStamped]:
        """
        llama.cpp 서버에 이미지를 전송하고 물체 위치를 반환.

        TODO: llama.cpp 서버 확정 후 구현
          1. rgb_msg, depth_msg → base64 인코딩
          2. HTTP POST → self.server_url
          3. 응답 파싱 → x, y, z 추출
          4. PoseStamped 메시지 생성 후 반환

        Returns:
            PoseStamped — 물체 위치. 실패 시 None.
        """
        raise NotImplementedError(
            "[vlm_node] llama.cpp 추론 미구현. "
            "모델 확정 후 _run_inference()를 구현하세요."
        )


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
