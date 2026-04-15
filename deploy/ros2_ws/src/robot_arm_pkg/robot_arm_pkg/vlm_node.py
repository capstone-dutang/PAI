"""
vlm_node.py

Qwen3.5-4B GGUF 모델을 llama.cpp 서버로 실행하여
집을 물체의 클래스를 추론하는 노드.

환경:
  Python   3.10
  ROS2     Humble (Humboldt, 22.04 LTS)

구독 토픽:
  /camera/rgb          (sensor_msgs/Image)  — 추론 시 최신 프레임 사용
  /task_command        (std_msgs/String)    — 터미널에서 발행
  /reflection_feedback (std_msgs/String)    — state_monitor로부터 재시도 요청 → 즉시 재추론
  /task_active         (std_msgs/Bool)      — False 수신 시 _task_command 초기화, 추론 중단

발행 토픽:
  /target_class (std_msgs/String) — 0.4Hz (2.5초마다), 재시도 시 즉시 발행
  예시: "cup", "bottle"

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
from std_msgs.msg import String, Bool


class VLMNode(Node):

    def __init__(self):
        super().__init__('vlm_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('model_path', '/home/user/models/vlm_qwen3.5_4b_q4.gguf')
        self.declare_parameter('server_url', 'http://localhost:8080')
        self.declare_parameter('infer_hz',   0.4)

        self.model_path = self.get_parameter('model_path').value
        self.server_url = self.get_parameter('server_url').value
        infer_hz        = self.get_parameter('infer_hz').value

        # ── 최신 프레임 캐시 ──────────────────────
        self._latest_rgb:   Optional[Image] = None
        self._task_command: Optional[str]   = None
        self._is_running  = False   # 중복 추론 방지 플래그
        self._task_active = False   # 태스크 활성 여부 (False 시 추론 중단)

        # ── 구독 ──────────────────────────────────
        self.create_subscription(Image,  '/camera/rgb',          self._rgb_callback,      10)
        self.create_subscription(String, '/task_command',        self._task_callback,     10)
        self.create_subscription(String, '/reflection_feedback', self._feedback_callback, 10)
        self.create_subscription(Bool,   '/task_active',         self._task_active_callback, 10)

        # ── 발행 ──────────────────────────────────
        self.class_pub = self.create_publisher(String, '/target_class', 10)

        # ── 추론 타이머 ───────────────────────────
        self.create_timer(1.0 / infer_hz, self._infer_callback)

        self.get_logger().info(
            f'vlm_node 시작 | 모델: {self.model_path} | 서버: {self.server_url} | {infer_hz}Hz\n'
            f'  구독: /camera/rgb  /task_command  /reflection_feedback  /task_active\n'
            f'  발행: /target_class'
        )

    # ── 구독 콜백 ─────────────────────────────────

    def _rgb_callback(self, msg: Image) -> None:
        self._latest_rgb = msg

    def _task_callback(self, msg: String) -> None:
        self._task_command = msg.data
        self.get_logger().info(f'태스크 수신: {msg.data}')

    def _feedback_callback(self, msg: String) -> None:
        """
        state_monitor로부터 재시도 요청 수신 → 즉시 재추론 트리거.

        0.4Hz 타이머를 기다리지 않고 바로 _infer_callback()을 호출해
        새 target_class를 발행한다.
        """
        self.get_logger().warn(f'재시도 요청 수신: {msg.data}')
        self._is_running = False   # 혹시 이전 추론이 걸려있으면 플래그 리셋
        self._infer_callback()     # 즉시 재추론

    def _task_active_callback(self, msg: Bool) -> None:
        """
        태스크 비활성화(False) 수신 → _task_command 초기화.
        다음 /task_command가 올 때까지 0.4Hz 추론 루프를 중단한다.
        """
        if self._task_active and not msg.data:
            self.get_logger().info('태스크 비활성화 → _task_command 초기화, 추론 중단')
            self._task_command = None
        self._task_active = msg.data

    # ── 추론 타이머 콜백 ──────────────────────────

    def _infer_callback(self) -> None:
        if self._latest_rgb is None:
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
            target_class = self._run_inference(self._latest_rgb, self._task_command)
            if target_class is not None:
                msg = String()
                msg.data = target_class
                self.class_pub.publish(msg)
                self.get_logger().info(f'target_class 발행: {target_class!r}')
        except Exception as e:
            self.get_logger().error(f'추론 실패: {e}')
        finally:
            self._is_running = False

    def _run_inference(
        self,
        rgb_msg: Image,
        task:    str,
    ) -> Optional[str]:
        """
        llama.cpp 서버에 이미지와 태스크 명령을 전송하고
        집을 물체의 클래스 이름을 반환.

        TODO: llama.cpp 서버 확정 후 구현
          1. rgb_msg → base64 인코딩
          2. HTTP POST → self.server_url
             payload: {"image": <base64>, "prompt": task}
          3. 응답 파싱 → 클래스 이름 문자열 추출
             예시 응답: {"class": "cup"}

        Returns:
            str — YOLO 클래스 이름 (예: "cup", "bottle"). 실패 시 None.
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
