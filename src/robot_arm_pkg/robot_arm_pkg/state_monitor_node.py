"""
state_monitor_node.py

joint_states를 모니터링하여 태스크 성공/실패를 판단하고,
실패 시 reflection_feedback으로 vlm_node에 재시도를 트리거하는 노드.

환경:
  Python   3.10
  ROS2     Humble (Humboldt, 22.04 LTS)

구독 토픽:
  /joint_states (sensor_msgs/JointState) — robot_controller 피드백

발행 토픽:
  /reflection_feedback (std_msgs/String) — vlm_node로 재시도 요청

파라미터:
  success_threshold  (float) — 목표 관절 위치 오차 허용 범위 (rad, 기본 0.05)
  timeout_sec        (float) — 태스크 타임아웃 시간 (초, 기본 30.0)
  max_retry          (int)   — 최대 재시도 횟수 (기본 3)
"""

from __future__ import annotations

import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class StateMonitorNode(Node):

    def __init__(self):
        super().__init__('state_monitor_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('success_threshold', 0.05)
        self.declare_parameter('timeout_sec',       30.0)
        self.declare_parameter('max_retry',         3)

        self.success_threshold = self.get_parameter('success_threshold').value
        self.timeout_sec       = self.get_parameter('timeout_sec').value
        self.max_retry         = self.get_parameter('max_retry').value

        # ── 상태 변수 ─────────────────────────────
        self._retry_count:    int                  = 0
        self._task_start_sec: Optional[float]      = None
        self._latest_joints:  Optional[JointState] = None

        # ── 구독 ──────────────────────────────────
        self.create_subscription(JointState, '/joint_states', self._joints_callback, 10)

        # ── 발행 ──────────────────────────────────
        self.feedback_pub = self.create_publisher(String, '/reflection_feedback', 10)

        # ── 모니터 타이머 (10Hz) ──────────────────
        self.create_timer(0.1, self._monitor_callback)

        self.get_logger().info(
            f'state_monitor_node 시작 | '
            f'threshold: {self.success_threshold} rad | '
            f'timeout: {self.timeout_sec}s | '
            f'max_retry: {self.max_retry}'
        )

    # ── 구독 콜백 ─────────────────────────────────

    def _joints_callback(self, msg: JointState) -> None:
        if self._task_start_sec is None:
            self._task_start_sec = time.time()
        self._latest_joints = msg

    # ── 모니터 콜백 (10Hz) ────────────────────────

    def _monitor_callback(self) -> None:
        if self._latest_joints is None or self._task_start_sec is None:
            return

        elapsed = time.time() - self._task_start_sec

        if elapsed > self.timeout_sec:
            self.get_logger().warn(f'태스크 타임아웃 ({elapsed:.1f}s)')
            self._handle_failure(reason='timeout')
            return

        result = self._evaluate(self._latest_joints)

        if result == 'success':
            self.get_logger().info('태스크 성공')
            self._reset()
        elif result == 'failure':
            self.get_logger().warn('태스크 실패 감지')
            self._handle_failure(reason='failure')

    def _evaluate(self, joints: JointState) -> str:
        """
        관절 상태를 보고 성공/실패/진행 중을 반환.

        TODO: 성공/실패 판단 기준 확정 후 구현
          예시) 목표 관절 위치와 현재 위치의 오차가 threshold 이하 → 'success'
               특정 이상 동작 감지 → 'failure'
               그 외 → 'in_progress'

        Returns:
            'success' | 'failure' | 'in_progress'
        """
        self.get_logger().debug(
            '_evaluate() 미구현. 성공/실패 기준 확정 후 구현하세요.',
            throttle_duration_sec=10.0
        )
        return 'in_progress'

    # ── 실패 처리 ─────────────────────────────────

    def _handle_failure(self, reason: str) -> None:
        if self._retry_count >= self.max_retry:
            self.get_logger().error(
                f'최대 재시도 횟수 초과 ({self.max_retry}회). 태스크 중단.'
            )
            self._reset()
            return

        self._retry_count += 1
        msg = String()
        msg.data = f'retry:{reason}:{self._retry_count}'
        self.feedback_pub.publish(msg)
        self.get_logger().warn(
            f'재시도 요청 발행 ({self._retry_count}/{self.max_retry}) — 원인: {reason}'
        )
        self._task_start_sec = time.time()

    def _reset(self) -> None:
        """태스크 완료(성공/포기) 후 상태 초기화."""
        self._retry_count    = 0
        self._task_start_sec = None
        self._latest_joints  = None


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = StateMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
