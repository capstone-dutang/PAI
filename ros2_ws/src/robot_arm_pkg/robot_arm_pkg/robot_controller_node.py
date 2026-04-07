"""
robot_controller_node.py

act_node의 joint_command를 받아 실제 LeRobot 모터를 구동하고
현재 관절 상태(joint_states)를 피드백하는 노드.

환경:
  Python   3.10
  ROS2     Humble (Humboldt, 22.04 LTS)

구독 토픽:
  /joint_command (sensor_msgs/JointState) — act_node로부터 수신

발행 토픽:
  /joint_states  (sensor_msgs/JointState) — 1000Hz

파라미터:
  publish_hz (float) — joint_states 발행 주기 (기본 1000Hz)
"""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class RobotControllerNode(Node):

    def __init__(self):
        super().__init__('robot_controller_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('publish_hz', 1000.0)
        publish_hz = self.get_parameter('publish_hz').value

        # ── LeRobot API 초기화 ────────────────────
        # TODO: LeRobot API 확정 후 구현
        #   self.robot = LeRobotAPI(...)
        #   self.robot.connect()
        self._robot = None
        self.get_logger().warn(
            '[robot_controller] LeRobot API 미확정. '
            '_send_command() 및 _read_joints() 를 구현하세요.'
        )

        # ── 최신 커맨드 캐시 ──────────────────────
        self._latest_cmd: Optional[JointState] = None

        # ── 구독 ──────────────────────────────────
        self.create_subscription(JointState, '/joint_command', self._cmd_callback, 10)

        # ── 발행 ──────────────────────────────────
        self.state_pub = self.create_publisher(JointState, '/joint_states', 10)

        # ── 타이머 ────────────────────────────────
        self.create_timer(1.0 / publish_hz, self._timer_callback)

        self.get_logger().info(f'robot_controller_node 시작 ({publish_hz}Hz)')

    # ── 구독 콜백 ─────────────────────────────────

    def _cmd_callback(self, msg: JointState) -> None:
        """act_node로부터 관절 명령 수신."""
        self._latest_cmd = msg
        self._send_command(msg)

    # ── 타이머 콜백 (1000Hz) ──────────────────────

    def _timer_callback(self) -> None:
        """현재 관절 상태를 읽어 발행."""
        try:
            state = self._read_joints()
            if state is not None:
                self.state_pub.publish(state)
        except NotImplementedError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f'관절 상태 읽기 실패: {e}')

    # ── 드라이버 인터페이스 ───────────────────────

    def _send_command(self, cmd: JointState) -> None:
        """
        관절 명령을 실제 모터로 전송.

        TODO: LeRobot API 확정 후 구현
          self._robot.set_joint_positions(cmd.position)
        """
        raise NotImplementedError(
            "[robot_controller] _send_command() 미구현. "
            "LeRobot API 확정 후 구현하세요."
        )

    def _read_joints(self) -> Optional[JointState]:
        """
        현재 관절 상태를 하드웨어에서 읽어 반환.

        TODO: LeRobot API 확정 후 구현
          positions = self._robot.get_joint_positions()
          → JointState 메시지로 변환 후 반환

        Returns:
            JointState — 현재 관절 상태. 실패 시 None.
        """
        raise NotImplementedError(
            "[robot_controller] _read_joints() 미구현. "
            "LeRobot API 확정 후 구현하세요."
        )

    def destroy_node(self) -> None:
        # TODO: LeRobot API 연결 해제
        # if self._robot: self._robot.disconnect()
        super().destroy_node()


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
