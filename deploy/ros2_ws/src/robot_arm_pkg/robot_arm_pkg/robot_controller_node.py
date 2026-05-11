"""
robot_controller_node.py

act_node의 joint_command를 받아 SO-101 팔로워 암을 구동하고
현재 관절 상태(joint_states)를 피드백하는 노드.

환경:
  Python   3.10
  ROS2     Humble (Humboldt, 22.04 LTS)
  LeRobot  SO-101 (so101_follower) — Feetech STS3215 모터

구독 토픽:
  /joint_command (sensor_msgs/JointState) — act_node로부터 수신

발행 토픽:
  /joint_states  (sensor_msgs/JointState) — 1000Hz

파라미터:
  robot_port (str)   — SO-101 팔로워 USB 포트 (기본 /dev/ttyACM0)
  robot_id   (str)   — 캘리브레이션 파일 식별 이름 (기본 follower)
  publish_hz (float) — joint_states 발행 주기 (기본 1000Hz)

SO-101 관절 순서 (6축):
  shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_roll, gripper
"""

from __future__ import annotations

from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# LeRobot 0.4.4 기준 경로
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower


# SO-101 관절 이름 — LeRobot SDK의 모터 이름과 일치.
# SDK는 get_observation()에서 {motor}.pos 형태로 위치를 반환하고,
# send_action()도 같은 dict 형식을 받는다.
SO101_JOINT_NAMES: List[str] = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]


class RobotControllerNode(Node):

    def __init__(self):
        super().__init__('robot_controller_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('robot_port', '/dev/ttyACM0')
        self.declare_parameter('robot_id',   'follower')
        self.declare_parameter('publish_hz', 1000.0)

        robot_port = self.get_parameter('robot_port').value
        robot_id   = self.get_parameter('robot_id').value
        publish_hz = self.get_parameter('publish_hz').value

        # ── SO-101 LeRobot SDK 초기화 ─────────────
        robot_cfg = SO101FollowerConfig(
            port=robot_port,
            id=robot_id,
        )
        self._robot = SO101Follower(robot_cfg)
        self._robot.connect()
        self.get_logger().info(
            f'SO-101 연결 완료 | 포트: {robot_port} | ID: {robot_id}'
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
        """act_node로부터 관절 명령 수신 후 즉시 모터로 전송."""
        self._latest_cmd = msg
        self._send_command(msg)

    # ── 타이머 콜백 (1000Hz) ──────────────────────

    def _timer_callback(self) -> None:
        """현재 관절 상태를 읽어 발행."""
        try:
            state = self._read_joints()
            if state is not None:
                self.state_pub.publish(state)
        except Exception as e:
            self.get_logger().error(f'관절 상태 읽기 실패: {e}')

    # ── LeRobot SDK 연동 ──────────────────────────

    def _send_command(self, cmd: JointState) -> None:
        """
        JointState 메시지를 받아 SO-101 모터로 전송.
        cmd.position 은 SO101_JOINT_NAMES 순서의 6개 값이어야 함.
        """
        action = {
            f'{name}.pos': float(pos)
            for name, pos in zip(SO101_JOINT_NAMES, cmd.position)
        }
        self._robot.send_action(action)

    def _read_joints(self) -> Optional[JointState]:
        """
        SO-101 현재 관절 상태를 읽어 JointState 메시지로 반환.

        SDK 반환 dict에서 {motor}.pos 키만 추려 6개 값을 모은다.
        (카메라 등록 시 obs에 이미지 키가 함께 들어올 수 있어 명시적으로 골라야 한다.)
        """
        obs = self._robot.get_observation()
        positions = [float(obs[f'{name}.pos']) for name in SO101_JOINT_NAMES]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = SO101_JOINT_NAMES
        msg.position     = positions
        return msg

    def destroy_node(self) -> None:
        """노드 종료 시 SO-101 연결 해제."""
        try:
            self._robot.disconnect()
            self.get_logger().info('SO-101 연결 해제 완료')
        except Exception as e:
            self.get_logger().warn(f'SO-101 연결 해제 중 오류: {e}')
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