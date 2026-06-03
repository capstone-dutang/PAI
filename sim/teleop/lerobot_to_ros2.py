#!/usr/bin/env python3
"""
SO-101 Leader → Isaac Sim follower 텔레오퍼레이션 브리지 (로컬 버전).
- LeRobot 0.4.4 API로 Leader joint state를 읽어
- ROS2 /joint_command 토픽으로 직접 publish한다.
- 같은 머신 내 Isaac Sim ROS2 Bridge가 이를 받아 follower에 적용한다.

EC2 버전(`lerobot_to_rosbridge.py`)이 WebSocket → rosbridge 경로로 보내는 것과 달리,
로컬 버전은 rclpy publisher로 ROS2 에 직접 publish 한다. 따라서 `rclpy` 와 `lerobot`
이 같은 env 에 설치돼 있어야 한다(권장: `ros_humble` env 에 lerobot 추가 설치).
셋업 절차는 `sim_local_teleop_setting.md` Part 4~5 참조.

실행:
    mamba activate ros_humble
    export ROS_DOMAIN_ID=42
    python lerobot_to_ros2.py --port /dev/ttyACM0 --id my_leader
"""
import argparse
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


# Isaac Sim 측 gripper joint limit (degree)
GRIPPER_MIN_DEG = -10.0
GRIPPER_MAX_DEG = 100.0

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def gripper_pct_to_rad(pct: float) -> float:
    """Leader gripper 0~100% → Isaac Sim gripper rad."""
    pct = max(0.0, min(100.0, pct))
    deg = GRIPPER_MIN_DEG + (GRIPPER_MAX_DEG - GRIPPER_MIN_DEG) * (pct / 100.0)
    return math.radians(deg)


class TeleopBridge(Node):
    def __init__(self, leader: SO101Leader, rate_hz: float):
        super().__init__("lerobot_teleop_bridge")
        self.leader = leader
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.rate_hz = rate_hz
        self.timer = self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(f"Teleop bridge started at {rate_hz} Hz")

    def tick(self):
        try:
            action = self.leader.get_action()
        except Exception as e:
            self.get_logger().warn(f"get_action failed: {e}")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_ORDER)

        positions = []
        for j in JOINT_ORDER:
            val = action[f"{j}.pos"]
            if j == "gripper":
                positions.append(gripper_pct_to_rad(val))
            else:
                positions.append(math.radians(val))
        msg.position = positions

        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="e.g. /dev/ttyACM0")
    parser.add_argument("--id", default="my_leader", help="calibration id")
    parser.add_argument("--rate", type=float, default=30.0, help="publish Hz")
    args = parser.parse_args()

    cfg = SO101LeaderConfig(port=args.port, id=args.id, use_degrees=True)
    leader = SO101Leader(cfg)
    leader.connect(calibrate=True)

    rclpy.init()
    node = TeleopBridge(leader, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        leader.disconnect()


if __name__ == "__main__":
    main()
