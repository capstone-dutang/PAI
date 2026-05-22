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

서비스:
  /go_home (std_srvs/Trigger) — 현재 자세에서 home_pose.yaml 자세로 선형 보간 복귀.
                                agent_node 가 task 종료 시 호출. 진행 중에는
                                /joint_command 수신 명령을 무시한다 (안전).

파라미터:
  robot_port         (str)   — SO-101 팔로워 USB 포트 (기본 /dev/ttyACM0)
  robot_id           (str)   — 캘리브레이션 파일 식별 이름 (기본 follower)
  publish_hz         (float) — joint_states 발행 주기 (기본 1000Hz)
  disable_torque     (bool)  — 캘리브레이션용 토크 비활성화
  home_pose_file     (str)   — home pose YAML 경로
                                (기본 ~/calibration/home_pose.yaml,
                                 record_real_teleop.py 와 공유)
  go_home_duration_s (float) — /go_home 보간 시간 (기본 2.0초)

SO-101 관절 순서 (6축):
  shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_roll, gripper
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

# LeRobot 0.4.4 기준 경로
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

# 공유 유틸 — record_real_teleop 과 동일한 home_pose.yaml 을 읽는다.
from robot_arm_pkg.utils.home_pose import DEFAULT_HOME_POSE_PATH, load_home_pose


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

# /joint_states 발행 시 URDF(라디안) 매칭을 위한 그리퍼 한계 — so101.urdf 의
# revolute joint "gripper" <limit lower upper> 와 동일하게 유지할 것.
_GRIPPER_URDF_LOWER = -0.174533
_GRIPPER_URDF_UPPER = 1.74533


class RobotControllerNode(Node):

    def __init__(self):
        super().__init__('robot_controller_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('robot_port',         '/dev/ttyACM0')
        self.declare_parameter('robot_id',           'follower')
        self.declare_parameter('publish_hz',         1000.0)
        self.declare_parameter('disable_torque',     False)
        self.declare_parameter('home_pose_file',     DEFAULT_HOME_POSE_PATH)
        self.declare_parameter('go_home_duration_s', 2.0)

        robot_port           = self.get_parameter('robot_port').value
        robot_id             = self.get_parameter('robot_id').value
        publish_hz           = self.get_parameter('publish_hz').value
        disable_torque       = self.get_parameter('disable_torque').value
        self._home_pose_file = self.get_parameter('home_pose_file').value

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

        # ── 캘리브레이션용 토크 비활성화 ──────────
        # connect()는 자동으로 토크를 켜서 모터가 자세를 유지함.
        # 캘리브레이션 중엔 수동으로 팔을 옮겨야 하므로 토크를 끈다.
        # encoder는 계속 동작하므로 /joint_states 발행에는 영향 없음.
        if disable_torque:
            try:
                if hasattr(self._robot, 'bus') and hasattr(self._robot.bus, 'disable_torque'):
                    self._robot.bus.disable_torque()
                elif hasattr(self._robot, 'disable_torque'):
                    self._robot.disable_torque()
                else:
                    for motor_name in SO101_JOINT_NAMES:
                        self._robot.bus.write('Torque_Enable', motor_name, 0)
                self.get_logger().info('토크 비활성화 완료 — 수동 이동 가능')
            except Exception as e:
                self.get_logger().warn(f'토크 비활성화 실패: {e}')

        # ── 최신 커맨드 캐시 ──────────────────────
        self._latest_cmd: Optional[JointState] = None

        # ── home pose 로드 ────────────────────────
        # 파일이 없으면 /go_home 호출 시 실패 응답을 반환한다 (자동 캡처 없음 —
        # 안전상 record_real_teleop.py 가 최초 캡처를 담당).
        self._home_pose: Optional[np.ndarray] = None
        self._load_home_pose()
        # _go_home_active 가 True 인 동안 외부 /joint_command 는 무시된다.
        self._go_home_active: bool = False

        # ── 구독 ──────────────────────────────────
        self.create_subscription(JointState, '/joint_command', self._cmd_callback, 10)

        # ── 발행 ──────────────────────────────────
        self.state_pub = self.create_publisher(JointState, '/joint_states', 10)

        # ── 서비스 ────────────────────────────────
        # /go_home: task 종료 후 follower 를 home_pose.yaml 자세로 복귀.
        # 진행 중 약 go_home_duration_s 초 동안 단일 스레드 콜백을 점유하므로
        # /joint_states 1000Hz 발행이 잠시 멈춘다 — 안전상 task 외 호출 금지.
        self.create_service(Trigger, '/go_home', self._on_go_home)

        # ── 타이머 ────────────────────────────────
        self.create_timer(1.0 / publish_hz, self._timer_callback)

        self.get_logger().info(
            f'robot_controller_node 시작 ({publish_hz}Hz) | '
            f'home_pose: {"loaded" if self._home_pose is not None else "MISSING"}'
        )

    # ── 구독 콜백 ─────────────────────────────────

    def _cmd_callback(self, msg: JointState) -> None:
        """act_node로부터 관절 명령 수신 후 즉시 모터로 전송.

        /go_home 진행 중에는 외부 명령을 무시한다 — 보간 도중 act_node 가
        명령을 계속 발행해 충돌하는 상황을 방지하기 위한 안전장치.
        """
        if self._go_home_active:
            self.get_logger().debug('/go_home 진행 중 — /joint_command 무시')
            return
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

        단위 변환: LeRobot SO-101 SDK 는 회전 5축을 도(degree), 그리퍼를
        0~100 정규화 값으로 반환한다. URDF/TF 는 라디안 기반이라
        robot_state_publisher 가 한계로 클램프되어 TF 가 멈추는 것을 막기 위해
        여기서 변환한다. (act_node·home_pose.yaml·데이터셋은 SDK 단위 그대로
        유지하므로 _send_command/_go_home 은 변환하지 않는다.)
        """
        obs = self._robot.get_observation()
        raw = [float(obs[f'{name}.pos']) for name in SO101_JOINT_NAMES]

        positions = [math.radians(v) for v in raw[:5]]
        gripper_frac = max(0.0, min(100.0, raw[5])) / 100.0
        positions.append(
            _GRIPPER_URDF_LOWER
            + gripper_frac * (_GRIPPER_URDF_UPPER - _GRIPPER_URDF_LOWER)
        )

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = SO101_JOINT_NAMES
        msg.position     = positions
        return msg

    # ── home pose / /go_home 서비스 ────────────────

    def _load_home_pose(self) -> None:
        """home_pose.yaml 을 읽어 self._home_pose 에 캐싱.

        파일이 없거나 형식이 잘못된 경우 None 으로 두고 warn 로그를 남긴다.
        record_real_teleop.py 가 최초 1회 실행 시 파일을 생성하므로, 본 노드 단독
        실행 시에는 파일이 없을 수 있다.
        """
        try:
            self._home_pose = load_home_pose(self._home_pose_file)
            self.get_logger().info(
                f'home_pose 로드: {self._home_pose_file} '
                f'→ {self._home_pose.tolist()}'
            )
        except FileNotFoundError:
            self._home_pose = None
            self.get_logger().warn(
                f'home_pose 파일 없음 ({self._home_pose_file}). '
                '/go_home 서비스는 실패 응답을 반환한다. '
                'record_real_teleop.py 를 한 번 실행하면 자동 생성된다.'
            )
        except Exception as e:
            self._home_pose = None
            self.get_logger().error(
                f'home_pose 로드 실패 ({self._home_pose_file}): {e}'
            )

    def _on_go_home(
        self,
        request:  Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """
        /go_home 서비스 핸들러 — 현재 자세에서 home_pose 로 선형 보간 복귀.

        호출 측 (agent_node) 책임:
          - 본 서비스를 부르기 전 act_node 가 /joint_command 발행을 멈춘
            상태여야 한다 (ExecutionStatus 수신 후, 다음 subtask 발행 전).
          - duration 동안 /joint_states 발행이 잠시 끊긴다 — 그 시간 동안
            추론 등 다른 작업을 하지 말 것.
        """
        if self._home_pose is None:
            # 런타임 중 파일이 새로 생겼을 수 있으니 한 번 재시도
            self._load_home_pose()
        if self._home_pose is None:
            response.success = False
            response.message = (
                f'home_pose 미로드 — {self._home_pose_file} 파일 확인 필요'
            )
            self.get_logger().error(response.message)
            return response

        duration_s = float(self.get_parameter('go_home_duration_s').value)
        try:
            self._go_home_blocking(duration_s)
            response.success = True
            response.message = f'home 복귀 완료 ({duration_s:.1f}s)'
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f'go_home 보간 실패: {e}'
            self.get_logger().error(response.message)
        return response

    def _go_home_blocking(self, duration_s: float) -> None:
        """현재 follower 자세 → self._home_pose 로 50Hz 선형 보간 명령 송신.

        보간 주기 50Hz 는 SO-101 모터 제어가 안정적으로 따라가는 수준에서
        선택. 보간 도중 _go_home_active=True 로 두어 외부 /joint_command 를
        차단한다.
        """
        cur_obs = self._robot.get_observation()
        cur = np.array(
            [float(cur_obs[f'{n}.pos']) for n in SO101_JOINT_NAMES],
            dtype=np.float32,
        )

        motion_hz = 50.0
        n_steps = max(1, int(round(motion_hz * duration_s)))
        dt = 1.0 / motion_hz

        self._go_home_active = True
        try:
            for i in range(1, n_steps + 1):
                alpha  = i / n_steps
                target = (1.0 - alpha) * cur + alpha * self._home_pose
                self._robot.send_action({
                    f'{n}.pos': float(v)
                    for n, v in zip(SO101_JOINT_NAMES, target)
                })
                time.sleep(dt)
        finally:
            self._go_home_active = False

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