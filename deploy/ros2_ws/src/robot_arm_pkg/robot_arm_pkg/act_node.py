"""
act_node.py

LeRobot ACT Policy (.pth 체크포인트) 로 관절 시퀀스를 추론하는 노드.

이전 설계와 달리 본 노드는 다음 의존을 모두 제거한다:
  - /target_class (vlm_node)
  - /object_pose  (yolo_node)
  - /task_active  (state_monitor_node)
  - observation.object_pose 채널

대신 agent_node가 보낸 SubtaskCommand(language_embedding 포함)를 받아
ACT 추론을 시작하고, chunk 실행이 끝나면 ExecutionStatus를 발행한다.
language_embedding은 SubtaskCommand 안에 사전 임베딩되어 도착하므로
act_node는 추론 시점에 임베딩 변환 지연이 없다.

환경:
  Python   3.10
  ROS2     Humble (Ubuntu 22.04 LTS)
  LeRobot  ACT Policy — .pth 체크포인트를 torch로 직접 추론

구독 토픽:
  /camera/rgb       (sensor_msgs/Image, BestEffort+KeepLast(1))
  /joint_states     (sensor_msgs/JointState) — robot_controller 피드백
  /subtask_command  (robot_arm_msgs/SubtaskCommand) — agent_node 발행

발행 토픽:
  /joint_command    (sensor_msgs/JointState) — 30Hz (subtask 활성 시)
  /execution_status (robot_arm_msgs/ExecutionStatus) — subtask 종료 시 1회

파라미터:
  model_path  (str)   — LeRobot ACT 체크포인트 경로 (.pth)
  chunk_size  (int)   — 한 번에 예측할 스텝 수 (기본 100)
  publish_hz  (float) — 발행 주기 (기본 30Hz)

학습 시 obs 스키마(LeRobotDataset)와 추론 시 obs dict 키는 일치해야 함:
  observation.images.top         — RGB 이미지 텐서
  observation.state              — 관절 상태 텐서 (6축, SO-101 순서)
  observation.language_embedding — 자연어 타겟 묘사 임베딩 (D차원)

TODO (ACT Co-training 완료 후 구현):
  _load_policy()    — 체크포인트 로드 및 normalizer 설정 확인
  _predict_chunk()  — 입력 텐서 전처리 + policy.select_action() 호출
"""

from __future__ import annotations

import os
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, JointState

from robot_arm_msgs.msg import SubtaskCommand, ExecutionStatus

# ExecutionStatus.status SSoT — 대문자 enum (agent_node 가 case 변환 없이 Gemma
# tool_result 로 전달, function_call 학습 분포도 대문자). 소문자 리터럴 금지.
from robot_arm_pkg.prompts import ExecStatus

# SO-101 관절 이름 — robot_controller_node.py 와 동일한 순서 유지
SO101_JOINT_NAMES: List[str] = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]


class ACTNode(Node):

    def __init__(self):
        super().__init__('act_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter(
            'model_path',
            os.path.expanduser('~/models/act/act.pth'),
        )
        self.declare_parameter('chunk_size', 100)
        self.declare_parameter('publish_hz', 30.0)

        self.model_path = self.get_parameter('model_path').value
        self.chunk_size = self.get_parameter('chunk_size').value
        publish_hz      = self.get_parameter('publish_hz').value

        # ── LeRobot Policy 로드 ───────────────────
        # TODO: ACT Co-training 완료 후 _load_policy() 구현
        self.policy = self._load_policy()

        # ── 입력 캐시 ─────────────────────────────
        self._latest_rgb:    Optional[Image]      = None
        self._latest_joints: Optional[JointState] = None

        # ── 활성 subtask 상태 ─────────────────────
        # _active_cmd: 현재 실행 중인 SubtaskCommand (None이면 idle)
        # _chunk:      현재 실행 중인 관절 시퀀스 (list of JointState)
        # _chunk_idx:  chunk 내 현재 스텝 인덱스
        self._active_cmd: Optional[SubtaskCommand]   = None
        self._chunk:      Optional[List[JointState]] = None
        self._chunk_idx:  int                        = 0

        # ── QoS ───────────────────────────────────
        # camera는 BestEffort + KeepLast(1) — 30Hz 스트림에서 latest-only.
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 구독 ──────────────────────────────────
        self.create_subscription(Image,          '/camera/rgb',      self._rgb_callback,     camera_qos)
        self.create_subscription(JointState,     '/joint_states',    self._joints_callback,  10)
        self.create_subscription(SubtaskCommand, '/subtask_command', self._subtask_callback, 10)

        # ── 발행 ──────────────────────────────────
        self.cmd_pub    = self.create_publisher(JointState,      '/joint_command',    10)
        self.status_pub = self.create_publisher(ExecutionStatus, '/execution_status', 10)

        # ── 타이머 ────────────────────────────────
        self.create_timer(1.0 / publish_hz, self._timer_callback)

        self.get_logger().info(
            f'act_node 시작 | 모델: {self.model_path} | '
            f'chunk_size: {self.chunk_size} | {publish_hz}Hz'
        )

    # ── Policy 로드 ───────────────────────────────

    def _load_policy(self):
        """
        LeRobot ACT 체크포인트를 로드하여 반환.

        TODO: ACT 학습 완료 후 구현
          체크포인트 구조와 normalizer 설정을 확인한 뒤 아래를 구현:

          from lerobot.policies.act.modeling_act import ACTPolicy
          policy = ACTPolicy.from_pretrained(self.model_path)
          policy.eval()
          return policy

        Returns:
            ACTPolicy 객체. 미구현 시 None.
        """
        self.get_logger().warn(
            '[act_node] _load_policy() 미구현. '
            'ACT Co-training 완료 후 LeRobot ACT 체크포인트 로드를 구현하세요.'
        )
        return None

    # ── 구독 콜백 ─────────────────────────────────

    def _rgb_callback(self, msg: Image) -> None:
        self._latest_rgb = msg

    def _joints_callback(self, msg: JointState) -> None:
        self._latest_joints = msg

    def _subtask_callback(self, msg: SubtaskCommand) -> None:
        """
        agent_node가 보낸 SubtaskCommand 수신 → 새 subtask 활성화.

        같은 subtask_id가 재수신되면 무시한다 (중복 방지).
        다른 subtask_id가 도착하면 현재 chunk를 폐기하고 새로 시작한다.
        """
        if self._active_cmd is not None and self._active_cmd.subtask_id == msg.subtask_id:
            return

        self.get_logger().info(
            f'SubtaskCommand 수신: id={msg.subtask_id!r} domain={msg.domain!r} '
            f'target={msg.target_description!r} embed_dim={len(msg.language_embedding)}'
        )
        self._active_cmd = msg
        self._chunk      = None
        self._chunk_idx  = 0

    # ── 타이머 콜백 (30Hz) ────────────────────────

    def _timer_callback(self) -> None:
        if self._active_cmd is None:
            return  # idle

        # chunk 미생성 시 새로 예측
        if self._chunk is None:
            self._chunk = self._predict_chunk(self._active_cmd)
            self._chunk_idx = 0
            if self._chunk is None:
                # 예측 실패 → 다음 주기 재시도. 영구 실패 처리는 추후 결정.
                return

        # chunk 소진 시 종료 → ExecutionStatus 발행
        if self._chunk_idx >= len(self._chunk):
            self._finish_subtask(status=ExecStatus.SUCCESS, failure_reason='', confidence=0.0)
            return

        cmd = self._chunk[self._chunk_idx]
        self.cmd_pub.publish(cmd)
        self._chunk_idx += 1

    # ── subtask 종료 처리 ─────────────────────────

    def _finish_subtask(self, status: str, failure_reason: str, confidence: float) -> None:
        """
        ExecutionStatus를 1회 발행하고 idle 상태로 복귀.

        status: ExecStatus.SUCCESS | FAIL | TIMEOUT (대문자 — SSoT prompts.ExecStatus).
                agent_node 가 case 변환 없이 Gemma tool_result 로 전달한다.
        """
        if self._active_cmd is None:
            return

        msg = ExecutionStatus()
        msg.subtask_id     = self._active_cmd.subtask_id
        msg.status         = status
        msg.failure_reason = failure_reason
        msg.confidence     = float(confidence)
        self.status_pub.publish(msg)

        self.get_logger().info(
            f'ExecutionStatus 발행: id={msg.subtask_id!r} status={status!r}'
        )

        self._active_cmd = None
        self._chunk      = None
        self._chunk_idx  = 0

    # ── 추론 ──────────────────────────────────────

    def _predict_chunk(self, cmd: SubtaskCommand) -> Optional[List[JointState]]:
        """
        현재 입력으로 chunk_size 스텝의 관절 시퀀스 예측.

        TODO: ACT 학습 완료 후 구현
          체크포인트의 normalizer 설정 확인 후 아래를 구현:

          1. 입력 텐서 구성 — normalizer 설정에 맞게 전처리
             키 이름은 학습 시 LeRobotDataset 스키마와 반드시 일치해야 함:
             obs = {
                 'observation.images.top':         _to_image_tensor(self._latest_rgb),
                 'observation.state':              _to_state_tensor(self._latest_joints),
                 'observation.language_embedding': torch.tensor(cmd.language_embedding),
             }

          2. LeRobot Policy 추론
             with torch.no_grad():
                 action_chunk = self.policy.select_action(obs)
             # action_chunk shape: (chunk_size, 6)

          3. 출력 텐서 → JointState 리스트 변환
             result = []
             for step in action_chunk:
                 m = JointState()
                 m.name     = SO101_JOINT_NAMES
                 m.position = step.tolist()
                 result.append(m)
             return result

        Returns:
            List[JointState] 길이 chunk_size. 미구현 또는 실패 시 None.
        """
        if self.policy is None:
            self.get_logger().warn(
                '[act_node] policy 미로드. _load_policy() 구현 후 사용 가능.',
                throttle_duration_sec=5.0,
            )
            return None

        if self._latest_rgb is None or self._latest_joints is None:
            self.get_logger().warn(
                '[act_node] RGB/joint_states 미수신, 예측 보류',
                throttle_duration_sec=2.0,
            )
            return None

        # TODO: 위 독스트링의 구현 내용 채우기
        raise NotImplementedError(
            "[act_node] _predict_chunk() 미구현. "
            "ACT Co-training 완료 후 LeRobot Policy 추론을 구현하세요."
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
