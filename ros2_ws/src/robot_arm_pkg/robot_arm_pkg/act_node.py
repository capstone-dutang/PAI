"""
act_node.py

LeRobot ACT Policy (.pth 체크포인트) 로 관절 시퀀스를 추론하는 노드.
VLM이 발행한 object_pose를 캐싱하고, 30Hz로 joint_command를 발행.

환경:
  Python   3.10
  ROS2     Humble (Humboldt, 22.04 LTS)
  LeRobot  ACT Policy — .pth 체크포인트를 torch로 직접 추론 (TensorRT 미사용)

구독 토픽:
  /camera/rgb    (sensor_msgs/Image)
  /camera/depth  (sensor_msgs/Image)
  /object_pose   (geometry_msgs/PoseStamped) — vlm_node에서 수신, 캐싱
  /joint_states  (sensor_msgs/JointState)    — robot_controller 피드백

발행 토픽:
  /joint_command (sensor_msgs/JointState)    — 30Hz

파라미터:
  model_path  (str)   — LeRobot ACT 체크포인트 경로 (.pth)
  chunk_size  (int)   — 한 번에 예측할 스텝 수 (기본 100)
  publish_hz  (float) — 발행 주기 (기본 30Hz)

observation 구성 (학습 시와 동일해야 함):
  observation.image        — RGB 이미지 텐서
  observation.depth        — Depth 맵 텐서
  observation.state        — 관절 상태 텐서 (6축, SO-101 순서)
  observation.object_pose  — VLM 추론 결과 (x, y, z)

TODO (Part E 완료 후 구현):
  _load_policy()  — 체크포인트 로드 및 normalizer 설정 확인
  _predict_chunk() 내부 — 입력 텐서 전처리 및 policy.select_action() 호출
  입력 전처리 방식은 체크포인트에 포함된 normalizer 설정에 따라 결정됨
"""

from __future__ import annotations

from typing import List, Optional

import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped

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
        self.declare_parameter('model_path', '/home/user/models/act_policy.pth')
        self.declare_parameter('chunk_size', 100)
        self.declare_parameter('publish_hz', 30.0)

        self.model_path = self.get_parameter('model_path').value
        self.chunk_size = self.get_parameter('chunk_size').value
        publish_hz      = self.get_parameter('publish_hz').value

        # ── LeRobot Policy 로드 ───────────────────
        # TODO: Part E 완료 후 _load_policy() 구현
        self.policy = self._load_policy()

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

    # ── Policy 로드 ───────────────────────────────

    def _load_policy(self):
        """
        LeRobot ACT 체크포인트를 로드하여 반환.

        TODO: Part E(ACT 학습) 완료 후 구현
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
            'Part E 완료 후 LeRobot ACT 체크포인트 로드를 구현하세요.'
        )
        return None

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

        TODO: Part E(ACT 학습) 완료 후 구현
          체크포인트의 normalizer 설정 확인 후 아래를 구현:

          1. 입력 텐서 구성 — normalizer 설정에 맞게 전처리
             obs = {
                 'observation.image':       _to_image_tensor(self._latest_rgb),
                 'observation.depth':       _to_depth_tensor(self._latest_depth),
                 'observation.state':       _to_state_tensor(self._latest_joints),
                 'observation.object_pose': _to_pose_tensor(self._latest_pose),
             }

          2. LeRobot Policy 추론
             with torch.no_grad():
                 action_chunk = self.policy.select_action(obs)
             # action_chunk shape: (chunk_size, 6)

          3. 출력 텐서 → JointState 리스트 변환
             result = []
             for step in action_chunk:
                 msg = JointState()
                 msg.name     = SO101_JOINT_NAMES
                 msg.position = step.tolist()
                 result.append(msg)
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

        # TODO: 위 독스트링의 구현 내용 채우기
        raise NotImplementedError(
            "[act_node] _predict_chunk() 미구현. "
            "Part E 완료 후 LeRobot Policy 추론을 구현하세요."
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
