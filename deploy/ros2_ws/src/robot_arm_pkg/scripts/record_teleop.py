"""
record_teleop.py

Part D — SO-101 teleop 기반 ACT 학습용 시연 데이터 수집 스크립트.

`agent_node` / Gemma LoRA는 사용하지 않는다. SO-101 리더(작업자가 직접 조작)의
관절 값을 SO-101 팔로워에 그대로 미러링하면서, 시점마다 다음을 기록한다:

  observation.state              (6,)   float32  — 팔로워 관절 위치
  observation.language_embedding (D,)   float32  — embedding_node가 산출한
                                                    자연어 타겟 묘사 임베딩
  observation.images.top         (H,W,3) uint8   — /camera/rgb (RealSense RGB)
  action                         (6,)   float32  — 리더 관절 값 (정답)

전제조건 (README "데이터 수집 (Part D)" 참조):
  1. realsense2_camera 드라이버 실행 (align_depth.enable:=true)
  2. camera_node 실행 (/camera/rgb 발행)
  3. embedding_node 실행 (/embed_text Service)
  4. SO-101 리더/팔로워 USB 포트 확인 + lerobot-calibrate 완료
  5. 임베딩 모델 결정 (README 미확정 #1)
     ─ Part C/E의 학습 단계, Part F의 배포 단계와 동일 모델이어야 함

CLI 사용 예 (README 명시):
  python record_teleop.py \\
    --robot-port      /dev/ttyACM0 \\
    --robot-id        follower \\
    --teleop-port     /dev/ttyACM1 \\
    --teleop-id       leader \\
    --dataset-repo-id myname/dataset_real \\
    --single-task     "pick and place" \\
    --num-episodes    50

미구현 항목 (구현 시 결정 필요):
  - 자연어 라벨 입력 방식 (README 미확정 #3, Option A/B/C 중 하나):
      Option A: 에피소드 시작 시 작업자가 자유 텍스트 입력 (다양성↑, 일관성↓)
      Option B: --single-task CLI 인자로 전체 통일 (일관성↑, 다양성↓)
      Option C: 도메인별 prompt 템플릿 + 작업자가 변수만 입력 (권장)
  - LeRobotDataset 0.4.4 features 스키마에 language_embedding 채널 추가 방법
    (임베딩 차원 D는 첫 _embed_text() 호출 응답에서 결정됨)
  - 에피소드 종료 조건 (키 입력 / 타임아웃 / 리더 정지 감지)
  - 30Hz 동기화 루프 (rclpy 콜백 / LeRobot read 주기)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image

from robot_arm_msgs.srv import EmbedText


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@dataclass
class RecordArgs:
    robot_port:       str
    robot_id:         str
    teleop_port:      str
    teleop_id:        str
    dataset_repo_id:  str
    single_task:      str
    num_episodes:     int
    fps:              int


def parse_args() -> RecordArgs:
    p = argparse.ArgumentParser(description='Part D — SO-101 teleop 시연 데이터 수집')
    p.add_argument('--robot-port',      required=True, help='SO-101 팔로워 USB 포트')
    p.add_argument('--robot-id',        default='follower', help='팔로워 캘리브레이션 ID')
    p.add_argument('--teleop-port',     required=True, help='SO-101 리더 USB 포트')
    p.add_argument('--teleop-id',       default='leader',   help='리더 캘리브레이션 ID')
    p.add_argument('--dataset-repo-id', required=True, help='LeRobotDataset repo id (예: user/dataset_real)')
    p.add_argument('--single-task',     required=True, help='기본 태스크 라벨 (Option B 사용 시)')
    p.add_argument('--num-episodes',    type=int, default=50, help='수집 에피소드 수')
    p.add_argument('--fps',             type=int, default=30, help='기록 주기 (Hz)')
    a = p.parse_args()
    return RecordArgs(
        robot_port      = a.robot_port,
        robot_id        = a.robot_id,
        teleop_port     = a.teleop_port,
        teleop_id       = a.teleop_id,
        dataset_repo_id = a.dataset_repo_id,
        single_task     = a.single_task,
        num_episodes    = a.num_episodes,
        fps             = a.fps,
    )


# ──────────────────────────────────────────────
# ROS2 보조 노드 — 카메라 캐싱 + embedding 서비스 클라이언트
# ──────────────────────────────────────────────

class RecordHelperNode(Node):
    """
    record_teleop가 ROS2와 상호작용하기 위한 헬퍼 노드.
      - /camera/rgb 구독 (BestEffort + KeepLast(1)) — latest-only 캐시
      - /embed_text 서비스 클라이언트
    """

    def __init__(self):
        super().__init__('record_teleop_helper')

        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._latest_rgb: Optional[Image] = None
        self.create_subscription(Image, '/camera/rgb', self._rgb_cb, camera_qos)

        self.embed_client = self.create_client(EmbedText, '/embed_text')

    def _rgb_cb(self, msg: Image) -> None:
        self._latest_rgb = msg

    def latest_rgb(self) -> Optional[Image]:
        return self._latest_rgb

    def call_embed_text(self, text: str, timeout_sec: float = 5.0) -> Optional[List[float]]:
        """
        /embed_text 서비스 호출. 응답의 dim==0 이면 모델 미로드/실패로 간주하고 None 반환.
        """
        if not self.embed_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error('/embed_text 서비스 연결 실패')
            return None

        req = EmbedText.Request()
        req.text = text
        future = self.embed_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        resp = future.result()
        if resp is None or resp.dim == 0:
            self.get_logger().error(
                f'/embed_text 응답 실패 또는 dim=0 (text={text!r})'
            )
            return None
        return list(resp.embedding)


# ──────────────────────────────────────────────
# 라벨링 (Option A/B/C 미결정 — 미구현)
# ──────────────────────────────────────────────

def prompt_episode_label(args: RecordArgs, episode_idx: int) -> str:
    """
    에피소드별 자연어 라벨을 결정한다.

    TODO (README 미확정 #3): 다음 중 하나로 확정 후 구현
      Option A: input()으로 작업자가 자유 텍스트 입력
      Option B: args.single_task 그대로 반환
      Option C: 도메인별 prompt 템플릿 + 변수만 입력받기 (권장)
                  예: 템플릿 "{color} {object}을(를) 집어"에서
                       color/object만 input으로 받고 채워넣기
    """
    raise NotImplementedError(
        "[record_teleop] 라벨링 방식 미결정 (README 미확정 #3). "
        "Option A/B/C 중 확정 후 prompt_episode_label()을 구현하세요."
    )


# ──────────────────────────────────────────────
# 데이터셋 / 로봇 연결 (미구현)
# ──────────────────────────────────────────────

def open_dataset(args: RecordArgs, embedding_dim: int):
    """
    LeRobotDataset (0.4.4)을 features 스키마와 함께 생성/오픈한다.

    TODO: features 스키마 확정 후 구현
      예시 features:
        {
            'observation.state':              {'dtype': 'float32', 'shape': (6,)},
            'observation.language_embedding': {'dtype': 'float32', 'shape': (embedding_dim,)},
            'observation.images.top':         {'dtype': 'video',   'shape': (H, W, 3)},
            'action':                         {'dtype': 'float32', 'shape': (6,)},
        }

      embedding_dim은 첫 prompt_episode_label() 호출 후 helper.call_embed_text()
      응답의 길이로 결정된다. 즉, 첫 에피소드 시작 시점에 데이터셋 features를 확정한다.

      Sim 데이터셋과의 호환을 위해 동일 임베딩 모델 + 동일 D 보장이 필수.
    """
    raise NotImplementedError(
        "[record_teleop] open_dataset() 미구현. "
        "LeRobotDataset 0.4.4 features 스키마 확정 후 구현하세요."
    )


def connect_so101(robot_port: str, robot_id: str, teleop_port: str, teleop_id: str):
    """
    SO-101 팔로워(robot) + 리더(teleop) 연결.

    TODO: LeRobot SDK 호출. robot_controller_node.py와 동일한 방식 사용.
      from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
      from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
      ...

    Returns:
        (robot, teleop) 튜플
    """
    raise NotImplementedError(
        "[record_teleop] connect_so101() 미구현."
    )


# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────

def record_episode(
    args:          RecordArgs,
    helper:        RecordHelperNode,
    robot,
    teleop,
    dataset,
    episode_idx:   int,
    language_text: str,
    embedding:     List[float],
) -> None:
    """
    1개 에피소드 기록.

    TODO: 30Hz 루프로 다음을 동시 수행
      - teleop.get_action()      → 리더 관절 값 (action)
      - robot.send_action(...)   → 팔로워 미러링
      - robot.get_observation()  → 팔로워 관절 값 (observation.state)
      - helper.latest_rgb()      → 카메라 RGB (observation.images.top)
      - embedding 그대로         → observation.language_embedding (에피소드 내 상수)
      - dataset.add_frame(...)

      종료 조건:
        - 작업자 키 입력 (예: SPACE / ENTER)
        - 또는 타임아웃 (args.fps 기준 N초)
        — 종료 정책 미결정. 운영 중 결정.
    """
    raise NotImplementedError(
        "[record_teleop] record_episode() 미구현. "
        "30Hz 동기화 루프 / 에피소드 종료 정책 확정 후 구현하세요."
    )


def main() -> None:
    args = parse_args()

    rclpy.init()
    helper = RecordHelperNode()

    try:
        # 1) 첫 에피소드 라벨 → embedding_dim 결정 → 데이터셋 features 확정
        first_label = prompt_episode_label(args, episode_idx=0)
        first_embed = helper.call_embed_text(first_label)
        if first_embed is None:
            helper.get_logger().error(
                '첫 에피소드 임베딩 획득 실패. embedding_node 상태 확인 후 재시도.'
            )
            return
        embedding_dim = len(first_embed)
        helper.get_logger().info(
            f'embedding_dim = {embedding_dim} (이후 모든 에피소드에서 동일해야 함)'
        )

        # 2) 데이터셋 / 로봇 연결
        dataset = open_dataset(args, embedding_dim)
        robot, teleop = connect_so101(
            args.robot_port, args.robot_id, args.teleop_port, args.teleop_id,
        )

        # 3) 에피소드 루프
        for ep in range(args.num_episodes):
            if ep == 0:
                label, embed = first_label, first_embed
            else:
                label = prompt_episode_label(args, ep)
                embed = helper.call_embed_text(label)
                if embed is None or len(embed) != embedding_dim:
                    helper.get_logger().error(
                        f'에피소드 {ep} 임베딩 차원 불일치 또는 실패. 스킵.'
                    )
                    continue

            record_episode(args, helper, robot, teleop, dataset, ep, label, embed)

    finally:
        helper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
