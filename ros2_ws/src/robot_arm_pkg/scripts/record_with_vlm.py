"""
record_with_vlm.py

LeRobot record 루프를 커스터마이징하여 VLM의 object_pose를
observation에 포함시키는 데이터 수집 스크립트.

실행 환경:
  Python   3.10
  LeRobot  SO-101 (so101_follower / so101_leader)
  ROS2     Humble — /object_pose 토픽 구독용

동작 방식:
  - LeRobot record 루프(30Hz)가 돌면서 매 프레임에 VLMPoseCache의
    최신 pose를 'observation.object_pose' 키로 삽입함
  - VLM은 0.4Hz(2.5초 주기)로 추론하므로, 두 루프의 주기 차이는
    캐싱으로 처리함 (최신 pose를 매 프레임에 그대로 삽입)
  - pose가 아직 수신되지 않은 프레임에는 zero tensor (shape: [3]) 삽입

실행 전 준비사항:
  1. vlm_node 실행 및 /object_pose 토픽 발행 중인지 확인
  2. SO-101 팔로워/리더 USB 포트 확인
  3. 캘리브레이션 완료 여부 확인 (lerobot-calibrate)

실행 예시:
  python record_with_vlm.py \
    --robot-port /dev/ttyACM0 \
    --robot-id   follower \
    --teleop-port /dev/ttyACM1 \
    --teleop-id   leader \
    --dataset-repo-id myname/my_dataset \
    --single-task "pick and place" \
    --num-episodes 50 \
    --fps 30

TODO (Part C VLM 완료 후):
  vlm_node가 실제 추론 결과를 /object_pose로 발행하면
  VLMPoseCache가 자동으로 캐싱함. 코드 수정 불필요.

TODO (카메라 확정 후):
  --cameras 인자에 실제 카메라 설정을 지정:
  예) --cameras "{ top: {type: realsense, width: 640, height: 480, fps: 30} }"
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# LeRobot 0.4.4 기준 경로
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ──────────────────────────────────────────────
# VLM pose 캐시 — ROS2 토픽 구독 + 최신값 보관
# ──────────────────────────────────────────────

class VLMPoseCache(Node):
    """
    /object_pose 토픽을 구독하여 최신 pose를 캐싱하는 ROS2 노드.

    VLM(0.4Hz)과 record 루프(30Hz)의 주기 차이를 캐싱으로 처리.
    pose 미수신 시 zero tensor 반환.
    """

    def __init__(self):
        super().__init__('vlm_pose_cache')
        self._latest: torch.Tensor = torch.zeros(3)  # (x, y, z)
        self._received = False

        self.create_subscription(
            PoseStamped,
            '/object_pose',
            self._callback,
            10,
        )
        self.get_logger().info('VLMPoseCache 시작 — /object_pose 구독 중')

    def _callback(self, msg: PoseStamped) -> None:
        self._latest = torch.tensor([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=torch.float32)
        if not self._received:
            self._received = True
            self.get_logger().info(
                f'첫 번째 object_pose 수신: {self._latest.tolist()}'
            )

    def get(self) -> torch.Tensor:
        """최신 pose 반환. 미수신 시 zero tensor (shape: [3]) 반환."""
        return self._latest.clone()

    @property
    def has_pose(self) -> bool:
        """최소 한 번이라도 pose를 수신했는지 여부."""
        return self._received


# ──────────────────────────────────────────────
# ROS2 스핀 — 별도 스레드에서 실행
# ──────────────────────────────────────────────

def _spin_ros(node: Node, stop_event: threading.Event) -> None:
    """VLMPoseCache 노드를 별도 스레드에서 spin."""
    while not stop_event.is_set():
        rclpy.spin_once(node, timeout_sec=0.01)


# ──────────────────────────────────────────────
# 메인 record 루프
# ──────────────────────────────────────────────

def record_with_vlm(
    robot_port:      str,
    robot_id:        str,
    teleop_port:     str,
    teleop_id:       str,
    dataset_repo_id: str,
    single_task:     str,
    num_episodes:    int,
    fps:             int,
    warmup_time_s:   float,
    episode_time_s:  float,
    reset_time_s:    float,
) -> None:

    # ── ROS2 초기화 ───────────────────────────
    rclpy.init()
    pose_cache = VLMPoseCache()

    stop_event = threading.Event()
    ros_thread = threading.Thread(
        target=_spin_ros,
        args=(pose_cache, stop_event),
        daemon=True,
    )
    ros_thread.start()

    # ── SO-101 초기화 ─────────────────────────
    robot_cfg = SO101FollowerConfig(port=robot_port, id=robot_id)
    robot     = SO101Follower(robot_cfg)
    robot.connect()

    teleop_cfg = SO101LeaderConfig(port=teleop_port, id=teleop_id)
    teleop     = SO101Leader(teleop_cfg)
    teleop.connect()

    # ── 데이터셋 초기화 ───────────────────────
    # TODO: 카메라 확정 후 cameras 파라미터 추가
    dataset = LeRobotDataset.create(
        repo_id=dataset_repo_id,
        fps=fps,
        features={
            'observation.state':        {'dtype': 'float32', 'shape': (6,)},
            'observation.object_pose':  {'dtype': 'float32', 'shape': (3,)},
            'action':                   {'dtype': 'float32', 'shape': (6,)},
        },
    )

    print(f'\n[record_with_vlm] 데이터 수집 시작')
    print(f'  태스크: {single_task}')
    print(f'  에피소드 수: {num_episodes}')
    print(f'  FPS: {fps}')

    try:
        for episode_idx in range(num_episodes):
            print(f'\n[에피소드 {episode_idx + 1}/{num_episodes}] 워밍업 {warmup_time_s}s...')
            time.sleep(warmup_time_s)

            # pose 수신 여부 경고
            if not pose_cache.has_pose:
                print(
                    '[경고] /object_pose 미수신. '
                    'vlm_node가 실행 중인지 확인하세요. zero tensor로 기록합니다.'
                )

            print(f'[에피소드 {episode_idx + 1}] 기록 시작 ({episode_time_s}s)...')
            start = time.time()
            frame_interval = 1.0 / fps

            while time.time() - start < episode_time_s:
                frame_start = time.time()

                # 관절 상태 및 리더 액션 읽기
                obs    = robot.get_observation()
                action = teleop.get_action()

                # object_pose 채널 삽입
                obs['observation.object_pose'] = pose_cache.get()

                # 데이터셋에 프레임 추가
                dataset.add_frame({
                    'observation.state':       obs['observation.state'],
                    'observation.object_pose': obs['observation.object_pose'],
                    'action':                  action,
                })

                # 팔로워에 리더 액션 전달
                robot.send_action(action)

                # fps 유지
                elapsed = time.time() - frame_start
                sleep_t = frame_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            # 에피소드 저장
            dataset.save_episode(task=single_task)
            print(f'[에피소드 {episode_idx + 1}] 저장 완료')

            if episode_idx < num_episodes - 1:
                print(f'리셋 대기 {reset_time_s}s...')
                time.sleep(reset_time_s)

    finally:
        stop_event.set()
        robot.disconnect()
        teleop.disconnect()
        pose_cache.destroy_node()
        rclpy.shutdown()
        print('\n[record_with_vlm] 데이터 수집 완료')


# ──────────────────────────────────────────────
# CLI 인자 파싱
# ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='LeRobot record with VLM object_pose'
    )
    parser.add_argument('--robot-port',      default='/dev/ttyACM0')
    parser.add_argument('--robot-id',        default='follower')
    parser.add_argument('--teleop-port',     default='/dev/ttyACM1')
    parser.add_argument('--teleop-id',       default='leader')
    parser.add_argument('--dataset-repo-id', required=True)
    parser.add_argument('--single-task',     default='pick and place')
    parser.add_argument('--num-episodes',    type=int,   default=50)
    parser.add_argument('--fps',             type=int,   default=30)
    parser.add_argument('--warmup-time-s',   type=float, default=5.0)
    parser.add_argument('--episode-time-s',  type=float, default=30.0)
    parser.add_argument('--reset-time-s',    type=float, default=10.0)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    record_with_vlm(
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        teleop_port=args.teleop_port,
        teleop_id=args.teleop_id,
        dataset_repo_id=args.dataset_repo_id,
        single_task=args.single_task,
        num_episodes=args.num_episodes,
        fps=args.fps,
        warmup_time_s=args.warmup_time_s,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
    )
