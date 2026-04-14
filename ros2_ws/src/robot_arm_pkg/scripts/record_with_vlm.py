"""
record_with_vlm.py

LeRobot record 루프를 커스터마이징하여 YOLO의 object_pose와
카메라 이미지(observation.images.top)를 observation에 포함시키는 데이터 수집 스크립트.

실행 환경:
  Python   3.10
  LeRobot  SO-101 (so101_follower / so101_leader)
  ROS2     Humble

전제조건:
  1. realsense2_camera 드라이버 실행 중:
       ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
  2. camera_node 실행 중 (/camera/rgb, /camera/depth 발행):
       ros2 run robot_arm_pkg camera_node
  3. llama.cpp 서버 실행 중 (vlm_node 추론용):
       ~/llama.cpp/llama-server -m ~/models/vlm/vlm_qwen3.5_4b_q4.gguf --host 0.0.0.0 --port 8080
  4. vlm_node 실행 중 (/target_class 발행):
       ros2 run robot_arm_pkg vlm_node --ros-args -p server_url:=http://localhost:8080
  5. yolo_node 실행 중 (/object_pose 30Hz 발행):
       ros2 run robot_arm_pkg yolo_node --ros-args -p model_path:=~/models/yolo/yolo_model.pt
  6. SO-101 팔로워/리더 USB 포트 확인
  7. LeRobot 캘리브레이션 완료 (lerobot-calibrate)

동작 방식:
  - 스크립트 시작 시 /task_command 와 /task_active=True 를 발행하여
    vlm_node(추론 시작)와 yolo_node(탐지 시작)를 활성화함
  - SensorCache 노드가 /object_pose(30Hz)와 /camera/rgb(30Hz)를 구독하여
    최신 값을 캐싱함
  - record 루프(30Hz)가 매 프레임마다 캐시에서 최신 값을 읽어 데이터셋에 삽입
  - pose 미수신 프레임: zero tensor (shape: [3]) 삽입
  - image 미수신 프레임: 검정 이미지 (H x W x 3) 삽입

실행 예시:
  python record_with_vlm.py \\
    --robot-port      /dev/ttyACM0 \\
    --robot-id        follower \\
    --teleop-port     /dev/ttyACM1 \\
    --teleop-id       leader \\
    --dataset-repo-id myname/my_dataset \\
    --single-task     "pick and place" \\
    --num-episodes    50
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from PIL import Image as PILImage

# LeRobot 0.4.4 기준 경로
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ──────────────────────────────────────────────
# 센서 캐시 — VLM pose + 카메라 이미지를 단일 노드로 관리
# ──────────────────────────────────────────────

class SensorCache(Node):
    """
    /object_pose 와 /camera/rgb 를 구독하여 최신 값을 캐싱하는 ROS2 노드.

    데이터 수집 시작 시 /task_command 와 /task_active=True 를 발행하여
    vlm_node(추론)와 yolo_node(탐지)를 활성화한다.
    미수신 시 각각 zero tensor / 검정 이미지 반환.
    """

    def __init__(self, image_height: int, image_width: int):
        super().__init__('sensor_cache')
        self._bridge = CvBridge()
        self._image_height = image_height
        self._image_width  = image_width

        # YOLO pose 상태
        self._latest_pose: torch.Tensor     = torch.zeros(3)
        self._pose_received: bool           = False

        # 카메라 이미지 상태
        self._latest_image: Optional[PILImage.Image] = None

        # 구독
        self.create_subscription(
            PoseStamped, '/object_pose', self._pose_callback, 10
        )
        self.create_subscription(
            Image, '/camera/rgb', self._image_callback, 10
        )

        # 발행 — vlm_node / yolo_node 활성화용
        self._task_cmd_pub    = self.create_publisher(String, '/task_command', 10)
        self._task_active_pub = self.create_publisher(Bool,   '/task_active',  10)

        self.get_logger().info(
            'SensorCache 시작\n'
            '  구독: /object_pose, /camera/rgb\n'
            '  발행: /task_command, /task_active'
        )

    # ── 콜백 ──────────────────────────────────

    def _pose_callback(self, msg: PoseStamped) -> None:
        self._latest_pose = torch.tensor([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=torch.float32)
        if not self._pose_received:
            self._pose_received = True
            self.get_logger().info(
                f'첫 번째 object_pose 수신: {self._latest_pose.tolist()}'
            )

    def _image_callback(self, msg: Image) -> None:
        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self._latest_image = PILImage.fromarray(cv_image)

    # ── 조회 ──────────────────────────────────

    def get_pose(self) -> torch.Tensor:
        """최신 pose 반환. 미수신 시 zero tensor (shape: [3]) 반환."""
        return self._latest_pose.clone()

    def get_image(self) -> PILImage.Image:
        """최신 RGB 이미지 반환. 미수신 시 검정 이미지 반환."""
        if self._latest_image is not None:
            return self._latest_image.copy()
        return PILImage.fromarray(
            np.zeros((self._image_height, self._image_width, 3), dtype=np.uint8)
        )

    @property
    def has_pose(self) -> bool:
        return self._pose_received

    @property
    def has_image(self) -> bool:
        return self._latest_image is not None

    # ── 태스크 제어 ───────────────────────────

    def start_task(self, task: str) -> None:
        """
        /task_command 와 /task_active=True 발행.

        vlm_node가 추론을 시작하고 yolo_node가 /object_pose 발행을 시작하도록
        트리거한다. 에피소드 루프 진입 전 한 번만 호출한다.
        """
        cmd_msg = String()
        cmd_msg.data = task
        self._task_cmd_pub.publish(cmd_msg)

        active_msg = Bool()
        active_msg.data = True
        self._task_active_pub.publish(active_msg)

        self.get_logger().info(
            f'/task_command={task!r}, /task_active=True 발행 → vlm_node/yolo_node 활성화'
        )

    def stop_task(self) -> None:
        """/task_active=False 발행 — 수집 종료 시 yolo_node 탐지 중단."""
        active_msg = Bool()
        active_msg.data = False
        self._task_active_pub.publish(active_msg)
        self.get_logger().info('/task_active=False 발행 → yolo_node 탐지 중단')


# ──────────────────────────────────────────────
# ROS2 스핀 — 별도 스레드에서 실행
# ──────────────────────────────────────────────

def _spin_ros(node: Node, stop_event: threading.Event) -> None:
    """SensorCache 노드를 별도 스레드에서 spin."""
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
    image_height:    int,
    image_width:     int,
) -> None:

    # ── ROS2 초기화 ───────────────────────────
    rclpy.init()
    sensor_cache = SensorCache(image_height=image_height, image_width=image_width)

    stop_event = threading.Event()
    ros_thread = threading.Thread(
        target=_spin_ros,
        args=(sensor_cache, stop_event),
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
    dataset = LeRobotDataset.create(
        repo_id=dataset_repo_id,
        fps=fps,
        features={
            'observation.state': {
                'dtype': 'float32',
                'shape': (6,),
                'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5'],
            },
            'observation.object_pose': {
                'dtype': 'float32',
                'shape': (3,),
                'names': ['x', 'y', 'z'],
            },
            'observation.images.top': {
                'dtype': 'image',
                'shape': (image_height, image_width, 3),
                'names': ['height', 'width', 'channel'],
            },
            'action': {
                'dtype': 'float32',
                'shape': (6,),
                'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5'],
            },
        },
    )

    print(f'\n[record_with_vlm] 데이터 수집 시작')
    print(f'  태스크: {single_task}')
    print(f'  에피소드 수: {num_episodes}')
    print(f'  FPS: {fps}')
    print(f'  이미지 해상도: {image_width}x{image_height}')

    # ── vlm_node / yolo_node 활성화 ───────────
    # publisher가 subscriber에 연결될 때까지 잠시 대기 후 발행
    time.sleep(0.5)
    sensor_cache.start_task(single_task)
    # vlm_node가 첫 번째 /target_class를 발행하고
    # yolo_node가 /object_pose를 발행하기까지 여유 시간 확보
    print('[record_with_vlm] vlm_node / yolo_node 활성화 대기 중 (3s)...')
    time.sleep(3.0)

    try:
        for episode_idx in range(num_episodes):
            print(f'\n[에피소드 {episode_idx + 1}/{num_episodes}] 워밍업 {warmup_time_s}s...')
            time.sleep(warmup_time_s)

            if not sensor_cache.has_pose:
                print(
                    '[경고] /object_pose 미수신. '
                    'yolo_node가 실행 중인지 확인하세요. zero tensor로 기록합니다.'
                )
            if not sensor_cache.has_image:
                print(
                    '[경고] /camera/rgb 미수신. '
                    'camera_node가 실행 중인지 확인하세요. 검정 이미지로 기록합니다.'
                )

            print(f'[에피소드 {episode_idx + 1}] 기록 시작 ({episode_time_s}s)...')
            start = time.time()
            frame_interval = 1.0 / fps

            while time.time() - start < episode_time_s:
                frame_start = time.time()

                # 관절 상태 및 리더 액션 읽기
                obs    = robot.get_observation()
                action = teleop.get_action()

                # 데이터셋에 프레임 추가
                dataset.add_frame({
                    'observation.state':       obs['observation.state'],
                    'observation.object_pose': sensor_cache.get_pose(),
                    'observation.images.top':  sensor_cache.get_image(),
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
        sensor_cache.stop_task()
        stop_event.set()
        robot.disconnect()
        teleop.disconnect()
        sensor_cache.destroy_node()
        rclpy.shutdown()
        print('\n[record_with_vlm] 데이터 수집 완료')


# ──────────────────────────────────────────────
# CLI 인자 파싱
# ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='LeRobot record with VLM object_pose + RealSense image'
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
    parser.add_argument('--image-height',    type=int,   default=480)
    parser.add_argument('--image-width',     type=int,   default=640)
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
        image_height=args.image_height,
        image_width=args.image_width,
    )
