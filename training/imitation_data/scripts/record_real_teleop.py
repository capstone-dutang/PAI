"""
record_real_teleop.py

실행 위치: Jetson (학습 준비 단계 — PAI 전체 clone + `source ~/ros2_ws/install/setup.bash` 필요).

SO-101 teleop 기반 ACT 학습용 시연 데이터 수집 스크립트.

`agent_node` / Gemma LoRA는 사용하지 않는다. SO-101 리더(작업자가 직접 조작)의
관절 값을 SO-101 팔로워에 그대로 미러링하면서, 시점마다 다음을 기록한다:

  observation.state              (6,)   float32  — 팔로워 관절 위치
  observation.language_embedding (384,) float32  — embedding_node가 산출한
                                                    자연어 타겟 묘사 임베딩
                                                    (paraphrase-multilingual-MiniLM-L12-v2)
  observation.images.top         (H,W,3) uint8   — /camera/rgb (RealSense RGB)
  action                         (6,)   float32  — 리더 관절 값 (정답)

언어 라벨링 — Option C (템플릿 + 변수 슬롯) 확정:
  작업자는 (object, attribute, spatial) 3개 슬롯을 deploy/config/label_vocab.yaml
  vocabulary 안에서만 선택한다. canonical 라벨은 자동 생성된다:

      "the {attribute} {object} {spatial}"
      예: "the blue can on the right side of the table"

  자유 텍스트 입력 금지 — 영어 템플릿 고정으로 Sim/Real 분포 일치 보장.
  수집 후 paraphrase는 별도 training/imitation_data/scripts/postprocess_real_labels.py로 후처리.

전제조건 (README "데이터 수집" 참조):
  1. realsense2_camera 드라이버 실행 (align_depth.enable:=true)
  2. camera_node 실행 (/camera/rgb 발행)
  3. embedding_node 실행 (/embed_text Service,
     paraphrase-multilingual-MiniLM-L12-v2)
  4. SO-101 리더/팔로워 USB 포트 확인 + lerobot-calibrate 완료
  5. deploy/config/label_vocab.yaml 배치

CLI 사용 예:
  python record_real_teleop.py \\
    --robot-port      /dev/ttyACM0 \\
    --robot-id        follower \\
    --teleop-port     /dev/ttyACM1 \\
    --teleop-id       leader \\
    --dataset-repo-id myname/dataset_real \\
    --domain          recycling \\
    --label-vocab     ~/ros2_ws/config/label_vocab.yaml \\
    --num-episodes    50 \\
    --fps             30 \\
    --img-h           480 \\
    --img-w           640 \\
    --home-pose-file  ~/calibration/home_pose.yaml

운영 / 측정 결정 사항은 본 파일 상단 "운영 / 측정 결정 사항" 블록에 정리되어 있다.
변경이 필요한 경우 그 블록 + 본 docstring + README "데이터 수집" 을 같이 갱신.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image

# LeRobot 0.4.4 — features/SO-101 SDK 경로 (robot_controller_node 와 동일 규약)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower      import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader,   SO101LeaderConfig

# 공유 유틸 — home_pose 파일 입출력 (robot_controller_node 와 같은 파일 공유)
from robot_arm_pkg.utils.home_pose import (
    DEFAULT_HOME_POSE_PATH,
    home_pose_exists,
    load_home_pose,
    save_home_pose,
)
# 공유 유틸 — /camera/rgb msg → numpy (cv_bridge 미사용, NumPy 2.x 호환)
from robot_arm_pkg.utils.ros_image import image_msg_to_rgb_array

from robot_arm_msgs.srv import EmbedText


# ──────────────────────────────────────────────
# 운영 / 측정 결정 사항 (변경 시 본 블록 + README "데이터 수집" 갱신)
# ──────────────────────────────────────────────
#
#   [1] USB 포트 점유 — record_real_teleop 단독 점유
#         record_real_teleop 실행 중에는 robot_controller_node 를 끄고 본 스크립트가
#         단독으로 follower 포트를 점유한다. LeRobot 표준 teleop 흐름과 일치.
#         동시에 robot_controller_node 가 떠 있으면 SDK 의 포트 open 단계에서
#         실패한다. README "데이터 수집" 실행 순서에도 명시.
#
#   [2] 이미지 소스 — ROS /camera/rgb
#         camera_node 가 발행하는 /camera/rgb 토픽을 helper 노드가 구독하여
#         최신 1프레임만 캐싱한다 (BestEffort + KeepLast(1)).
#         (LeRobot 자체 카메라 플러그인 직접 사용 안 함)
#
#   [3] 이미지 해상도 (H, W)
#         --img-h / --img-w CLI 인자로 결정. 기본 480 x 640 (RealSense 기본).
#         Sim 측 dataset feature shape 과 반드시 일치해야 Co-training 가능.
#
#   [4] 에피소드 종료 정책 — 'q' + Enter
#         녹화 루프 중 stdin 에 'q'(또는 'Q') + Enter 가 들어오면 정상 종료.
#         KeyboardInterrupt (Ctrl+C) 도 같은 경로로 흡수. 자동 타임아웃은
#         사용하지 않음 — 시연 자유도 우선. Linux/macOS 전용 (Jetson 타깃).
#
#   [5] 에피소드 간 reset — follower 자동 home 복귀 (파일 영구화)
#         녹화 직후 follower 를 *home pose* 로 자동 복귀시킨다.
#         home pose 는 --home-pose-file 이 가리키는 YAML (기본
#         ~/calibration/home_pose.yaml) 에서 로드한다.
#         파일이 없으면 connect 직후의 follower 자세를 캡처해 *자동 저장* 한다.
#         → 새 home 으로 바꾸려면 follower 를 새 자세에 두고 파일을 삭제한 뒤
#           record_real_teleop 을 한 번 실행하면 된다.
#         → 같은 파일을 robot_controller_node 의 /go_home 서비스가 공유하므로
#           추론(배포) 단계에서도 task 종료 후 동일 자세로 복귀한다.
#         복귀는 RETURN_HOME_DURATION_S 초 동안 선형 보간 (현재값 → home).
#
#   [6] fps — Jetson 실측 후 확정
#         기본 30Hz (--fps). Jetson 측정 결과 30Hz 미달 시 15Hz 등으로 낮춰
#         재기록 필요. Sim 측 fps 와 학습 시 일치 또는 resample 처리 필요.
#         본 루프는 deadline 기반 스케줄을 사용하므로 한 주기 이상 밀리면 자동
#         재동기화한다 (drift 방지).
#
# ──────────────────────────────────────────────


EXPECTED_EMBED_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2

# SO-101 관절 이름 — robot_controller_node.py 와 동일 순서.
# LeRobot SDK 의 motor 키와 일치 (get_observation/send_action 은 {name}.pos 형식).
SO101_JOINT_NAMES: List[str] = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]

# 이미지 기본값 (operational decision [3])
DEFAULT_IMG_H = 480
DEFAULT_IMG_W = 640

# Home pose 복귀 시간 (operational decision [5])
RETURN_HOME_DURATION_S = 2.0

# LeRobot 데이터셋 카메라 채널 키 (act_node._predict_chunk obs 키와 일치 필수)
CAMERA_KEY = 'observation.images.top'

# /camera/rgb msg → numpy (rgb8) 변환은 robot_arm_pkg.utils.ros_image 에서 처리.
# cv_bridge 의존을 제거하여 NumPy 1.x / 2.x 양쪽 환경에서 동작하게 한다.


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
    domain:           str
    label_vocab:      str
    num_episodes:     int
    fps:              int
    labels_jsonl:     str
    img_h:            int
    img_w:            int
    home_pose_file:   str


def parse_args() -> RecordArgs:
    p = argparse.ArgumentParser(description='SO-101 teleop 시연 데이터 수집')
    p.add_argument('--robot-port',      required=True, help='SO-101 팔로워 USB 포트')
    p.add_argument('--robot-id',        default='follower', help='팔로워 캘리브레이션 ID')
    p.add_argument('--teleop-port',     required=True, help='SO-101 리더 USB 포트')
    p.add_argument('--teleop-id',       default='leader',   help='리더 캘리브레이션 ID')
    p.add_argument('--dataset-repo-id', required=True,
                   help='LeRobotDataset repo id (예: user/dataset_real)')
    p.add_argument('--domain',          required=True,
                   help='현재 수집할 도메인 (label_vocab.yaml의 domains 키와 일치)')
    p.add_argument('--label-vocab',     default='~/ros2_ws/config/label_vocab.yaml',
                   help='라벨 vocabulary YAML 경로')
    p.add_argument('--num-episodes',    type=int, default=50, help='수집 에피소드 수')
    p.add_argument('--fps',             type=int, default=30,
                   help='기록 주기 (Hz). operational decision [6] — Jetson 실측 후 변경.')
    p.add_argument('--labels-jsonl',    default='',
                   help='canonical 라벨 + 슬롯 메타데이터 기록 JSONL 경로. '
                        '비어 있으면 dataset.root 옆에 labels.jsonl 로 자동 생성. '
                        '이 파일은 postprocess_real_labels.py 입력으로 사용된다.')
    p.add_argument('--img-h',           type=int, default=DEFAULT_IMG_H,
                   help='이미지 높이 H (operational decision [3]). '
                        'Sim 측 feature shape 과 일치해야 Co-training 가능.')
    p.add_argument('--img-w',           type=int, default=DEFAULT_IMG_W,
                   help='이미지 너비 W (operational decision [3]).')
    p.add_argument('--home-pose-file',  default=DEFAULT_HOME_POSE_PATH,
                   help='home pose YAML 경로 (operational decision [5]). '
                        '존재하면 그 자세로 에피소드 간 자동 복귀. '
                        '없으면 connect 직후 follower 자세를 캡처해 같은 경로에 저장. '
                        f'기본 {DEFAULT_HOME_POSE_PATH} — robot_controller_node /go_home 과 공유.')
    a = p.parse_args()
    return RecordArgs(
        robot_port      = a.robot_port,
        robot_id        = a.robot_id,
        teleop_port     = a.teleop_port,
        teleop_id       = a.teleop_id,
        dataset_repo_id = a.dataset_repo_id,
        domain          = a.domain,
        label_vocab     = os.path.expanduser(a.label_vocab),
        num_episodes    = a.num_episodes,
        fps             = a.fps,
        labels_jsonl    = os.path.expanduser(a.labels_jsonl) if a.labels_jsonl else '',
        img_h           = a.img_h,
        img_w           = a.img_w,
        home_pose_file  = os.path.expanduser(a.home_pose_file),
    )


# ──────────────────────────────────────────────
# Vocabulary 로드 / 검증
# ──────────────────────────────────────────────

@dataclass
class LabelVocab:
    template:    str
    spatials:    List[str]
    objects:     Dict[str, List[str]]          # object_name -> attributes
    domain:      str = ''
    raw:         dict = field(default_factory=dict)


def load_vocab(path: str, domain: str) -> LabelVocab:
    """label_vocab.yaml 로드 + 지정 도메인 추출.

    스키마 검증:
      - template, 전역 spatial_qualifiers, domains[domain].objects 필수
      - 각 object는 attributes 리스트 필수
      - LabelVocab.spatials 는 (전역 spatial_qualifiers + 도메인별 spatial_qualifiers)
        병합 결과. 도메인별 항목은 "next to the {other_object}" 같은 관계형
        spatial 로, 해당 도메인 안에서만 의미가 있다.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    template        = raw.get('template')
    global_spatials = raw.get('spatial_qualifiers') or []
    domains         = raw.get('domains') or {}

    if not template or '{attribute}' not in template \
       or '{object}' not in template or '{spatial}' not in template:
        raise ValueError(
            f'label_vocab template이 비어 있거나 슬롯 누락 ({path}). '
            f'"the {{attribute}} {{object}} {{spatial}}" 형식 필요.'
        )
    if not global_spatials:
        raise ValueError(f'전역 spatial_qualifiers가 비어 있음 ({path}).')
    if domain not in domains:
        raise ValueError(
            f"도메인 '{domain}'이 label_vocab에 없음. 가능한 도메인: {list(domains)}"
        )

    domain_block      = domains[domain] or {}
    domain_spatials   = domain_block.get('spatial_qualifiers') or []
    merged_spatials   = list(global_spatials) + list(domain_spatials)

    obj_block = domain_block.get('objects') or {}
    objects: Dict[str, List[str]] = {}
    for obj_name, obj_def in obj_block.items():
        attrs = (obj_def or {}).get('attributes') or []
        if not attrs:
            raise ValueError(
                f"객체 '{obj_name}'에 attributes가 비어 있음 (domain={domain})."
            )
        objects[obj_name] = list(attrs)

    if not objects:
        raise ValueError(f"도메인 '{domain}'에 등록된 객체가 없음.")

    return LabelVocab(
        template=template,
        spatials=merged_spatials,
        objects=objects,
        domain=domain,
        raw=raw,
    )


# ──────────────────────────────────────────────
# 슬롯 입력 (Option C)
# ──────────────────────────────────────────────

def _pick(prompt: str, choices: List[str]) -> str:
    """번호 또는 값 입력으로 choices 중 하나를 선택.

    자유 텍스트 금지 — choices 안에 없으면 재시도.
    """
    while True:
        print(f"\n{prompt}")
        for i, c in enumerate(choices):
            print(f"  [{i}] {c}")
        raw = input(f"선택 (0-{len(choices)-1} 또는 값 입력): ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"  ! 범위 밖 — 0~{len(choices)-1}")
            continue
        if raw in choices:
            return raw
        print(f"  ! vocabulary 외 입력 거부. 위 목록에서만 선택 가능.")


def prompt_episode_label(
    vocab:       LabelVocab,
    episode_idx: int,
) -> Dict[str, str]:
    """
    Option C 슬롯 입력으로 canonical 라벨 생성.

    Returns:
        {
            'object':    str,
            'attribute': str,
            'spatial':   str,
            'canonical': str,   # 템플릿 채워진 영어 라벨
            'domain':    str,
            'episode':   int,
        }
    """
    print('\n' + '=' * 60)
    print(f'  [Episode {episode_idx}] 라벨 슬롯 입력 (domain={vocab.domain})')
    print('=' * 60)

    obj_names = list(vocab.objects.keys())
    obj   = _pick('1) Object',    obj_names)
    attr  = _pick('2) Attribute', vocab.objects[obj])
    spat  = _pick('3) Spatial',   vocab.spatials)

    canonical = vocab.template.format(attribute=attr, object=obj, spatial=spat)
    print(f"\n  canonical: {canonical}")

    return {
        'object':    obj,
        'attribute': attr,
        'spatial':   spat,
        'canonical': canonical,
        'domain':    vocab.domain,
        'episode':   episode_idx,
    }


# ──────────────────────────────────────────────
# 라벨 메타데이터 JSONL 기록 (postprocess 입력)
# ──────────────────────────────────────────────

def _resolve_labels_jsonl(args: RecordArgs, dataset: Optional[LeRobotDataset] = None) -> str:
    """
    labels.jsonl 저장 경로 결정.

      1) --labels-jsonl 가 명시되면 그 경로 사용.
      2) dataset 이 주어지면 dataset.root / 'labels.jsonl' (권장).
         postprocess_real_labels.py 도 같은 곳을 기본 입력으로 본다.
      3) 둘 다 아니면 CWD 에 labels_<repo>.jsonl 폴백.
    """
    if args.labels_jsonl:
        path = args.labels_jsonl
    elif dataset is not None:
        path = str(Path(dataset.root) / 'labels.jsonl')
    else:
        name = args.dataset_repo_id.replace('/', '_')
        path = os.path.join(os.getcwd(), f'labels_{name}.jsonl')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def append_label_record(jsonl_path: str, record: dict) -> None:
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ──────────────────────────────────────────────
# ROS2 보조 노드 — 카메라 캐싱 + embedding 서비스 클라이언트
# ──────────────────────────────────────────────

class RecordHelperNode(Node):
    """
    record_real_teleop가 ROS2와 상호작용하기 위한 헬퍼 노드.
      - /camera/rgb 구독 (BestEffort + KeepLast(1)) — latest-only 캐시
      - /embed_text 서비스 클라이언트

    main() 에서 별도 daemon 스레드의 SingleThreadedExecutor 에 add_node 되어
    메인 루프의 시리얼 IO / 디스크 IO 와 무관하게 콜백이 즉시 처리된다.
    이전 구조 (메인 루프에서 rclpy.spin_once(timeout_sec=0.0) 호출) 에서는
    시리얼/PNG 저장이 메인 스레드를 점유하는 동안 /camera/rgb 콜백이 밀려
    같은 이미지가 dataset 에 중복 적재되어 저장된 video 가 'freeze + 정상'
    패턴으로 보이는 문제가 발생했다.
    """

    def __init__(self):
        super().__init__('record_real_teleop_helper')

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

        본 노드는 main() 의 백그라운드 executor 에서 spin 되고 있으므로
        여기서 spin_until_future_complete 를 호출하면 이중 spin 으로 충돌한다.
        대신 future.done() 폴링으로 결과를 기다린다.
        """
        if not self.embed_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error('/embed_text 서비스 연결 실패')
            return None

        req = EmbedText.Request()
        req.text = text
        future = self.embed_client.call_async(req)
        deadline = time.perf_counter() + timeout_sec
        while not future.done() and time.perf_counter() < deadline:
            time.sleep(0.01)
        resp = future.result() if future.done() else None
        if resp is None or resp.dim == 0:
            self.get_logger().error(
                f'/embed_text 응답 실패 또는 dim=0 (text={text!r})'
            )
            return None
        if resp.dim != EXPECTED_EMBED_DIM:
            self.get_logger().error(
                f'/embed_text dim={resp.dim} 예상값 {EXPECTED_EMBED_DIM}와 불일치. '
                f'embedding_node 모델 확인 필요.'
            )
            return None
        return list(resp.embedding)


# ──────────────────────────────────────────────
# 데이터셋 / 로봇 연결
# ──────────────────────────────────────────────

def open_dataset(args: RecordArgs, embedding_dim: int) -> LeRobotDataset:
    """
    LeRobotDataset 0.4.4 인스턴스를 features 스키마와 함께 생성하거나
    기존 데이터셋을 그대로 열어 *resume* 한다.

    동작:
      - ~/.cache/huggingface/lerobot/<repo_id> 가 존재하면 LeRobotDataset() 로
        기존 데이터셋을 열어 신규 에피소드를 append (resume).
        language_embedding shape 만 검증하고 그 외 채널은 add_frame 의
        validate_frame 에 위임한다.
      - 존재하지 않으면 LeRobotDataset.create() 로 새로 생성.
        LeRobotDataset.create() 는 내부에서 exist_ok=False 로 mkdir 하므로
        기존 폴더가 남아 있으면 FileExistsError 가 난다 → 위 분기로 사전 차단.

    features 스키마 (lerobot.datasets.utils 의 표준 컨벤션 따름):
      observation.state              : float32 (6,)              — 팔로워 관절 위치
      observation.language_embedding : float32 (embedding_dim,)  — paraphrase MiniLM D=384
      observation.images.top         : video   (H, W, 3)         — RGB (rgb8 통일)
      action                         : float32 (6,)              — 리더 관절 값

    참고:
      - dtype='video' + use_videos=True 가 표준. add_frame 시점엔 PNG 임시 저장,
        save_episode 시점에 일괄 비디오 인코딩 (parallel_encoding=True 가 기본).
      - 1D 비의미 벡터(language_embedding)는 names=None (DEFAULT_FEATURES 규약).
      - 이미지 채널 names 는 ['height','width','channels'] (utils.py:661 형식).
    """
    features = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (6,),
            'names': list(SO101_JOINT_NAMES),
        },
        'observation.language_embedding': {
            'dtype': 'float32',
            'shape': (embedding_dim,),
            'names': None,
        },
        CAMERA_KEY: {
            'dtype': 'video',   # operational decision: video 인코딩 채택
            'shape': (args.img_h, args.img_w, 3),
            'names': ['height', 'width', 'channels'],
        },
        'action': {
            'dtype': 'float32',
            'shape': (6,),
            'names': list(SO101_JOINT_NAMES),
        },
    }

    # LeRobotDataset 기본 저장 위치 — root 인자 미지정 시 사용.
    default_root = Path.home() / '.cache' / 'huggingface' / 'lerobot' / args.dataset_repo_id

    # 이전 run 이 LeRobotDataset.create() 만 하고 save_episode() 전에 종료된 경우
    # 디렉터리는 존재하지만 meta/tasks.parquet 같은 핵심 메타파일이 없다. 그 상태에서
    # LeRobotDataset() 으로 열려고 하면 load_metadata 가 FileNotFoundError 를 내고,
    # LeRobot 의 Hub fallback 이 동작해 404/401 로 죽는다. 이 케이스를 미리 정리.
    tasks_parquet = default_root / 'meta' / 'tasks.parquet'
    if default_root.exists() and not tasks_parquet.exists():
        print(
            f'[record_real_teleop] 기존 폴더 {default_root} 가 미완성 데이터셋 '
            '(meta/tasks.parquet 없음 — 이전 run 이 첫 save_episode 전에 종료된 흔적). '
            '자동 삭제 후 새로 생성합니다.'
        )
        shutil.rmtree(default_root)

    if default_root.exists():
        print(
            f'[record_real_teleop] 기존 데이터셋 발견 → resume: {default_root}\n'
            '                새로 시작하려면 위 경로를 삭제 후 재실행.'
        )
        dataset = LeRobotDataset(repo_id=args.dataset_repo_id)

        # language_embedding shape 일치 사전 검증 — 임베딩 모델 변경 시 즉시 차단.
        existing = dataset.features.get('observation.language_embedding')
        if existing is None:
            raise RuntimeError(
                f'기존 데이터셋에 observation.language_embedding 채널이 없음 '
                f'({default_root}). 호환되는 데이터셋이 아님 — 다른 '
                '--dataset-repo-id 사용 또는 위 경로 삭제 후 재실행.'
            )
        ex_shape = tuple(existing['shape'])
        if ex_shape != (embedding_dim,):
            raise RuntimeError(
                f'기존 데이터셋 language_embedding shape={ex_shape} != '
                f'({embedding_dim},). 임베딩 모델이 다른 데이터셋이므로 '
                '동일 repo 에 append 불가 — 다른 repo_id 사용 또는 데이터셋 삭제 필요.'
            )
        return dataset

    return LeRobotDataset.create(
        repo_id=args.dataset_repo_id,
        fps=args.fps,
        features=features,
        robot_type='so101_follower',
        use_videos=True,
    )


def connect_so101(
    robot_port: str, robot_id: str, teleop_port: str, teleop_id: str,
) -> Tuple[SO101Follower, SO101Leader]:
    """
    SO-101 팔로워(robot) + 리더(teleop) 동시 연결.

    operational decision [1]:
      본 스크립트가 follower 포트를 단독 점유한다.
      robot_controller_node 가 동시에 실행 중이면 SDK 의 포트 open 단계에서
      실패한다 — 그 경우 robot_controller_node 를 먼저 종료할 것.
    """
    follower = SO101Follower(SO101FollowerConfig(port=robot_port, id=robot_id))
    follower.connect()

    leader = SO101Leader(SO101LeaderConfig(port=teleop_port, id=teleop_id))
    leader.connect()

    return follower, leader


# ──────────────────────────────────────────────
# 헬퍼 — 종료 신호 / 변환 / home 복귀
# ──────────────────────────────────────────────

def _user_wants_quit() -> bool:
    """
    operational decision [4]:
      stdin 에 'q' (또는 'Q') + Enter 가 들어왔는지 비차단 검사.
      Linux/macOS 전용 (Jetson 타깃). Windows 에서는 항상 False.
    """
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except (ValueError, OSError):
        return False
    if not ready:
        return False
    line = sys.stdin.readline()
    return bool(line) and line.strip().lower() == 'q'


def _action_dict_to_vec(d: dict) -> np.ndarray:
    """SDK 의 {motor}.pos dict → SO101_JOINT_NAMES 순서의 float32 벡터."""
    return np.array(
        [float(d[f'{n}.pos']) for n in SO101_JOINT_NAMES],
        dtype=np.float32,
    )


def _rgb_msg_to_array(msg: Image) -> np.ndarray:
    """
    /camera/rgb (sensor_msgs/Image) → numpy (H,W,3) uint8 RGB.

    operational decision [2] + bridging encoding:
      RealSense 가 bgr8 로 발행하든 rgb8 로 발행하든 numpy 레벨에서 rgb8 로
      통일한다 (Sim/Real 일치). cv_bridge 의존을 제거하기 위해
      robot_arm_pkg.utils.ros_image.image_msg_to_rgb_array 로 직접 변환.
    """
    return image_msg_to_rgb_array(msg)


def _capture_home_pose(robot: SO101Follower) -> np.ndarray:
    """
    operational decision [5]:
      connect 직후 follower 자세를 home pose 로 그대로 저장.
      스크립트 실행 전 작업자가 팔을 원하는 rest 자세에 두고 켜야 한다.
    """
    obs = robot.get_observation()
    return _action_dict_to_vec(obs)


def _return_to_home(
    robot:      SO101Follower,
    home_pose:  np.ndarray,
    fps:        int,
    duration_s: float = RETURN_HOME_DURATION_S,
) -> None:
    """
    operational decision [5]:
      현재 follower 자세 → home_pose 로 fps Hz 선형 보간 명령 전송.
      duration_s 초에 걸쳐 동작. 카메라 spin 은 멈춰도 무관 (녹화 종료 후이므로).
    """
    cur = _action_dict_to_vec(robot.get_observation())
    n_steps = max(1, int(round(fps * duration_s)))
    dt = 1.0 / fps
    for i in range(1, n_steps + 1):
        alpha  = i / n_steps
        target = (1.0 - alpha) * cur + alpha * home_pose
        robot.send_action({
            f'{n}.pos': float(v) for n, v in zip(SO101_JOINT_NAMES, target)
        })
        time.sleep(dt)


# ──────────────────────────────────────────────
# 메인 — 에피소드 녹화 루프
# ──────────────────────────────────────────────

def record_episode(
    args:          RecordArgs,
    helper:        RecordHelperNode,
    robot:         SO101Follower,
    teleop:        SO101Leader,
    dataset:       LeRobotDataset,
    episode_idx:   int,
    label_record:  dict,
    embedding:     List[float],
    home_pose:     np.ndarray,
) -> bool:
    """
    1개 에피소드 녹화 → 저장 → follower home pose 복귀.

    Returns:
        True  — 사용자가 저장 선택, save_episode() 호출 완료 (디스크에 에피소드 존재).
        False — 사용자가 저장 거부, clear_episode_buffer() 로 버퍼만 폐기 (디스크 무변경).

        호출자(main)는 이 값으로 retry 시 delete_episode() 호출 여부를 결정한다.
        save=False 인데 delete_episode() 를 부르면 직전 정상 저장 에피소드를 잘못
        삭제하거나 LeRobotDataset 상태가 깨질 수 있어 반드시 분기 필요.

    각 틱(=1/fps 초) 마다:
      1) 종료 신호 확인 ('q'+Enter) — operational decision [4]
      2) 카메라 콜백 처리 (latest-only)
      3) 리더 action 읽어 팔로워에 미러링
      4) 팔로워 관절 상태 + 최신 RGB 프레임 수집
      5) frame dict 구성 + dataset.add_frame
         frame['task'] 가 필수 (add_frame 이 pop 하여 episode_buffer.task 에 적재).
      6) deadline 기반 sleep — drift 가 한 주기 이상이면 재동기화 (decision [6])

    종료 후:
      - dataset.save_episode()  (parallel_encoding=True 가 기본)
      - follower 를 home_pose 로 복귀 (decision [5])
    """
    embed_vec = np.asarray(embedding, dtype=np.float32)
    if embed_vec.shape != (EXPECTED_EMBED_DIM,):
        raise ValueError(
            f'embedding shape {embed_vec.shape} != ({EXPECTED_EMBED_DIM},)'
        )
    task_str = label_record['canonical']
    dt = 1.0 / args.fps

    # 카메라 첫 프레임 대기 — camera_node 가 살아 있는지 확인
    # 본 노드의 콜백은 main() 의 백그라운드 executor 가 처리하므로
    # 여기서 별도 spin 없이 캐시만 polling 한다.
    print(f'\n[ep {episode_idx}] /camera/rgb 첫 프레임 대기...')
    wait_deadline = time.perf_counter() + 5.0
    while helper.latest_rgb() is None:
        if time.perf_counter() > wait_deadline:
            raise RuntimeError(
                '[record_real_teleop] /camera/rgb 미수신 (5s timeout). '
                'camera_node 와 realsense2_camera 가 실행 중인지 확인.'
            )
        time.sleep(0.05)

    print(f'[ep {episode_idx}] task = {task_str!r}')
    print(f'[ep {episode_idx}] Enter 로 녹화 시작.')
    input()

    print(f'[ep {episode_idx}] 녹화 중 — "q"+Enter 또는 Ctrl+C 로 종료')
    n_frames    = 0
    t_start     = time.perf_counter()
    next_dl     = t_start
    quit_flag   = False
    try:
        while True:
            # 1) 종료 신호
            if _user_wants_quit():
                quit_flag = True
                break

            # 2) 카메라 콜백은 main() 의 백그라운드 executor 가 비동기 갱신.
            #    여기서 spin_once 호출하지 않음 — 메인 스레드는 시리얼/디스크 IO
            #    동안 ROS 콜백을 못 돌리므로 같은 RGB 가 재사용되는 freeze 가
            #    발생했었음. 백그라운드 spin 으로 캐시는 항상 최신 프레임.

            # 3) 리더 → 팔로워 미러링
            leader_action = teleop.get_action()
            robot.send_action(leader_action)

            # 4) 팔로워 상태 + 최신 RGB
            follower_obs = robot.get_observation()
            rgb_msg      = helper.latest_rgb()
            if rgb_msg is None:
                # 카메라 일시 단절 — 이번 틱 스킵하고 다음 deadline 으로
                next_dl += dt
                _sleep_until(next_dl)
                continue

            # 5) frame 구성 + LeRobot 데이터셋 적재
            #    add_frame 은 frame['task'] 를 pop 하여 episode_buffer.task 에 누적.
            frame = {
                'observation.state':              _action_dict_to_vec(follower_obs),
                'observation.language_embedding': embed_vec,
                CAMERA_KEY:                       _rgb_msg_to_array(rgb_msg),
                'action':                         _action_dict_to_vec(leader_action),
                'task':                           task_str,
            }
            dataset.add_frame(frame)
            n_frames += 1

            # 6) fps 유지 — deadline 기반 (decision [6] drift 재동기화)
            next_dl += dt
            slack = next_dl - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -dt:
                # 한 주기 이상 밀린 경우 — 누적 drift 누락하고 현재 시점으로 재설정
                next_dl = time.perf_counter()
    except KeyboardInterrupt:
        quit_flag = True
        print(f'\n[ep {episode_idx}] KeyboardInterrupt — 종료')

    elapsed = time.perf_counter() - t_start
    realized_fps = n_frames / elapsed if elapsed > 0 else 0.0
    print(
        f'[ep {episode_idx}] 녹화 종료 — frames={n_frames} '
        f'elapsed={elapsed:.2f}s realized_fps={realized_fps:.2f} '
        f'(quit={"q" if quit_flag else "?"})'
    )

    # 저장 여부 확인
    save = input(f'[ep {episode_idx}] 이 에피소드를 저장할까요? (y/n): ').strip().lower()
    if save != 'y':
        dataset.clear_episode_buffer()
        print(f'[ep {episode_idx}] 에피소드 버퍼 폐기.')
        return False

    # 에피소드 저장 (parallel_encoding=True 가 기본 — 백그라운드 비디오 인코딩)
    dataset.save_episode()
    print(f'[ep {episode_idx}] save_episode 호출 완료 — 인코딩 백그라운드 진행')

    # follower home pose 복귀 (decision [5])
    print(f'[ep {episode_idx}] follower → home pose 복귀 중 '
          f'({RETURN_HOME_DURATION_S:.1f}s)...')
    _return_to_home(robot, home_pose, args.fps)
    print(f'[ep {episode_idx}] 복귀 완료. 장면을 reset 한 뒤 다음 라벨 입력으로 진행.')
    return True


def _sleep_until(deadline_perf: float) -> None:
    """time.perf_counter() 기준 deadline 까지 sleep. 음수면 즉시 반환."""
    remaining = deadline_perf - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def main() -> None:
    args = parse_args()

    # 1) Vocabulary 로드 (Option C 슬롯 검증용)
    vocab = load_vocab(args.label_vocab, args.domain)
    print(f'[record_real_teleop] vocab loaded ({args.label_vocab}, domain={args.domain})')

    rclpy.init()
    helper = RecordHelperNode()

    # 카메라 콜백을 메인 루프와 분리된 daemon 스레드의 SingleThreadedExecutor 에서
    # 처리한다. 메인 스레드가 SO-101 시리얼 IO + add_frame 의 PNG 저장에 점유되어
    # /camera/rgb 콜백이 밀려 동일 프레임이 dataset 에 중복 적재되던 문제를 차단.
    executor = SingleThreadedExecutor()
    executor.add_node(helper)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot:  Optional[SO101Follower] = None
    teleop: Optional[SO101Leader]   = None

    try:
        # 2) 첫 에피소드 라벨 → 임베딩 차원 검증 (384 강제)
        first_rec   = prompt_episode_label(vocab, episode_idx=0)
        first_embed = helper.call_embed_text(first_rec['canonical'])
        if first_embed is None:
            helper.get_logger().error(
                '첫 에피소드 임베딩 획득 실패. embedding_node 상태 확인 후 재시도.'
            )
            return
        embedding_dim = len(first_embed)
        if embedding_dim != EXPECTED_EMBED_DIM:
            helper.get_logger().error(
                f'embedding_dim={embedding_dim} 예상 {EXPECTED_EMBED_DIM}와 다름. 중단.'
            )
            return
        helper.get_logger().info(f'embedding_dim = {embedding_dim} (확정)')

        # 3) 데이터셋 생성 — labels.jsonl 은 dataset.root 옆에 두는 것을 기본으로
        dataset = open_dataset(args, embedding_dim)
        labels_jsonl = _resolve_labels_jsonl(args, dataset)
        print(f'[record_real_teleop] dataset.root = {dataset.root}')
        print(f'[record_real_teleop] label log    = {labels_jsonl}')

        # 4) SO-101 연결 (operational decision [1] — single-owner)
        robot, teleop = connect_so101(
            args.robot_port, args.robot_id, args.teleop_port, args.teleop_id,
        )

        # 5) Home pose 결정 (operational decision [5] — 파일 영구화)
        #    파일이 있으면 그 자세를 그대로 사용.
        #    없으면 connect 직후 follower 자세를 캡처해 같은 경로에 저장 →
        #    이후 record_real_teleop / robot_controller_node /go_home 이 공유.
        if home_pose_exists(args.home_pose_file):
            home_pose = load_home_pose(args.home_pose_file)
            print(f'[record_real_teleop] home pose 로드: {args.home_pose_file} '
                  f'→ {home_pose.tolist()}')
        else:
            home_pose = _capture_home_pose(robot)
            save_home_pose(args.home_pose_file, home_pose)
            print(f'[record_real_teleop] home pose 신규 캡처 후 저장: {args.home_pose_file} '
                  f'→ {home_pose.tolist()}')
            print('[record_real_teleop] 향후 변경 시: follower 를 새 자세에 두고 '
                  f'{args.home_pose_file} 삭제 후 record_real_teleop 재실행.')

        # 6) 에피소드 루프
        start_ep = dataset.num_episodes
        for ep in range(start_ep, start_ep + args.num_episodes):
            if ep == 0:
                rec, embed = first_rec, first_embed
            else:
                rec   = prompt_episode_label(vocab, ep)
                embed = helper.call_embed_text(rec['canonical'])
                if embed is None or len(embed) != EXPECTED_EMBED_DIM:
                    helper.get_logger().error(
                        f'에피소드 {ep} 임베딩 실패. 스킵.'
                    )
                    continue

            while True:
                append_label_record(labels_jsonl, rec)
                saved = record_episode(
                    args, helper, robot, teleop, dataset,
                    ep, rec, embed, home_pose,
                )
                retry = input(f'[ep {ep}] 이 에피소드를 다시 촬영할까요? (y/n): ').strip().lower()
                if retry == 'y':
                    # saved=True: 디스크 에피소드 삭제 후 재촬영.
                    # saved=False: 이미 clear_episode_buffer() 됐으므로 추가 정리 불필요
                    #   — 이때 delete_episode() 를 부르면 직전 정상 에피소드를
                    #   잘못 지우거나 데이터셋 상태가 깨진다.
                    if saved:
                        dataset.delete_episode()
                    print(f'[ep {ep}] 재촬영 — 라벨 다시 입력합니다.')
                    rec   = prompt_episode_label(vocab, ep)
                    embed = helper.call_embed_text(rec['canonical'])
                    if embed is None or len(embed) != EXPECTED_EMBED_DIM:
                        print(f'[ep {ep}] 임베딩 실패. 스킵.')
                        break
                else:
                    break

    finally:
        # SO-101 안전 해제 — 포트 점유 해제 보장 (decision [1])
        if teleop is not None:
            try:
                teleop.disconnect()
            except Exception as e:
                print(f'[record_real_teleop] teleop disconnect 오류: {e}')
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as e:
                print(f'[record_real_teleop] robot disconnect 오류: {e}')
        # 백그라운드 executor 종료 → spin 스레드 회수 → 노드/rclpy 정리 순서
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        helper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
