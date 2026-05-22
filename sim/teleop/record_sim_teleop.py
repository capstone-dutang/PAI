"""
record_sim_teleop.py

Isaac Sim teleop 기반 ACT 학습용 *Sim* 시연 데이터 수집 스크립트 (스켈레톤).

실물 SO-101 Leader 를 작업자가 직접 조작하면 Isaac Sim 안의 follower(디지털
트윈)가 이를 미러링하고, 그 과정의 관측·행동을 LeRobotDataset 으로 녹화한다.
규칙기반 IK 전문가 방식(구 sim/generate_act_dataset.py)을 대체하며, Real 측
deploy/.../scripts/record_teleop.py 와 대칭 구조를 갖는다.

기록 채널 (Real record_teleop.py 와 동일 스펙 — Co-training 가능 조건):

  observation.state              (6,)   float32  — Isaac Sim follower 관절 위치
                                                    (sim rad → SDK .pos 단위 변환, [D3])
  observation.language_embedding (D,)   float32  — placeholder zero 벡터.
                                                    수집 후 generate_language_labels.py
                                                    가 실제 임베딩으로 채움 ([D9])
  observation.images.top         (H,W,3) uint8   — Isaac Sim 카메라 RGB
  action                         (6,)   float32  — 실물 Leader 관절 값 (정답).
                                                    LeRobot SDK .pos 단위 그대로 — Real 과
                                                    동일 경로이므로 변환 불요

언어 라벨:
  multi_target_scenes.yaml 의 장면 정의에서 타겟 객체를 선택하고, label_vocab.yaml
  의 템플릿으로 canonical 라벨 "the {attribute} {object} {spatial}" 을 만든다.
  이 canonical 라벨 + 슬롯 메타데이터는 labels.jsonl 로 기록되어 후처리
  (generate_language_labels.py) 입력으로 쓰인다. 임베딩 자체는 본 스크립트에서
  넣지 않는다 ([D9] 참조).

전제조건:
  1. Isaac Sim teleop 인프라 구동 — sim/teleop/sim_{local,cloud}_teleop_setting.md
  2. Isaac Sim Action Graph 가 아래를 갖출 것 ([D4] — 현재 셋업 가이드엔 없음):
       - /joint_command (JointState) 구독 → Articulation Controller (drive target)
       - /joint_reset (JointState) 구독 → SetJointPositions (에피소드 시작 강제)
       - follower joint_states (JointState) 발행
       - 카메라 RGB (Image) 발행
  3. 실물 SO-101 Leader USB 포트 확인 + lerobot-calibrate 완료
  4. SO-101 follower 캘리브레이션 yaml (observation.state 단위 변환용, [D3])
  5. label_vocab.yaml / multi_target_scenes.yaml / dr_config.yaml 배치
  6. deploy/calibration/home_pose.yaml — record_teleop.py 가 캡처한 실물 follower
     rest 자세. 에피소드 시작 강제 자세로 재사용한다 ([D7]).

CLI 사용 예:
  python sim/teleop/record_sim_teleop.py \\
    --teleop-port      COM5 \\
    --teleop-id        leader \\
    --dataset-repo-id  myname/dataset_sim \\
    --domain           recycling \\
    --label-vocab      deploy/config/label_vocab.yaml \\
    --scenes           sim/config/multi_target_scenes.yaml \\
    --dr-config        sim/config/dr_config.yaml \\
    --follower-calib   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json \\
    --num-episodes     500 \\
    --mode             local

실행 환경:
  Isaac Sim teleop 환경(ros_humble mamba env — rclpy + lerobot)에서 실행한다.
  deploy 측 robot_arm_pkg / robot_arm_msgs 는 이 환경에 없으므로 import 하지
  않는다 (record_teleop.py 와 코드 공유 불가 — 필요한 로직은 본 파일에 자립).

상태: 스켈레톤 — 인터페이스/CLI/파라미터만 작성 완료. 핵심 로직은
      NotImplementedError 로 표시하고, 미확정 항목은 아래 "확정 필요 항목"
      블록 [D1]~[D9] 에 모았다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
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
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState

# LeRobot 0.4.4 — 데이터셋 + Leader SDK (브리지 lerobot_to_rosbridge.py 와 동일 경로)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


# ──────────────────────────────────────────────
# 운영 / 측정 결정 사항 (Real record_teleop.py 와 대응)
# ──────────────────────────────────────────────
#
#   [S1] 관측 소스 — Isaac Sim ROS2 토픽
#         카메라 RGB / follower joint_states 를 Isaac Sim 이 ROS2 로 발행하고,
#         본 스크립트의 헬퍼 노드가 BestEffort + KeepLast(1) 로 구독해 최신
#         1프레임만 캐싱한다. (Real 의 camera_node /camera/rgb 구독에 대응)
#
#   [S2] 이미지 해상도 (H, W)
#         --img-h / --img-w 로 결정. 기본 480 x 640.
#         Real dataset feature shape 과 반드시 일치해야 Co-training 가능.
#
#   [S3] 에피소드 종료 정책 — 'q' + Enter
#         녹화 루프 중 stdin 에 'q'(또는 'Q') + Enter → 정상 종료.
#         KeyboardInterrupt 도 같은 경로로 흡수. Linux/macOS 전용.
#
#   [S4] 에피소드 reset — 씬 재구성 + DR + follower 초기 자세 강제
#         에피소드 시작 시 deploy/calibration/home_pose.yaml(실물 follower rest
#         자세)을 읽어 /joint_reset 으로 발행하고, Isaac Sim 측이 SetJointPositions
#         (kinematic teleport)로 follower 를 그 자세에 강제 고정한다 — drive target
#         보간이 아니라 순간 고정이므로 매 에피소드가 동일한 초기 상태에서
#         시작한다. 씬 재구성/DR 은 SceneManager 가 담당. ([D2], [D7])
#
#   [S5] fps
#         기본 30Hz (--fps). Real record_teleop.py 의 fps 와 일치시켜야
#         학습 시 resample 이 불필요하다. ([D8])
#
#   [S6] action / observation 단위
#         action(Leader 관절값)은 LeRobot SDK .pos 단위 그대로 저장 — Real 과
#         동일 경로라 변환 불요. observation.state(Isaac Sim follower 관절)는
#         Isaac Sim 이 rad 로 주므로 SDK .pos 단위로 변환 후 저장한다. ([D3])
#
# ──────────────────────────────────────────────
# 확정 필요 항목 — 스켈레톤 단계에서 결정 불가. 본문 NotImplementedError 가 참조.
# ──────────────────────────────────────────────
#
#   [D1] 통신 방식 / 스크립트 실행 위치
#         - local : Isaac Sim·Leader·녹화 스크립트가 한 머신. rclpy 직접 통신.
#                   Leader 는 SDK 로 직접 read. action 이 native SDK 단위.
#         - cloud : Isaac Sim 은 EC2, Leader 는 로컬 PC. rclpy DDS 가 인터넷을
#                   넘지 못함 → rosbridge WebSocket 경유 또는 녹화 스크립트를
#                   EC2 에서 돌리고 /joint_command 를 구독해 action 을 얻는 방식
#                   등 후보가 있음. 본 스켈레톤은 local 모드를 1차 경로로 작성.
#
#   [D2] Isaac Sim 씬 / DR 제어 메커니즘 (SceneManager)
#         녹화 스크립트는 Isaac Sim USD 스테이지를 직접 못 만짐. 후보:
#           (a) Isaac Sim 측 동반 스크립트가 ROS2 service/topic 으로 명령 수신
#           (b) 녹화 스크립트를 Isaac Sim Python 안에서 실행 (env 결합 문제)
#           (c) 씬/DR 은 Isaac Sim 측에서 수동, 녹화만 본 스크립트
#         multi_target_scenes.yaml / dr_config.yaml 스키마도 확정 전 (README
#         미구현 항목).
#
#   [D3] observation.state 단위 변환 (_sim_jointstate_to_sdk_units)
#         Isaac Sim follower joint position(rad) → LeRobot SDK .pos 단위.
#         follower 캘리브레이션 yaml 의 affine(zero offset / range) 필요.
#         구 generate_act_dataset.py [5] 의 단위 변환·결합 규칙 참고.
#         (재캘리브레이션 시 기존 Sim 데이터셋 무효 — 동일 yaml 고정 필수)
#
#   [D4] Isaac Sim Action Graph 확장
#         현재 teleop 셋업 가이드의 Action Graph 는 /joint_command 구독 +
#         /clock 발행만 있음. 아래를 추가해야 본 스크립트가 동작한다:
#           - follower joint_states 발행 (observation.state 입력)
#           - 카메라 RGB 발행 (observation.images.top 입력)
#           - /joint_reset 구독 → 에피소드 시작 강제 자세 핸들러. 이 핸들러는
#             한 물리 스텝 안에서 원자적으로 아래 3가지를 모두 적용해야 한다:
#               (1) set_joint_positions(target)
#               (2) set_joint_velocities(0)        — 잔여 속도 제거
#               (3) Articulation drive target = target  — PD 목표 갱신
#             (1) 만 적용하면 다음 물리 스텝에서 PD 가 직전 drive target
#             (이전 /joint_command, 첫 reset 시 USD default)으로 팔을 끌어간다.
#             표준 OmniGraph 노드만으로 3가지를 한 tick 에 묶기 어려우면
#             Isaac Sim 측 Python 핸들러 또는 커스텀 노드를 쓴다.
#
#   [D5] 타겟 선택 전략 (select_episode_target)
#         작업자 수동 선택(Real Option C 와 동일) vs 자동 분포 보장
#         (multi-target shortcut 방지 가이드라인 — 같은 객체 다른 spatial ≥30%
#         등)을 코드가 강제. 현재는 단순 무작위 선택으로 두고 [D5] 로 표시.
#
#   [D6] gripper 매핑 상수
#         Leader 0~100% ↔ Isaac Sim gripper rad. 현재 브리지와 같은
#         -10°~100° 범위를 쓰지만 실측 확정 필요.
#
#   [D7] sim follower 초기 자세 — 확정
#         에피소드 시작 시 deploy/calibration/home_pose.yaml(record_teleop.py 가
#         캡처한 실물 follower rest 자세)을 읽어 /joint_reset → SetJointPositions
#         로 강제한다. Sim/Real 의 에피소드 시작 자세가 일치하며 별도 측정이
#         불필요하다. SDK .pos → rad 변환은 load_initial_pose 참조 ([D3] 연동).
#
#   [D8] 에피소드 길이 / fps
#         프레임 수 목표와 fps 를 Real record_teleop.py 와 일치시킬 것.
#
#   [D9] language_embedding 처리 방식
#         현재 채택: placeholder zero 벡터로 녹화 → generate_language_labels.py
#         가 후처리로 실제 임베딩 적재 (README §수집 절차 step 2).
#         대안: Real record_teleop.py 처럼 녹화 중 embedding_node /embed_text
#         로 즉시 임베딩. 채택 시 헬퍼 노드에 서비스 클라이언트 추가 필요.
#
# ──────────────────────────────────────────────


EXPECTED_EMBED_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 (README 확정)

# SO-101 관절 이름 — record_teleop.py / 브리지와 동일 순서.
SO101_JOINT_NAMES: List[str] = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]

# Isaac Sim SO-101 USD (so101_new_calib) joint limit — degree.
# Physics Inspector 실측값. 초기 자세 강제 시 arm 관절을 이 범위로 clamp 한다.
# (실물 SO-101 은 가동범위가 약간 더 넓어 home_pose 가 한계를 넘어 측정될 수 있음)
SO101_JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    'shoulder_pan':  (-110.0,   110.0),
    'shoulder_lift': (-100.0,   100.0),
    'elbow_flex':    ( -96.83,   96.83),
    'wrist_flex':    ( -95.0,    95.0),
    'wrist_roll':    (-157.211, 162.789),
    'gripper':       ( -10.0,   100.0),
}

# 이미지 기본값 (operational decision [S2])
DEFAULT_IMG_H = 480
DEFAULT_IMG_W = 640

# LeRobot 데이터셋 카메라 채널 키 (Real record_teleop.py 와 일치 필수)
CAMERA_KEY = 'observation.images.top'

# 에피소드 시작 강제 자세 적용 후 물리 안정화 대기 초 (operational decision [S4])
RESET_SETTLE_S = 0.5

# gripper 매핑 상수 ([D6] — 브리지 lerobot_to_rosbridge.py 와 동일값, 실측 확정 필요)
GRIPPER_MIN_DEG = -10.0   # Leader 0%
GRIPPER_MAX_DEG = 100.0   # Leader 100%

# 기본 토픽 이름 ([D4] — Isaac Sim Action Graph 가 발행/구독해야 함)
DEFAULT_JOINT_COMMAND_TOPIC = '/joint_command'   # 녹화 → Isaac Sim (drive target)
DEFAULT_JOINT_RESET_TOPIC   = '/joint_reset'     # 녹화 → Isaac Sim (SetJointPositions)
DEFAULT_SIM_CAMERA_TOPIC    = '/camera/rgb'      # Isaac Sim → 녹화 스크립트
DEFAULT_SIM_JOINTSTATE_TOPIC = '/joint_states'   # Isaac Sim follower 실제 상태


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@dataclass
class SimRecordArgs:
    teleop_port:           str
    teleop_id:             str
    dataset_repo_id:       str
    domain:                str
    label_vocab:           str
    scenes:                str
    dr_config:             str
    follower_calib:        str
    num_episodes:          int
    fps:                   int
    labels_jsonl:          str
    img_h:                 int
    img_w:                 int
    embedding_dim:         int
    mode:                  str            # 'local' | 'cloud'  ([D1])
    seed:                  int
    home_pose_file:        str
    joint_command_topic:   str
    joint_reset_topic:     str
    sim_camera_topic:      str
    sim_jointstate_topic:  str
    ec2_ip:                str            # mode == 'cloud' 일 때만 사용 ([D1])
    ec2_port:              int


def parse_args() -> SimRecordArgs:
    p = argparse.ArgumentParser(description='Isaac Sim teleop Sim 시연 데이터 수집')
    p.add_argument('--teleop-port',     required=True, help='실물 SO-101 Leader USB 포트')
    p.add_argument('--teleop-id',       default='leader', help='Leader 캘리브레이션 ID')
    p.add_argument('--dataset-repo-id', required=True,
                   help='LeRobotDataset repo id (예: user/dataset_sim)')
    p.add_argument('--domain',          required=True,
                   help='수집 도메인 (label_vocab.yaml domains 키와 일치)')
    p.add_argument('--label-vocab',     default='deploy/config/label_vocab.yaml',
                   help='라벨 vocabulary YAML 경로')
    p.add_argument('--scenes',          default='sim/config/multi_target_scenes.yaml',
                   help='multi-target 장면 구성 YAML 경로')
    p.add_argument('--dr-config',       default='sim/config/dr_config.yaml',
                   help='Domain Randomization 설정 YAML 경로')
    p.add_argument('--follower-calib',  default='',
                   help='SO-101 follower 캘리브레이션 yaml ([D3] observation.state '
                        '단위 변환용). 배포 로봇과 동일 파일이어야 함.')
    p.add_argument('--num-episodes',    type=int, default=500,
                   help='수집 에피소드 수 (순수 teleop 전환 후 목표 ~500)')
    p.add_argument('--fps',             type=int, default=30,
                   help='기록 주기 (Hz). Real record_teleop.py 와 일치 ([D8]).')
    p.add_argument('--labels-jsonl',    default='',
                   help='canonical 라벨 + 슬롯 메타데이터 JSONL 경로. 비어 있으면 '
                        'dataset.root 옆에 labels.jsonl 로 자동 생성. '
                        'generate_language_labels.py 입력으로 사용된다.')
    p.add_argument('--img-h',           type=int, default=DEFAULT_IMG_H,
                   help='이미지 높이 H ([S2]). Real feature shape 과 일치 필수.')
    p.add_argument('--img-w',           type=int, default=DEFAULT_IMG_W,
                   help='이미지 너비 W ([S2]).')
    p.add_argument('--embedding-dim',   type=int, default=EXPECTED_EMBED_DIM,
                   help=f'language_embedding placeholder 차원 D (기본 {EXPECTED_EMBED_DIM}). '
                        'generate_language_labels.py 의 임베딩 모델과 일치 필수 ([D9]).')
    p.add_argument('--mode',            default='local', choices=['local', 'cloud'],
                   help='local: rclpy 직접 통신 / cloud: EC2 (rosbridge 경유) — [D1]')
    p.add_argument('--seed',            type=int, default=0,
                   help='DR / 타겟 선택 재현성 시드')
    p.add_argument('--home-pose-file',   default='deploy/calibration/home_pose.yaml',
                   help='에피소드 시작 강제 자세 — record_teleop.py 가 캡처한 실물 '
                        'follower home_pose.yaml 경로 ([D7]).')
    p.add_argument('--joint-command-topic',  default=DEFAULT_JOINT_COMMAND_TOPIC,
                   help='녹화 스크립트 → Isaac Sim follower 구동 토픽 (drive target)')
    p.add_argument('--joint-reset-topic',    default=DEFAULT_JOINT_RESET_TOPIC,
                   help='녹화 스크립트 → Isaac Sim SetJointPositions 강제 토픽 ([D4])')
    p.add_argument('--sim-camera-topic',     default=DEFAULT_SIM_CAMERA_TOPIC,
                   help='Isaac Sim 카메라 RGB 토픽 ([D4])')
    p.add_argument('--sim-jointstate-topic', default=DEFAULT_SIM_JOINTSTATE_TOPIC,
                   help='Isaac Sim follower 실제 관절 상태 토픽 ([D4])')
    p.add_argument('--ec2-ip',          default='',
                   help='mode=cloud 일 때 EC2 rosbridge 호스트 ([D1])')
    p.add_argument('--ec2-port',        type=int, default=9090,
                   help='mode=cloud 일 때 rosbridge 포트 ([D1])')
    a = p.parse_args()
    return SimRecordArgs(
        teleop_port          = a.teleop_port,
        teleop_id            = a.teleop_id,
        dataset_repo_id      = a.dataset_repo_id,
        domain               = a.domain,
        label_vocab          = os.path.expanduser(a.label_vocab),
        scenes               = os.path.expanduser(a.scenes),
        dr_config            = os.path.expanduser(a.dr_config),
        follower_calib       = os.path.expanduser(a.follower_calib) if a.follower_calib else '',
        num_episodes         = a.num_episodes,
        fps                  = a.fps,
        labels_jsonl         = os.path.expanduser(a.labels_jsonl) if a.labels_jsonl else '',
        img_h                = a.img_h,
        img_w                = a.img_w,
        embedding_dim        = a.embedding_dim,
        mode                 = a.mode,
        seed                 = a.seed,
        home_pose_file       = os.path.expanduser(a.home_pose_file),
        joint_command_topic  = a.joint_command_topic,
        joint_reset_topic    = a.joint_reset_topic,
        sim_camera_topic     = a.sim_camera_topic,
        sim_jointstate_topic = a.sim_jointstate_topic,
        ec2_ip               = a.ec2_ip,
        ec2_port             = a.ec2_port,
    )


# ──────────────────────────────────────────────
# Config 로드 — vocabulary / scenes / DR
# ──────────────────────────────────────────────

@dataclass
class LabelVocab:
    """label_vocab.yaml — record_teleop.py 의 LabelVocab 와 동일 스키마."""
    template:  str
    spatials:  List[str]
    objects:   Dict[str, List[str]]        # object_name -> attributes
    domain:    str = ''
    raw:       dict = field(default_factory=dict)


def load_vocab(path: str, domain: str) -> LabelVocab:
    """label_vocab.yaml 로드 + 지정 도메인 추출 + 스키마 검증.

    record_teleop.py 의 load_vocab 와 동일 규약 — Sim/Real 라벨 분포 일치를
    위해 같은 vocabulary 파일을 공유한다.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    template = raw.get('template')
    spatials = raw.get('spatial_qualifiers') or []
    domains  = raw.get('domains') or {}

    if not template or '{attribute}' not in template \
       or '{object}' not in template or '{spatial}' not in template:
        raise ValueError(
            f'label_vocab template 이 비어 있거나 슬롯 누락 ({path}). '
            '"the {attribute} {object} {spatial}" 형식 필요.'
        )
    if not spatials:
        raise ValueError(f'spatial_qualifiers 가 비어 있음 ({path}).')
    if domain not in domains:
        raise ValueError(
            f"도메인 '{domain}' 이 label_vocab 에 없음. 가능: {list(domains)}"
        )

    obj_block = domains[domain].get('objects') or {}
    objects: Dict[str, List[str]] = {}
    for obj_name, obj_def in obj_block.items():
        attrs = (obj_def or {}).get('attributes') or []
        if not attrs:
            raise ValueError(f"객체 '{obj_name}' 의 attributes 가 비어 있음.")
        objects[obj_name] = list(attrs)
    if not objects:
        raise ValueError(f"도메인 '{domain}' 에 등록된 객체가 없음.")

    return LabelVocab(template=template, spatials=list(spatials),
                      objects=objects, domain=domain, raw=raw)


@dataclass
class SceneConfig:
    """multi_target_scenes.yaml 파싱 결과.

    NOTE: multi_target_scenes.yaml 스키마는 아직 확정 전(README 미구현 항목, [D2]).
          여기서는 raw dict 보관 + 문서화된 예시 스키마 기준 최소 추출만 한다.
          스키마 확정 시 본 dataclass 와 load_scenes 를 갱신할 것.
    """
    scenes:              List[dict]
    multi_target_ratio:  float
    raw:                 dict = field(default_factory=dict)


def load_scenes(path: str) -> SceneConfig:
    """multi_target_scenes.yaml 로드 (얕은 파싱)."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    scenes = raw.get('scenes') or []
    if not scenes:
        raise ValueError(f'multi_target_scenes.yaml 에 scenes 정의가 없음 ({path}).')
    return SceneConfig(
        scenes=list(scenes),
        multi_target_ratio=float(raw.get('multi_target_ratio', 0.0)),
        raw=raw,
    )


@dataclass
class DRConfig:
    """dr_config.yaml 파싱 결과.

    NOTE: dr_config.yaml 은 DR 파라미터 실측값이 미정(README 미구현, [D2]).
          raw dict 만 보관하고 실제 적용은 SceneManager 가 담당한다.
    """
    raw: dict = field(default_factory=dict)


def load_dr_config(path: str) -> DRConfig:
    """dr_config.yaml 로드."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    return DRConfig(raw=raw)


# ──────────────────────────────────────────────
# 타겟 선택 — multi-target 장면에서 에피소드 타겟 결정
# ──────────────────────────────────────────────

def select_episode_target(
    scene_cfg:   SceneConfig,
    vocab:       LabelVocab,
    episode_idx: int,
    rng:         np.random.Generator,
) -> Dict[str, str]:
    """
    이번 에피소드의 장면 + 타겟 객체를 선택하고 canonical 라벨을 생성한다.

    Returns:
        {
            'scene':     str,   # 선택된 장면 이름
            'object':    str,
            'attribute': str,
            'spatial':   str,
            'canonical': str,   # "the {attribute} {object} {spatial}"
            'domain':    str,
            'episode':   int,
        }

    [D5] 미확정 — 선택 전략:
        현재는 단순 무작위 선택. multi-target shortcut 방지 가이드라인
        (같은 객체 다른 spatial ≥30%, 한 장면 후보 2+ 다른 라벨 ≥40% 등)을
        코드가 강제하도록 분포 추적 로직을 추가할지 결정 필요.
        또는 Real Option C 처럼 작업자 수동 선택으로 바꿀 수도 있음.
    """
    raise NotImplementedError(
        '[D5] 타겟 선택 전략 미확정 — 무작위 선택 vs 분포 보장 vs 작업자 수동 선택. '
        'scene_cfg.scenes 와 vocab(objects/attributes/spatials)로 canonical 라벨을 '
        f'구성하도록 구현할 것. (template={vocab.template!r})'
    )


# ──────────────────────────────────────────────
# 라벨 메타데이터 JSONL — generate_language_labels.py 입력
# ──────────────────────────────────────────────

def _resolve_labels_jsonl(args: SimRecordArgs,
                          dataset: Optional[LeRobotDataset] = None) -> str:
    """labels.jsonl 저장 경로 결정 (record_teleop.py 와 동일 규약)."""
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
# ROS2 헬퍼 노드 — Isaac Sim 과의 토픽 통신
# ──────────────────────────────────────────────

class SimRecordHelperNode(Node):
    """
    record_sim_teleop 의 ROS2 통신 헬퍼 노드 (mode='local' 기준).

      - {joint_command_topic} publisher  → Isaac Sim follower 구동 (drive target)
      - {joint_reset_topic}   publisher  → Isaac Sim SetJointPositions (강제 자세)
      - {sim_camera_topic}    subscriber → 카메라 RGB latest-only 캐시
      - {sim_jointstate_topic} subscriber → follower 실제 관절 상태 latest-only 캐시

    Real record_teleop.py 의 RecordHelperNode 와 동일하게, main() 의 백그라운드
    SingleThreadedExecutor 에서 spin 되어 메인 루프의 시리얼/디스크 IO 와 무관하게
    콜백이 즉시 처리된다 (동일 프레임 중복 적재 방지).

    [D1] mode='cloud' 인 경우 rclpy DDS 가 EC2 까지 닿지 않으므로 본 노드를
         rosbridge WebSocket 클라이언트로 대체해야 한다 — 별도 결정 필요.
    """

    def __init__(self, args: SimRecordArgs):
        super().__init__('record_sim_teleop_helper')

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._latest_rgb: Optional[Image] = None
        self._latest_js:  Optional[JointState] = None

        self._cmd_pub = self.create_publisher(
            JointState, args.joint_command_topic, 10)
        self._reset_pub = self.create_publisher(
            JointState, args.joint_reset_topic, 10)
        self.create_subscription(
            Image, args.sim_camera_topic, self._rgb_cb, sensor_qos)
        self.create_subscription(
            JointState, args.sim_jointstate_topic, self._js_cb, sensor_qos)

    def _rgb_cb(self, msg: Image) -> None:
        self._latest_rgb = msg

    def _js_cb(self, msg: JointState) -> None:
        self._latest_js = msg

    def latest_rgb(self) -> Optional[Image]:
        return self._latest_rgb

    def latest_joint_state(self) -> Optional[JointState]:
        return self._latest_js

    def publish_joint_command(self, positions_rad: List[float]) -> None:
        """SO101_JOINT_NAMES 순서의 rad 위치를 /joint_command 로 발행 (drive target)."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(SO101_JOINT_NAMES)
        msg.position = [float(v) for v in positions_rad]
        self._cmd_pub.publish(msg)

    def publish_joint_reset(self, positions_rad: List[float]) -> None:
        """SO101_JOINT_NAMES 순서의 rad 위치를 /joint_reset 으로 발행.

        velocity 0 을 함께 실어, Isaac Sim 측 핸들러가 위치 강제와 더불어
        관절 속도도 0 으로 만들도록 의도를 명시한다 (잔여 속도 drift 방지).
        Isaac Sim 측 /joint_reset 핸들러의 원자적 처리 요구사항은 [D4] 참조.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(SO101_JOINT_NAMES)
        msg.position = [float(v) for v in positions_rad]
        msg.velocity = [0.0] * len(SO101_JOINT_NAMES)
        self._reset_pub.publish(msg)


# ──────────────────────────────────────────────
# Isaac Sim 씬 / DR 제어 — SceneManager
# ──────────────────────────────────────────────

class SceneManager:
    """
    Isaac Sim 의 multi-target 씬 구성 + Domain Randomization 을 담당.

    [D2] 미확정 — 녹화 스크립트는 Isaac Sim USD 스테이지를 직접 못 만진다.
         아래 메서드들의 실제 구현은 Isaac Sim 연동 방식 결정에 의존한다:
           (a) Isaac Sim 측 동반 스크립트 + ROS2 service/topic 명령
           (b) 본 스크립트를 Isaac Sim Python 안에서 실행
           (c) 씬/DR 은 Isaac Sim 측 수동, 본 클래스는 no-op
         또한 multi_target_scenes.yaml / dr_config.yaml 스키마도 확정 전.
    """

    def __init__(self, scene_cfg: SceneConfig, dr_cfg: DRConfig,
                 rng: np.random.Generator):
        self.scene_cfg = scene_cfg
        self.dr_cfg = dr_cfg
        self.rng = rng

    def setup_episode_scene(self, episode_idx: int, target: Dict[str, str]) -> None:
        """target 에 맞춰 multi-target 객체 배치 (캔/페트병/종이 등)."""
        raise NotImplementedError(
            '[D2] Isaac Sim 씬 구성 메커니즘 미확정 — multi_target_scenes.yaml '
            '기반 객체 배치를 Isaac Sim 에 적용하는 방법 결정 필요.'
        )

    def apply_domain_randomization(self, episode_idx: int) -> None:
        """dr_config.yaml 기준 조명·텍스처·FOV·질량·마찰·PD 게인·노이즈 무작위화."""
        raise NotImplementedError(
            '[D2] DR 적용 메커니즘 미확정 — dr_config.yaml 스키마 확정 후 구현.'
        )

    def reset(self) -> None:
        """다음 에피소드를 위한 씬 정리 (객체 제거 등)."""
        raise NotImplementedError('[D2] 씬 reset 메커니즘 미확정.')


# ──────────────────────────────────────────────
# 데이터셋 / Leader 연결
# ──────────────────────────────────────────────

def open_dataset(args: SimRecordArgs) -> LeRobotDataset:
    """
    LeRobotDataset 0.4.4 생성 또는 기존 데이터셋 resume.

    features 스키마는 Real record_teleop.py 의 open_dataset 과 반드시 일치해야
    Co-training 이 가능하다:
      observation.state              : float32 (6,)
      observation.language_embedding : float32 (embedding_dim,)  — placeholder, [D9]
      observation.images.top         : video   (H, W, 3)
      action                         : float32 (6,)
    """
    features = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (6,),
            'names': list(SO101_JOINT_NAMES),
        },
        'observation.language_embedding': {
            'dtype': 'float32',
            'shape': (args.embedding_dim,),
            'names': None,
        },
        CAMERA_KEY: {
            'dtype': 'video',
            'shape': (args.img_h, args.img_w, 3),
            'names': ['height', 'width', 'channels'],
        },
        'action': {
            'dtype': 'float32',
            'shape': (6,),
            'names': list(SO101_JOINT_NAMES),
        },
    }

    default_root = (Path.home() / '.cache' / 'huggingface' / 'lerobot'
                    / args.dataset_repo_id)

    if default_root.exists():
        print(f'[record_sim_teleop] 기존 데이터셋 발견 → resume: {default_root}')
        dataset = LeRobotDataset(repo_id=args.dataset_repo_id)
        existing = dataset.features.get('observation.language_embedding')
        if existing is None or tuple(existing['shape']) != (args.embedding_dim,):
            raise RuntimeError(
                'resume 대상 데이터셋의 language_embedding shape 이 '
                f'({args.embedding_dim},) 와 불일치 — 다른 repo_id 사용 또는 삭제 필요.'
            )
        return dataset

    return LeRobotDataset.create(
        repo_id=args.dataset_repo_id,
        fps=args.fps,
        features=features,
        robot_type='so101_follower',   # 디지털 트윈 — Real 과 동일 타입 표기
        use_videos=True,
    )


def connect_leader(args: SimRecordArgs) -> SO101Leader:
    """
    실물 SO-101 Leader 연결 (LeRobot SDK).

    [D1] mode='cloud': Isaac Sim 이 EC2 에 있고 Leader 가 로컬 PC 에 있는 구성에서
         녹화 스크립트의 실행 위치/Leader 접근 경로가 미확정. 현재는 local 만 구현.
    """
    if args.mode == 'cloud':
        raise NotImplementedError(
            '[D1] cloud 모드 미구현 — EC2 Isaac Sim + 로컬 Leader 구성에서 '
            'Leader 접근 및 토픽 통신(rosbridge) 방식 결정 필요.'
        )
    leader = SO101Leader(SO101LeaderConfig(
        port=args.teleop_port, id=args.teleop_id, use_degrees=True))
    leader.connect()
    return leader


# ──────────────────────────────────────────────
# 헬퍼 — 종료 신호 / 단위 변환 / home 복귀
# ──────────────────────────────────────────────

def _user_wants_quit() -> bool:
    """stdin 'q'+Enter 비차단 검사 ([S3], Linux/macOS 전용)."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except (ValueError, OSError):
        return False
    if not ready:
        return False
    line = sys.stdin.readline()
    return bool(line) and line.strip().lower() == 'q'


def _leader_action_to_vec(action: dict) -> np.ndarray:
    """
    Leader SDK 의 {motor}.pos dict → SO101_JOINT_NAMES 순서 (6,) float32.

    이 값이 dataset 의 `action` 채널에 그대로 저장된다 — Real record_teleop.py
    의 action 과 동일 경로/단위 (LeRobot SDK .pos). 변환 불요 ([S6]).
    """
    return np.array(
        [float(action[f'{n}.pos']) for n in SO101_JOINT_NAMES],
        dtype=np.float32,
    )


def _gripper_pct_to_rad(pct: float) -> float:
    """Leader gripper 0~100% → Isaac Sim gripper rad ([D6])."""
    pct = max(0.0, min(100.0, pct))
    deg = GRIPPER_MIN_DEG + (GRIPPER_MAX_DEG - GRIPPER_MIN_DEG) * (pct / 100.0)
    return math.radians(deg)


def _leader_action_to_joint_command(action: dict) -> List[float]:
    """
    Leader SDK action → Isaac Sim /joint_command 용 rad 위치 리스트.

    5개 revolute joint 는 degree → rad, gripper 는 0~100% → rad.
    브리지 lerobot_to_rosbridge.py 의 변환과 동일 ([D6] gripper 상수).
    """
    positions: List[float] = []
    for j in SO101_JOINT_NAMES:
        val = float(action[f'{j}.pos'])
        if j == 'gripper':
            positions.append(_gripper_pct_to_rad(val))
        else:
            positions.append(math.radians(val))
    return positions


def _sim_jointstate_to_sdk_units(js: JointState, follower_calib: dict) -> np.ndarray:
    """
    Isaac Sim follower JointState(rad) → LeRobot SDK .pos 단위 (6,) float32.

    이 값이 dataset 의 `observation.state` 채널에 저장된다. Real 의
    observation.state(팔로워 SDK .pos)와 같은 단위여야 Co-training 이 가능하다.

    [D3] 미확정 — 변환식:
        Isaac Sim joint position (rad)
          → degree (× 180/π)
          → follower 캘리브레이션 yaml 의 affine(zero offset / range) 적용
          → LeRobot SDK 가 .pos 로 반환하는 값과 동일 스케일
        gripper 는 SDK 의 0~100 정규화 규약에 맞춰 별도 매핑.
        구 generate_act_dataset.py [5] 의 단위 변환·결합 규칙 참고:
          - 배포 로봇과 동일 캘리브레이션 yaml 고정
          - 재캘리브레이션 시 기존 Sim 데이터셋 무효화
          - home pose 한 프레임 ±1% sanity check 후 본격 수집
    """
    # 1) Isaac Sim JointState 의 name 순서를 SO101_JOINT_NAMES 순서로 정렬
    name_to_pos = dict(zip(js.name, js.position))
    try:
        rad = np.array([float(name_to_pos[n]) for n in SO101_JOINT_NAMES],
                       dtype=np.float64)
    except KeyError as e:
        raise RuntimeError(
            f'sim joint_states 에 관절 {e} 누락 — Isaac Sim Action Graph '
            'joint_states 발행 설정 확인 ([D4]).'
        ) from e

    # 2) rad → SDK .pos 단위 변환 ── [D3] 미확정
    raise NotImplementedError(
        '[D3] observation.state 단위 변환 미구현 — follower 캘리브레이션 yaml '
        f'(affine) 기반 rad → SDK .pos 변환 구현 필요. (입력 rad={rad.tolist()})'
    )


def _sim_rgb_to_array(msg: Image) -> np.ndarray:
    """
    Isaac Sim 카메라 Image → numpy (H,W,3) uint8 RGB.

    cv_bridge 미사용 (sim teleop env 에 없음). encoding 은 numpy 레벨에서 rgb8 로
    통일한다. Isaac Sim ROS2 카메라의 실제 encoding 확인 후 분기 보강할 것 ([D4]).
    """
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    enc = (msg.encoding or '').lower()
    if enc in ('rgb8', 'rgba8'):
        rgb = arr[:, :, :3]
    elif enc in ('bgr8', 'bgra8'):
        rgb = arr[:, :, 2::-1]
    else:
        raise NotImplementedError(
            f'[D4] 미지원 이미지 encoding {msg.encoding!r} — Isaac Sim 카메라 '
            '발행 encoding 확인 후 분기 추가.'
        )
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def load_initial_pose(path: str) -> List[float]:
    """
    에피소드 시작 강제 자세를 home_pose.yaml 에서 로드 → SO101_JOINT_NAMES
    순서 rad 리스트.

    실물 데이터 수집(record_teleop.py)이 캡처해 둔 follower rest 자세를 그대로
    재사용한다 — Sim/Real 의 에피소드 시작 자세를 일치시키고 별도 측정을
    생략한다 ([D7]). 파일 스키마는 robot_arm_pkg.utils.home_pose 와 동일:

        positions:
          shoulder_pan:  <float>   # LeRobot SDK .pos (arm: deg)
          shoulder_lift: <float>
          elbow_flex:    <float>
          wrist_flex:    <float>
          wrist_roll:    <float>
          gripper:       <float>   # LeRobot SDK .pos (gripper: 0~100)

    단위 변환 + clamp:
      home_pose.yaml 의 arm 값은 LeRobot SDK .pos = degree 이고 Isaac Sim USD
      joint 와 같은 degree 척도이다 → math.radians() 로 rad 변환. gripper 는
      0~100% 라 _gripper_pct_to_rad() 로 변환한다 (teleop 브리지 /
      _leader_action_to_joint_command 와 동일한 SDK→rad 규약).

      실물 SO-101 은 sim USD 보다 가동범위가 약간 넓어, home_pose 캡처 시
      관절이 USD 한계를 넘어 측정될 수 있다 (예: shoulder_lift -111° vs USD
      한계 -100°). 이런 값을 그대로 SetJointPositions 하면 한계 밖이 되므로
      arm 각 관절을 SO101_JOINT_LIMITS_DEG 로 clamp 하고, clamp 가 발생하면
      경고를 출력한다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'home_pose.yaml 을 찾을 수 없음: {path}\n'
            'record_teleop.py 가 생성한 ~/calibration/home_pose.yaml 을 '
            'deploy/calibration/ 로 복사하거나 --home-pose-file 로 경로 지정.'
        )
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    positions = data.get('positions')
    if not isinstance(positions, dict):
        raise ValueError(f'home_pose.yaml 에 positions: 딕셔너리가 없음 ({path}).')
    missing = [n for n in SO101_JOINT_NAMES if n not in positions]
    if missing:
        raise ValueError(
            f'home_pose.yaml positions 에 관절 {missing} 누락 ({path}).'
        )

    positions_rad: List[float] = []
    for n in SO101_JOINT_NAMES:
        sdk_val = float(positions[n])
        if n == 'gripper':
            # gripper 0~100% → rad. _gripper_pct_to_rad 가 입력을 0~100 으로
            # 이미 clamp 하므로 출력은 USD gripper 한계 안에 들어온다.
            positions_rad.append(_gripper_pct_to_rad(sdk_val))
            continue
        lo, hi = SO101_JOINT_LIMITS_DEG[n]
        clamped = min(hi, max(lo, sdk_val))
        if clamped != sdk_val:
            print(f'[record_sim_teleop] 경고 — {n} home_pose {sdk_val:.2f}° 가 '
                  f'Isaac Sim USD 한계 [{lo}, {hi}]° 밖 → {clamped:.2f}° 로 clamp')
        positions_rad.append(math.radians(clamped))
    return positions_rad


def _force_initial_pose(helper: SimRecordHelperNode,
                        initial_pose_rad: List[float],
                        episode_idx: int) -> None:
    """
    에피소드 시작 시 follower 를 initial_pose 로 강제 고정 ([S4]).

    set_joint_positions 만으로는 다음 물리 스텝에서 PD 가 *직전 drive target*
    (이전 에피소드의 마지막 /joint_command, 첫 reset 시엔 USD default)으로 팔을
    끌어간다. 따라서 reset 은 위치 강제 + 속도 0 + drive target 갱신을 함께,
    한 물리 스텝 안에서 원자적으로 적용해야 하며 그 책임은 Isaac Sim 측
    /joint_reset 핸들러에 있다 ([D4]) — ROS2 토픽만으로는 tick 단위 적용
    순서를 보장할 수 없기 때문이다.

    아래에서 /joint_command 를 함께 발행하는 것은, Isaac Sim 핸들러가 drive
    target 까지 갱신하지 않는 경우를 대비한 fallback 이다 (핸들러가 3가지를
    원자적으로 처리하면 이 publish 는 무해한 중복).

    녹화는 이 함수 반환 후 작업자가 Enter 를 누른 시점부터 시작되므로, 토픽
    적용 순서로 인한 일시적 transient 가 있더라도 RESET_SETTLE_S 안정화
    구간에 흡수되어 데이터셋에는 들어가지 않는다. 반환 직전 joint_states 로
    초기 자세 도달을 약식 검증한다.
    """
    helper.publish_joint_command(initial_pose_rad)   # drive target = 초기 자세
    helper.publish_joint_reset(initial_pose_rad)     # kinematic teleport
    time.sleep(RESET_SETTLE_S)

    # 약식 검증 — /joint_reset 이 SetJointPositions 로 라우팅됐는지 확인.
    js = helper.latest_joint_state()
    if js is None:
        print(f'[ep {episode_idx}] 경고 — joint_states 미수신, 초기 자세 검증 불가.')
        return
    name_to_pos = dict(zip(js.name, js.position))
    try:
        actual = np.array([float(name_to_pos[n]) for n in SO101_JOINT_NAMES],
                          dtype=np.float64)
    except KeyError:
        print(f'[ep {episode_idx}] 경고 — joint_states 관절 누락, 초기 자세 검증 불가.')
        return
    target = np.asarray(initial_pose_rad, dtype=np.float64)
    max_err_deg = math.degrees(float(np.max(np.abs(actual - target))))
    if max_err_deg > 2.0:
        print(f'[ep {episode_idx}] 경고 — 초기 자세 강제 후 최대 오차 '
              f'{max_err_deg:.1f}° (>2°). /joint_reset → SetJointPositions '
              f'Action Graph 배선 확인 ([D4]).')


# ──────────────────────────────────────────────
# 메인 — 에피소드 녹화 루프
# ──────────────────────────────────────────────

def record_episode(
    args:          SimRecordArgs,
    helper:        SimRecordHelperNode,
    leader:        SO101Leader,
    dataset:       LeRobotDataset,
    follower_calib: dict,
    episode_idx:   int,
    label_record:  dict,
    initial_pose_rad: List[float],
) -> bool:
    """
    1개 에피소드 녹화 → 저장.

    Returns:
        True  — 사용자가 저장 선택, save_episode() 완료.
        False — 저장 거부, clear_episode_buffer() 로 버퍼 폐기.
        (호출자는 이 값으로 retry 시 delete_episode() 호출 여부 결정)

    시작 시: initial_pose 로 follower 강제 고정 (_force_initial_pose, [S4]).

    각 틱(1/fps 초)마다:
      1) 종료 신호 확인 ('q'+Enter) — [S3]
      2) 카메라/관절 콜백은 백그라운드 executor 가 비동기 갱신 — [S1]
      3) Leader action 읽기 → rad 변환 → /joint_command 발행 (sim follower 미러링)
      4) sim follower 관절 상태 + 최신 RGB 수집
      5) frame dict 구성 + dataset.add_frame
           observation.state              ← _sim_jointstate_to_sdk_units  [D3]
           observation.language_embedding ← placeholder zero 벡터          [D9]
           observation.images.top         ← _sim_rgb_to_array
           action                         ← _leader_action_to_vec
      6) deadline 기반 sleep (drift 재동기화) — [S5]

    종료 후: save_episode().
    """
    # observation.language_embedding 은 placeholder zero 벡터 ([D9]).
    # 실제 임베딩은 수집 후 generate_language_labels.py 가 채운다.
    embed_vec = np.zeros(args.embedding_dim, dtype=np.float32)
    task_str = label_record['canonical']
    dt = 1.0 / args.fps

    # 카메라 / joint_states 첫 프레임 대기 — Isaac Sim Action Graph 동작 확인
    print(f'\n[ep {episode_idx}] Isaac Sim 관측 첫 프레임 대기...')
    wait_deadline = time.perf_counter() + 5.0
    while helper.latest_rgb() is None or helper.latest_joint_state() is None:
        if time.perf_counter() > wait_deadline:
            raise RuntimeError(
                '[record_sim_teleop] Isaac Sim 관측 미수신 (5s timeout). '
                'Action Graph 의 카메라/joint_states 발행 설정 확인 ([D4]).'
            )
        time.sleep(0.05)

    # follower 를 initial_pose 로 강제 고정 ([S4]) — 매 에피소드 동일 초기 상태
    print(f'[ep {episode_idx}] follower 초기 자세 강제 (SetJointPositions)...')
    _force_initial_pose(helper, initial_pose_rad, episode_idx)

    print(f'[ep {episode_idx}] task = {task_str!r}')
    print(f'[ep {episode_idx}] Leader 를 초기 자세 근처로 옮긴 뒤 Enter 로 녹화 시작.')
    input()

    print(f'[ep {episode_idx}] 녹화 중 — "q"+Enter 또는 Ctrl+C 로 종료')
    n_frames  = 0
    t_start   = time.perf_counter()
    next_dl   = t_start
    quit_flag = False
    try:
        while True:
            # 1) 종료 신호
            if _user_wants_quit():
                quit_flag = True
                break

            # 3) Leader → sim follower 미러링
            leader_action = leader.get_action()
            helper.publish_joint_command(
                _leader_action_to_joint_command(leader_action))

            # 4) sim follower 상태 + 최신 RGB
            js_msg  = helper.latest_joint_state()
            rgb_msg = helper.latest_rgb()
            if js_msg is None or rgb_msg is None:
                next_dl += dt
                _sleep_until(next_dl)
                continue

            # 5) frame 구성 + LeRobot 데이터셋 적재
            frame = {
                'observation.state':              _sim_jointstate_to_sdk_units(
                                                      js_msg, follower_calib),
                'observation.language_embedding': embed_vec,
                CAMERA_KEY:                       _sim_rgb_to_array(rgb_msg),
                'action':                         _leader_action_to_vec(leader_action),
                'task':                           task_str,
            }
            dataset.add_frame(frame)
            n_frames += 1

            # 6) fps 유지 — deadline 기반 (drift 재동기화)
            next_dl += dt
            slack = next_dl - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -dt:
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

    dataset.save_episode()
    print(f'[ep {episode_idx}] save_episode 호출 완료 — 인코딩 백그라운드 진행')
    # follower 초기 자세 복귀는 다음 에피소드 시작의 _force_initial_pose 가 담당.
    return True


def _sleep_until(deadline_perf: float) -> None:
    """time.perf_counter() 기준 deadline 까지 sleep. 음수면 즉시 반환."""
    remaining = deadline_perf - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _load_follower_calib(path: str) -> dict:
    """follower 캘리브레이션 yaml 로드 ([D3] 변환 입력)."""
    if not path:
        raise RuntimeError(
            '--follower-calib 미지정 — observation.state 단위 변환 ([D3]) 에 '
            'follower 캘리브레이션 yaml 이 필요하다.'
        )
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()

    # 1) Config 로드
    vocab     = load_vocab(args.label_vocab, args.domain)
    scene_cfg = load_scenes(args.scenes)
    dr_cfg    = load_dr_config(args.dr_config)
    follower_calib = _load_follower_calib(args.follower_calib)
    rng = np.random.default_rng(args.seed)
    print(f'[record_sim_teleop] config 로드 완료 (domain={args.domain}, '
          f'mode={args.mode}, scenes={len(scene_cfg.scenes)})')

    if args.embedding_dim != EXPECTED_EMBED_DIM:
        print(f'[record_sim_teleop] 경고 — embedding_dim={args.embedding_dim} != '
              f'{EXPECTED_EMBED_DIM}. generate_language_labels.py 의 임베딩 모델과 '
              '일치하는지 확인할 것 ([D9]).')

    # 2) ROS2 초기화 + 백그라운드 executor
    rclpy.init()
    helper = SimRecordHelperNode(args)
    executor = SingleThreadedExecutor()
    executor.add_node(helper)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    scene_manager = SceneManager(scene_cfg, dr_cfg, rng)

    leader: Optional[SO101Leader] = None
    try:
        # 3) Leader 연결 ([D1] — cloud 모드는 미구현)
        leader = connect_leader(args)

        # 4) 데이터셋 + 라벨 로그
        dataset = open_dataset(args)
        labels_jsonl = _resolve_labels_jsonl(args, dataset)
        print(f'[record_sim_teleop] dataset.root = {dataset.root}')
        print(f'[record_sim_teleop] label log    = {labels_jsonl}')

        # 5) 에피소드 시작 강제 자세 로드 — 실물 home_pose.yaml 재사용 ([S4], [D7])
        initial_pose_rad = load_initial_pose(args.home_pose_file)
        print(f'[record_sim_teleop] 초기 자세 로드: {args.home_pose_file}')

        # 6) 에피소드 루프
        start_ep = dataset.num_episodes
        for ep in range(start_ep, start_ep + args.num_episodes):
            # 6-1) 타겟 선택 + Isaac Sim 씬 구성 + DR ([D2], [D5])
            target = select_episode_target(scene_cfg, vocab, ep, rng)
            scene_manager.setup_episode_scene(ep, target)
            scene_manager.apply_domain_randomization(ep)

            # 6-2) 녹화 (retry 루프)
            while True:
                append_label_record(labels_jsonl, target)
                saved = record_episode(
                    args, helper, leader, dataset,
                    follower_calib, ep, target, initial_pose_rad,
                )
                retry = input(
                    f'[ep {ep}] 이 에피소드를 다시 촬영할까요? (y/n): '
                ).strip().lower()
                if retry == 'y':
                    if saved:
                        dataset.delete_episode()
                    print(f'[ep {ep}] 재촬영 — 타겟/씬 다시 구성합니다.')
                    target = select_episode_target(scene_cfg, vocab, ep, rng)
                    scene_manager.setup_episode_scene(ep, target)
                    scene_manager.apply_domain_randomization(ep)
                else:
                    break

            # 6-3) 다음 에피소드용 씬 정리 ([D2])
            scene_manager.reset()

    finally:
        if leader is not None:
            try:
                leader.disconnect()
            except Exception as e:
                print(f'[record_sim_teleop] leader disconnect 오류: {e}')
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        helper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
