"""
record_sim_teleop.py

실행 위치: Isaac Sim Python (외부 PC 로컬 또는 EC2 클라우드).

Isaac Sim teleop 기반 ACT 학습용 Sim 시연 데이터 수집 (Isaac Sim 직접 API).

[D1] 로컬 통합 — Isaac Sim · SO-101 Leader · 본 스크립트를 한 Ubuntu 머신에서
실행한다. 본 스크립트는 Isaac Sim 의 standalone Python 앱(SimulationApp)으로
동작하며, follower · 카메라 · 관절을 ROS2 가 아니라 Isaac Sim API 로 직접
접근한다. (이전 ROS2/rclpy 설계 — SimRecordHelperNode, /joint_command·
/joint_reset 토픽, --mode cloud — 는 [D1] 확정으로 폐기.)

기록 채널 (Real record_real_teleop.py 와 동일 스펙 — Co-training 조건):
  observation.state              (6,)    float32 — follower 관절 (SDK .pos 단위, [D3])
  observation.language_embedding (D,)    float32 — placeholder zero ([D9])
  observation.images.top         (H,W,3) uint8   — Isaac Sim 카메라 RGB
  action                         (6,)    float32 — 실물 Leader 관절 값 (SDK .pos)

실행 — 반드시 Isaac Sim 의 python 으로 (시스템 python 아님):
  ~/isaacsim/python.sh training/imitation_data/scripts/record_sim_teleop.py \\
      --scene-usd        sim/scenes/base_scene.usd \\
      --teleop-port      /dev/ttyACM0 \\
      --dataset-repo-id  myname/dataset_sim \\
      --domain           recycling

전제조건:
  1. Isaac Sim 5.1 + 그 python 환경에 lerobot 0.4.4 설치
     (~/isaacsim/python.sh -m pip install "lerobot[feetech]==0.4.4")
  2. sim/scenes/base_scene.usd (robot + Camera_RGB + bins) 준비
  3. 실물 SO-101 Leader USB 연결 + lerobot-calibrate 완료
  4. deploy/calibration/home_pose.yaml,
     deploy/config/label_vocab.yaml, sim/config/{multi_target_scenes,dr_config}.yaml

설계 결정 [D1]~[D9] — 모두 확정 (이전 README "Sim teleop 수집" 표에서 이관):
  D1 로컬 통합 — Isaac Sim·SO-101 Leader·본 스크립트를 한 Ubuntu 머신에서
     실행. 본 스크립트는 Isaac Sim Python 내에서 돌고, Leader 는 LeRobot SDK
     로 직접 read, follower·카메라·관절은 Isaac Sim API 로 직접 접근한다.
  D2 SceneManager 가 Isaac Sim API 직접 호출로 multi-target 씬 구성 + DR
     적용 (D1 귀결 — ROS2 서비스/동반 스크립트 불필요).
  D3 observation.state 변환 — arm 은 plain deg↔rad, gripper 는 %-매핑,
     affine 불필요. round-trip 검증.
  D4 ROS2 Action Graph 불필요 — D1=로컬 통합으로 녹화가 Isaac Sim 직접 API
     를 쓰므로 ROS2 Action Graph 확장이 무효화된다.
  D5 자동 분포 보장 — TargetSelector 가 multi/single 비율·장면·타겟 카테고리
     를 누적 분포로 균형 배분해 multi-target shortcut 을 코드로 방지한다.
  D6 gripper 0~100% ↔ -10°~100° (USD joint limit 으로 확인).
  D7 sim follower 초기 자세 — deploy/calibration/home_pose.yaml 재사용,
     set_joint_positions 강제 + USD limit clamp.
  D8 fps=30 (Real record_real_teleop.py 와 일치), 에피소드 길이는 작업자
     'q' 종료.
  D9 language_embedding — 녹화 시 placeholder zero, 수집 후
     generate_language_labels.py 가 채운다.

상태 — 핵심 로직 구현 완료. SimInterface(Isaac Sim 래퍼)·SceneManager(객체 배치
·DR)·TargetSelector(자동 분포 타겟 선택, D5) 모두 구현됨. Isaac Sim 5.1 API
호출은 best-effort 이므로 실기 검증 필요. DR 의 texture/friction/pd_gain/
observation_noise 는 미구현(dr_config.yaml 주석 — 의도적 보류).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# CLI — SimulationApp 기동보다 먼저 파싱해야 한다
# ──────────────────────────────────────────────

@dataclass
class SimRecordArgs:
    scene_usd:       str
    teleop_port:     str
    teleop_id:       str
    dataset_repo_id: str
    domain:          str
    label_vocab:     str
    scenes:          str
    dr_config:       str
    home_pose_file:  str
    num_episodes:    int
    fps:             int
    img_h:           int
    img_w:           int
    embedding_dim:   int
    labels_jsonl:    str
    seed:            int
    headless:        bool
    robot_prim:      str
    camera_prim:     str


def parse_args() -> SimRecordArgs:
    p = argparse.ArgumentParser(description='Isaac Sim teleop Sim 시연 데이터 수집')
    p.add_argument('--scene-usd',       required=True, help='베이스 씬 USD (robot+camera+bins)')
    p.add_argument('--teleop-port',     required=True, help='실물 SO-101 Leader USB 포트')
    p.add_argument('--teleop-id',       default='leader', help='Leader 캘리브레이션 ID')
    p.add_argument('--dataset-repo-id', required=True, help='LeRobotDataset repo id')
    p.add_argument('--domain',          required=True, help='수집 도메인 (label_vocab domains 키)')
    p.add_argument('--label-vocab',     default='deploy/config/label_vocab.yaml')
    p.add_argument('--scenes',          default='sim/config/multi_target_scenes.yaml')
    p.add_argument('--dr-config',       default='sim/config/dr_config.yaml')
    p.add_argument('--home-pose-file',  default='deploy/calibration/home_pose.yaml')
    p.add_argument('--num-episodes',    type=int, default=500)
    p.add_argument('--fps',             type=int, default=30, help='[D8] Real 과 일치')
    p.add_argument('--img-h',           type=int, default=480)
    p.add_argument('--img-w',           type=int, default=640)
    p.add_argument('--embedding-dim',   type=int, default=384,
                   help='[D9] placeholder 차원 — generate_language_labels.py 모델과 일치')
    p.add_argument('--labels-jsonl',    default='')
    p.add_argument('--seed',            type=int, default=0)
    p.add_argument('--headless',        action='store_true',
                   help='렌더 없이 실행 (teleop 에는 비권장 — 작업자가 못 봄)')
    p.add_argument('--robot-prim',      default='/World/so101_new_calib',
                   help='씬 내 SO-101 articulation prim 경로')
    p.add_argument('--camera-prim',     default='/World/Camera_RGB',
                   help='씬 내 RGB 카메라 prim 경로')
    a = p.parse_args()
    return SimRecordArgs(
        scene_usd       = os.path.expanduser(a.scene_usd),
        teleop_port     = a.teleop_port,
        teleop_id       = a.teleop_id,
        dataset_repo_id = a.dataset_repo_id,
        domain          = a.domain,
        label_vocab     = os.path.expanduser(a.label_vocab),
        scenes          = os.path.expanduser(a.scenes),
        dr_config       = os.path.expanduser(a.dr_config),
        home_pose_file  = os.path.expanduser(a.home_pose_file),
        num_episodes    = a.num_episodes,
        fps             = a.fps,
        img_h           = a.img_h,
        img_w           = a.img_w,
        embedding_dim   = a.embedding_dim,
        labels_jsonl    = os.path.expanduser(a.labels_jsonl) if a.labels_jsonl else '',
        seed            = a.seed,
        headless        = a.headless,
        robot_prim      = a.robot_prim,
        camera_prim     = a.camera_prim,
    )


ARGS = parse_args()


# ──────────────────────────────────────────────
# Isaac Sim 기동 — 반드시 다른 isaacsim/omni/pxr/lerobot import 보다 먼저
# ──────────────────────────────────────────────

from isaacsim import SimulationApp                            # noqa: E402

simulation_app = SimulationApp({'headless': ARGS.headless})

# SimulationApp 생성 이후에만 import 가능 ──────────────────────
import numpy as np                                            # noqa: E402
import yaml                                                   # noqa: E402

# Isaac Sim 5.1 기준 import 경로. 버전이 다르면 조정 필요
#   (4.x: omni.isaac.* 네임스페이스 / 5.x: isaacsim.*)
from isaacsim.core.api import World                           # noqa: E402
from isaacsim.core.prims import SingleArticulation            # noqa: E402
from isaacsim.core.utils.stage import open_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction      # noqa: E402
from isaacsim.sensors.camera import Camera                    # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics     # noqa: E402

# LeRobot — Isaac Sim python 환경에 설치돼 있어야 함 (전제조건 1)
from lerobot.datasets.lerobot_dataset import LeRobotDataset                  # noqa: E402
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig   # noqa: E402


# ──────────────────────────────────────────────
# 운영 노트 ([D1]~[D9] 확정 반영)
# ──────────────────────────────────────────────
#  - [D1] Isaac Sim standalone app — follower/카메라/관절 전부 직접 API.
#  - [D3] observation.state: sim rad → SDK .pos = plain deg↔rad (_sim_jointpos_to_sdk).
#  - [D7] 에피소드 시작 시 home_pose.yaml 자세로 set_joint_positions 강제.
#  - [D8] fps=30 — World(physics_dt = rendering_dt = 1/fps).
#  - [D9] observation.language_embedding 은 placeholder zero — 수집 후
#         generate_language_labels.py 가 실제 임베딩으로 채운다.
#  - 종료 정책: 녹화 중 'q'+Enter (Linux). 대기 중에도 world.step() 으로
#    뷰포트를 갱신해 작업자가 sim 을 실시간으로 보며 Leader 를 정렬할 수 있게 한다.

EXPECTED_EMBED_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2

# SO-101 관절 이름 — record_real_teleop.py / 브리지와 동일 순서
SO101_JOINT_NAMES: List[str] = [
    'shoulder_pan', 'shoulder_lift', 'elbow_flex',
    'wrist_flex', 'wrist_roll', 'gripper',
]

# Isaac Sim SO-101 USD joint limit — degree (Physics Inspector 실측값)
SO101_JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    'shoulder_pan':  (-110.0,   110.0),
    'shoulder_lift': (-100.0,   100.0),
    'elbow_flex':    ( -96.83,   96.83),
    'wrist_flex':    ( -95.0,    95.0),
    'wrist_roll':    (-157.211, 162.789),
    'gripper':       ( -10.0,   100.0),
}

# gripper 매핑 상수 ([D6] — USD joint limit 으로 확정)
GRIPPER_MIN_DEG = -10.0    # Leader 0%
GRIPPER_MAX_DEG = 100.0    # Leader 100%

CAMERA_KEY = 'observation.images.top'

# 초기 자세 강제 후 물리 안정화 스텝 수
RESET_SETTLE_STEPS = 15


# ──────────────────────────────────────────────
# Config 로드 — vocabulary / scenes / DR / 초기 자세
# ──────────────────────────────────────────────

@dataclass
class LabelVocab:
    template: str
    spatials: List[str]
    objects:  Dict[str, List[str]]
    domain:   str = ''
    raw:      dict = field(default_factory=dict)


def load_vocab(path: str, domain: str) -> LabelVocab:
    """label_vocab.yaml 로드 + 도메인 추출 (record_real_teleop.py 와 동일 규약).

    LabelVocab.spatials = (전역 spatial_qualifiers + 도메인별 spatial_qualifiers).
    REGION_SPATIALS 5개는 전역 측에 있어야 하며, _validate 가 이를 검증한다.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    template        = raw.get('template')
    global_spatials = raw.get('spatial_qualifiers') or []
    domains         = raw.get('domains') or {}
    if not template or '{attribute}' not in template \
       or '{object}' not in template or '{spatial}' not in template:
        raise ValueError(f'label_vocab template 슬롯 누락 ({path}).')
    if not global_spatials:
        raise ValueError(f'전역 spatial_qualifiers 가 비어 있음 ({path}).')
    if domain not in domains:
        raise ValueError(f"도메인 '{domain}' 이 label_vocab 에 없음. 가능: {list(domains)}")
    domain_block    = domains[domain] or {}
    domain_spatials = domain_block.get('spatial_qualifiers') or []
    merged_spatials = list(global_spatials) + list(domain_spatials)
    objects: Dict[str, List[str]] = {}
    for obj_name, obj_def in (domain_block.get('objects') or {}).items():
        attrs = (obj_def or {}).get('attributes') or []
        if not attrs:
            raise ValueError(f"객체 '{obj_name}' attributes 비어 있음.")
        objects[obj_name] = list(attrs)
    if not objects:
        raise ValueError(f"도메인 '{domain}' 에 등록된 객체 없음.")
    return LabelVocab(template, merged_spatials, objects, domain, raw)


@dataclass
class SceneConfig:
    scenes:             List[dict]
    multi_target_ratio: float
    placement:          dict          # 객체 배치 영역·규칙 (SceneManager 사용)
    object_library:     str           # 래퍼 USD 디렉터리
    raw:                dict = field(default_factory=dict)


def load_scenes(path: str, domain: str) -> SceneConfig:
    """multi_target_scenes.yaml 의 domains.<domain> 블록 로드 + placement 검증.

    yaml 스키마 (2026-05-25):
        domains:
          <domain>:
            object_library: ...
            placement: {spawn_region, reach, table_top_z, min_separation,
                        exclusion_zones, max_sample_attempts}
            scenes: [...]
            multi_target_ratio: ...
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    domains = raw.get('domains') or {}
    if domain not in domains:
        raise ValueError(
            f"multi_target_scenes.yaml 에 도메인 '{domain}' 블록 없음. "
            f"가능: {list(domains)} ({path}).")
    dom = domains[domain] or {}
    scenes = dom.get('scenes') or []
    if not scenes:
        raise ValueError(
            f"도메인 '{domain}' 에 scenes 정의 없음 ({path}).")
    placement = dom.get('placement')
    if not isinstance(placement, dict) or 'spawn_region' not in placement:
        raise ValueError(
            f"도메인 '{domain}' 의 placement.spawn_region 없음 ({path}).")
    sr = placement['spawn_region']
    if 'x_range' not in sr or 'y_range' not in sr:
        raise ValueError(
            f"placement.spawn_region 에 x_range/y_range 필요 ({path}).")
    object_library = os.path.expanduser(
        dom.get('object_library', f'sim/assets/objects/{domain}'))
    return SceneConfig(
        scenes=list(scenes),
        multi_target_ratio=float(dom.get('multi_target_ratio', 0.0)),
        placement=placement,
        object_library=object_library,
        raw=dom,
    )


@dataclass
class DRConfig:
    raw: dict = field(default_factory=dict)


def load_dr_config(path: str) -> DRConfig:
    """dr_config.yaml 로드 (스키마 미확정 — raw 보관)."""
    with open(path, 'r', encoding='utf-8') as f:
        return DRConfig(yaml.safe_load(f) or {})


def load_initial_pose(path: str) -> List[float]:
    """
    에피소드 시작 강제 자세를 home_pose.yaml 에서 로드 → SO101_JOINT_NAMES 순서 rad.

    home_pose.yaml(`positions:`) 값은 LeRobot SDK .pos (arm=deg, gripper=0~100%).
    arm 은 math.radians(), gripper 는 _gripper_pct_to_rad() 로 변환하고, USD joint
    limit 을 넘는 값은 clamp 한다 ([D7]).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'home_pose.yaml 을 찾을 수 없음: {path} '
            '(record_real_teleop.py 가 생성한 ~/calibration/home_pose.yaml 을 '
            'deploy/calibration/ 로 복사)'
        )
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    positions = data.get('positions')
    if not isinstance(positions, dict):
        raise ValueError(f'home_pose.yaml 에 positions: 딕셔너리 없음 ({path}).')
    missing = [n for n in SO101_JOINT_NAMES if n not in positions]
    if missing:
        raise ValueError(f'home_pose.yaml positions 관절 누락 {missing} ({path}).')

    out: List[float] = []
    for n in SO101_JOINT_NAMES:
        sdk_val = float(positions[n])
        if n == 'gripper':
            out.append(_gripper_pct_to_rad(sdk_val))
            continue
        lo, hi = SO101_JOINT_LIMITS_DEG[n]
        clamped = min(hi, max(lo, sdk_val))
        if clamped != sdk_val:
            print(f'[record_sim_teleop] 경고 — {n} home_pose {sdk_val:.2f}° 가 '
                  f'USD 한계 [{lo}, {hi}]° 밖 → {clamped:.2f}° 로 clamp')
        out.append(math.radians(clamped))
    return out


# ──────────────────────────────────────────────
# 단위 변환 ([D3], [D6])
# ──────────────────────────────────────────────

def _gripper_pct_to_rad(pct: float) -> float:
    """Leader gripper 0~100% → rad ([D6])."""
    pct = max(0.0, min(100.0, pct))
    deg = GRIPPER_MIN_DEG + (GRIPPER_MAX_DEG - GRIPPER_MIN_DEG) * (pct / 100.0)
    return math.radians(deg)


def _leader_action_to_vec(action: dict) -> np.ndarray:
    """
    Leader SDK {motor}.pos → (6,) float32. dataset 의 `action` 채널에 그대로 저장.
    Real record_real_teleop.py 와 동일 경로/단위 (LeRobot SDK .pos) — 변환 불요.
    """
    return np.array([float(action[f'{n}.pos']) for n in SO101_JOINT_NAMES],
                    dtype=np.float32)


def _leader_to_sim_rad(action: dict) -> List[float]:
    """Leader SDK action → Isaac Sim follower 구동용 rad (SO101_JOINT_NAMES 순서)."""
    out: List[float] = []
    for n in SO101_JOINT_NAMES:
        val = float(action[f'{n}.pos'])
        out.append(_gripper_pct_to_rad(val) if n == 'gripper' else math.radians(val))
    return out


def _sim_jointpos_to_sdk(positions_rad) -> np.ndarray:
    """
    Isaac Sim follower 관절(rad) → LeRobot SDK .pos 단위 (6,) float32 — [D3].

    [D3] 확정: arm 은 plain deg↔rad (affine 불필요), gripper 는 %-매핑.
    dataset 의 `observation.state` 채널에 저장되며 Real 의 observation.state
    (follower SDK .pos)와 같은 단위여야 Co-training 이 가능하다.
    검증: 초기 자세 강제 후 이 함수 출력이 home_pose.yaml 값과 일치하는지 확인.
    """
    out: List[float] = []
    for n, rad in zip(SO101_JOINT_NAMES, positions_rad):
        deg = math.degrees(float(rad))
        if n == 'gripper':
            pct = (deg - GRIPPER_MIN_DEG) / (GRIPPER_MAX_DEG - GRIPPER_MIN_DEG) * 100.0
            out.append(pct)
        else:
            out.append(deg)
    return np.array(out, dtype=np.float32)


# ──────────────────────────────────────────────
# 타겟 선택 / 라벨 메타데이터
# ──────────────────────────────────────────────

# Sim 자동 라벨링에 쓰는 절대 영역 spatial — vocab spatial_qualifiers 의 부분집합.
# 관계형("next to the X")은 작업자 수동 라벨용이라 자동 도출에서 제외한다.
REGION_SPATIALS: Dict[str, str] = {
    'left':   'on the left side of the table',
    'right':  'on the right side of the table',
    'center': 'in the center of the table',
    'front':  'in front of the robot',
    'back':   'at the back of the table',
}


def _category_of(wrapper_name: str) -> str:
    """래퍼 USD 파일명 prefix = 카테고리 (can_master_chef.usd → 'can')."""
    return Path(wrapper_name).stem.split('_')[0]


def _frac(value: float, rng_pair) -> float:
    """value 를 [lo,hi] 구간 내 정규화 위치 [0,1] 로 — 구간 밖은 clamp."""
    lo, hi = float(rng_pair[0]), float(rng_pair[1])
    if hi <= lo:
        return 0.5
    return min(1.0, max(0.0, (float(value) - lo) / (hi - lo)))


class TargetSelector:
    """
    매 에피소드의 장면·배치 객체·grasp 타겟을 자동 선택한다 ([D5]).

    [D5] 자동 분포 보장 — multi-target shortcut learning 을 막기 위해 누적 분포를
    추적하며 선택을 배분한다:
      - scene multi/single 비율을 multi_target_scenes.yaml 의 multi_target_ratio
        로 수렴시킨다 (commit 된 에피소드 기준 결정적 밸런싱).
      - scene·타겟 카테고리는 최소 사용분을 우선 선택해 다양성을 강제한다.
      - spatial 다양성은 SceneManager 의 무작위 배치에서 자연히 확보된다
        (별도 카운트 불요).

    2단계 사용 — 객체 배치 전/후로 나뉜다:
      1. select(episode_idx, rng)            장면·객체·타겟 결정 (배치 전).
      2. SceneManager.setup_episode_scene()  실제 (x,y) 배치.
      3. finalize(target, placements)        실 배치 좌표로 spatial·canonical 확정.
    저장에 성공하면 commit(target) 으로 누적 분포에 반영한다 (재촬영·폐기분 제외).

    select() 가 채우는 target 계약 (SceneManager·record_episode 가 소비):
        {
          'episode': int, 'domain': str,
          'scene':         <multi_target_scenes.yaml scene name>,
          'objects':       [<래퍼 USD 파일명>, ...],   # setup_episode_scene 입력
          'target_object': <objects 중 grasp 대상 래퍼>,
          'target_slot':   <objects 내 인덱스>,
          'object':        <vocab object 명사 = 카테고리>,
          'attribute':     None,                       # [Q1] Gemma 후처리가 채움
          'multi_target':  bool,
          'spatial':   None,    # finalize 가 채움
          'canonical': None,    # finalize 가 채움
        }
    finalize() 후 spatial/canonical 이 확정된다. canonical 은 [Q1] 결정대로
    "the {object} {spatial}" — {attribute}(색)는 Sim 객체가 고정 YCB 메시이므로
    generate_language_labels.py(Gemma)가 렌더 이미지를 보고 채운다.
    """

    def __init__(self, scene_cfg: SceneConfig, vocab: LabelVocab):
        self.scene_cfg = scene_cfg
        self.vocab = vocab
        self._scenes: List[dict] = list(scene_cfg.scenes)
        self._ratio = float(scene_cfg.multi_target_ratio)

        # 객체 라이브러리 스캔 — 카테고리(파일명 prefix)별 래퍼 USD 목록
        lib = Path(scene_cfg.object_library)
        if not lib.is_dir():
            raise FileNotFoundError(
                f'object_library 디렉터리 없음: {lib} — '
                'sim/scene_setup/setup_objects.py 를 먼저 실행할 것.'
            )
        self._by_category: Dict[str, List[str]] = {}
        for f in sorted(lib.glob('*.usd')):
            self._by_category.setdefault(_category_of(f.name), []).append(f.name)

        # scene 을 multi-target(객체 2+) / single-target 으로 분류
        self._multi_scenes: List[dict] = []
        self._single_scenes: List[dict] = []
        for sc in self._scenes:
            (self._multi_scenes if self._scene_total(sc) >= 2
             else self._single_scenes).append(sc)

        self._validate(vocab)

        # 누적 분포 — commit 된(저장·유지된) 에피소드만 카운트
        self._n_multi = 0
        self._n_single = 0
        self._scene_counts: Dict[str, int] = {sc['name']: 0 for sc in self._scenes}
        self._target_cat_counts: Dict[str, int] = {c: 0 for c in self._by_category}

    # ── 검증 ─────────────────────────────────────────

    @staticmethod
    def _scene_total(scene: dict) -> int:
        return sum(int(o.get('count', 1)) for o in (scene.get('objects') or []))

    def _validate(self, vocab: LabelVocab) -> None:
        """scene 카테고리 ↔ 래퍼 라이브러리 ↔ label_vocab 동기화 검증 ([D5])."""
        if not self._scenes:
            raise ValueError('multi_target_scenes.yaml 에 scene 정의가 없음.')
        scene_cats = set()
        for sc in self._scenes:
            objs = sc.get('objects') or []
            if not objs:
                raise ValueError(f"scene '{sc.get('name')}' 에 objects 가 없음.")
            for o in objs:
                scene_cats.add(o['category'])
        for c in sorted(scene_cats):
            if c not in self._by_category:
                raise RuntimeError(
                    f"scene 카테고리 '{c}' 의 래퍼가 object_library 에 없음 — "
                    'setup_objects.py 의 DOMAINS 매핑 과 파일명 prefix 확인.'
                )
            if c not in vocab.objects:
                raise RuntimeError(
                    f"scene 카테고리 '{c}' 가 label_vocab 도메인 '{vocab.domain}' "
                    'objects 에 없음 — multi_target_scenes.yaml 과 '
                    'label_vocab.yaml 의 객체 어휘를 동기화할 것.'
                )
        for s in REGION_SPATIALS.values():
            if s not in vocab.spatials:
                raise RuntimeError(
                    f"label_vocab spatial_qualifiers 에 '{s}' 없음 — "
                    'Sim 자동 라벨링에 필요한 영역 spatial 이 누락됨.'
                )

    # ── 공개 API ─────────────────────────────────────

    def select(self, episode_idx: int, rng: np.random.Generator) -> Dict:
        """이번 에피소드의 장면·배치 객체·grasp 타겟 결정 (배치 전 1단계)."""
        scene = self._pick_scene(rng)
        objects, slot_cats = self._expand_objects(scene, rng)
        target_slot = self._pick_target_slot(slot_cats, rng)
        return {
            'episode':       int(episode_idx),
            'domain':        self.vocab.domain,
            'scene':         scene['name'],
            'objects':       objects,
            'target_object': objects[target_slot],
            'target_slot':   target_slot,
            'object':        slot_cats[target_slot],
            'attribute':     None,                       # [Q1] Gemma 후처리
            'multi_target':  len(objects) >= 2,
            'spatial':       None,                       # finalize 가 채움
            'canonical':     None,                       # finalize 가 채움
        }

    def finalize(self, target: Dict, placements: List[Dict]) -> None:
        """실 배치 좌표(placements)로 spatial·canonical 을 확정 (배치 후 3단계)."""
        slot = target['target_slot']
        if slot >= len(placements):
            raise RuntimeError(
                f'placements({len(placements)}) 가 target_slot({slot}) 보다 적음.')
        tp = placements[slot]
        tx, ty = float(tp['x']), float(tp['y'])

        # 같은 카테고리 형제 = 혼동 가능 객체 → spatial 이 반드시 변별해야 한다
        target_cat = target['object']
        siblings = [
            (float(p['x']), float(p['y']))
            for i, p in enumerate(placements)
            if i != slot and _category_of(target['objects'][i]) == target_cat
        ]
        region = self._disambiguating_region((tx, ty), siblings) \
            if siblings else None
        if region is None:                               # 형제 없음 → 영역 기반
            spawn = self.scene_cfg.placement['spawn_region']
            region = self._region_of(_frac(tx, spawn['x_range']),
                                     _frac(ty, spawn['y_range']))
        spatial = REGION_SPATIALS[region]
        target['spatial'] = spatial
        target['canonical'] = f"the {target['object']} {spatial}"

    def commit(self, target: Dict) -> None:
        """저장 성공한 에피소드를 누적 분포에 반영 (재촬영·폐기분은 호출 안 함)."""
        if target.get('multi_target'):
            self._n_multi += 1
        else:
            self._n_single += 1
        self._scene_counts[target['scene']] = \
            self._scene_counts.get(target['scene'], 0) + 1
        cat = target['object']
        self._target_cat_counts[cat] = self._target_cat_counts.get(cat, 0) + 1

    # ── 내부 — 선택 ──────────────────────────────────

    def _pick_scene(self, rng: np.random.Generator) -> dict:
        """multi_target_ratio 로 수렴하게 scene-type 결정 후 최소 사용 scene 선택."""
        total = self._n_multi + self._n_single
        if not self._multi_scenes:
            want_multi = False
        elif not self._single_scenes:
            want_multi = True
        elif total == 0:
            want_multi = bool(rng.random() < self._ratio)
        else:
            want_multi = (self._n_multi / total) < self._ratio
        group = self._multi_scenes if want_multi else self._single_scenes
        # 최소 commit 사용 scene 우선 (random tiebreak)
        min_used = min(self._scene_counts[sc['name']] for sc in group)
        cands = [sc for sc in group
                 if self._scene_counts[sc['name']] == min_used]
        return cands[int(rng.integers(len(cands)))]

    def _expand_objects(self, scene: dict,
                        rng: np.random.Generator) -> Tuple[List[str], List[str]]:
        """scene 의 (category,count) 명세를 구체 래퍼 USD 목록으로 전개."""
        objects: List[str] = []
        slot_cats: List[str] = []
        for spec in scene['objects']:
            cat = spec['category']
            count = int(spec.get('count', 1))
            pool = self._by_category[cat]
            replace = count > len(pool)   # 래퍼가 모자라면 중복 허용 (가능하면 distinct)
            idx = rng.choice(len(pool), size=count, replace=replace)
            for i in np.atleast_1d(idx):
                objects.append(pool[int(i)])
                slot_cats.append(cat)
        return objects, slot_cats

    def _pick_target_slot(self, slot_cats: List[str],
                          rng: np.random.Generator) -> int:
        """타겟 카테고리 누적 사용분이 가장 적은 슬롯을 우선 선택."""
        present = sorted(set(slot_cats))
        min_used = min(self._target_cat_counts.get(c, 0) for c in present)
        cand_cats = [c for c in present
                     if self._target_cat_counts.get(c, 0) == min_used]
        chosen_cat = cand_cats[int(rng.integers(len(cand_cats)))]
        slots = [i for i, c in enumerate(slot_cats) if c == chosen_cat]
        return slots[int(rng.integers(len(slots)))]

    # ── 내부 — spatial 도출 ──────────────────────────

    @staticmethod
    def _disambiguating_region(target_xy: Tuple[float, float],
                               siblings: List[Tuple[float, float]]
                               ) -> Optional[str]:
        """
        같은 카테고리 형제와 target 을 변별하는 영역 spatial 키를 도출.

        형제와 더 크게 벌어진 축을 채택하고, 그 축에서 target 이 모든 형제보다
        한쪽 극단일 때만 해당 region 을 반환한다 (아니면 None → 영역 fallback).
        축 규약: x+ = 로봇에서 멀어짐(back), y+ = 로봇 기준 왼쪽(left).
        ⚠️ base_scene.usd 실측 후 규약이 어긋나면 아래 부호를 뒤집을 것.
        """
        tx, ty = target_xy
        max_dx = max(abs(tx - sx) for sx, _ in siblings)
        max_dy = max(abs(ty - sy) for _, sy in siblings)
        if max_dy >= max_dx:
            if all(ty > sy for _, sy in siblings):
                return 'left'
            if all(ty < sy for _, sy in siblings):
                return 'right'
        else:
            if all(tx < sx for sx, _ in siblings):
                return 'front'
            if all(tx > sx for sx, _ in siblings):
                return 'back'
        return None

    @staticmethod
    def _region_of(fx: float, fy: float) -> str:
        """spawn_region 내 정규화 위치(fx,fy ∈ [0,1])를 영역 spatial 키로 매핑."""
        near, far = 0.34, 0.66
        if near <= fx <= far and near <= fy <= far:
            return 'center'
        if abs(fy - 0.5) >= abs(fx - 0.5):
            return 'left' if fy >= 0.5 else 'right'
        return 'back' if fx >= 0.5 else 'front'


def _resolve_labels_jsonl(args: SimRecordArgs,
                          dataset: Optional[LeRobotDataset] = None) -> str:
    if args.labels_jsonl:
        path = args.labels_jsonl
    elif dataset is not None:
        path = str(Path(dataset.root) / 'labels.jsonl')
    else:
        path = os.path.join(os.getcwd(),
                            f"labels_{args.dataset_repo_id.replace('/', '_')}.jsonl")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def append_label_record(jsonl_path: str, record: dict) -> None:
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ──────────────────────────────────────────────
# SimInterface — Isaac Sim 직접 API 래퍼
# ──────────────────────────────────────────────

class SimInterface:
    """
    Isaac Sim 의 follower articulation · 카메라 · 물리 스텝을 직접 API 로 감싼다.
    ROS2 토픽/Action Graph 를 거치지 않는다 ([D1], [D4]).

    NOTE: 아래 Isaac Sim 5.1 API 호출은 best-effort 다 — World/SingleArticulation/
          Camera 의 정확한 시그니처는 설치된 Isaac Sim 버전에서 검증할 것.
    """

    def __init__(self, args: SimRecordArgs):
        self.args = args
        if not os.path.exists(args.scene_usd):
            raise FileNotFoundError(f'씬 USD 없음: {args.scene_usd}')

        # 씬 USD 로드 → World 생성 (fps 에 맞춰 물리/렌더 dt 설정, [D8])
        open_stage(args.scene_usd)
        self.world = World(physics_dt=1.0 / args.fps,
                           rendering_dt=1.0 / args.fps)
        self.stage = get_current_stage()   # SceneManager 의 USD 조작용

        self.robot = SingleArticulation(prim_path=args.robot_prim, name='so101_follower')
        self.camera = Camera(prim_path=args.camera_prim,
                             resolution=(args.img_w, args.img_h))

        # 물리 초기화 — articulation/camera 핸들 유효화
        self.world.reset()
        self.robot.initialize()
        self.camera.initialize()

        # articulation DOF 순서 → SO101_JOINT_NAMES 인덱스 매핑
        self._dof_index = self._build_dof_index()
        self._n_dof = len(self.robot.dof_names)

    def _build_dof_index(self) -> Dict[str, int]:
        dof_names = list(self.robot.dof_names)
        idx: Dict[str, int] = {}
        for n in SO101_JOINT_NAMES:
            if n not in dof_names:
                raise RuntimeError(
                    f"articulation '{self.args.robot_prim}' 에 관절 '{n}' 없음. "
                    f'DOF: {dof_names}'
                )
            idx[n] = dof_names.index(n)
        return idx

    def step(self, render: bool = True) -> None:
        self.world.step(render=render)

    def get_follower_joint_positions(self) -> np.ndarray:
        """follower 관절 위치(rad) — SO101_JOINT_NAMES 순서."""
        q = self.robot.get_joint_positions()
        return np.array([float(q[self._dof_index[n]]) for n in SO101_JOINT_NAMES],
                        dtype=np.float64)

    def drive_follower(self, positions_rad: List[float]) -> None:
        """SO101_JOINT_NAMES 순서 rad 를 follower drive target 으로 적용."""
        full = np.zeros(self._n_dof, dtype=np.float32)
        for n, rad in zip(SO101_JOINT_NAMES, positions_rad):
            full[self._dof_index[n]] = rad
        self.robot.apply_action(ArticulationAction(joint_positions=full))

    def force_initial_pose(self, positions_rad: List[float]) -> None:
        """
        [D7] follower 를 초기 자세로 강제. 한 번에 위치 강제 + 속도 0 + drive
        target 일치 — set_joint_positions 만 하면 다음 스텝에서 PD 가 직전 drive
        target 으로 끌어가므로 셋을 함께 적용한다.
        """
        full = np.zeros(self._n_dof, dtype=np.float32)
        for n, rad in zip(SO101_JOINT_NAMES, positions_rad):
            full[self._dof_index[n]] = rad
        self.robot.set_joint_positions(full)
        self.robot.set_joint_velocities(np.zeros(self._n_dof, dtype=np.float32))
        self.robot.apply_action(ArticulationAction(joint_positions=full))

    def get_camera_rgb(self) -> Optional[np.ndarray]:
        """카메라 RGB (H,W,3) uint8. 워밍업 전이면 None."""
        rgba = self.camera.get_rgba()
        if rgba is None or getattr(rgba, 'size', 0) == 0:
            return None
        return np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)

    def close(self) -> None:
        try:
            self.world.stop()
        except Exception:
            pass


# ──────────────────────────────────────────────
# SceneManager — Isaac Sim USD 스테이지 직접 조작 ([D2])
# ──────────────────────────────────────────────

class SceneManager:
    """
    매 에피소드 multi-target 객체 배치를 Isaac Sim USD API 로 직접 수행한다 ([D2]).

    배치 — `multi_target_scenes.yaml` 의 `placement` 규칙:
      - spawn_region(테이블 사각형) ∩ reach(가동범위 환형섹터, 선택적) 안에서
        (x,y) rejection sampling — reach 미정의면 사각형만
      - 빈(bin) footprint·기배치 객체와 겹치면 재샘플 (min_separation)
      - z 는 world bounding box 로 객체 바닥을 table_top_z 에 맞춤
      - yaw 무작위 (DR 다양성)
    객체는 `/World/Objects` 아래에 래퍼 USD 를 reference 하여 생성하고,
    reset() 으로 다음 에피소드 전에 제거한다.

    DR(apply_domain_randomization)은 dr_config.yaml 기준으로 조명·카메라·질량을
    무작위화한다. texture/friction/pd_gain/노이즈는 미구현 (dr_config.yaml 주석).
    """

    OBJECTS_ROOT = '/World/Objects'

    def __init__(self, scene_cfg: SceneConfig, dr_cfg: DRConfig,
                 sim: SimInterface, rng: np.random.Generator):
        self.scene_cfg = scene_cfg
        self.dr_cfg = dr_cfg
        self.sim = sim
        self.rng = rng
        self._placed: List[str] = []

        p = scene_cfg.placement
        self._spawn           = p['spawn_region']
        self._table_top_z     = float(p.get('table_top_z', 0.0))
        self._min_sep         = float(p.get('min_separation', 0.08))
        self._exclusion_zones = list(p.get('exclusion_zones') or [])
        self._max_attempts    = int(p.get('max_sample_attempts', 100))
        self._lib = Path(scene_cfg.object_library)

        # reach (annular sector, 선택적) — 사각형 spawn_region ∩ 환형 가동범위.
        # 없으면 None → _in_reach 가 항상 True (이전 사각형-only 동작).
        reach_cfg = p.get('reach')
        if isinstance(reach_cfg, dict):
            cx, cy   = reach_cfg['center']
            r_lo, r_hi   = reach_cfg['r_range']
            th_lo, th_hi = reach_cfg['theta_range_deg']
            self._reach = {
                'cx': float(cx), 'cy': float(cy),
                'r_min': float(r_lo), 'r_max': float(r_hi),
                'th_min': math.radians(float(th_lo)),
                'th_max': math.radians(float(th_hi)),
            }
        else:
            self._reach = None

        # 배치 객체 컨테이너 Xform
        UsdGeom.Xform.Define(self.sim.stage, self.OBJECTS_ROOT)

        # DR — 기동 시 베이스 값 1회 캡처 (에피소드 누적 drift 방지)
        self._dr = dr_cfg.raw or {}
        self._light_bases = self._capture_light_bases()
        self._camera_focal_base = self._capture_camera_focal()
        if (self._dr.get('lighting') or {}).get('enabled') and not self._light_bases:
            print('[record_sim_teleop] 경고 — 씬에서 UsdLux 광원을 찾지 못함, '
                  '조명 DR 무효.')
        if (self._dr.get('camera') or {}).get('enabled') \
           and self._camera_focal_base is None:
            print(f'[record_sim_teleop] 경고 — {self.sim.args.camera_prim} 의 '
                  'focalLength 를 읽지 못함, 카메라 DR 무효.')

    # ── 공개 API ─────────────────────────────────────

    def setup_episode_scene(self, episode_idx: int,
                            target: Dict) -> List[Dict]:
        """
        target['objects'](배치할 래퍼 USD 파일명 리스트)를 spawn_region 안에
        배치한다. target 계약은 TargetSelector.select 의 docstring 참조.

        Returns: 슬롯 순서의 배치 기록 [{'slot','wrapper','prim_path','x','y'}, ...]
                 — TargetSelector.finalize 가 spatial 도출에 사용한다.

        NOTE: 동적으로 추가한 rigid body 는 이후 world.step() 에서 PhysX 에
              등록된다 — record_episode 의 초기 안정화 스텝이 처리한다.
        """
        wrappers = target.get('objects')
        if not wrappers:
            raise ValueError("target['objects'] (배치할 래퍼 목록)이 비어 있음.")
        placed_xy: List[Tuple[float, float]] = []
        placements: List[Dict] = []
        for slot, wrapper_name in enumerate(wrappers):
            xy = self._sample_xy(placed_xy)
            placed_xy.append(xy)
            prim_path = self._place_object(wrapper_name, xy, episode_idx, slot)
            self._placed.append(prim_path)
            placements.append({
                'slot': slot, 'wrapper': wrapper_name, 'prim_path': prim_path,
                'x': xy[0], 'y': xy[1],
            })
        return placements

    def apply_domain_randomization(self, episode_idx: int) -> None:
        """
        dr_config.yaml 기준 도메인 무작위화 — setup_episode_scene 직후 호출.

        각 섹션(lighting/camera/mass)은 선택적이며 키 부재 또는 enabled:false 면
        건너뛴다. jitter 는 기동 시 캡처한 베이스 값(조명·카메라) 또는 래퍼 USD
        의 baked 값(질량)에 곱해지므로 에피소드가 쌓여도 분포가 떠내려가지 않는다.
        texture/friction/pd_gain/observation_noise 는 미구현 — dr_config.yaml 주석.
        """
        light_cfg = self._dr.get('lighting')
        if isinstance(light_cfg, dict) and light_cfg.get('enabled', True):
            self._randomize_lighting(light_cfg)
        cam_cfg = self._dr.get('camera')
        if isinstance(cam_cfg, dict) and cam_cfg.get('enabled', True):
            self._randomize_camera(cam_cfg)
        mass_cfg = self._dr.get('mass')
        if isinstance(mass_cfg, dict) and mass_cfg.get('enabled', True):
            self._randomize_mass(mass_cfg)

    def reset(self) -> None:
        """배치한 객체 prim 을 모두 제거 (다음 에피소드 준비)."""
        for prim_path in self._placed:
            if self.sim.stage.GetPrimAtPath(prim_path).IsValid():
                self.sim.stage.RemovePrim(Sdf.Path(prim_path))
        self._placed = []

    # ── 내부 ─────────────────────────────────────────

    def _sample_xy(self, existing: List[Tuple[float, float]]) -> Tuple[float, float]:
        """spawn_region(사각형) ∩ reach(환형섹터, 선택적) 안 rejection sampling."""
        xr = self._spawn['x_range']
        yr = self._spawn['y_range']
        for _ in range(self._max_attempts):
            x = float(self.rng.uniform(xr[0], xr[1]))
            y = float(self.rng.uniform(yr[0], yr[1]))
            if any((x - ex) ** 2 + (y - ey) ** 2 < self._min_sep ** 2
                   for ex, ey in existing):
                continue
            if self._in_excluded(x, y):
                continue
            if not self._in_reach(x, y):
                continue
            return (x, y)
        raise RuntimeError(
            f'spawn 위치 샘플 실패 ({self._max_attempts}회) — spawn_region·reach '
            '가 너무 좁거나 객체 수 대비 min_separation 이 큼.'
        )

    def _in_excluded(self, x: float, y: float) -> bool:
        """초기 배치 금지 영역(bin/target_area 등) 안인지 검사."""
        for b in self._exclusion_zones:
            bx, by = b['x_range'], b['y_range']
            if bx[0] <= x <= bx[1] and by[0] <= y <= by[1]:
                return True
        return False

    def _in_reach(self, x: float, y: float) -> bool:
        """reach 환형섹터 안인지 — reach 미정의면 항상 True (제약 없음).

        축 규약: x+ = 로봇 정면(forward), y+ = 로봇 기준 왼쪽.
        theta = atan2(dy, dx) — 정면(+x)이 0, +90°가 왼쪽(+y).
        ⚠️ theta_range 가 ±π 를 가로지르는 wrap-around 는 미지원.
        """
        if self._reach is None:
            return True
        r = self._reach
        dx, dy = x - r['cx'], y - r['cy']
        rad = math.hypot(dx, dy)
        if rad < r['r_min'] or rad > r['r_max']:
            return False
        theta = math.atan2(dy, dx)
        return r['th_min'] <= theta <= r['th_max']

    def _place_object(self, wrapper_name: str, xy: Tuple[float, float],
                      episode_idx: int, slot: int) -> str:
        """래퍼 USD 를 reference 한 prim 을 생성·배치하고 prim 경로를 반환."""
        wrapper = self._lib / wrapper_name
        if not wrapper.exists():
            raise FileNotFoundError(
                f'객체 래퍼 없음: {wrapper} — '
                'sim/scene_setup/setup_objects.py 를 먼저 실행할 것.'
            )
        prim_path = (f'{self.OBJECTS_ROOT}/ep{episode_idx}'
                     f'_{slot}_{Path(wrapper_name).stem}')
        prim = self.sim.stage.DefinePrim(prim_path, 'Xform')
        prim.GetReferences().AddReference(str(wrapper.resolve()))

        x, y = xy
        rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0),
                          float(self.rng.uniform(0.0, 360.0)))   # yaw 무작위
        xform = UsdGeom.Xformable(prim)
        # ClearXformOpOrder 호출 안 함 — 래퍼 USD 가 baked 한 ScaleOp
        # (setup_objects.py 의 DOMAINS[domain]['scale']) 이 보존돼야 한다.
        # AddTransformOp 은 기존 xformOpOrder 에 append 되어 [..., transform]
        # 순으로 적용된다.
        #
        # SimReady 래퍼는 referenced asset 에서 [translate, rotateXYZ, scale]
        # 을 inherit 받는다. inherit 된 translate/rotateXYZ 가 non-zero 이면
        # 새 transform 과 합성되어 placement 가 의도와 어긋남. defensive 하게
        # identity 로 reset (ScaleOp 는 보존 — 래퍼의 baked domain scale 유지).
        # YCB 래퍼는 translate/rotateXYZ 가 없어 본 분기 no-op.
        existing_ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
        for op_type in (UsdGeom.XformOp.TypeTranslate,
                        UsdGeom.XformOp.TypeRotateXYZ):
            op = existing_ops.get(op_type)
            if op is None:
                continue
            curr = op.Get()
            if curr is not None:
                op.Set(type(curr)(0, 0, 0))

        xop = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

        # 임시로 table_top_z 에 두고 → world bbox 로 객체 바닥을 테이블에 정렬
        xop.Set(Gf.Matrix4d().SetTransform(rot, Gf.Vec3d(x, y, self._table_top_z)))
        rng_box = UsdGeom.Imageable(prim).ComputeWorldBound(
            Usd.TimeCode.Default(),
            UsdGeom.Tokens.default_, UsdGeom.Tokens.render).ComputeAlignedRange()
        if rng_box.IsEmpty():
            final_z = self._table_top_z + 0.10   # bbox 실패 → 띄우고 물리 settle
        else:
            final_z = 2.0 * self._table_top_z - rng_box.GetMin()[2]
        xop.Set(Gf.Matrix4d().SetTransform(rot, Gf.Vec3d(x, y, final_z)))
        return prim_path

    # ── DR 베이스 캡처 (기동 시 1회) ─────────────────

    def _capture_light_bases(
            self) -> Dict[str, Tuple[float, Tuple[float, float, float]]]:
        """씬의 모든 UsdLux 광원 베이스 intensity/color 를 캡처."""
        bases: Dict[str, Tuple[float, Tuple[float, float, float]]] = {}
        for prim in self.sim.stage.Traverse():
            if not prim.HasAPI(UsdLux.LightAPI):
                continue
            light = UsdLux.LightAPI(prim)
            intensity = light.GetIntensityAttr().Get()
            color = light.GetColorAttr().Get()
            bases[str(prim.GetPath())] = (
                float(intensity) if intensity is not None else 1.0,
                tuple(color) if color is not None else (1.0, 1.0, 1.0),
            )
        return bases

    def _capture_camera_focal(self) -> Optional[float]:
        """camera_prim 의 베이스 focalLength 를 캡처 (카메라/속성 없으면 None)."""
        prim = self.sim.stage.GetPrimAtPath(self.sim.args.camera_prim)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            return None
        fl = UsdGeom.Camera(prim).GetFocalLengthAttr().Get()
        return float(fl) if fl is not None else None

    # ── DR 적용 (에피소드별) ─────────────────────────

    def _randomize_lighting(self, cfg: dict) -> None:
        """모든 광원의 intensity/color 를 베이스 값 * uniform 배율로 설정."""
        i_lo, i_hi = cfg.get('intensity_mult', [1.0, 1.0])
        c_lo, c_hi = cfg.get('color_jitter', [1.0, 1.0])
        for path, (base_i, base_c) in self._light_bases.items():
            prim = self.sim.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            light = UsdLux.LightAPI(prim)
            light.GetIntensityAttr().Set(
                base_i * float(self.rng.uniform(i_lo, i_hi)))
            light.GetColorAttr().Set(Gf.Vec3f(
                base_c[0] * float(self.rng.uniform(c_lo, c_hi)),
                base_c[1] * float(self.rng.uniform(c_lo, c_hi)),
                base_c[2] * float(self.rng.uniform(c_lo, c_hi)),
            ))

    def _randomize_camera(self, cfg: dict) -> None:
        """camera_prim 의 focalLength 를 베이스 값 * uniform 배율로 설정."""
        if self._camera_focal_base is None:
            return
        lo, hi = cfg.get('focal_length_mult', [1.0, 1.0])
        prim = self.sim.stage.GetPrimAtPath(self.sim.args.camera_prim)
        if not prim.IsValid():
            return
        UsdGeom.Camera(prim).GetFocalLengthAttr().Set(
            self._camera_focal_base * float(self.rng.uniform(lo, hi)))

    def _randomize_mass(self, cfg: dict) -> None:
        """이번 에피소드 배치 객체의 MassAPI mass 를 baked 값 * uniform 배율로 설정.

        객체는 매 에피소드 래퍼 USD 를 새로 reference 하므로 mass 어트리뷰트는
        항상 baked 값에서 시작한다 — 별도 베이스 캡처가 필요 없다.
        """
        lo, hi = cfg.get('mult', [1.0, 1.0])
        for prim_path in self._placed:
            prim = self.sim.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
            base = mass_attr.Get() if mass_attr else None
            if base is None or base <= 0.0:
                continue
            mass_attr.Set(float(base) * float(self.rng.uniform(lo, hi)))


# ──────────────────────────────────────────────
# 데이터셋 / Leader 연결
# ──────────────────────────────────────────────

def open_dataset(args: SimRecordArgs) -> LeRobotDataset:
    """
    LeRobotDataset 0.4.4 생성 또는 resume. features 스펙은 Real
    record_real_teleop.py 와 반드시 일치해야 Co-training 이 가능하다.
    """
    features = {
        'observation.state': {
            'dtype': 'float32', 'shape': (6,), 'names': list(SO101_JOINT_NAMES),
        },
        'observation.language_embedding': {
            'dtype': 'float32', 'shape': (args.embedding_dim,), 'names': None,
        },
        CAMERA_KEY: {
            'dtype': 'video', 'shape': (args.img_h, args.img_w, 3),
            'names': ['height', 'width', 'channels'],
        },
        'action': {
            'dtype': 'float32', 'shape': (6,), 'names': list(SO101_JOINT_NAMES),
        },
    }
    default_root = (Path.home() / '.cache' / 'huggingface' / 'lerobot'
                    / args.dataset_repo_id)
    if default_root.exists():
        print(f'[record_sim_teleop] 기존 데이터셋 resume: {default_root}')
        dataset = LeRobotDataset(repo_id=args.dataset_repo_id)
        existing = dataset.features.get('observation.language_embedding')
        if existing is None or tuple(existing['shape']) != (args.embedding_dim,):
            raise RuntimeError(
                f'resume 데이터셋 language_embedding shape 불일치 — '
                f'({args.embedding_dim},) 필요. 다른 repo_id 또는 삭제.'
            )
        return dataset
    return LeRobotDataset.create(
        repo_id=args.dataset_repo_id, fps=args.fps, features=features,
        robot_type='so101_follower', use_videos=True,
    )


def connect_leader(args: SimRecordArgs) -> SO101Leader:
    """실물 SO-101 Leader 연결 (LeRobot SDK, USB)."""
    leader = SO101Leader(SO101LeaderConfig(
        port=args.teleop_port, id=args.teleop_id, use_degrees=True))
    leader.connect()
    return leader


# ──────────────────────────────────────────────
# 헬퍼 — 종료 신호 / sim step 하며 대기
# ──────────────────────────────────────────────

def _user_wants_quit() -> bool:
    """stdin 'q'+Enter 비차단 검사 (Linux/macOS). Windows 에서는 항상 False."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except (ValueError, OSError):
        return False
    if not ready:
        return False
    line = sys.stdin.readline()
    return bool(line) and line.strip().lower() == 'q'


def _stdin_line_ready() -> bool:
    """stdin 에 한 줄이 들어왔는지 비차단 검사 (Enter 대기용)."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except (ValueError, OSError):
        return False
    if not ready:
        return False
    sys.stdin.readline()
    return True


def _wait_for_enter(sim: SimInterface) -> None:
    """작업자 Enter 대기 — 대기 동안에도 world.step() 으로 뷰포트를 갱신한다."""
    while True:
        sim.step()
        if _stdin_line_ready():
            return


# ──────────────────────────────────────────────
# 메인 — 에피소드 녹화 루프
# ──────────────────────────────────────────────

def record_episode(args:           SimRecordArgs,
                   sim:            SimInterface,
                   leader:         SO101Leader,
                   dataset:        LeRobotDataset,
                   episode_idx:    int,
                   target:         Dict,
                   initial_pose:   List[float]) -> bool:
    """
    1개 에피소드 녹화 → 저장.

    Returns: True  = save_episode() 완료 / False = 버퍼 폐기.

    시작: follower 를 home_pose 자세로 강제 ([D7]) → 안정화 → 작업자 Enter.
    각 틱: Leader 읽기 → follower 구동 → world.step() → 관측 수집 → add_frame.
    종료: 'q'+Enter ([D8]) → save 여부 확인.
    """
    embed_vec = np.zeros(args.embedding_dim, dtype=np.float32)   # [D9] placeholder
    task_str = target['canonical']

    # 초기 자세 강제 + 물리 안정화 ([D7])
    print(f'[ep {episode_idx}] follower 초기 자세 강제...')
    sim.force_initial_pose(initial_pose)
    for _ in range(RESET_SETTLE_STEPS):
        sim.step()

    # 카메라 워밍업 — 첫 RGB 가 나올 때까지 step
    deadline = time.perf_counter() + 5.0
    while sim.get_camera_rgb() is None:
        sim.step()
        if time.perf_counter() > deadline:
            raise RuntimeError(f'카메라 RGB 미수신 — {args.camera_prim} 확인.')

    print(f'[ep {episode_idx}] task = {task_str!r}')
    print(f'[ep {episode_idx}] Leader 를 초기 자세 근처로 옮긴 뒤 Enter 로 시작.')
    _wait_for_enter(sim)

    print(f'[ep {episode_idx}] 녹화 중 — "q"+Enter 로 종료')
    n_frames = 0
    t_start = time.perf_counter()
    try:
        while True:
            if _user_wants_quit():
                break
            # Leader → follower 미러링
            leader_action = leader.get_action()
            sim.drive_follower(_leader_to_sim_rad(leader_action))
            sim.step()
            # 관측 수집
            rgb = sim.get_camera_rgb()
            if rgb is None:
                continue
            frame = {
                'observation.state':              _sim_jointpos_to_sdk(
                                                      sim.get_follower_joint_positions()),
                'observation.language_embedding': embed_vec,
                CAMERA_KEY:                       rgb,
                'action':                         _leader_action_to_vec(leader_action),
                'task':                           task_str,
            }
            dataset.add_frame(frame)
            n_frames += 1
    except KeyboardInterrupt:
        print(f'\n[ep {episode_idx}] KeyboardInterrupt — 종료')

    elapsed = time.perf_counter() - t_start
    fps_real = n_frames / elapsed if elapsed > 0 else 0.0
    print(f'[ep {episode_idx}] 녹화 종료 — frames={n_frames} '
          f'elapsed={elapsed:.1f}s realized_fps={fps_real:.1f}')

    if input(f'[ep {episode_idx}] 저장할까요? (y/n): ').strip().lower() != 'y':
        dataset.clear_episode_buffer()
        print(f'[ep {episode_idx}] 버퍼 폐기.')
        return False
    dataset.save_episode()
    print(f'[ep {episode_idx}] save_episode 완료.')
    return True


def main() -> None:
    args = ARGS

    # 1) Config
    vocab     = load_vocab(args.label_vocab, args.domain)
    scene_cfg = load_scenes(args.scenes, args.domain)
    dr_cfg    = load_dr_config(args.dr_config)
    initial_pose = load_initial_pose(args.home_pose_file)
    rng = np.random.default_rng(args.seed)
    print(f'[record_sim_teleop] config 로드 완료 (domain={args.domain})')

    # 2) Isaac Sim 인터페이스 + 씬/DR 매니저 + 타겟 셀렉터
    sim = SimInterface(args)
    scene_manager = SceneManager(scene_cfg, dr_cfg, sim, rng)
    selector = TargetSelector(scene_cfg, vocab)                  # [D5]

    leader: Optional[SO101Leader] = None
    try:
        # 3) Leader + 데이터셋
        leader = connect_leader(args)
        dataset = open_dataset(args)
        labels_jsonl = _resolve_labels_jsonl(args, dataset)
        print(f'[record_sim_teleop] dataset.root = {dataset.root}')

        # 4) 에피소드 루프
        start_ep = dataset.num_episodes
        for ep in range(start_ep, start_ep + args.num_episodes):
            while True:
                # 선택([D5]) → 배치([D2]) → 배치 기반 라벨 확정 → DR([D2])
                target = selector.select(ep, rng)
                placements = scene_manager.setup_episode_scene(ep, target)
                selector.finalize(target, placements)
                scene_manager.apply_domain_randomization(ep)

                saved = record_episode(args, sim, leader, dataset,
                                       ep, target, initial_pose)
                retry = input(f'[ep {ep}] 다시 촬영할까요? (y/n): ').strip().lower()
                scene_manager.reset()                            # 배치 객체 정리
                if retry == 'y':
                    if saved:
                        dataset.delete_episode()
                    continue                                     # 같은 ep 재시도
                if saved:
                    selector.commit(target)                      # 누적 분포 반영
                    append_label_record(labels_jsonl, target)
                break
    finally:
        if leader is not None:
            try:
                leader.disconnect()
            except Exception as e:
                print(f'[record_sim_teleop] leader disconnect 오류: {e}')
        sim.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        simulation_app.close()
