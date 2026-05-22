"""
record_sim_teleop.py

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
  ~/isaacsim/python.sh sim/teleop/record_sim_teleop.py \\
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

설계 결정 ([D1]~[D9] 모두 확정 — README "Sim teleop 수집" 표 참조):
  D1 로컬 통합·직접 API    D2 SceneManager 가 USD 스테이지 직접 조작
  D3 plain deg↔rad         D4 ROS2 Action Graph 불필요
  D5 자동 분포 타겟 선택    D6 gripper 0~100% ↔ -10°~100°
  D7 home_pose.yaml 초기자세  D8 fps=30  D9 embedding placeholder

상태 — 스켈레톤. 아래는 핵심 로직이 미구현(NotImplementedError):
  - SceneManager : multi-target 씬 구성 + DR (multi_target_scenes.yaml /
                   dr_config.yaml 스키마 확정 필요)
  - select_episode_target : 자동 분포 보장 로직 (D5)
  SimInterface 의 Isaac Sim API 호출은 5.1 기준 best-effort — 실기 검증 필요.
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
from isaacsim.core.utils.stage import open_stage              # noqa: E402
from isaacsim.core.utils.types import ArticulationAction      # noqa: E402
from isaacsim.sensors.camera import Camera                    # noqa: E402

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
    """label_vocab.yaml 로드 + 도메인 추출 (record_real_teleop.py 와 동일 규약)."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    template = raw.get('template')
    spatials = raw.get('spatial_qualifiers') or []
    domains  = raw.get('domains') or {}
    if not template or '{attribute}' not in template \
       or '{object}' not in template or '{spatial}' not in template:
        raise ValueError(f'label_vocab template 슬롯 누락 ({path}).')
    if not spatials:
        raise ValueError(f'spatial_qualifiers 가 비어 있음 ({path}).')
    if domain not in domains:
        raise ValueError(f"도메인 '{domain}' 이 label_vocab 에 없음. 가능: {list(domains)}")
    objects: Dict[str, List[str]] = {}
    for obj_name, obj_def in (domains[domain].get('objects') or {}).items():
        attrs = (obj_def or {}).get('attributes') or []
        if not attrs:
            raise ValueError(f"객체 '{obj_name}' attributes 비어 있음.")
        objects[obj_name] = list(attrs)
    if not objects:
        raise ValueError(f"도메인 '{domain}' 에 등록된 객체 없음.")
    return LabelVocab(template, list(spatials), objects, domain, raw)


@dataclass
class SceneConfig:
    scenes:             List[dict]
    multi_target_ratio: float
    raw:                dict = field(default_factory=dict)


def load_scenes(path: str) -> SceneConfig:
    """multi_target_scenes.yaml 로드 (스키마 미확정 — 얕은 파싱)."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    scenes = raw.get('scenes') or []
    if not scenes:
        raise ValueError(f'multi_target_scenes.yaml 에 scenes 정의 없음 ({path}).')
    return SceneConfig(list(scenes), float(raw.get('multi_target_ratio', 0.0)), raw)


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

def select_episode_target(scene_cfg: SceneConfig, vocab: LabelVocab,
                          episode_idx: int, rng: np.random.Generator) -> Dict[str, str]:
    """
    이번 에피소드의 장면 + 타겟 객체를 선택하고 canonical 라벨을 만든다.

    Returns: {scene, object, attribute, spatial, canonical, domain, episode}

    [D5] 자동 분포 보장 — multi-target shortcut 방지 가이드라인(같은 객체 다른
    spatial ≥30%, 한 장면 후보 2+ 다른 라벨 ≥40% 등)을 만족하도록 누적 분포를
    추적하며 타겟을 배분한다. 구현은 multi_target_scenes.yaml 스키마 확정 필요.
    """
    raise NotImplementedError(
        '[D5] 자동 분포 보장 타겟 선택 미구현 — 누적 분포 추적 + '
        'multi_target_scenes.yaml(스키마 미확정) 기반 장면/타겟 배분 필요.'
    )


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
    매 에피소드 multi-target 씬 구성 + Domain Randomization 을 Isaac Sim API 로
    직접 수행한다 ([D2] — recorder 가 Isaac Sim 안에서 실행되므로 ROS2 서비스/
    동반 스크립트 불필요).

    미구현: 실제 객체 배치/DR 로직은 multi_target_scenes.yaml · dr_config.yaml
            스키마 확정과 sim/assets/objects/recycling/ 래퍼 준비가 선행돼야 한다.
    """

    def __init__(self, scene_cfg: SceneConfig, dr_cfg: DRConfig,
                 sim: SimInterface, rng: np.random.Generator):
        self.scene_cfg = scene_cfg
        self.dr_cfg = dr_cfg
        self.sim = sim
        self.rng = rng

    def setup_episode_scene(self, episode_idx: int, target: Dict[str, str]) -> None:
        """target 에 맞춰 multi-target 객체를 USD 스테이지에 배치 (래퍼 USD reference)."""
        raise NotImplementedError(
            '[D2] 씬 구성 미구현 — sim/assets/objects/recycling/ 래퍼를 스테이지에 '
            'reference + 배치. multi_target_scenes.yaml 스키마 확정 필요.'
        )

    def apply_domain_randomization(self, episode_idx: int) -> None:
        """dr_config.yaml 기준 조명·텍스처·FOV·질량·마찰·PD 게인·노이즈 무작위화."""
        raise NotImplementedError(
            '[D2] DR 적용 미구현 — dr_config.yaml 스키마 확정 후 Isaac Sim API/'
            'Replicator 로 구현.'
        )

    def reset(self) -> None:
        """다음 에피소드를 위한 씬 정리 (배치 객체 제거 등)."""
        raise NotImplementedError('[D2] 씬 reset 미구현.')


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
                   target:         Dict[str, str],
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
    scene_cfg = load_scenes(args.scenes)
    dr_cfg    = load_dr_config(args.dr_config)
    initial_pose = load_initial_pose(args.home_pose_file)
    rng = np.random.default_rng(args.seed)
    print(f'[record_sim_teleop] config 로드 완료 (domain={args.domain})')

    # 2) Isaac Sim 인터페이스 + 씬/DR 매니저
    sim = SimInterface(args)
    scene_manager = SceneManager(scene_cfg, dr_cfg, sim, rng)

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
            target = select_episode_target(scene_cfg, vocab, ep, rng)   # [D5]
            scene_manager.setup_episode_scene(ep, target)               # [D2]
            scene_manager.apply_domain_randomization(ep)                # [D2]

            while True:
                append_label_record(labels_jsonl, target)
                saved = record_episode(args, sim, leader, dataset,
                                       ep, target, initial_pose)
                retry = input(f'[ep {ep}] 다시 촬영할까요? (y/n): ').strip().lower()
                if retry == 'y':
                    if saved:
                        dataset.delete_episode()
                    target = select_episode_target(scene_cfg, vocab, ep, rng)
                    scene_manager.setup_episode_scene(ep, target)
                    scene_manager.apply_domain_randomization(ep)
                else:
                    break
            scene_manager.reset()
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
