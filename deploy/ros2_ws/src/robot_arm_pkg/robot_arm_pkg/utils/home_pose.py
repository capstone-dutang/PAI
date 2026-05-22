"""
robot_arm_pkg.utils.home_pose — SO-101 follower home pose 파일 입출력 유틸.

데이터 수집과 추론/배포 모두 같은 home pose 로 task 종료 후
복귀하기 위해 YAML 파일 한 개를 공유한다.

표준 경로 :  ~/calibration/home_pose.yaml
표준 스키마:
    positions:
      shoulder_pan:   <float>
      shoulder_lift:  <float>
      elbow_flex:     <float>
      wrist_flex:     <float>
      wrist_roll:     <float>
      gripper:        <float>

사용처:
  - record_real_teleop.py            : 첫 실행 시 follower 의 현재 자세를 캡처해 저장.
                                  이후 실행은 같은 파일을 읽어 에피소드 간 자동 복귀.
  - robot_controller_node.py    : 시작 시 파일을 읽어 /go_home Trigger 서비스의
                                  목표값으로 사용.
  - sim 추론 (검증 단계, 추후)  : 동일 스키마를 그대로 읽어 sim rollout 종료 시
                                  복귀 지점으로 사용.

home pose 변경 절차:
  1) follower 를 원하는 새 rest 자세에 둔다.
  2) ~/calibration/home_pose.yaml 을 삭제한다.
  3) record_real_teleop 을 한 번 실행 — 새 자세가 자동 캡처되어 저장된다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import yaml


# SO-101 관절 순서 — robot_controller_node.py / record_real_teleop.py 의 정의와 동일해야 한다.
# 본 모듈을 단일 소스로 통일하는 리팩터는 추후 작업.
SO101_JOINT_NAMES: List[str] = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]

DEFAULT_HOME_POSE_PATH = '~/calibration/home_pose.yaml'


def load_home_pose(path: str) -> np.ndarray:
    """
    home_pose.yaml → SO101_JOINT_NAMES 순서로 정렬된 (6,) float32 배열.

    Raises:
        FileNotFoundError: 파일 없음.
        KeyError:          positions 딕셔너리에 필수 관절 키 누락.
        ValueError:        YAML 구조가 예상과 다름.
    """
    abs_path = os.path.expanduser(path)
    with open(abs_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f'home_pose.yaml 루트가 dict 가 아님 ({abs_path})')

    positions = raw.get('positions')
    if not isinstance(positions, dict):
        raise ValueError(
            f'home_pose.yaml 에 positions: 딕셔너리가 없음 ({abs_path})'
        )

    missing = [n for n in SO101_JOINT_NAMES if n not in positions]
    if missing:
        raise KeyError(
            f'home_pose.yaml positions 누락 관절: {missing} ({abs_path})'
        )

    return np.array(
        [float(positions[n]) for n in SO101_JOINT_NAMES],
        dtype=np.float32,
    )


def save_home_pose(path: str, pose: np.ndarray) -> None:
    """
    (6,) float32 → home_pose.yaml. pose 는 SO101_JOINT_NAMES 순서.
    부모 디렉터리는 자동 생성한다.
    """
    if pose.shape != (len(SO101_JOINT_NAMES),):
        raise ValueError(
            f'pose shape {pose.shape} != ({len(SO101_JOINT_NAMES)},)'
        )

    abs_path = os.path.expanduser(path)
    Path(abs_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'positions': {
            name: float(pose[i])
            for i, name in enumerate(SO101_JOINT_NAMES)
        },
    }
    with open(abs_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def home_pose_exists(path: str) -> bool:
    """파일 존재 여부 — record_real_teleop 의 '있으면 load, 없으면 capture+save' 분기에 사용."""
    return os.path.exists(os.path.expanduser(path))
