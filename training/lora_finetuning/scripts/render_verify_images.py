"""
render_verify_images.py

verify 시나리오 sidecar (`*.scenarios.jsonl`) 를 입력으로 Isaac Sim 에서
RGB PNG 를 생성한다 — generate_verify_qa.py(LLM 환경) 의 2-pass 짝.

실행 환경 (필수):
  ~/isaacsim/python.sh training/lora_finetuning/scripts/render_verify_images.py ...
  Isaac Sim standalone python — generate_verify_qa.py(openai SDK 환경) 와 다른
  환경. 본 스크립트는 OPENAI_API_KEY 등 LLM 의존성 없음.

  추가 의존성: numpy, PyYAML, Pillow (PNG 저장).
    ~/isaacsim/python.sh -m pip install pillow

freeze 결정 (2026-05-25, C2 → Sim 렌더 분리 후 본 파일에 박힘):
  값은 모듈 상단 상수가 캐노니컬 위치. 다른 곳(README/memory)에서 재기재 금지.

  · 팔 자세        : home_pose 단일 (deploy/calibration/home_pose.yaml).
                     agent_node 의 verify 직전 /go_home freeze 와 정합.
  · gripper        : home_pose 그대로 (운영 시 verify 시점에 빈 gripper 라는
                     전제) — verify 렌더도 동일.
  · distractor 개수 : 시나리오 당 1개, target 과 다른 카테고리 (multi-target
                     shortcut 방지). DISTRACTOR_COUNT 상수.
  · target 동역학   : kinematic — 정확한 위치/자세 유지 (status 별 의미 보존).
                     KINEMATIC_TARGET / TARGET_Z_OFFSET.
  · distractor 동역학: dynamic — 자연스러운 안착 (테이블 위 random).
  · 위치 — SUCCESS  : destination 중심 ± SUCCESS_JITTER_M.
  · 위치 — PARTIAL : destination 한 변(4 중 랜덤) 위 50% 안쪽 지점 + 30° tilt.
                     PARTIAL_EDGE_OFFSET_RATIO / PARTIAL_TILT_DEG.
  · 위치 — FAIL    : spawn_region 내 exclusion_zones 회피 임의 (x, y).
  · 위치 — UNCLEAR : target 은 destination 안 (SUCCESS-비슷), distractor 가
                     camera-target 직선상으로 옮겨 가림 (xy plane,
                     UNCLEAR_OCCLUDER_OFFSET_M).
  · 물리 settle    : SETTLE_STEPS (kinematic target 은 영향 받지 않음).
  · 렌더 step      : RENDER_STEPS (캡처 직전).
  · PNG 출력       : RGB 채널만, 기본 640×480 (RealSense 일치).
  · 시나리오 root   : /World/VerifyScenario — 시나리오 사이 하위 prim 전체 제거.

CLI:
  ~/isaacsim/python.sh training/lora_finetuning/scripts/render_verify_images.py \\
    --scenarios       training/lora_finetuning/data/recycling/recycling_verify_qa.jsonl.scenarios.jsonl \\
    --scene-usd       sim/scenes/base_scene.usd \\
    --domain          recycling \\
    --image-dir       training/lora_finetuning/data/recycling/verify_images/

옵션:
  --multi-target   sim/config/multi_target_scenes.yaml  (placement/exclusion 좌표 출처)
  --home-pose      deploy/calibration/home_pose.yaml
  --img-w / --img-h  RealSense 기본 640×480
  --robot-prim / --camera-prim  base_scene 의 prim 경로
  --headless       대량 batch 권장 (viewport off)
  --seed           distractor / FAIL 위치 샘플 시드

설계 노트 (Sim 환경에서 첫 실행 시 디버깅 필요할 수 있는 부분):
  - SimulationApp 은 import 순서가 까다로움 — 본 파일은 SimulationApp 인스턴스
    생성 후에 isaacsim.* / pxr 을 import 한다.
  - 래퍼 USD 의 ScaleOp 보존 패턴은 record_sim_teleop.py 의 _place_object 와
    동일 — translate/rotate 만 추가하고 ClearXformOpOrder 호출 안 함.
  - 카메라의 world 위치는 stage 에서 직접 조회 (UNCLEAR occluder 방향용).
  - 객체 pivot 위치는 자산마다 달라 z_offset 보정이 필요할 수 있음
    (TARGET_Z_OFFSET 으로 일괄 보정 — 본 값은 첫 실행 후 조정).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# freeze 결정값 (캐노니컬 위치 — 본 파일)
# ──────────────────────────────────────────────

DISTRACTOR_COUNT = 1                     # 시나리오 당 distractor 개수

KINEMATIC_TARGET = True                  # target 객체 위치/자세 고정
TARGET_Z_OFFSET = 0.01                   # m — 테이블 표면 위 1cm (pivot 클리핑 방지)
DISTRACTOR_DROP_Z = 0.10                 # m — 테이블 위 10cm 에서 떨어뜨림 (settle)

SUCCESS_JITTER_M = 0.01                  # m — destination 중심 ± jitter

PARTIAL_EDGE_OFFSET_RATIO = 0.45         # destination 한 변에서 안쪽 절반 지점
PARTIAL_TILT_DEG = 30.0                  # z축 회전

FAIL_MAX_SAMPLE_TRIES = 50

UNCLEAR_OCCLUDER_OFFSET_M = 0.10         # camera 쪽 10cm 로 distractor 이동

SETTLE_STEPS = 60
RENDER_STEPS = 2

SCENARIO_ROOT = '/World/VerifyScenario'
CAMERA_PRIM_DEFAULT = '/World/Camera_RGB'
ROBOT_PRIM_DEFAULT  = '/World/so101_new_calib'

# SO-101 관절 — record_sim_teleop 의 SO101_JOINT_NAMES 와 같은 순서.
SO101_JOINT_NAMES = [
    'shoulder_pan', 'shoulder_lift', 'elbow_flex',
    'wrist_flex',   'wrist_roll',    'gripper',
]
GRIPPER_MIN_DEG = 0.0
GRIPPER_MAX_DEG = 100.0


# ──────────────────────────────────────────────
# 헬퍼 (Sim 미의존)
# ──────────────────────────────────────────────

def _wrapper_category(filename: str) -> str:
    """래퍼 USD 파일명 prefix → 카테고리 (can_master_chef.usd → 'can').

    record_sim_teleop._category_of 와 동일 규약.
    """
    return Path(filename).stem.split('_')[0]


def _gripper_pct_to_rad(pct: float) -> float:
    """home_pose.yaml 의 gripper 0~100 % → rad
    (record_sim_teleop._gripper_pct_to_rad 와 동일)."""
    pct = max(0.0, min(100.0, pct))
    deg = GRIPPER_MIN_DEG + (GRIPPER_MAX_DEG - GRIPPER_MIN_DEG) * (pct / 100.0)
    return math.radians(deg)


def _zone_center_size(zone: dict) -> Tuple[float, float, float, float]:
    xr = zone['x_range']
    yr = zone['y_range']
    cx = (xr[0] + xr[1]) / 2.0
    cy = (yr[0] + yr[1]) / 2.0
    sx = xr[1] - xr[0]
    sy = yr[1] - yr[0]
    return cx, cy, sx, sy


def _in_zone(x: float, y: float, zone: dict) -> bool:
    xr = zone['x_range']
    yr = zone['y_range']
    return xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]


def _in_any_zone(x: float, y: float, zones: List[dict]) -> bool:
    return any(_in_zone(x, y, z) for z in zones)


def _build_destinations(scene_cfg: dict) -> Dict[str, dict]:
    """exclusion_zones 중 'destination' 키가 있는 항목만 추출.

    반환: {'<category>' or '_shared': {cx, cy, sx, sy}}
      - recycling: 카테고리(can/plastic/paper) 별 bin
      - tableware: '_shared' (모든 카테고리 공용)
    """
    result: Dict[str, dict] = {}
    for z in scene_cfg.get('placement', {}).get('exclusion_zones') or []:
        dest = z.get('destination')
        if dest is None:
            continue
        cx, cy, sx, sy = _zone_center_size(z)
        result[dest] = {'cx': cx, 'cy': cy, 'sx': sx, 'sy': sy}
    return result


def _destination_for(destinations: Dict[str, dict], target_object: str) -> dict:
    if target_object in destinations:
        return destinations[target_object]
    if '_shared' in destinations:
        return destinations['_shared']
    raise RuntimeError(
        f"destination 없음: target={target_object}, "
        f"keys={list(destinations.keys())}")


# ──────────────────────────────────────────────
# Sim Renderer (Isaac Sim 의존)
# ──────────────────────────────────────────────

class SimVerifyRenderer:
    """Isaac Sim standalone app 으로 verify 시나리오 PNG 를 생성.

    수명 주기:
      __init__  : SimulationApp 시작, base scene 열기, robot home_pose 적용,
                   camera/articulation 초기화 (비싸므로 1회만).
      render(s) : 시나리오 1건 — 객체 배치 → 물리 settle → 캡처 → PNG → 정리.
      close()   : SimulationApp 종료.

    스레드 안전 아님. 단일 프로세스/인스턴스로만 사용.
    """

    def __init__(
        self,
        scene_usd:      str,
        scene_cfg:      dict,
        object_lib:     Path,
        home_pose_path: str,
        img_w:          int = 640,
        img_h:          int = 480,
        robot_prim:     str = ROBOT_PRIM_DEFAULT,
        camera_prim:    str = CAMERA_PRIM_DEFAULT,
        headless:       bool = True,
        rng:            Optional[random.Random] = None,
    ) -> None:
        self._scene_cfg   = scene_cfg
        # 절대경로 강제 — _place() 가 USD reference 에 박는 path 가 상대이면
        # base_scene 의 디렉터리(sim/scenes/) 기준으로 잘못 resolve 된다
        # (예: 'sim/assets/...' → 'sim/scenes/sim/assets/...' → Could not open).
        self._object_lib  = Path(object_lib).resolve()
        self._img_w       = img_w
        self._img_h       = img_h
        self._robot_prim  = robot_prim
        self._camera_prim = camera_prim
        self._rng         = rng or random.Random()

        self._destinations = _build_destinations(scene_cfg)

        p = scene_cfg.get('placement') or {}
        self._spawn       = p['spawn_region']
        self._exclusion   = list(p.get('exclusion_zones') or [])
        self._table_top_z = float(p.get('table_top_z', 0.0))

        # 도메인 카테고리 (distractor 선택용) — object_lib 의 wrapper 파일명 prefix
        self._all_categories = sorted({
            _wrapper_category(f.name) for f in object_lib.glob('*.usd')
        })

        # ── Isaac Sim 시작 ──────────────────────────────────────
        # SimulationApp 이전에 isaacsim.* 를 import 하면 빈 모듈만 나옴 — 순서 중요.
        from isaacsim import SimulationApp  # noqa: PLC0415
        self._app = SimulationApp({
            'headless': headless,
            'width':    img_w,
            'height':   img_h,
        })

        # Sim 환경 올라온 후만 import 가능
        import omni.usd                                                # noqa: PLC0415
        from pxr import Usd, UsdGeom, UsdPhysics, Gf                    # noqa: PLC0415
        from isaacsim.sensors.camera import Camera                      # noqa: PLC0415
        from isaacsim.core.api.world import World                       # noqa: PLC0415
        from isaacsim.core.prims import SingleArticulation              # noqa: PLC0415
        from isaacsim.core.utils.stage import open_stage                # noqa: PLC0415

        self._omni_usd   = omni.usd
        self._Usd        = Usd
        self._UsdGeom    = UsdGeom
        self._UsdPhysics = UsdPhysics
        self._Gf         = Gf

        # base scene
        open_stage(scene_usd)
        self._world = World(stage_units_in_meters=1.0)
        self._world.reset()
        self._stage = omni.usd.get_context().get_stage()

        # robot articulation + home_pose 강제
        self._robot = SingleArticulation(prim_path=robot_prim)
        self._robot.initialize()
        home = self._load_home_pose(home_pose_path)
        import numpy as np  # noqa: PLC0415
        self._robot.set_joint_positions(np.asarray(home, dtype=np.float32))
        self._robot.set_joint_velocities(np.zeros(len(home), dtype=np.float32))

        # camera
        self._camera = Camera(
            prim_path=camera_prim,
            resolution=(img_w, img_h),
            frequency=30,
        )
        self._camera.initialize()

        # camera world 위치 (UNCLEAR occluder 방향 계산용)
        cam_prim = self._stage.GetPrimAtPath(camera_prim)
        cam_m4 = UsdGeom.Xformable(cam_prim) \
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        self._camera_world_xy = (float(cam_m4[3][0]), float(cam_m4[3][1]))

        # 시나리오 컨테이너 root
        UsdGeom.Xform.Define(self._stage, SCENARIO_ROOT)

        # 카메라 warm-up — 첫 시나리오 호출 시 camera.get_rgba() 가 무효 프레임
        # (float32 또는 None) 을 반환하는 케이스 회피. 빈 씬 상태에서 3 step.
        for _ in range(3):
            self._world.step(render=True)

    # ── 공개 API ─────────────────────────────────────────────

    def render(self, scenario: dict) -> None:
        """1 시나리오 → PNG 저장. scenario 는 sidecar 한 줄의 dict."""
        self._clear_scenario_root()
        sid    = scenario.get('scenario_id', '?')
        status = scenario['status']

        # 1) target 위치 결정 + 배치 (kinematic)
        tx, ty, t_rot = self._target_position(scenario)
        target_wrapper = self._pick_wrapper(scenario['target_object'])
        target_path = self._place(
            name='target',
            wrapper=target_wrapper,
            x=tx, y=ty,
            rot_deg=t_rot,
            z=self._table_top_z + TARGET_Z_OFFSET,
        )
        if KINEMATIC_TARGET:
            self._set_kinematic(target_path)

        # 검증용 로그 — placement 좌표 / 자산 / status 의미
        dest = _destination_for(self._destinations, scenario['target_object'])
        print(
            f'[place] {sid} status={status} '
            f'target={scenario["target_object"]} ({target_wrapper.name}) '
            f'pos=({tx:+.3f},{ty:+.3f}) rot={t_rot:+.0f}° '
            f'dest_center=({dest["cx"]:+.3f},{dest["cy"]:+.3f})',
            file=sys.stderr,
        )

        # 2) distractor 배치 (dynamic — settle 시 자연 안착)
        for i in range(DISTRACTOR_COUNT):
            dcat = self._pick_distractor_category(scenario['target_object'])
            if dcat is None:
                break
            dwrap = self._pick_wrapper(dcat)
            if status == 'UNCLEAR':
                dx, dy = self._unclear_occluder_xy(tx, ty)
                # UNCLEAR 의 occluder 도 kinematic — 가림 위치 보장
                dpath = self._place(f'distractor_{i}', dwrap, dx, dy, 0.0,
                                    z=self._table_top_z + TARGET_Z_OFFSET)
                self._set_kinematic(dpath)
                d_kind = 'occluder(kinematic)'
            else:
                dx, dy = self._sample_free_xy()
                self._place(f'distractor_{i}', dwrap, dx, dy, 0.0,
                            z=self._table_top_z + DISTRACTOR_DROP_Z)
                d_kind = 'free(dynamic)'
            print(
                f'[place] {sid}   distractor={dcat} ({dwrap.name}) '
                f'pos=({dx:+.3f},{dy:+.3f}) kind={d_kind}',
                file=sys.stderr,
            )

        # 3) 물리 settle (target kinematic 은 영향 받지 않음)
        for _ in range(SETTLE_STEPS):
            self._world.step(render=False)

        # 4) 렌더 step + 캡처
        for _ in range(RENDER_STEPS):
            self._world.step(render=True)
        rgba = self._camera.get_rgba()
        if rgba is None:
            raise RuntimeError(
                f"camera.get_rgba() None — scenario={scenario.get('scenario_id')}. "
                "카메라 초기화 / 시뮬레이션 상태 확인.")

        # 5) PNG 저장
        out_path = scenario['image_abspath']
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self._save_png(rgba, out_path)

    def close(self) -> None:
        if getattr(self, '_app', None) is not None:
            self._app.close()
            self._app = None

    # ── 내부: status 별 target 위치 ────────────────────────

    def _target_position(self, scenario: dict) -> Tuple[float, float, float]:
        dest = _destination_for(self._destinations, scenario['target_object'])
        cx, cy = dest['cx'], dest['cy']
        sx, sy = dest['sx'], dest['sy']
        status = scenario['status']

        if status == 'SUCCESS':
            jx = self._rng.uniform(-SUCCESS_JITTER_M, SUCCESS_JITTER_M)
            jy = self._rng.uniform(-SUCCESS_JITTER_M, SUCCESS_JITTER_M)
            return cx + jx, cy + jy, 0.0

        if status == 'PARTIAL':
            side = self._rng.choice(['+x', '-x', '+y', '-y'])
            ex = sx * PARTIAL_EDGE_OFFSET_RATIO
            ey = sy * PARTIAL_EDGE_OFFSET_RATIO
            if   side == '+x': pos = (cx + ex, cy)
            elif side == '-x': pos = (cx - ex, cy)
            elif side == '+y': pos = (cx, cy + ey)
            else:              pos = (cx, cy - ey)
            return pos[0], pos[1], PARTIAL_TILT_DEG

        if status == 'FAIL':
            x, y = self._sample_free_xy()
            return x, y, 0.0

        # UNCLEAR — target 은 destination 안 (SUCCESS-like), 가림은 distractor 가
        return cx, cy, 0.0

    def _sample_free_xy(self) -> Tuple[float, float]:
        """spawn_region 안에서 exclusion_zones 회피 random (x, y)."""
        for _ in range(FAIL_MAX_SAMPLE_TRIES):
            x = self._rng.uniform(self._spawn['x_range'][0],
                                  self._spawn['x_range'][1])
            y = self._rng.uniform(self._spawn['y_range'][0],
                                  self._spawn['y_range'][1])
            if not _in_any_zone(x, y, self._exclusion):
                return x, y
        raise RuntimeError(
            f'spawn 위치 샘플 실패 ({FAIL_MAX_SAMPLE_TRIES}회). '
            'spawn_region·exclusion_zones 확인.')

    def _unclear_occluder_xy(self, tx: float, ty: float) -> Tuple[float, float]:
        """target 에서 camera 쪽으로 UNCLEAR_OCCLUDER_OFFSET_M 만큼 이동한 (x, y)."""
        cx, cy = self._camera_world_xy
        dx, dy = cx - tx, cy - ty
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return tx, ty
        nx, ny = dx / L, dy / L
        return tx + nx * UNCLEAR_OCCLUDER_OFFSET_M, ty + ny * UNCLEAR_OCCLUDER_OFFSET_M

    # ── 내부: 자산 선택 ───────────────────────────────────

    def _pick_wrapper(self, category: str) -> Path:
        matches = sorted(self._object_lib.glob(f'{category}_*.usd'))
        if not matches:
            raise RuntimeError(
                f"category '{category}' 래퍼 USD 없음 in {self._object_lib}")
        return Path(self._rng.choice(matches))

    def _pick_distractor_category(self, target_category: str) -> Optional[str]:
        others = [c for c in self._all_categories if c != target_category]
        if not others:
            return None
        return self._rng.choice(others)

    # ── 내부: 객체 배치 / 정리 ─────────────────────────────

    def _place(
        self,
        name:    str,
        wrapper: Path,
        x:       float,
        y:       float,
        rot_deg: float,
        z:       float,
    ) -> str:
        """wrapper USD 를 reference 로 추가 + translate/rotate 부여.

        Wrapper 의 ScaleOp 는 그대로 두기 위해 ClearXformOpOrder 호출 안 함.
        Wrapper 종류에 따라 xformOp 보유 상태가 다름:
          - YCB (recycling): scale 만 — translate/rotateZ 를 새로 Add
          - SimReady (tableware): [translate, rotateXYZ, scale] 가 inherit 되어
            이미 존재 — AddTranslateOp 가 'already exists' 로 실패.
        해결: 기존 op 가 있으면 Set, 없으면 Add. rotateZ 가 없고 rotateXYZ 가
        있으면 (SimReady 케이스) (0, 0, rot_deg) 로 Set.
        """
        UsdGeom = self._UsdGeom
        Gf      = self._Gf
        prim_path = f'{SCENARIO_ROOT}/{name}'

        existing = self._stage.GetPrimAtPath(prim_path)
        if existing and existing.IsValid():
            self._stage.RemovePrim(prim_path)

        xf = UsdGeom.Xform.Define(self._stage, prim_path)
        xf.GetPrim().GetReferences().AddReference(str(wrapper))

        xform = UsdGeom.Xformable(xf.GetPrim())
        existing_ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}

        # translate — 있으면 set, 없으면 add
        if UsdGeom.XformOp.TypeTranslate in existing_ops:
            existing_ops[UsdGeom.XformOp.TypeTranslate].Set(Gf.Vec3f(x, y, z))
        else:
            xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))

        # rotation — rotateXYZ 가 있으면 (0,0,rot_deg) set,
        # 없고 rotateZ 가 있으면 set, 둘 다 없으면 rotateZ add
        if UsdGeom.XformOp.TypeRotateXYZ in existing_ops:
            existing_ops[UsdGeom.XformOp.TypeRotateXYZ].Set(
                Gf.Vec3f(0.0, 0.0, rot_deg))
        elif UsdGeom.XformOp.TypeRotateZ in existing_ops:
            existing_ops[UsdGeom.XformOp.TypeRotateZ].Set(rot_deg)
        else:
            xform.AddRotateZOp().Set(rot_deg)

        return prim_path

    def _set_kinematic(self, prim_path: str) -> None:
        """RigidBodyAPI 가 적용된 prim 을 kinematic 으로 설정 (위치 고정)."""
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        if prim.HasAPI(self._UsdPhysics.RigidBodyAPI):
            rb = self._UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr().Set(True)

    def _clear_scenario_root(self) -> None:
        root = self._stage.GetPrimAtPath(SCENARIO_ROOT)
        if not root.IsValid():
            return
        for child in list(root.GetChildren()):
            self._stage.RemovePrim(child.GetPath())

    # ── 내부: I/O ────────────────────────────────────────

    def _save_png(self, rgba, out_path: str) -> None:
        from PIL import Image  # noqa: PLC0415
        import numpy as np     # noqa: PLC0415
        # Isaac Sim 카메라는 호출 시점/렌더링 상태에 따라 uint8 [0,255] 또는
        # float32 normalized [0,1] 을 반환할 수 있다. PIL PNG 저장은 uint8 RGB
        # 만 받으므로 항상 변환한다 (mode 'F' 에러 회피).
        arr = np.asarray(rgba)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        Image.fromarray(arr).save(out_path)

    def _load_home_pose(self, path: str) -> List[float]:
        """home_pose.yaml positions (SDK .pos: arm=deg, gripper=0~100%) → rad.

        SO101_JOINT_NAMES 순서. record_sim_teleop.load_initial_pose 와 같은 규약.
        """
        import yaml  # noqa: PLC0415
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        positions = data.get('positions')
        if not isinstance(positions, dict):
            raise ValueError(f'home_pose.yaml 에 positions 딕셔너리 없음 ({path}).')
        out: List[float] = []
        for n in SO101_JOINT_NAMES:
            if n not in positions:
                raise ValueError(f'home_pose.yaml positions 관절 누락: {n}')
            v = float(positions[n])
            out.append(_gripper_pct_to_rad(v) if n == 'gripper'
                       else math.radians(v))
        return out


# ──────────────────────────────────────────────
# CLI / main
# ──────────────────────────────────────────────

@dataclass
class RenderArgs:
    scenarios:    str
    scene_usd:    str
    domain:       str
    label_vocab:  str
    multi_target: str
    home_pose:    str
    image_dir:    str
    img_w:        int
    img_h:        int
    robot_prim:   str
    camera_prim:  str
    headless:     bool
    seed:         int


def parse_args() -> RenderArgs:
    p = argparse.ArgumentParser(
        description='verify 시나리오 → Isaac Sim 렌더링')
    p.add_argument('--scenarios',    required=True,
                   help='generate_verify_qa.py 가 만든 *.scenarios.jsonl')
    p.add_argument('--scene-usd',    required=True, help='base scene USD')
    p.add_argument('--domain',       required=True)
    p.add_argument('--label-vocab',  default='deploy/config/label_vocab.yaml',
                   help='(현재 사용 안 함, 향후 확장용)')
    p.add_argument('--multi-target', default='sim/config/multi_target_scenes.yaml')
    p.add_argument('--home-pose',    default='deploy/calibration/home_pose.yaml')
    p.add_argument('--image-dir',    default='',
                   help='지정 시 시나리오의 image_abspath 를 무시하고 본 dir / '
                        'image_relpath 로 강제 (이동된 경로용).')
    p.add_argument('--img-w',        type=int, default=640)
    p.add_argument('--img-h',        type=int, default=480)
    p.add_argument('--robot-prim',   default=ROBOT_PRIM_DEFAULT)
    p.add_argument('--camera-prim',  default=CAMERA_PRIM_DEFAULT)
    p.add_argument('--headless',     action='store_true',
                   help='viewport 끔 — 대량 batch 권장.')
    p.add_argument('--seed',         type=int, default=0)
    a = p.parse_args()
    return RenderArgs(
        scenarios    = os.path.expanduser(a.scenarios),
        scene_usd    = os.path.expanduser(a.scene_usd),
        domain       = a.domain,
        label_vocab  = os.path.expanduser(a.label_vocab),
        multi_target = os.path.expanduser(a.multi_target),
        home_pose    = os.path.expanduser(a.home_pose),
        image_dir    = os.path.expanduser(a.image_dir) if a.image_dir else '',
        img_w        = a.img_w,
        img_h        = a.img_h,
        robot_prim   = a.robot_prim,
        camera_prim  = a.camera_prim,
        headless     = a.headless,
        seed         = a.seed,
    )


def _load_scenarios(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_scene_cfg(yaml_path: str, domain: str) -> dict:
    import yaml  # noqa: PLC0415
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    domains = (raw or {}).get('domains') or {}
    if domain not in domains:
        raise ValueError(
            f"domain '{domain}' 이 multi_target_scenes.yaml 에 없음. "
            f"가능: {list(domains)}")
    return domains[domain]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    scenarios = _load_scenarios(args.scenarios)
    if not scenarios:
        print('[render_verify_images] scenarios 비어 있음 — 종료',
              file=sys.stderr)
        return

    scene_cfg = _load_scene_cfg(args.multi_target, args.domain)
    object_lib = Path(os.path.expanduser(
        scene_cfg.get('object_library', f'sim/assets/objects/{args.domain}')))
    if not object_lib.is_dir():
        raise FileNotFoundError(
            f'object_library 디렉터리 없음: {object_lib} — '
            'sim/scene_setup/setup_objects.py 를 먼저 실행할 것.')

    print(
        f'[render_verify_images] start — domain={args.domain} '
        f'scenarios={len(scenarios)} object_lib={object_lib} '
        f'scene_usd={args.scene_usd} headless={args.headless}',
        file=sys.stderr,
    )

    renderer = SimVerifyRenderer(
        scene_usd      = args.scene_usd,
        scene_cfg      = scene_cfg,
        object_lib     = object_lib,
        home_pose_path = args.home_pose,
        img_w          = args.img_w,
        img_h          = args.img_h,
        robot_prim     = args.robot_prim,
        camera_prim    = args.camera_prim,
        headless       = args.headless,
        rng            = rng,
    )

    n_done = 0
    n_fail = 0
    try:
        for i, s in enumerate(scenarios):
            if args.image_dir:
                s['image_abspath'] = os.path.join(args.image_dir, s['image_relpath'])
            try:
                renderer.render(s)
                n_done += 1
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                import traceback  # noqa: PLC0415
                # 처음 3건만 traceback — 대량 batch 시 로그 폭주 방지.
                # 메시지 없는 ErrorException 등 generic 예외의 실제 발생 지점 노출.
                tb = traceback.format_exc() if n_fail <= 3 else ''
                print(
                    f'[render_verify_images] {s.get("scenario_id", i)} '
                    f'fail: {type(e).__name__}: {e}\n{tb}',
                    file=sys.stderr,
                )
            if (i + 1) % 50 == 0 or (i + 1) == len(scenarios):
                print(
                    f'[render_verify_images] progress {i+1}/{len(scenarios)} '
                    f'done={n_done} fail={n_fail}',
                    file=sys.stderr,
                )
    finally:
        renderer.close()

    print(
        f'[render_verify_images] done — done={n_done}/{len(scenarios)} '
        f'fail={n_fail}',
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
