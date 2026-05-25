"""
probe_reach.py

SO-101 가동범위 측정 도구 — Isaac Sim Script Editor 에서 실행.

base_scene.usd 를 연 채 본 스크립트를 실행하면 그리퍼 tip 의 world (x,y) 를
매 프레임 읽어 (r, theta) [로봇 base 기준 극좌표] 의 min/max 를 누적 출력한다.
다시 실행하면 정지·요약 + multi_target_scenes.yaml > placement > reach 스니펫
을 출력한다 (토글 동작).

사용 흐름:
  1. Isaac Sim GUI 에서 base_scene.usd 를 연다.
  2. 다음 중 하나로 그리퍼를 움직일 준비:
       - Articulation Inspector 의 joint slider 로 follower jog
       - 실물 Leader teleop (대응 환경이 GUI 와 함께 동작할 때)
  3. 본 스크립트 실행 → "▶ Probe started" 출력 후 매 N 프레임마다 min/max 출력.
  4. 가동범위 극단(정면 최대·최소·좌·우) 까지 그리퍼 이동.
  5. 본 스크립트 재실행 → "⏹ Probe stopped" + reach yaml 스니펫 출력 →
     multi_target_scenes.yaml > placement > reach 에 붙여넣기.

⚠️ GRIPPER_PRIM 은 그리퍼 tip 의 world 위치를 대표할 prim 으로 설정할 것
   (예: 그리퍼 마지막 link, 또는 별도 tip Xform).
"""
from pxr import Usd, UsdGeom
import omni.usd
import omni.kit.app
import math
import builtins

# ───── 사용자 조정 영역 ─────
ROBOT_PRIM   = "/World/so101_new_calib"
GRIPPER_PRIM = "/World/so101_new_calib/gripper_link"   # ⚠️ 실제 prim 경로로 수정
PRINT_EVERY  = 30        # N 프레임마다 진행 출력 (60 fps → 약 0.5초)
# ─────────────────────────────

_STATE_KEY = "_probe_reach_state"
TAG = "[probe_reach]"


def _start():
    stage = omni.usd.get_context().get_stage()

    robot = stage.GetPrimAtPath(ROBOT_PRIM)
    if not robot.IsValid():
        print(f"{TAG} ❌ ROBOT_PRIM 없음: {ROBOT_PRIM}")
        return
    grip = stage.GetPrimAtPath(GRIPPER_PRIM)
    if not grip.IsValid():
        print(f"{TAG} ❌ GRIPPER_PRIM 없음: {GRIPPER_PRIM}")
        print(f"{TAG}   상단 GRIPPER_PRIM 을 실제 그리퍼 tip prim 경로로 수정할 것.")
        return

    t0 = UsdGeom.Xformable(robot).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()).ExtractTranslation()
    cx, cy = float(t0[0]), float(t0[1])
    print(f"{TAG} robot base = ({cx:+.4f}, {cy:+.4f})")
    print(f"{TAG} gripper    = {GRIPPER_PRIM}")

    state = {
        'robot_xy': (cx, cy),
        'grip_prim': grip,
        'r_min':  math.inf,  'r_max':  -math.inf,
        'th_min': math.inf,  'th_max': -math.inf,
        'x_min':  math.inf,  'x_max':  -math.inf,
        'y_min':  math.inf,  'y_max':  -math.inf,
        'frames': 0,
    }

    def _on_update(event):
        try:
            tg = UsdGeom.Xformable(state['grip_prim']).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()).ExtractTranslation()
            gx, gy = float(tg[0]), float(tg[1])
        except Exception as e:
            print(f"{TAG} read 실패: {e}")
            return
        dx, dy = gx - state['robot_xy'][0], gy - state['robot_xy'][1]
        r  = math.hypot(dx, dy)
        th = math.degrees(math.atan2(dy, dx))
        if r  < state['r_min']:  state['r_min']  = r
        if r  > state['r_max']:  state['r_max']  = r
        if th < state['th_min']: state['th_min'] = th
        if th > state['th_max']: state['th_max'] = th
        if gx < state['x_min']:  state['x_min']  = gx
        if gx > state['x_max']:  state['x_max']  = gx
        if gy < state['y_min']:  state['y_min']  = gy
        if gy > state['y_max']:  state['y_max']  = gy
        state['frames'] += 1
        if state['frames'] % PRINT_EVERY == 0:
            print(f"{TAG} frame {state['frames']:5d}  "
                  f"r∈[{state['r_min']:.3f},{state['r_max']:.3f}]  "
                  f"θ∈[{state['th_min']:+6.1f}°,{state['th_max']:+6.1f}°]")

    sub = omni.kit.app.get_app().get_update_event_stream() \
        .create_subscription_to_pop(_on_update, name="probe_reach")
    setattr(builtins, _STATE_KEY, {'sub': sub, 'data': state})
    print(f"{TAG} ▶ Probe started — 그리퍼를 가동범위 극단으로 옮길 것")
    print(f"{TAG}   본 스크립트 재실행 = 정지 + reach yaml 스니펫 출력")


def _stop():
    s = getattr(builtins, _STATE_KEY)
    s['sub'].unsubscribe()
    delattr(builtins, _STATE_KEY)
    d = s['data']
    if d['frames'] == 0:
        print(f"{TAG} ⏹ Probe stopped — 샘플 0개. 그리퍼가 움직였는지 확인.")
        return
    print(f"{TAG} ⏹ Probe stopped after {d['frames']} frames")
    print(f"{TAG}   gripper x ∈ [{d['x_min']:+.4f}, {d['x_max']:+.4f}]  (world)")
    print(f"{TAG}   gripper y ∈ [{d['y_min']:+.4f}, {d['y_max']:+.4f}]")
    print(f"{TAG}   r ∈ [{d['r_min']:.4f}, {d['r_max']:.4f}]   (base 거리)")
    print(f"{TAG}   θ ∈ [{d['th_min']:+.2f}°, {d['th_max']:+.2f}°]")
    cx, cy = d['robot_xy']
    print("")
    print("=" * 64)
    print("multi_target_scenes.yaml > placement > reach 스니펫:")
    print("=" * 64)
    print("  reach:")
    print(f"    center:          [{cx:.4f}, {cy:.4f}]")
    print(f"    r_range:         [{d['r_min']:.4f}, {d['r_max']:.4f}]")
    print(f"    theta_range_deg: [{d['th_min']:.2f}, {d['th_max']:.2f}]")
    print("=" * 64)


# 토글 — 이미 실행 중이면 정지, 아니면 시작
if hasattr(builtins, _STATE_KEY):
    _stop()
else:
    _start()
