"""
measure_placement.py

base_scene.usd 의 테이블·빈(bin) 배치를 실측해 multi_target_scenes.yaml 의
placement 값을 산출한다. Isaac Sim Script Editor 에서 base_scene.usd 를 연 채
실행한다 (setup_recycling_*.py 와 동일 사용법).

산출 (multi_target_scenes.yaml > placement):
  table_top_z    테이블 윗면 world z           ← 자동 측정
  bin_exclusion  3개 빈의 world footprint       ← 자동 측정
  spawn_region   테이블 윗면 ∩ 로봇 가동범위   ← 자동 측정 불가
                 본 스크립트는 테이블 footprint(상한)만 출력한다. 가동범위는
                 실물 Leader 로 테이블 네 모서리를 teleop 해 grasp 가능 여부를
                 확인하며 직접 좁혀야 한다.

결과를 multi_target_scenes.yaml 에 기입하고 placeholder ⚠️ 주석을 제거할 것.
"""
from pxr import Usd, UsdGeom, Gf
import omni.usd

# ───── 사용자 조정 영역 ─────
TABLE_PRIM  = "/World/Table"               # 테이블 prim — 씬에 맞게 수정
BINS_PARENT = "/World/Bins"
BIN_NAMES   = ["bin_plastic", "bin_can", "bin_paper"]
ROBOT_PRIM  = "/World/so101_new_calib"
BIN_MARGIN  = 0.02                         # 빈 footprint 둘레 여유 (m)
# ─────────────────────────────

stage = omni.usd.get_context().get_stage()


def world_range(prim):
    """prim 의 world axis-aligned bbox (Gf.Range3d). geometry 없으면 None."""
    bbox = UsdGeom.Imageable(prim).ComputeWorldBound(
        Usd.TimeCode.Default(), UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
    rng = bbox.ComputeAlignedRange()
    return None if rng.IsEmpty() else rng


print("=" * 66)

# ── 0. 스테이지 단위·축 ────────────────────────────────────
mpu = UsdGeom.GetStageMetersPerUnit(stage)
up  = UsdGeom.GetStageUpAxis(stage)
print(f"stage metersPerUnit = {mpu}   (1.0 이어야 아래 수치가 m)")
print(f"stage upAxis        = {up}    (Z 이어야 z-up)")
if abs(mpu - 1.0) > 1e-6:
    print("⚠️ metersPerUnit != 1.0 — 출력 수치를 m 로 환산할 것")

# ── 로봇 base 위치 (축 규약 검증용) ────────────────────────
robot = stage.GetPrimAtPath(ROBOT_PRIM)
if robot.IsValid():
    t = UsdGeom.Xformable(robot).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()).ExtractTranslation()
    print(f"\nrobot base ≈ ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})  [{ROBOT_PRIM}]")
else:
    print(f"\n⚠️ robot prim 없음: {ROBOT_PRIM}")
print("축 규약 확인 — record_sim_teleop.py 의 TargetSelector 는 x+ = 로봇에서")
print("멀어짐(back), y+ = 로봇 기준 왼쪽(left) 으로 가정한다. 뷰포트에서")
print("어긋나면 _region_of / _disambiguating_region 의 부호를 뒤집을 것.")

# ── 1. table_top_z + 테이블 footprint ─────────────────────
print("\n" + "-" * 66)
table = stage.GetPrimAtPath(TABLE_PRIM)
table_rng = world_range(table) if table.IsValid() else None
if not table.IsValid():
    print(f"⚠️ TABLE_PRIM 없음: {TABLE_PRIM} — 상단 TABLE_PRIM 을 수정할 것.")
    world = stage.GetPrimAtPath("/World")
    if world.IsValid():
        print("/World 하위 prim:")
        for c in world.GetChildren():
            print(f"    {c.GetPath()}   ({c.GetTypeName()})")
elif table_rng is None:
    print(f"⚠️ '{TABLE_PRIM}' bbox 가 비어 있음 (하위 geometry 확인).")
else:
    tmin, tmax = table_rng.GetMin(), table_rng.GetMax()
    print(f"table '{TABLE_PRIM}' world bbox:")
    print(f"    x: [{tmin[0]:+.4f}, {tmax[0]:+.4f}]")
    print(f"    y: [{tmin[1]:+.4f}, {tmax[1]:+.4f}]")
    print(f"    z: [{tmin[2]:+.4f}, {tmax[2]:+.4f}]")
    print(f"  >> table_top_z = {tmax[2]:.4f}")
    print(f"  >> 테이블 footprint (spawn_region 상한):")
    print(f"       x_range [{tmin[0]:.4f}, {tmax[0]:.4f}]  "
          f"y_range [{tmin[1]:.4f}, {tmax[1]:.4f}]")

# ── 2. bin_exclusion ──────────────────────────────────────
print("\n" + "-" * 66)
print(f"bin_exclusion (footprint + 여유 {BIN_MARGIN} m):")
bin_lines = []
for name in BIN_NAMES:
    path = f"{BINS_PARENT}/{name}"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(f"  ⚠️ {name}: prim 없음 ({path})")
        continue
    br = world_range(prim)
    if br is None:
        print(f"  ⚠️ {name}: bbox 비어 있음")
        continue
    bmin, bmax = br.GetMin(), br.GetMax()
    x0, x1 = bmin[0] - BIN_MARGIN, bmax[0] + BIN_MARGIN
    y0, y1 = bmin[1] - BIN_MARGIN, bmax[1] + BIN_MARGIN
    print(f"  {name:12s} x:[{bmin[0]:+.4f}, {bmax[0]:+.4f}]  "
          f"y:[{bmin[1]:+.4f}, {bmax[1]:+.4f}]")
    bin_lines.append(
        f"    - {{x_range: [{x0:.4f}, {x1:.4f}], "
        f"y_range: [{y0:.4f}, {y1:.4f}]}}   # {name}")

# ── 3. YAML 스니펫 ────────────────────────────────────────
print("\n" + "=" * 66)
print("multi_target_scenes.yaml > placement 붙여넣기용 스니펫")
print("(spawn_region 은 위 테이블 footprint 안에서 teleop 도달 검증 후 좁힐 것)")
print("-" * 66)
print("placement:")
print("  spawn_region:")
print("    x_range: [?, ?]   # 테이블 footprint ∩ 로봇 가동범위 (teleop 검증)")
print("    y_range: [?, ?]")
print(f"  table_top_z: {table_rng.GetMax()[2]:.4f}" if table_rng
      else "  table_top_z: ?")
print("  min_separation: 0.08")
print("  bin_exclusion:")
for line in bin_lines:
    print(line)
print("  max_sample_attempts: 100")
print("=" * 66)
