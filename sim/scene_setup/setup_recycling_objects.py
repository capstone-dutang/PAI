"""
Generate PAI wrapper USDs for YCB recycling objects.
Run in Isaac Sim Script Editor (pxr is guaranteed available).
"""
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

# ───── 설정 ─────
# 경로는 실행 환경에 맞게 수정 (EC2 / 로컬 Isaac Sim 설치 위치에 따라 다름).
YCB_ROOT   = "/home/ubuntu/isaacsim_assets/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned"
OUTPUT_DIR = "/home/ubuntu/PAI/sim/assets/objects/recycling"

# (래퍼파일, YCB원본, 카테고리, 질량[kg], 콜라이더타입)
MAPPING = [
    # cans
    ("can_master_chef.usd",   "002_master_chef_can.usd",   "can",     0.120, "convexHull"),
    ("can_tomato_soup.usd",   "005_tomato_soup_can.usd",   "can",     0.100, "convexHull"),
    ("can_tuna.usd",          "007_tuna_fish_can.usd",     "can",     0.080, "convexHull"),
    ("can_potted_meat.usd",   "010_potted_meat_can.usd",   "can",     0.110, "convexHull"),
    # plastic
    ("plastic_bleach.usd",    "021_bleach_cleanser.usd",   "plastic", 0.150, "convexDecomposition"),
    ("plastic_pitcher.usd",   "019_pitcher_base.usd",      "plastic", 0.130, "convexDecomposition"),
    ("plastic_bowl.usd",      "024_bowl.usd",              "plastic", 0.100, "convexHull"),
    # paper boxes
    ("paper_cracker_box.usd", "003_cracker_box.usd",       "paper",   0.090, "convexHull"),
    ("paper_sugar_box.usd",   "004_sugar_box.usd",         "paper",   0.080, "convexHull"),
    ("paper_pudding_box.usd", "008_pudding_box.usd",       "paper",   0.060, "convexHull"),
]
# ─────────────────

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
created, skipped = [], []

for wrapper_name, ycb_name, category, mass_kg, collider_type in MAPPING:
    ycb_path = Path(YCB_ROOT) / ycb_name
    if not ycb_path.exists():
        print(f"  SKIP {wrapper_name}: {ycb_path} 없음")
        skipped.append(wrapper_name)
        continue

    out_path = Path(OUTPUT_DIR) / wrapper_name
    if out_path.exists():
        out_path.unlink()   # 재실행 시 덮어쓰기

    stage = Usd.Stage.CreateNew(str(out_path))

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    obj = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(obj.GetPrim())
    prim = obj.GetPrim()

    # ① YCB 원본 reference
    prim.GetReferences().AddReference(str(ycb_path))

    # ② Physics APIs
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)

    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_col.CreateApproximationAttr().Set(collider_type)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(mass_kg)

    # ③ PAI semantic
    prim.CreateAttribute(
        "semantic:Semantics:params:semanticType",
        Sdf.ValueTypeNames.String
    ).Set("class")
    prim.CreateAttribute(
        "semantic:Semantics:params:semanticData",
        Sdf.ValueTypeNames.String
    ).Set(category)

    stage.GetRootLayer().Save()
    print(f"  OK  {wrapper_name:28s} <- {ycb_name:30s} [{category:7s}] "
          f"mass={mass_kg}kg col={collider_type}")
    created.append(wrapper_name)

print(f"\n생성 {len(created)}개 / 스킵 {len(skipped)}개")
print(f"위치: {OUTPUT_DIR}")
