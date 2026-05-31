"""
Generate PAI wrapper USDs for YCB objects — 도메인 통합 버전.

이전의 setup_recycling_objects.py · setup_tableware_objects.py 두 스크립트를
하나로 합쳤다. 두 스크립트가 공유하던 골격(YCB reference → Physics →
ScaleOp baked → PhysicsMaterial → Semantic)을 함수로 추출하고, 도메인별
차이(SCALE · MAPPING · FRICTION)만 DOMAINS 딕셔너리에 모았다.

도메인을 추가하려면 DOMAINS 에 entry 만 추가하면 된다. 기본은 ENABLED_DOMAINS
에 적힌 모든 도메인을 한 번에 처리한다 — 하나만 재생성하려면 그 도메인 하나로
줄이면 된다.

Run in Isaac Sim Script Editor (pxr is guaranteed available).

사전 준비 — YCB 원본 에셋 다운로드 (`YCB_ROOT` 경로에 없으면 실행 전 1회 수행):

    pip install awscli
    aws s3 sync --no-sign-request \\
      --exclude "*.thumb.usd" --exclude ".thumbs/*" \\
      s3://omniverse-content-production/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned/ \\
      ~/isaacsim_assets/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned/

NVIDIA Omniverse 퍼블릭 버킷이라 인증 불필요(`--no-sign-request`). 썸네일은
제외(불필요·용량 절감). 모든 도메인이 같은 YCB 폴더를 공유한다.
"""
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Sdf

# ───── 공통 설정 ─────
# 경로는 실행 환경에 맞게 수정 (EC2 / 로컬 Isaac Sim 설치 위치에 따라 다름).
YCB_ROOT    = "/home/ubuntu/isaacsim_assets/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned"
OUTPUT_BASE = "/home/ubuntu/PAI/sim/assets/objects"   # 도메인별 하위 디렉터리 자동 생성

# 처리할 도메인 — 둘 다 / 하나만 / 새 도메인 추가 가능.
ENABLED_DOMAINS = ["recycling", "tableware"]

# ───── 도메인별 설정 ─────
# scale    : 객체 크기 배율. SO-101 가동범위(반경 ~0.35 m) 에 맞춰 축소.
#            래퍼 USD 의 /Object Xform 에 ScaleOp 로 baked, 질량도 부피
#            비례(scale**3) 로 같이 축소. record_sim_teleop.py 의
#            _place_object 가 ClearXformOpOrder 를 호출하지 않아 래퍼의
#            scale op 이 보존된 채 placement transform 이 [scale, transform]
#            순으로 쌓인다.
# mapping  : (래퍼파일, YCB원본, 카테고리, 질량[kg], 콜라이더타입)
# friction : 카테고리 → {static, dynamic, restitution} (소재 차이 반영)

DOMAINS = {
    "recycling": {
        "scale": 0.5,
        "mapping": [
            # cans
            ("can_master_chef.usd",   "002_master_chef_can.usd",   "can",     0.120, "convexHull"),
            ("can_tomato_soup.usd",   "005_tomato_soup_can.usd",   "can",     0.100, "convexHull"),
            ("can_tuna.usd",          "007_tuna_fish_can.usd",     "can",     0.080, "convexHull"),
            ("can_potted_meat.usd",   "010_potted_meat_can.usd",   "can",     0.110, "convexHull"),
            # plastic
            ("plastic_bleach.usd",    "021_bleach_cleanser.usd",   "plastic", 0.150, "convexDecomposition"),
            ("plastic_pitcher.usd",   "019_pitcher_base.usd",      "plastic", 0.130, "convexDecomposition"),
            ("plastic_bowl.usd",      "024_bowl.usd",              "plastic", 0.100, "convexDecomposition"),
            # paper boxes
            ("paper_cracker_box.usd", "003_cracker_box.usd",       "paper",   0.090, "convexHull"),
            ("paper_sugar_box.usd",   "004_sugar_box.usd",         "paper",   0.080, "convexHull"),
            ("paper_pudding_box.usd", "008_pudding_box.usd",       "paper",   0.060, "convexHull"),
        ],
        "friction": {
            "can":     {"static": 1.5, "dynamic": 1.3, "restitution": 0.0},   # 알루미늄, 약간 미끄러움
            "plastic": {"static": 1.6, "dynamic": 1.4, "restitution": 0.1},   # 플라스틱, 보통
            "paper":   {"static": 1.8, "dynamic": 1.6, "restitution": 0.0},   # 종이/판지, 잘 잡힘
        },
    },
    "tableware": {
        "scale": 0.7,
        "mapping": [
            # 접시류 (세라믹, 평평) — 접시 포개기(nesting) 필요하면 convexDecomposition 으로 교체.
            ("plate_dinner.usd",  "029_plate.usd",  "plate",   0.250, "convexHull"),
            # 그릇류 (세라믹, 오목 → 내부 형상 보존 위해 분해 콜라이더)
            ("bowl_round.usd",    "024_bowl.usd",   "bowl",    0.147, "convexDecomposition"),
            # 수저류 (스테인리스, 얇음 → convexHull 이 그래스핑에 더 안정적)
            ("cutlery_fork.usd",  "030_fork.usd",   "cutlery", 0.040, "convexHull"),
            ("cutlery_spoon.usd", "031_spoon.usd",  "cutlery", 0.035, "convexHull"),
            ("cutlery_knife.usd", "032_knife.usd",  "cutlery", 0.035, "convexHull"),
        ],
        "friction": {
            "plate":   {"static": 1.4, "dynamic": 1.2, "restitution": 0.1},   # 유약 세라믹(접시)
            "bowl":    {"static": 1.4, "dynamic": 1.2, "restitution": 0.1},   # 유약 세라믹(그릇)
            "cutlery": {"static": 1.1, "dynamic": 0.9, "restitution": 0.05},  # 스테인리스, 미끄러움
        },
    },
}
# ─────────────────


def make_wrapper(stage, prim_path, ycb_path, category, mass_kg, collider_type,
                 scale, friction):
    """단일 래퍼 USD 의 /Object prim 작성 — stage 생성/저장은 호출자가 담당."""
    obj = UsdGeom.Xform.Define(stage, prim_path)
    stage.SetDefaultPrim(obj.GetPrim())
    prim = obj.GetPrim()

    # ① YCB 원본 reference
    prim.GetReferences().AddReference(str(ycb_path))

    # ② Physics APIs (collider + baked mass = 원본 mass · scale**3)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(collider_type)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(mass_kg * scale ** 3)

    # ScaleOp baked — record_sim_teleop._place_object 가 ClearXformOpOrder 를
    # 호출하지 않아 이 op 이 placement transform 앞에 보존된다.
    UsdGeom.Xformable(prim).AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(scale, scale, scale))

    # ③ PhysicsMaterial 생성·바인딩 (카테고리별 마찰)
    mat_path = f"{prim_path}/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_mat.CreateStaticFrictionAttr().Set(friction["static"])
    physics_mat.CreateDynamicFrictionAttr().Set(friction["dynamic"])
    physics_mat.CreateRestitutionAttr().Set(friction["restitution"])
    UsdShade.MaterialBindingAPI(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )

    # ④ PAI semantic — class = 카테고리
    prim.CreateAttribute(
        "semantic:Semantics:params:semanticType",
        Sdf.ValueTypeNames.String).Set("class")
    prim.CreateAttribute(
        "semantic:Semantics:params:semanticData",
        Sdf.ValueTypeNames.String).Set(category)


def process_domain(domain_name, cfg):
    """한 도메인의 모든 매핑을 돌며 래퍼 USD 를 생성한다."""
    out_dir = Path(OUTPUT_BASE) / domain_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = cfg["scale"]
    friction_table = cfg["friction"]
    created, skipped = [], []

    print(f"\n── 도메인 '{domain_name}' (scale={scale}) → {out_dir}")
    for wrapper_name, ycb_name, category, mass_kg, collider_type in cfg["mapping"]:
        ycb_path = Path(YCB_ROOT) / ycb_name
        if not ycb_path.exists():
            print(f"  SKIP {wrapper_name}: {ycb_path} 없음")
            skipped.append(wrapper_name)
            continue

        out_path = out_dir / wrapper_name
        if out_path.exists():
            out_path.unlink()   # 재실행 시 덮어쓰기

        stage = Usd.Stage.CreateNew(str(out_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        fric = friction_table[category]
        make_wrapper(stage, "/Object", ycb_path, category, mass_kg, collider_type,
                     scale, fric)
        stage.GetRootLayer().Save()

        print(f"  OK  {wrapper_name:28s} <- {ycb_name:30s} [{category:7s}] "
              f"scale={scale} mass={mass_kg * scale ** 3:.4f}kg (orig {mass_kg}) "
              f"col={collider_type} fric=({fric['static']},{fric['dynamic']})")
        created.append(wrapper_name)

    print(f"  → 생성 {len(created)}개 / 스킵 {len(skipped)}개")
    return created, skipped


# 메인 — ENABLED_DOMAINS 만 처리
all_created, all_skipped = [], []
for domain_name in ENABLED_DOMAINS:
    if domain_name not in DOMAINS:
        print(f"⚠️ '{domain_name}' 가 DOMAINS 에 정의되지 않음 — 스킵")
        continue
    c, s = process_domain(domain_name, DOMAINS[domain_name])
    all_created.extend(c)
    all_skipped.extend(s)

print(f"\n=== 합계: 생성 {len(all_created)}개 / 스킵 {len(all_skipped)}개 ===")
