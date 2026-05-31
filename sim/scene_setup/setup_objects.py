"""
Generate PAI wrapper USDs for domain objects — recycling + tableware 통합.

도메인별 차이(asset_base · SCALE · MAPPING · FRICTION) 만 DOMAINS 딕셔너리에
모았다. 공유 골격(원본 reference → Physics → ScaleOp baked → PhysicsMaterial
→ Semantic) 은 make_wrapper / process_domain 으로 추출. 새 도메인을 추가하려면
DOMAINS entry 한 줄 추가하면 된다.

소스 에셋:
  recycling — YCB (NVIDIA Omniverse 퍼블릭 버킷)
  tableware — SimReady (omni.client 로 로컬화 후 reference)

Run in Isaac Sim Script Editor (pxr is guaranteed available).

────────────────────────────────────────────────────────────────────────────
사전 준비 — 에셋이 로컬에 없으면 실행 전 1회:

[YCB · recycling — `YCB_ROOT` 기준]
    pip install awscli
    aws s3 sync --no-sign-request \\
      --exclude "*.thumb.usd" --exclude ".thumbs/*" \\
      s3://omniverse-content-production/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned/ \\
      ~/isaacsim_assets/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned/

NVIDIA Omniverse 퍼블릭 버킷이라 인증 불필요(`--no-sign-request`). 썸네일은
제외(불필요·용량 절감).

[SimReady · tableware — `SIMREADY_ROOT` 기준]
    LOCALIZE_SIMREADY=True 로 두고 1회 실행하면, mapping 에서 실제 쓰는 prop
    폴더 + 공용 materials(SimPBR.mdl 등) 만 omni.client 로 로컬 미러링한다
    (원격과 동일 트리 → 상대참조 보존; common_assets 전체는 받지 않아 빠름).
    이후 skip_existing 으로 재실행도 빠르다. 강제 재다운로드는 로컬 폴더 삭제 후 실행.
────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

# ───── 공통 설정 ─────
# 경로는 실행 환경에 맞게 수정 (EC2 / 로컬 Isaac Sim 설치 위치에 따라 다름).
YCB_ROOT      = "/home/ubuntu/isaacsim_assets/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned"
SIMREADY_ROOT = "/home/ubuntu/isaacsim_assets/Assets/simready_content/common_assets/props"
OUTPUT_BASE   = "/home/ubuntu/PAI/sim/assets/objects"   # 도메인별 하위 디렉터리 자동 생성

# 처리할 도메인 — 둘 다 / 하나만 / 새 도메인 추가 가능.
ENABLED_DOMAINS = ["recycling", "tableware"]

# ───── 도메인별 설정 ─────
# asset_base : 원본 USD 가 모여 있는 디렉터리 (로컬 절대경로 또는 URL).
#              mapping 의 source 는 asset_base 기준 상대경로.
# scale      : 객체 크기 배율. SO-101 가동범위(반경 ~0.35 m) 에 맞춰 축소.
#              래퍼 USD 의 /Object Xform 에 ScaleOp 로 baked, 질량도 부피
#              비례(scale**3) 로 같이 축소. 보존 메커니즘은 make_wrapper 주석 참조.
# mapping    : (래퍼파일, asset_base 기준 상대경로, 카테고리, 질량[kg], 콜라이더타입)
# friction   : 카테고리 → {static, dynamic, restitution} (소재 차이 반영)

DOMAINS = {
    "recycling": {
        "asset_base": YCB_ROOT,
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
        "asset_base": SIMREADY_ROOT,
        "scale": 0.7,
        # SimReady 식기는 실물 크기(meters) — 0.7 이면 접시 ≈0.18 m, 포크 ≈0.14 m.
        # 품목별로 어긋나면 entry 에 per-item scale 을 추가해 분리하면 된다.
        # 카테고리는 vocab object 명사(plate/fork/spoon)와 1:1 — wrapper 파일명
        # prefix 도 카테고리와 같아야 한다(record_sim_teleop._category_of 가 split
        # '_'[0]). cup/bowl 은 vocab 에서 제거 (자산 가용성 부족, 2026-05-25 scope).
        "mapping": [
            ("plate_dinner.usd", "plate_large/plate_large.usd", "plate", 0.250, "convexHull"),
            ("fork_big.usd",     "fork_big/fork_big.usd",       "fork",  0.050, "convexHull"),
            ("spoon_big.usd",    "spoon_big/spoon_big.usd",     "spoon", 0.050, "convexHull"),
        ],
        "friction": {
            "plate": {"static": 1.4, "dynamic": 1.2, "restitution": 0.1},    # 유약 세라믹
            "fork":  {"static": 1.1, "dynamic": 0.9, "restitution": 0.05},   # 스테인리스
            "spoon": {"static": 1.1, "dynamic": 0.9, "restitution": 0.05},   # 스테인리스
        },
        # 방법 B 로컬화 대상: 실제 쓰는 prop 폴더(mapping 기준) + 공용 materials.
        # remote_props 는 asset_base(SIMREADY_ROOT)에 대응하는 원격 prop 루트.
        "localize": {
            "remote_props": "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props",
            "shared": [
                "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/materials",
            ],
        },
    },
}
# ─────────────────


# ───── SimReady 의존성 로컬화 (방법 B — 사용 에셋만) ─────
# prop 머티리얼이 SimPBR.mdl 을 ../../../materials/SimPBR.mdl 로 참조하는데
# 그 공용 materials 폴더가 로컬에 없어 셰이더(SdrShaderNode)를 못 만든다.
# (1) mapping 에서 실제 쓰는 prop 폴더만 + (2) 공용 materials 폴더를
# 원격과 '동일 구조' 로 로컬 미러링해 상대참조를 보존. common_assets 전체를
# 받지 않으므로 빠르다.
# 사용: LOCALIZE_SIMREADY=True 로 1회 실행 → 이후 skip_existing 으로 재실행도 빠름.
#       (강제 재다운로드는 해당 로컬 폴더를 지우고 다시 실행)
LOCALIZE_SIMREADY = False
LOCAL_ROOT        = "/home/ubuntu/isaacsim_assets"   # 원격 'Assets/...' 구조를 이 아래로 미러링


def _to_local(url):
    import os
    from urllib.parse import urlparse
    return os.path.join(LOCAL_ROOT, urlparse(url).path.lstrip("/"))


def _copy_tree(remote_dir, skip_existing=True):
    """omni.client 로 remote_dir 하위를 로컬에 트리 구조 그대로 재귀 복사.
    skip_existing=True 면 이미 있는 파일은 건너뛴다(재실행 속도 향상)."""
    import os
    import omni.client
    remote_dir = remote_dir.rstrip("/")
    res, entries = omni.client.list(remote_dir + "/")
    if res != omni.client.Result.OK:
        print("    LIST FAIL", res, remote_dir)
        return 0, 1, 0
    ok = fail = skip = 0
    for e in entries:
        child = remote_dir + "/" + e.relative_path
        if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:   # 폴더 → 재귀
            o, f, s = _copy_tree(child, skip_existing)
            ok += o
            fail += f
            skip += s
        else:
            local = _to_local(child)
            if skip_existing and os.path.exists(local):
                skip += 1
                continue
            os.makedirs(os.path.dirname(local), exist_ok=True)
            r = omni.client.copy(child, local, omni.client.CopyBehavior.OVERWRITE)
            if r == omni.client.Result.OK:
                ok += 1
            else:
                fail += 1
                print("    COPY FAIL", r, child)
    return ok, fail, skip


def localize_domain_assets(domain_name, cfg):
    """cfg['localize'] 가 있으면, mapping 에서 실제 쓰는 prop 폴더 + 공용
    의존성(materials 등) 만 로컬로 미러링한다(방법 B). 안 쓰는 에셋은 받지 않음."""
    import posixpath
    lz = cfg.get("localize")
    if not lz:
        return
    print(f"── {domain_name} 의존성 로컬화")
    ok = fail = skip = 0

    # (1) mapping 에 등장하는 prop 폴더만 (중복 제거)
    folders = sorted({posixpath.dirname(src_rel)
                      for (_, src_rel, *_r) in cfg["mapping"]
                      if posixpath.dirname(src_rel)})
    for fld in folders:
        url = f"{lz['remote_props'].rstrip('/')}/{fld}"
        print(f"  prop  : {fld}")
        o, f, s = _copy_tree(url)
        ok += o
        fail += f
        skip += s

    # (2) prop 들이 공유하는 의존성(SimPBR.mdl 등) — 1회만
    for shared in lz.get("shared", []):
        print(f"  shared: {shared.rstrip('/').rsplit('/', 1)[-1]}")
        o, f, s = _copy_tree(shared)
        ok += o
        fail += f
        skip += s

    print(f"  → 복사 {ok} / 건너뜀 {skip} / 실패 {fail}")


def is_url(s: str) -> bool:
    return s.startswith(("omniverse://", "http://", "https://"))


def make_wrapper(stage, prim_path, source, category, mass_kg, collider_type,
                 scale, friction):
    """단일 래퍼 USD 의 /Object prim 작성 — stage 생성/저장은 호출자 책임.

    source: 원본 USD 의 절대경로 또는 URL. AddReference 에 그대로 전달.
    """
    obj = UsdGeom.Xform.Define(stage, prim_path)
    stage.SetDefaultPrim(obj.GetPrim())
    prim = obj.GetPrim()

    # ① 원본 reference
    prim.GetReferences().AddReference(source)

    # SimReady 에셋이 PhysicsVariant 를 가지면 None 으로 고정 — 래퍼가 직접
    # RigidBodyAPI 를 적용하므로 이중 적용 방지. YCB 등 없으면 no-op.
    vset = prim.GetVariantSets().GetVariantSet("PhysicsVariant")
    if "None" in vset.GetVariantNames():
        vset.SetVariantSelection("None")

    # ② Physics APIs (collider + baked mass = 원본 mass × scale**3)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(collider_type)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(mass_kg * scale ** 3)

    # ScaleOp baked — record_sim_teleop._place_object 가 ClearXformOpOrder 를
    # 호출하지 않아 이 op 이 placement transform 앞에 보존된다 ([scale, transform]
    # 순으로 적용). SimReady 에셋은 reference 시 자체 xformOp(translate/rotate
    # /scale) 을 /Object 에 합성하므로 scale op 이 이미 있으면 기존 값에 곱해
    # override, 없으면(YCB 등) 새로 추가. translate/rotate 는 그대로 둔다.
    xf = UsdGeom.Xformable(prim)
    scale_op = next((op for op in xf.GetOrderedXformOps()
                     if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)
    if scale_op is not None:
        v = scale_op.Get() or Gf.Vec3f(1.0, 1.0, 1.0)
        scale_op.Set(type(v)(v[0] * scale, v[1] * scale, v[2] * scale))
    else:
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
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
    asset_base = cfg["asset_base"]
    scale = cfg["scale"]
    friction_table = cfg["friction"]
    created, skipped = [], []

    print(f"\n── 도메인 '{domain_name}' (scale={scale}) → {out_dir}")
    for wrapper_name, src_rel, category, mass_kg, collider_type in cfg["mapping"]:
        source = f"{asset_base}/{src_rel}"

        # 로컬 절대경로 만 존재 검사 (원격 URL 은 건너뜀 — reference 시점에 해결)
        if not is_url(source) and not Path(source).exists():
            print(f"  SKIP {wrapper_name}: {source} 없음")
            skipped.append(wrapper_name)
            continue

        out_path = str(out_dir / wrapper_name)

        # 같은 Script Editor 세션에서 재실행 시 USD 가 layer 를 identifier 로
        # 캐시하고 있어 CreateNew 가 'A layer already exists' 로 실패한다
        # (파일을 unlink 해도 캐시는 남음). FindOrOpen 으로 캐시·디스크 모두
        # 회수 후 Clear() 하여 재사용; 없으면 새로 만든다.
        layer = Sdf.Layer.FindOrOpen(out_path)
        if layer:
            layer.Clear()
        else:
            layer = Sdf.Layer.CreateNew(out_path)
        stage = Usd.Stage.Open(layer)

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        fric = friction_table[category]
        make_wrapper(stage, "/Object", source, category, mass_kg, collider_type,
                     scale, fric)
        stage.GetRootLayer().Save()

        print(f"  OK  {wrapper_name:28s} <- {src_rel:30s} [{category:7s}] "
              f"scale={scale} mass={mass_kg * scale ** 3:.4f}kg (orig {mass_kg}) "
              f"col={collider_type} fric=({fric['static']},{fric['dynamic']})")
        created.append(wrapper_name)

    print(f"  → 생성 {len(created)}개 / 스킵 {len(skipped)}개")
    return created, skipped


# 메인 — ENABLED_DOMAINS 만 처리
if LOCALIZE_SIMREADY:
    print("\n=== SimReady 의존성 로컬화 (사용 에셋만) ===")
    for domain_name in ENABLED_DOMAINS:
        cfg = DOMAINS.get(domain_name)
        if cfg:
            localize_domain_assets(domain_name, cfg)

all_created, all_skipped = [], []
for domain_name in ENABLED_DOMAINS:
    if domain_name not in DOMAINS:
        print(f"⚠️ '{domain_name}' 가 DOMAINS 에 정의되지 않음 — 스킵")
        continue
    c, s = process_domain(domain_name, DOMAINS[domain_name])
    all_created.extend(c)
    all_skipped.extend(s)

print(f"\n=== 합계: 생성 {len(all_created)}개 / 스킵 {len(all_skipped)}개 ===")