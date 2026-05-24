"""
Add 3 recycling bins (plastic / can / paper) to base scene.
Open-top boxes made of 5 cuboids each.
Run in Script Editor with base_scene.usd open.
"""
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import omni.usd

# ───── 사용자 조정 영역 ─────
BINS_PARENT = "/World/Bins"

# 빈 내부 치수 (m)
INNER_W = 0.15
INNER_D = 0.105
INNER_H = 0.10
WALL_T  = 0.005   # 벽 두께

# 빈 정의 — recycling_kb.json과 라벨 정합 필요
BINS = [
    {
        "name":  "bin_plastic",
        "color": (0.90, 0.39, 0.08),   # 주황
        "pos":   (-0.3475, -0.305, 0.0),
        "semantic": "plastic",
    },
    {
        "name":  "bin_can",
        "color": (0.35, 0.00, 0.43),   # 보라
        "pos":   (-0.2325, -0.305, 0.0),
        "semantic": "can",
    },
    {
        "name":  "bin_paper",
        "color": (0.19, 0.07, 0.00),   # 갈색
        "pos":   (-0.1175, -0.305, 0.0),
        "semantic": "paper",
    },
]
# ─────────────────────────────

stage = omni.usd.get_context().get_stage()

# 컨테이너 Xform
UsdGeom.Xform.Define(stage, BINS_PARENT)


def add_semantic_label(prim, label):
    """Attach a Replicator-compatible 'class' semantic to a prim."""
    sem = "Semantics"
    prim.CreateAttribute(f"semantic:{sem}:params:semanticType",
                         Sdf.ValueTypeNames.String).Set("class")
    prim.CreateAttribute(f"semantic:{sem}:params:semanticData",
                         Sdf.ValueTypeNames.String).Set(label)


def make_face(stage, parent_path, name, size_xyz, pos_xyz, color):
    """Single cuboid face — UsdGeom.Cube (size=1) + scale + translate."""
    face_path = f"{parent_path}/{name}"
    cube = UsdGeom.Cube.Define(stage, face_path)
    cube.GetSizeAttr().Set(1.0)

    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos_xyz))
    xf.AddScaleOp().Set(Gf.Vec3f(*size_xyz))

    # 색상 (displayColor — DR로 나중에 override 가능)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    # Static collider — 빈은 움직이지 않으니 RigidBody는 안 붙임
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def make_bin(stage, parent_path, spec):
    """Open-top box: floor + 4 walls."""
    name, color, pos, semantic = spec["name"], spec["color"], spec["pos"], spec["semantic"]
    bin_path = f"{parent_path}/{name}"

    # Bin Xform — 사용자가 GUI에서 이걸 움직이면 5개 face가 같이 따라옴
    bin_xf = UsdGeom.Xform.Define(stage, bin_path)
    bxf = UsdGeom.Xformable(bin_xf.GetPrim())
    bxf.ClearXformOpOrder()
    bxf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    bxf.AddRotateZOp().Set(90.0)

    iw, id_, ih, wt = INNER_W, INNER_D, INNER_H, WALL_T
    outer_w = iw + 2*wt
    outer_d = id_ + 2*wt

    faces = [
        # name,        size (x,y,z),         center (x,y,z)
        ("floor",      (outer_w, outer_d, wt), (0, 0, wt/2)),
        ("wall_pos_x", (wt, outer_d, ih),      ( iw/2 + wt/2, 0, ih/2 + wt)),
        ("wall_neg_x", (wt, outer_d, ih),      (-iw/2 - wt/2, 0, ih/2 + wt)),
        ("wall_pos_y", (iw, wt, ih),           (0,  id_/2 + wt/2, ih/2 + wt)),
        ("wall_neg_y", (iw, wt, ih),           (0, -id_/2 - wt/2, ih/2 + wt)),
    ]
    for face_name, size, fpos in faces:
        make_face(stage, bin_path, face_name, size, fpos, color)

    # Semantic은 bin 단위로 부착 — 5개 face 전체가 하나의 객체로 인식됨
    add_semantic_label(bin_xf.GetPrim(), semantic)

    print(f"  ✓ {name:12s} @ {pos}  color={color}  semantic='{semantic}'")
    return bin_xf


print(f"Creating bins under {BINS_PARENT}:")
for spec in BINS:
    make_bin(stage, BINS_PARENT, spec)

print("\n✅ 3 bins added. Move each /World/Bins/bin_* in viewport to position.")
print("   Stage tree → bin Xform 선택 → 노란 화살표(Translate gizmo)로 드래그")
