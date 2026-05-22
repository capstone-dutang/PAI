"""
Configure RGB camera from PAI calibration yamls.
Run in Script Editor with base_scene.usd open.
"""
import yaml
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf
import omni.usd

# ───── 사용자 설정 ─────
# 경로는 실행 환경에 맞게 수정. EC2(클라우드): /home/ubuntu/PAI/...,
# 로컬 Ubuntu 셋업: ~/projects/PAI/... (sim/teleop/sim_local_teleop_setting.md 참조)
INTRINSICS_YAML = "/home/ubuntu/PAI/deploy/calibration/camera_intrinsics.yaml"
EXTRINSICS_YAML = "/home/ubuntu/PAI/deploy/calibration/camera_robot_transform.yaml"
SO101_BASE_PRIM = "/World/so101_new_calib/base_link"
CAMERA_PRIM     = "/World/Camera_RGB"
FOCAL_LENGTH_MM = 24.0
CLIPPING_NEAR   = 0.05
CLIPPING_FAR    = 10.0

# ───── 1. Intrinsics 로드 + 변환 ─────
with open(INTRINSICS_YAML) as f:
    intr = yaml.safe_load(f)

W, H = intr["width"], intr["height"]
K    = intr["k"]
fx, fy = K[0], K[4]
cx, cy = K[2], K[5]

f_mm  = FOCAL_LENGTH_MM
h_ap  = W * f_mm / fx
v_ap  = H * f_mm / fy
h_off = (cx - W/2) * h_ap / W
v_off = (cy - H/2) * v_ap / H

print(f"[intrinsics] fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
print(f"[USD param]  focal={f_mm}mm hAp={h_ap:.4f} vAp={v_ap:.4f}")
print(f"             hOff={h_off:.5f} vOff={v_off:.5f}")

# ───── 2. Extrinsics 로드 ─────
with open(EXTRINSICS_YAML) as f:
    extr = yaml.safe_load(f)

tt = extr["transform"]["translation"]
rr = extr["transform"]["rotation"]
trans_base_opt = np.array([tt["x"], tt["y"], tt["z"]])
qx, qy, qz, qw = rr["x"], rr["y"], rr["z"], rr["w"]

def quat_to_R(qx, qy, qz, qw):
    """ROS quaternion (x,y,z,w) → 3x3 rotation matrix (column-vector convention)."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])

R_base_opt = quat_to_R(qx, qy, qz, qw)

# ROS optical (+Z forward, +Y down) → USD camera (-Z forward, +Y up)
# X축 180° 회전: y,z 부호 반전
R_x180 = np.diag([1.0, -1.0, -1.0])
R_base_usdcam = R_base_opt @ R_x180

# ───── 3. SO-101 base의 world pose 조회 ─────
stage = omni.usd.get_context().get_stage()
so101 = stage.GetPrimAtPath(SO101_BASE_PRIM)
assert so101.IsValid(), f"{SO101_BASE_PRIM} not found in stage"

xformable = UsdGeom.Xformable(so101)
M_gf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

# Gf.Matrix4d는 row-vector form. numpy 후 .T로 column-vector form 변환
M_np = np.array([[M_gf[i, j] for j in range(4)] for i in range(4)]).T
R_world_base = M_np[:3, :3]
t_world_base = M_np[:3,  3]

# ───── 4. 합성: T_world_cam_USD ─────
R_world_cam = R_world_base @ R_base_usdcam
t_world_cam = R_world_base @ trans_base_opt + t_world_base

print(f"[so101 world pos] {t_world_base}")
print(f"[camera world pos] {t_world_cam}")

# ───── 5. Camera prim 생성/업데이트 ─────
cam_path = Sdf.Path(CAMERA_PRIM)
cam_prim = stage.GetPrimAtPath(cam_path)
if not cam_prim.IsValid():
    cam_geom = UsdGeom.Camera.Define(stage, cam_path)
else:
    cam_geom = UsdGeom.Camera(cam_prim)

cam_geom.GetFocalLengthAttr().Set(f_mm)
cam_geom.GetHorizontalApertureAttr().Set(h_ap)
cam_geom.GetVerticalApertureAttr().Set(v_ap)
cam_geom.GetHorizontalApertureOffsetAttr().Set(h_off)
cam_geom.GetVerticalApertureOffsetAttr().Set(v_off)
cam_geom.GetClippingRangeAttr().Set(Gf.Vec2f(CLIPPING_NEAR, CLIPPING_FAR))
cam_geom.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)

# USD row-vector form 매트릭스 (top-left 3x3는 R의 전치, 마지막 행이 translation)
M = Gf.Matrix4d(
    R_world_cam[0,0], R_world_cam[1,0], R_world_cam[2,0], 0.0,
    R_world_cam[0,1], R_world_cam[1,1], R_world_cam[2,1], 0.0,
    R_world_cam[0,2], R_world_cam[1,2], R_world_cam[2,2], 0.0,
    t_world_cam[0],   t_world_cam[1],   t_world_cam[2],   1.0,
)

cam_xf = UsdGeom.Xformable(cam_geom.GetPrim())
cam_xf.ClearXformOpOrder()
xop = cam_xf.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble)
xop.Set(M)

print("✅ Camera_RGB configured.")
print("   Viewport menu → Cameras → Camera_RGB 로 시점 전환 후 검증하세요.")
