"""
SO-101 그리퍼 핑거에 collision approximation + PhysicsMaterial 추가.
Run in Isaac Sim Script Editor — so101_new_calib.usd 가 현재 스테이지로
열려 있어야 한다.

사전 준비:
  1) SO-101 URDF 임포트 완료 (sim_cloud_teleop_setting.md Part 2 참조)
  2) so101_new_calib.usd 를 Isaac Sim 에서 열어둔 상태

기본 URDF 임포트만으로는 그리퍼 핑거 메시에 collision approximation 이 부정확
하거나 PhysicsMaterial 이 없어서 객체 파지 시 미끄러진다. 이 스크립트는:
  - 핑거 prim 에 CollisionAPI + MeshCollisionAPI(convexDecomposition) 부여
  - GripperPhysicsMaterial 생성·바인딩 (높은 마찰계수)
재실행 안전(HasAPI 체크), so101_new_calib.usd 를 in-place 수정한다.
"""
import omni.usd
from pxr import UsdPhysics, UsdShade

stage = omni.usd.get_context().get_stage()

FINGER_PRIMS = [
    "/so101_new_calib/moving_jaw_so101_v1_link",
    "/so101_new_calib/gripper_frame_link",
]

gripper_friction = {"static": 2.0, "dynamic": 1.8, "restitution": 0.0}

mat_path = "/so101_new_calib/GripperPhysicsMaterial"
material = UsdShade.Material.Define(stage, mat_path)
phys_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
phys_mat.CreateStaticFrictionAttr().Set(gripper_friction["static"])
phys_mat.CreateDynamicFrictionAttr().Set(gripper_friction["dynamic"])
phys_mat.CreateRestitutionAttr().Set(gripper_friction["restitution"])

for fp in FINGER_PRIMS:
    prim = stage.GetPrimAtPath(fp)
    if not prim.IsValid():
        print(f"  SKIP: {fp} not found")
        continue

    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    mc = UsdPhysics.MeshCollisionAPI(prim)
    mc.CreateApproximationAttr().Set("convexDecomposition")

    UsdShade.MaterialBindingAPI(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )

    print(f"  OK {fp}: decomposition + friction({gripper_friction['static']})")

stage.Save()
print("\n✅ Gripper fingers fixed. so101_new_calib.usd updated.")
