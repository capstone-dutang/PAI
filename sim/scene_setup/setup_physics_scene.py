"""
PhysicsScene 솔버 파라미터 튜닝 (Isaac Sim manipulation 권장값).
Run in Isaac Sim Script Editor — 튜닝 대상 씬(보통 base_scene.usd)이 현재
스테이지로 열려 있어야 한다.

사전 준비:
  1) 튜닝 대상 씬 USD 가 Isaac Sim 에서 열린 상태
  2) PhysicsScene 이 없으면 /World/physicsScene 으로 생성

튜닝 항목 (manipulation/grasp 안정성 우선):
  - 200 Hz 시뮬레이션 (dt = 5ms) — 마찰·접촉 수렴 향상
  - position iter 16~64, velocity iter 2~8 — 강한 접촉에서 관통 방지
  - CCD on — 빠른 움직임에서 터널링 방지
  - Stabilization on — 다중 접촉 적층 안정화
재실행 안전, 현재 스테이지를 in-place 저장한다.
"""
from pxr import UsdPhysics, PhysxSchema
import omni.usd

stage = omni.usd.get_context().get_stage()

physics_scene = None
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsScene":
        physics_scene = prim
        break

if physics_scene is None:
    print("PhysicsScene 없음 — 생성")
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene").GetPrim()

physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene)
physx_scene.CreateTimeStepsPerSecondAttr().Set(200)        # dt = 1/200 = 0.005s
physx_scene.CreateMinPositionIterationCountAttr().Set(16)
physx_scene.CreateMaxPositionIterationCountAttr().Set(64)
physx_scene.CreateMinVelocityIterationCountAttr().Set(2)
physx_scene.CreateMaxVelocityIterationCountAttr().Set(8)
physx_scene.CreateEnableCCDAttr().Set(True)                # Continuous Collision Detection
physx_scene.CreateEnableStabilizationAttr().Set(True)

print("OK Physics tuned: 200Hz, pos iter 16-64, vel iter 2-8, CCD on, stabilization on")

omni.usd.get_context().save_stage()
print("\n✅ PhysicsScene updated. Stage saved.")
