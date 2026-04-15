# =============================================================================
# generate_act_dataset.py
# -----------------------------------------------------------------------------
# 목적: Isaac Sim에서 ACT Co-training용 시연 데이터(Sim dataset)를 생성한다.
#       Part B-2 단계. 결과물은 Part D의 Real dataset과 함께 Part E에서
#       LeRobot 0.4.4 멀티 데이터셋 방식으로 공동 학습된다.
#
# 전체 파이프라인 내 위치:
#   Part B-2 (이 스크립트)  →  Sim LeRobotDataset (~1000 에피소드)
#   Part D                  →  Real LeRobotDataset (50~100 에피소드)
#   Part E                  →  ACT Co-training (Sim + Real)
#
# 실행 환경: 외부 PC (Ubuntu, GPU VRAM 16GB+, Isaac Sim 설치됨)
# =============================================================================


# -----------------------------------------------------------------------------
# [1] Isaac Sim 초기화
# -----------------------------------------------------------------------------
# - SimulationApp을 headless 모드로 시작 (대량 에피소드 생성 시 필수)
# - Isaac Sim 버전에 따라 import 경로가 다름:
#     * 4.0 이전: omni.isaac.core
#     * 4.2+    : isaacsim.core
#   → 사용자에게 버전 확인 필요 (현재 미결정)
# - 물리 시뮬레이션 dt = 1/30 (실기 camera_node 30Hz와 일치)


# -----------------------------------------------------------------------------
# [2] 씬 구성
# -----------------------------------------------------------------------------
# - SO-101 로봇 팔 로드:
#     * USD 경로 필요 (현재 미확보. ros2_ws/src/robot_arm_pkg/urdf/so101.urdf를
#       URDF Importer로 USD 변환해야 함)
#     * joint 6개, end-effector link 이름 확인 필요
#     * Action space는 position control (실기 SO-101과 반드시 일치)
# - RealSense 모사 카메라:
#     * base_link 기준 extrinsic은 calibration/camera_robot_transform.yaml의
#       Eye-to-Hand 값과 동일하게 배치 (object_pose 분포 일치 목적)
#     * intrinsics는 calibration/camera_intrinsics.yaml과 일치
#     * RGB + Depth 출력
# - 테이블 + 대상 물체:
#     * 물체 종류/USD는 미결정 (사용자 태스크 정의 후 확정)
#     * 작업 공간 범위(x, y, z bounds)도 미결정


# -----------------------------------------------------------------------------
# [3] 스크립트 기반 IK 전문가 (Expert Policy)
# -----------------------------------------------------------------------------
# - 4단계 waypoint 시퀀스:
#     1) home            : 홈 포즈
#     2) pre_grasp       : 물체 위 +10cm (approach)
#     3) grasp           : 물체 위치 + 그리퍼 close
#     4) lift            : grasp 지점 +10cm
#     5) place           : 목표 drop 위치
# - 물체 GT pose는 Isaac Sim API로 직접 읽어옴 (센서 추론 불필요)
# - IK 솔버: Lula IK 또는 cuRobo 중 택1 (미결정)
# - 각 waypoint 사이는 joint-space 선형 보간 + 30Hz 샘플링


# -----------------------------------------------------------------------------
# [4] Domain Randomization 적용
# -----------------------------------------------------------------------------
# - 매 에피소드 시작 시 config/dr_config.yaml을 로드하여 아래 항목 랜덤화:
#     * 시각: 조명 방향/강도/색온도, 물체·테이블·배경 텍스처, 카메라 FOV±
#     * 물리: 물체 질량·마찰, 모터 PD 게인, joint 마찰, latency
#     * 관측: joint_state 가우시안 노이즈, 이미지 blur/JPEG artifact,
#             depth의 hole/speckle noise (RealSense 특유)
# - 물체 초기 위치도 작업 공간 내에서 랜덤 샘플링


# -----------------------------------------------------------------------------
# [5] LeRobotDataset 저장
# -----------------------------------------------------------------------------
# - lerobot 0.4.4의 LeRobotDataset.create()로 생성
# - features 스펙 (README의 Part D 데이터셋 구조와 동일해야 Co-training 가능):
#     * observation.state        : (6,)  float32  - 팔로워 관절 위치
#     * observation.object_pose  : (3,)  float32  - base_link frame 3D 좌표
#                                                    (Sim에서는 GT, Real에서는
#                                                    YOLO 추론값. 의도된 도메인
#                                                    갭으로 Co-training이 보정)
#     * action                   : (6,)  float32  - 다음 step 목표 관절 위치
#     * observation.images.top   : RGB   uint8    - RealSense color 모사
# - 한 에피소드 = 한 번의 pick-and-place 성공 궤적
# - 에피소드 길이(프레임 수) 목표: 미결정 (대략 3~5초 × 30Hz = 90~150 frames 예상)
# - 총 에피소드 수: 약 1000 (사용자 조정 가능)


# -----------------------------------------------------------------------------
# [6] CLI 인자 (예정)
# -----------------------------------------------------------------------------
# --num-episodes      : 생성할 에피소드 수 (default 1000)
# --dataset-repo-id   : LeRobot 저장 경로/ID
# --dr-config         : config/dr_config.yaml 경로
# --so101-usd         : SO-101 USD 파일 경로
# --object-usd-dir    : 대상 물체 USD 디렉터리
# --seed              : 재현성 시드


# -----------------------------------------------------------------------------
# TODO (구현 전 사용자 확인 필요)
# -----------------------------------------------------------------------------
# [ ] Isaac Sim 버전 (API import 경로 결정)
# [ ] SO-101 USD 파일 보유 여부 및 경로
# [ ] IK 솔버 선택 (Lula IK / cuRobo)
# [ ] 대상 물체 종류 및 USD
# [ ] 작업 공간 범위 (x, y, z bounds)
# [ ] 에피소드 길이 목표 (프레임 수)
# [ ] camera_intrinsics.yaml 실측값 확보 여부