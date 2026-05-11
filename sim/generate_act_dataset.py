# =============================================================================
# generate_act_dataset.py
# -----------------------------------------------------------------------------
# 목적: Isaac Sim에서 ACT Co-training용 시연 데이터(Sim dataset)를 생성한다.
#       Part B 단계. 결과물은 Part D의 Real dataset과 함께 Part E에서
#       LeRobot 0.4.4 멀티 데이터셋 방식으로 공동 학습된다.
#
# 전체 파이프라인 내 위치:
#   Part B (이 스크립트 + generate_language_labels.py)
#                          →  Sim LeRobotDataset (~1000 에피소드)
#   Part D                 →  Real LeRobotDataset (50~100 에피소드)
#   Part E                 →  ACT Co-training (Sim + Real)
#
# 실행 환경: 외부 PC (Ubuntu, GPU VRAM 16GB+, Isaac Sim 설치됨)
#
# 본 스크립트는 시연 궤적과 시각 관측을 생성한다. 자연어 라벨은 후속 단계인
# generate_language_labels.py가 별도로 부여한다 (Gemma 자체 라벨링 + 외부 LLM
# paraphrase 혼합).
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
# [2] 씬 구성 (multi-target 포함)
# -----------------------------------------------------------------------------
# - SO-101 로봇 팔 로드:
#     * 저장소에 URDF와 STL 자산이 포함됨:
#         deploy/ros2_ws/src/robot_arm_pkg/urdf/so101.urdf
#         deploy/ros2_ws/src/robot_arm_pkg/urdf/assets/*.stl
#       (setup.py가 share에도 복사하도록 활성화되어 있음)
#     * Isaac Sim에서 사용하려면 URDF Importer로 USD 변환 1회 필요.
#     * joint 6개, end-effector link 이름 확인 필요
#     * Action space는 position control (실기 SO-101과 반드시 일치)
# - RealSense 모사 카메라:
#     * intrinsics는 deploy/calibration/camera_intrinsics.yaml과 일치
#       (Part A 캘리브레이션 후 산출되며 Jetson에서는 ~/calibration/ 하위로 배치)
#     * RGB 출력 (depth는 ACT 학습 obs에 포함되지 않음)
#     * 카메라 위치는 학습/배포 일관성을 위해 일정하게 유지하되, DR로 약간의
#       offset을 준다 (실기 캘리브레이션 정밀도 요구가 낮아진 만큼 분포는 폭넓게)
# - 테이블 + multi-target 객체 배치:
#     * sim/config/multi_target_scenes.yaml의 장면 템플릿에 따라 2개 이상의
#       후보 객체를 배치 (예: 캔 + 페트병 + 종이)
#     * 같은 객체 배치에 대해 다른 타겟 명령으로 여러 에피소드 생성하여
#       ACT가 RGB shortcut learning으로 정답을 맞히지 못하게 함 (README Part B)
#     * multi_target_ratio (예: 0.6)에 따라 일반 단일 객체 장면과 혼합


# -----------------------------------------------------------------------------
# [3] 스크립트 기반 IK 전문가 (Expert Policy)
# -----------------------------------------------------------------------------
# - 4단계 waypoint 시퀀스 (README Part B 참조):
#     1) pre_grasp       : 물체 위 +10cm (approach)
#     2) grasp           : 물체 위치 + 그리퍼 close
#     3) lift            : grasp 지점 +10cm
#     4) place           : 목표 drop 위치
# - 물체 GT pose는 Isaac Sim API로 직접 읽어옴 (센서 추론 불필요)
# - IK 솔버: Lula IK 또는 cuRobo 중 택1 (미결정)
# - 각 waypoint 사이는 joint-space 선형 보간 + 30Hz 샘플링
# - multi-target 장면에서는 에피소드 단위 타겟 객체를 한 개 선택하여 그것에
#   대해서만 grasp 시퀀스를 실행한다.


# -----------------------------------------------------------------------------
# [4] Domain Randomization 적용
# -----------------------------------------------------------------------------
# - 매 에피소드 시작 시 sim/config/dr_config.yaml을 로드.
# - 적용 항목 (README Part B 명시):
#     * 시각: 조명·텍스처·카메라 FOV
#     * 물리: 물체 질량·마찰, 모터 PD 게인
#     * 관측: joint 노이즈, 이미지 노이즈
# - depth 관련 DR 항목은 제거되었다 (depth는 ACT 학습 obs에서 빠짐).
# - 물체 초기 위치/yaw도 작업 공간 내에서 랜덤 샘플링.


# -----------------------------------------------------------------------------
# [5] LeRobotDataset 저장
# -----------------------------------------------------------------------------
# - lerobot 0.4.4의 LeRobotDataset.create()로 생성.
# - features 스펙 (Part D record_teleop.py와 반드시 일치 — Co-training 가능 조건):
#     * observation.state              : (6,)   float32  - 팔로워 관절 위치
#     * observation.language_embedding : (D,)   float32  - 자연어 타겟 묘사 임베딩
#                                                            (본 스크립트에서는 0벡터로 placeholder
#                                                            생성, 이후 generate_language_labels.py
#                                                            가 실제 임베딩으로 채움)
#     * observation.images.top         : (H,W,3) uint8   - RealSense color 모사
#     * action                         : (6,)   float32  - 다음 step 목표 관절 위치
# - 임베딩 차원 D는 generate_language_labels.py가 사용하는 embedding_node 모델에
#   의해 결정되며, Part D/F와 동일해야 한다. 본 스크립트 단계에서는 D를 결정할 수
#   없으므로 placeholder zero 벡터의 길이만이라도 동일 D로 맞추도록 후속 라벨링
#   스크립트가 features를 갱신하거나 사전에 D를 인자로 받아 생성한다 (방식 미결정).
# - 한 에피소드 = 한 번의 pick-and-place 성공 궤적.
# - 에피소드 길이(프레임 수) 목표: 미결정 (대략 3~5초 × 30Hz = 90~150 frames 예상).
# - 총 에피소드 수: 약 1000 (사용자 조정 가능).


# -----------------------------------------------------------------------------
# [6] CLI 인자 (예정)
# -----------------------------------------------------------------------------
# --num-episodes      : 생성할 에피소드 수 (default 1000)
# --dataset-repo-id   : LeRobot 저장 경로/ID (또는 --output 경로)
# --dr-config         : sim/config/dr_config.yaml 경로
# --scenes            : sim/config/multi_target_scenes.yaml 경로
# --so101-usd         : SO-101 USD 파일 경로
# --object-usd-dir    : 대상 물체 USD 디렉터리
# --seed              : 재현성 시드


# -----------------------------------------------------------------------------
# 미구현 / 결정 필요 항목
# -----------------------------------------------------------------------------
# [ ] Isaac Sim 버전 (API import 경로 결정)
# [ ] SO-101 USD (저장소의 URDF/STL에서 변환 1회 수행 + 산출 경로 확정)
# [ ] IK 솔버 선택 (Lula IK / cuRobo)
# [ ] 대상 물체 종류 및 USD (multi_target_scenes.yaml의 class 항목과 매칭)
# [ ] 작업 공간 범위 (x, y, z bounds)
# [ ] 에피소드 길이 목표 (프레임 수)
# [ ] camera_intrinsics.yaml 실측값 확보 여부
# [ ] language_embedding 채널의 D를 사전 인자로 받을지 / 후속 라벨링 단계에서
#     features를 갱신할지 정책 결정
