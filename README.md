# PAI — 환경 도메인 적응형 VLA 로봇 시스템

RGB 카메라와 **Gemma 4 E2B VLA 에이전트**로 환경 도메인(도서관·분리수거장 등)을 인식하고, 도메인별 **LoRA 어댑터**를 hot-swap하여 SO-101 로봇 팔을 제어한다. 중앙 `agent_node`가 함수 호출(`look_scene` → `execute_subtask` → `verify_result`) 루프로 전체 흐름을 오케스트레이션하며, Gemma는 자연어로 타겟을 묘사하고 별도 `embedding_node`가 이를 임베딩하여 **language-conditioned ACT Policy**의 observation 채널로 주입한다. Gemma 도메인 LoRA는 **PDF→QA→QLoRA→GGUF** 파이프라인으로, ACT Policy는 Isaac Sim 기반 **Sim+Real Co-training**으로 학습한다. Depth 정보는 ACT 학습에는 사용하지 않으며 `verify_result` 단계의 **grasp 성공 검증** 보조 신호로만 활용한다.

---

## 전체 파이프라인

```
Part A  하드웨어 세팅 (카메라 캘리브레이션)
  │
  ├─▶ Part B  Isaac Sim ACT 시연 데이터 생성 (외부 PC)
  │           스크립트 IK 전문가 + Domain Randomization
  │           pre_grasp → grasp → lift → place 4단계 waypoints
  │           multi-target 장면 구성 (같은 시각에 여러 후보 객체 배치)
  │           언어 라벨: Gemma 4 E2B 자체 라벨링 + 외부 LLM paraphrase 혼합
  │           → Sim LeRobotDataset (~1000 에피소드)
  │
  ├─▶ Part C  Gemma 도메인 LoRA 학습 (외부 PC)
  │     ├─ C-1  도메인별 PDF/문서 → LLM Q&A 자동 생성 → 필터링
  │              3종 데이터 혼합: 도메인 지식 / 함수 호출 / 검증 시나리오
  │     ├─ C-2  Unsloth QLoRA 파인튜닝 (예: RTX 5070급)
  │     └─ C-3  GGUF 변환 → adapter_<domain>.gguf (각 수십 MB)
  │              adapter_library.gguf, adapter_recycling.gguf, ...
  │
  ▼
Part D  실제 시연 데이터 수집 (Jetson, 축소 운영)
        teleop 기반 record_teleop.py (agent_node 미사용)
        → Real LeRobotDataset (50~100 에피소드)
        (observation.state + observation.language_embedding + observation.images.top + action)
  │
  └─▶ Part E  ACT Co-training (외부 PC, LeRobot 0.4.4 멀티 데이터셋)
                Sim + Real 혼합 학습 → act.pth
                    │
                    ▼
              Part F  통합 배포 (Sim 환경 + Jetson 실물 동일 노드 구성)
                      llama-server (Gemma 4 E2B base + LoRA hot-swap, systemd 상시)
                      + agent_node (오케스트레이션 함수 호출 루프)
                      + embedding_node (텍스트 → 벡터 임베딩 서비스)
                      + act_node
                    │
                    ▼
              Part G  검증
                      Sim rollout / Jetson 실물 모두 동일 노드 구성으로 평가
                      (학습에 쓰지 않은 DR 파라미터로 held-out)
```

> Part B와 Part C는 독립적으로 병렬 진행 가능. Part D는 ACT 학습 데이터 수집이
> 목적이므로 `agent_node` 없이 teleop 기반으로 진행한다.
>
> Sim과 Jetson 실물은 **동일 노드 구성**(camera_node, agent_node,
> llama_server_manager, embedding_node, act_node, robot_controller_node)으로
> 운영하며 검증도 같은 파이프라인에서 수행한다. Gemma LoRA는 도메인 1개만
> 적재되고, 사용자 명령 → 도메인 분류 → llama-server `/lora-adapter` HTTP
> 요청으로 hot-swap된다.
>
> ACT는 Sim-to-Real Co-training 방식으로 학습한다. Sim 측 Domain Randomization:
> 조명·텍스처·카메라 FOV, 질량·마찰·PD 게인, joint/이미지 노이즈.
> Sim action space는 실기와 동일하게 position control.

---

## 아키텍처 핵심 결정 사항

### Language-conditioned VLA 패턴 (RT-2 / OpenVLA / π0 계열)

`agent_node`의 `look_scene` 함수는 Gemma 4 E2B VLM에 RGB 이미지를 입력하여 **자연어 타겟 묘사**(예: "blue can on the right side of the table")를 출력한다. 이 텍스트는 `embedding_node`에서 벡터로 변환되어 `SubtaskCommand`에 실려 `act_node`로 전달되고, ACT는 이 임베딩을 observation 채널 중 하나로 사용한다. 이전 설계의 3D 좌표 산출 방식은 폐기되었다.

**이 결정의 결과**:
- `agent_node`는 depth/intrinsics/hand-eye calibration에 의존하지 않는다 (좌표 산출 책임 없음)
- ACT가 RGB + 언어 임베딩으로 perception을 직접 학습 (end-to-end 표현 학습)
- Sim/Real 라벨링 일관성 부담이 줄어든다 (좌표 정확도가 아닌 자연어 일치만 요구)
- depth는 ACT 학습 obs에서 제외되며, `verify_result`의 grasp 검증 보조 신호로만 사용된다

### 자율성 범위

한 태스크 진행 중에는 도메인이 고정되며, `agent_node`는 동일 도메인 내에서 `look_scene → execute_subtask → verify_result` 루프만 반복한다. 자율 도메인 재분류, 동적 함수 생성 등 자율성 확장은 향후 작업으로 유보한다. `verify_result`는 최대 3회까지 반복하며, 그 이상은 사용자 개입 대기 상태로 진입한다.

---

## 실행 환경

### Jetson (데이터 수집 / 배포)

| 항목 | 내용 |
|------|------|
| 하드웨어 | NVIDIA Jetson Orin Nano 8GB |
| JetPack | 6.2.1 |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.6 |
| ROS2 | Humble Hawksbill |
| Python | 3.10 |
| 빌드 시스템 | ament_python |

```
~/
├── ros2_ws/
│   └── src/
│       ├── robot_arm_msgs/        # 커스텀 메시지 패키지 (신규)
│       └── robot_arm_pkg/
│
├── models/
│   ├── gemma/
│   │   ├── gemma_4_e2b_q4.gguf   # base 모델, 상시 적재
│   │   └── adapters/              # 도메인별 LoRA, hot-swap
│   │       ├── adapter_library.gguf
│   │       └── adapter_recycling.gguf
│   ├── embedding/                 # 텍스트 임베딩 모델 (모델 미확정, TBD)
│   │   └── <model_files>          # 예: MiniLM-L6 / DistilBERT / CLIP text encoder
│   └── act/
│       └── act.pth                # ACT Policy 체크포인트 (단일)
│
├── calibration/
│   ├── camera_intrinsics.yaml
│   ├── camera_robot_transform.yaml
│   └── grasp_roi.yaml             # verify_result용 엔드이펙터 ROI 정의 (신규)
│
└── llama.cpp/
    └── llama-server               # systemd 서비스, --jinja 활성
```

### 외부 PC (학습 / Isaac Sim)

| 항목 | 내용 |
|------|------|
| OS | Ubuntu |
| GPU VRAM | 16GB 이상 |
| CUDA | 12.6 |
| 주요 작업 | Gemma 도메인 LoRA 학습 (Part C, Unsloth QLoRA), ACT 모방학습 (Part E), Isaac Sim 시뮬레이션 |

---

## 하드웨어

| 항목 | 내용 |
|------|------|
| 카메라 | Intel RealSense (RGB-D) |
| 로봇 팔 | SO-101 (HuggingFace LeRobot) |
| 리더 팔 | SO-101 Leader (데이터 수집 시 사용) |

> RealSense의 depth 스트림은 `verify_result`의 grasp 검증에만 사용된다.
> ACT 학습 obs에는 RGB만 포함된다.

---

## Jetson 환경 설치

저장소 루트의 스크립트로 환경을 구성합니다.

| 스크립트 | 대상 작업 |
|----------|-----------|
| `deploy/scripts/setup_jetson_base.sh` | 모방 데이터 수집 (Part D), 실제 로봇 구동 (Part F) |
| `deploy/scripts/setup_calibration.sh` | 카메라 캘리브레이션 (Part A) — base 설치 후 실행 |

```bash
# 1. 기본 환경 설치 (Part D, F 공통)
chmod +x deploy/scripts/setup_jetson_base.sh
./deploy/scripts/setup_jetson_base.sh

# 2. 캘리브레이션 환경 추가 설치 (Part A, 필요 시)
chmod +x deploy/scripts/setup_calibration.sh
./deploy/scripts/setup_calibration.sh
```

### 소프트웨어 의존성 상세

#### apt 패키지

| 패키지 | 설치 스크립트 | 용도 |
|--------|--------------|------|
| `python3-opencv`, `libopencv-dev` | base, calibration | OpenCV |
| `ros-humble-realsense2-camera/description` | base | RealSense 드라이버 |
| `ros-humble-message-filters` | base | 카메라 토픽 동기화 |
| `ros-humble-cv-bridge` | base | ROS2 ↔ OpenCV 변환 |
| `ros-humble-robot-state-publisher` | calibration | URDF → TF |
| `ros-humble-rqt`, `ros-humble-rqt-gui` | calibration | EasyHandeye2 GUI |
| `aruco_opencv` (소스 빌드) | calibration | ArUco 마커 검출 |
| `easy_handeye2` (소스 빌드) | calibration | Hand-Eye 캘리브레이션 |

#### Python 패키지 (`requirements.txt`)

| 패키지 | 버전 | 용도 |
|--------|------|------|
| setuptools | ≥58.2.0 | ROS2 Humble 빌드 호환 |
| numpy | ≥1.21.0 | 이미지/텐서 데이터 처리 |
| Pillow | - | 카메라 이미지 처리 |
| requests | ≥2.28.0 | llama-server HTTP 통신 (agent_node, llama_server_manager) |
| torch | ≥2.0.0 | ACT Policy 추론, SO-101 모터 제어, 임베딩 모델 추론 |
| lerobot | 0.4.4 | SO-101 SDK, LeRobotDataset, ACTPolicy |
| PyYAML | ≥6.0 | calibration / config YAML 파싱 |
| (TBD) 임베딩 라이브러리 | TBD | sentence-transformers / transformers — 임베딩 모델 확정 후 추가 |

---

## 저장소 구조

본 저장소는 **데이터 생성(sim) / 학습(training) / 배포(deploy)** 3가지 목적을
top-level 폴더로 분리하여 관리합니다.

```
PAI/
├── deploy/                                     # [배포 목적] Jetson에서 실행
│   ├── scripts/
│   │   ├── setup_jetson_base.sh                # 기본 환경 설치 (Part D, F)
│   │   └── setup_calibration.sh                # 캘리브레이션 환경 설치 (Part A)
│   ├── config/                                 # Jetson 런타임 설정
│   │   ├── domain_map.yaml                     # 사용자 명령 → 도메인 매핑
│   │   └── grasp_roi.yaml                      # verify_result용 엔드이펙터 ROI
│   ├── calibration/                            # 실기 캘리브레이션 결과
│   └── ros2_ws/
│       └── src/
│           ├── robot_arm_msgs/                 # 커스텀 메시지 패키지 (신규)
│           │   ├── msg/
│           │   │   ├── SubtaskCommand.msg      # agent_node → act_node
│           │   │   ├── ExecutionStatus.msg     # act_node → agent_node
│           │   │   ├── DomainSelection.msg     # agent_node → llama_server_manager
│           │   │   └── AgentLog.msg            # agent_node 디버그 로그
│           │   ├── srv/
│           │   │   └── EmbedText.srv           # agent_node → embedding_node (신규)
│           │   ├── CMakeLists.txt
│           │   └── package.xml
│           └── robot_arm_pkg/
│               ├── robot_arm_pkg/
│               │   ├── camera_node.py          # RGB-D 카메라 입력
│               │   ├── agent_node.py           # Gemma 함수 호출 오케스트레이션 (신규)
│               │   ├── embedding_node.py       # 텍스트 → 벡터 임베딩 서비스 (신규)
│               │   ├── llama_server_manager.py # llama-server LoRA hot-swap (신규)
│               │   ├── act_node.py             # ACT Policy 추론 (LeRobot .pth)
│               │   └── robot_controller_node.py# SO-101 팔로워 모터 제어 (LeRobot SDK)
│               ├── scripts/
│               │   └── record_teleop.py        # 데이터 수집 스크립트 (Part D, 신규)
│               ├── launch/
│               ├── resource/
│               ├── package.xml
│               ├── setup.py
│               └── requirements.txt
│
├── sim/                                        # [데이터 생성 목적] Isaac Sim 전용
│   ├── generate_act_dataset.py                 # Part B: ACT 시연 데이터 생성
│   ├── generate_language_labels.py             # Part B: Sim 에피소드 언어 라벨링 (신규)
│   └── config/
│       ├── dr_config.yaml                      # Domain Randomization 설정
│       └── multi_target_scenes.yaml            # multi-target 장면 구성 (신규)
│
├── training/                                   # [학습 목적] 외부 PC 전용
│   └── scripts/
│       ├── generate_qa_from_pdf.py             # Part C-1: 도메인 PDF → Q&A (도메인 지식)
│       ├── generate_function_call_qa.py        # Part C-1: 함수 호출 시나리오 합성 (신규)
│       ├── generate_verify_qa.py               # Part C-1: 검증 시나리오 합성 (신규)
│       ├── merge_qa_dataset.py                 # Part C-1: 3종 Q&A 혼합 (신규)
│       ├── train_gemma_qlora.py                # Part C-2: Unsloth QLoRA 학습
│       ├── convert_lora_to_gguf.sh             # Part C-3: GGUF 변환
│       └── train_act_cotraining.sh             # Part E: ACT Co-training
│
├── shared/                                     # 3목적 공통 자산 (예정: URDF 등)
├── docs/
└── README.md
```

> `deploy/ros2_ws/src/robot_arm_pkg/robot_arm_pkg/` 이중 폴더는 ament_python 패키지의
> 표준 레이아웃이며, 바깥쪽은 ROS2 패키지명 / 안쪽은 Python 모듈 경로입니다.

---

## ROS2 노드 및 토픽

### 배포 단계 노드 구성

| 노드 | 실행 주기 | 역할 |
|------|-----------|------|
| `camera_node` | 30 Hz | RGB-D 카메라 프레임 발행 |
| `agent_node` | 이벤트 (사용자 명령 + ACT 완료/실패) | Gemma 4 E2B 함수 호출 루프(`look_scene` → `execute_subtask` → `verify_result`) 오케스트레이션. llama-server HTTP 호출. `verify_result` 시 depth 기반 grasp 보조 검증 수행 |
| `embedding_node` | 이벤트 (Service 요청) | 자연어 타겟 묘사 → 벡터 임베딩 변환 (Service: `/embed_text`) |
| `llama_server_manager` | 이벤트 (`/domain_selection` 수신) | llama-server에 LoRA adapter hot-swap HTTP 요청 후 `/selected_domain` 발행 |
| `act_node` | 30 Hz (활성 시) | ACT Policy로 관절 시퀀스 예측. `SubtaskCommand` 수신 시 활성화, 종료/실패 시 `ExecutionStatus` 발행 |
| `robot_controller_node` | 1000 Hz | SO-101 팔로워 모터 구동 및 관절 상태 피드백 |

> **agent_node 트리거 이벤트**:
> - 외부 트리거: `/user_command` 수신 → 도메인 분류 → LoRA hot-swap → 함수 호출 루프 진입
> - 내부 트리거: `/execution_status` 수신 → 결과에 따라 `verify_result` 호출 → 다음 subtask 또는 종료

> **agent_node의 카메라 토픽 구독**: RGB는 `look_scene`용, depth는 `verify_result`의
> grasp 검증용. 둘 다 BestEffort + KeepLast(1) QoS로 latest-only 캐시.

> 이전 설계의 `vlm_node` / `yolo_node` / `state_monitor_node`는 `agent_node`의
> 함수 호출 툴(`look_scene`, `verify_result`)로 흡수되어 제거된다.

### 커스텀 메시지 (`robot_arm_msgs`)

| 메시지 | 주요 필드 | 발행자 → 구독자 |
|--------|----------|-----------------|
| `SubtaskCommand` | `string subtask_id`, `string domain`, `string target_description`, `float32[] language_embedding` | `agent_node` → `act_node` |
| `ExecutionStatus` | `string subtask_id`, `string status` (success/fail/timeout), `string failure_reason`, `float32 confidence` | `act_node` → `agent_node` |
| `DomainSelection` | `string domain`, `float32 scale` | `agent_node` → `llama_server_manager` |
| `AgentLog` | `string timestamp`, `string thought`, `string tool_called`, `string tool_args`, `string tool_result` | `agent_node` → 디버그 |

> **`SubtaskCommand` 변경 사항**: 이전 설계의 `string target_class` /
> `geometry_msgs/Point target_position` 필드는 제거되고, 자연어 묘사
> (`target_description`) + 사전 임베딩 벡터(`language_embedding`)로 대체된다.
> `agent_node`가 `embedding_node` Service를 호출하여 임베딩을 미리 받아 두고
> `act_node`로 전달하므로, `act_node`는 추론 시점에 임베딩 변환 지연이 없다.

### 커스텀 서비스 (`robot_arm_msgs`)

| 서비스 | 요청 | 응답 |
|--------|------|------|
| `EmbedText` | `string text` | `float32[] embedding`, `int32 dim` |

### 토픽 구조

```
사용자
  └─ /user_command (std_msgs/String) ──▶ [agent_node]

[camera_node]
  ├─ /camera/rgb   (sensor_msgs/Image, 30Hz) ──▶ [agent_node] (latest-only)
  │                                          ──▶ [act_node]   (ACT obs용)
  └─ /camera/depth (sensor_msgs/Image, 30Hz) ──▶ [agent_node] (latest-only, verify_result용)

[agent_node]
  ├─ /domain_selection (robot_arm_msgs/DomainSelection) ──▶ [llama_server_manager]
  ├─ /subtask_command  (robot_arm_msgs/SubtaskCommand)  ──▶ [act_node]
  ├─ /agent_log        (robot_arm_msgs/AgentLog)        ──▶ (디버그 콘솔/로그 파일)
  └─ Service Client: /embed_text                        ──▶ [embedding_node]

[embedding_node]
  └─ Service Server: /embed_text (robot_arm_msgs/EmbedText)

[llama_server_manager]
  └─ /selected_domain (std_msgs/String) ──▶ [agent_node] (스왑 완료 통보)

[act_node]
  ├─ /joint_command    (sensor_msgs/JointState)         ──▶ [robot_controller_node]
  └─ /execution_status (robot_arm_msgs/ExecutionStatus) ──▶ [agent_node]

[robot_controller_node] (1000Hz)
  └─ /joint_states (sensor_msgs/JointState) ──┬──▶ [act_node]
                                               └──▶ [agent_node]
```

> **QoS 정책**: 카메라 토픽(`/camera/rgb`, `/camera/depth`)은 BestEffort +
> KeepLast(1)로 구독한다. `agent_node`는 30Hz 스트림을 모두 처리하지 않고
> 함수 호출 시점의 최신 1프레임만 사용한다.

> **LoRA 스왑 트리거**: 현재 설계에서는 사용자 명령(`/user_command`)이
> 도메인 라벨로 매핑되어 Gemma LoRA가 hot-swap된다. 자율 재분류(예: 신뢰도
> 기반 도메인 재선택)는 향후 확장 항목으로 유보. 한 태스크 진행 중에는 도메인이
> 고정되며, `agent_node`는 동일 도메인 내에서 `look_scene → execute_subtask →
> verify_result` 루프만 반복한다. `verify_result`는 최대 3회까지 시도되며,
> 그 이상은 사용자 개입 대기 상태로 진입한다.

### `verify_result` 함수 명세

`verify_result`는 다음을 입력으로 받아 enum 결과를 출력한다.

**입력**:
- 현재 RGB 프레임 (`/camera/rgb` latest)
- 현재 depth 프레임 (`/camera/depth` latest)
- 원본 user_command 문자열
- 직전 subtask 정보 (subtask_id, target_description)

**처리**:
1. **Gemma 시각 검증**: RGB + 원본 명령을 Gemma에 입력 → 자연어 reasoning + status 후보
2. **Depth grasp 보조 검증**: 엔드이펙터 ROI(config의 `grasp_roi.yaml`) 내 depth 변화를 측정해 grasp 성공 여부 산출
3. **종합 판정**: 두 신호를 결합하여 단일 enum 산출

**출력**:
- `status`: `SUCCESS` / `FAIL` / `PARTIAL` / `UNCLEAR`
- `reasoning`: Gemma의 자연어 설명 + grasp 검증 결과 요약

> Gemma 시각 검증은 의미 수준의 판정(통에 잘 들어갔는가)을 담당하고,
> depth 검증은 결정론적 grasp 성공/실패 신호를 제공한다. 두 신호가 상충할 때의
> 정책(예: depth 우선 / Gemma 우선)은 운영 중 튜닝 대상.

---

## 빌드 및 실행

### 빌드

```bash
# 저장소 루트 기준 (개발 PC에서 확인용)
cd deploy/ros2_ws
colcon build --packages-select robot_arm_msgs robot_arm_pkg
source install/setup.bash

# Jetson 실기 배포 시: deploy/ros2_ws 내용을 Jetson의 ~/ros2_ws 로 복사 후 빌드
```

### 배포 실행 (Part F)

```bash
# llama-server (Gemma 4 E2B base, --jinja로 함수 호출 활성화)
# systemd 서비스로 상시 구동 권장
~/llama.cpp/llama-server \
  -m ~/models/gemma/gemma_4_e2b_q4.gguf \
  --jinja \
  --host 0.0.0.0 --port 8080

# 카메라
ros2 run robot_arm_pkg camera_node

# llama-server LoRA 스왑 매니저
ros2 run robot_arm_pkg llama_server_manager \
  --ros-args -p server_url:=http://localhost:8080 \
             -p adapter_dir:=~/models/gemma/adapters/

# 임베딩 노드 (텍스트 → 벡터 Service)
ros2 run robot_arm_pkg embedding_node \
  --ros-args -p model_path:=~/models/embedding/

# 에이전트 (Gemma 함수 호출 오케스트레이션)
ros2 run robot_arm_pkg agent_node \
  --ros-args -p server_url:=http://localhost:8080 \
             -p domain_map:=~/ros2_ws/config/domain_map.yaml \
             -p grasp_roi:=~/calibration/grasp_roi.yaml \
             -p max_verify_iterations:=3

# ACT Policy
ros2 run robot_arm_pkg act_node \
  --ros-args -p model_path:=~/models/act/act.pth \
             -p chunk_size:=100

# 로봇 컨트롤러 (SO-101 팔로워)
ros2 run robot_arm_pkg robot_controller_node \
  --ros-args -p robot_port:=/dev/ttyACM0 \
             -p robot_id:=follower

# 사용자 명령 발행
ros2 topic pub /user_command std_msgs/String "data: '쓰레기 좀 정리해줘'"
```

---

## 카메라-로봇 캘리브레이션 (Part A)

Hand-Eye 캘리브레이션 (Eye-to-Hand 방식): 카메라 고정, SO-101 엔드이펙터에 ArUco 마커 부착.

> 본 설계에서 ACT는 RGB만 사용하고 `agent_node`도 3D 좌표를 산출하지 않으므로,
> 캘리브레이션 정밀도 요구 수준은 이전 설계 대비 낮아진다. 다만 카메라 위치의
> **일관성**(학습 시점과 배포 시점 간)이 중요하므로 캘리브레이션 자체는 여전히 수행한다.
> `verify_result`의 grasp ROI 정의에 카메라-로봇 변환이 사용된다.

### Jetson 사전 설치

```bash
# setup_jetson_base.sh 실행 후 아래 스크립트 실행
./deploy/scripts/setup_calibration.sh
```

aruco_opencv, easy_handeye2 소스 빌드 및 관련 apt 패키지가 자동으로 설치됩니다.

### SO-101 URDF 배치

```bash
# LeRobot 저장소에서 SO-101 URDF 복사
# 경로 예시 (LeRobot 설치 위치에 따라 다를 수 있음):
#   ~/.local/lib/python3.10/site-packages/lerobot/robots/so101/urdf/
cp <lerobot_urdf_path>/so101.urdf ~/ros2_ws/src/robot_arm_pkg/urdf/so101.urdf

# setup.py의 urdf 주석 해제 후 재빌드
cd ~/ros2_ws && colcon build --packages-select robot_arm_pkg
```

> URDF의 엔드이펙터 링크 이름을 확인하여 `calibration.launch.py`의
> `robot_effector_frame` 파라미터와 일치시킬 것.

### 캘리브레이션 실행

```bash
# 패키지 빌드 및 소스
cd ~/PAI/deploy/ros2_ws
colcon build --packages-select robot_arm_pkg
source install/setup.bash

# 캘리브레이션 launch (마커 크기 5cm 기준)
ros2 launch robot_arm_pkg calibration.launch.py \
  marker_id:=0 \
  marker_size:=0.05 \
  robot_port:=/dev/ttyACM0

# 별도 터미널: rqt GUI에서 EasyHandeye2 플러그인 실행
ros2 run rqt_gui rqt_gui --force-discover
```

### 캘리브레이션 절차

1. SO-101 엔드이펙터에 ArUco 마커(ID=0, 5cm) 부착
2. rqt → EasyHandeye2 플러그인 열기
3. 로봇을 수동으로 이동 → 카메라가 마커를 인식하면 **Take Sample** 클릭
4. 10~15개 포즈에서 반복 (포즈 간 충분한 회전 변화 필요)
5. **Compute** → 변환 행렬 확인 → **Save**
6. 결과 파일 복사:

```bash
cp ~/.ros/easy_handeye2/eye_to_hand_calibration.yaml \
   ~/calibration/camera_robot_transform.yaml

# camera_intrinsics는 RealSense camera_info 토픽에서 추출
ros2 topic echo /camera/color/camera_info --once > ~/calibration/camera_intrinsics.yaml
```

### `grasp_roi.yaml` 작성

`verify_result`의 depth 기반 grasp 보조 검증에 사용되는 ROI를 정의한다.
SO-101 엔드이펙터 위치 근처에서 grasp 성공 시 depth 값이 변화하는 영역을 측정하여
설정한다. 카메라 frame 기준 좌표를 사용한다.

```yaml
# ~/calibration/grasp_roi.yaml (예시)
roi:
  frame: camera        # camera | base_link
  x_range: [-0.05, 0.05]   # m
  y_range: [-0.05, 0.05]
  z_range: [0.20, 0.40]
depth_change_threshold: 0.02  # m, 이 이상의 변화가 있으면 grasp 성공으로 간주
```

---

## 데이터 수집 (Part D)

ACT 학습용 시연 데이터를 teleop으로 수집한다. `agent_node` 및 LoRA는 사용하지 않는다.

### 실행 전 준비

1. SO-101 리더/팔로워 USB 포트 확인
2. LeRobot 캘리브레이션 완료 (`lerobot-calibrate`)
3. 카메라 캘리브레이션 결과 (`camera_intrinsics.yaml`, `camera_robot_transform.yaml`) 배치
4. 임베딩 모델 결정 및 배치 (Part C/E와 동일 모델 사용 필수)

### 실행 순서

```bash
# 1. RealSense 드라이버 실행
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# 2. camera_node 실행 (/camera/rgb 발행)
ros2 run robot_arm_pkg camera_node

# 3. embedding_node 실행 (수집 시 자연어 라벨 → 임베딩 변환에 사용)
ros2 run robot_arm_pkg embedding_node \
  --ros-args -p model_path:=~/models/embedding/

# 4. teleop 데이터 수집 스크립트 실행
python ~/ros2_ws/src/robot_arm_pkg/scripts/record_teleop.py \
  --robot-port      /dev/ttyACM0 \
  --robot-id        follower \
  --teleop-port     /dev/ttyACM1 \
  --teleop-id       leader \
  --dataset-repo-id myname/dataset_real \
  --single-task     "pick and place" \
  --num-episodes    50
```

> **언어 라벨 부여 방법 (확정 필요 — 3가지 옵션)**:
>
> - Option A: 에피소드 시작 시 작업자가 자유 텍스트 입력 (다양성↑, 일관성↓)
> - Option B: `--single-task` CLI 인자로 전체 통일 (일관성↑, 다양성↓)
> - Option C: 도메인별 prompt 템플릿 + 작업자가 변수만 입력 (균형형, 권장)
>
> 결정 후 `record_teleop.py`에 해당 방식 구현. 학습 시 일관성을 위해 같은
> 작업의 라벨이 너무 다르지 않도록 가이드라인 필요.

> **임베딩 일관성**: 수집 단계에서 사용한 임베딩 모델은 Part C/E의 학습 단계와
> Part F의 배포 단계에서 **반드시 동일한 모델**이어야 한다. 임베딩 모델 변경 시
> 전체 데이터셋 재라벨링 필요.

### 데이터셋 구조

| 채널 | shape / dtype | 내용 |
|------|-------|------|
| `observation.state` | (6,) float32 | 팔로워 관절 위치 (SO-101 6축) |
| `observation.language_embedding` | (D,) float32 | 자연어 타겟 묘사의 임베딩 벡터. D는 임베딩 모델에 따라 결정 |
| `observation.images.top` | (H,W,3) uint8 | RealSense RGB (`/camera/color/image_raw`) |
| `action` | (6,) float32 | 리더 관절 값 (정답 행동) |

> 이전 설계의 `observation.target_position` 채널은 제거되었다. depth는
> `verify_result` 단계에서만 사용되고 ACT 학습 obs에 포함되지 않는다.

> Sim 데이터셋(`dataset_sim`)도 동일 feature 스펙 + 동일 임베딩 모델을 따라야
> Part E Co-training이 가능하다. 임베딩 차원 D가 일치하지 않으면 학습이 실패한다.

---

## 학습 파이프라인 — Gemma 도메인 LoRA (Part C)

### C-1: Q&A 데이터 생성 (3종 혼합)

도메인별로 **3개 카테고리의 Q&A**를 생성하고 혼합하여 LoRA 학습에 사용한다.
LoRA는 학습 데이터의 분포를 그대로 모방하므로, 배포 시 `agent_node`가 요구하는
3가지 작업 형식(도메인 지식 답변 / 함수 호출 / 시각 검증)을 모두 학습 데이터에
반영해야 한다.

| 카테고리 | 목적 | 형식 | 생성 방법 |
|---|---|---|---|
| **Cat-A: 도메인 지식 Q&A** | LoRA의 도메인 적응 | 자연어 Q&A | PDF/문서에서 추출 (`generate_qa_from_pdf.py`) |
| **Cat-B: 함수 호출 시나리오** | `look_scene`/`execute_subtask` 정확도 | 사용자 명령 → JSON 함수 호출 | 외부 LLM으로 합성 (`generate_function_call_qa.py`) |
| **Cat-C: 검증 시나리오** | `verify_result` 정확도 | 이미지 + 명령 → enum + reasoning | Sim 렌더링 + 자동 라벨링 (`generate_verify_qa.py`) |

#### 비율 옵션 (확정 미정)

학습 후 평가 결과를 보며 조정. 다음 3가지 후보 중 선택:

| 옵션 | Cat-A | Cat-B | Cat-C | 특징 |
|------|-------|-------|-------|------|
| **균형형** | 50% | 30% | 20% | 표준, 도메인 지식 우선 |
| **함수호출 강화형** | 40% | 40% | 20% | 함수 호출 정확도 우선 |
| **검증 강화형** | 40% | 30% | 30% | grasp/verify 안정성 우선 |

#### Cat-A 예시 (도메인 지식)

```json
{
  "messages": [
    {"role": "user", "content": "유리병은 어떻게 분리수거하나요?"},
    {"role": "assistant", "content": "유리병은 내용물을 비우고 깨끗이 헹군 후 유리류 수거함에 넣습니다. 깨진 유리는 신문지로 싸서 일반 쓰레기로 배출하세요."}
  ]
}
```

#### Cat-B 예시 (함수 호출)

```json
{
  "messages": [
    {"role": "system", "content": "사용 가능 함수: look_scene(prompt), execute_subtask(target_description), verify_result(original_command)"},
    {"role": "user", "content": "쓰레기 좀 정리해줘"},
    {"role": "assistant", "content": "{\"function\": \"look_scene\", \"args\": {\"prompt\": \"분리수거할 물체를 찾아주세요\"}}"}
  ]
}
```

#### Cat-C 예시 (검증)

```json
{
  "messages": [
    {"role": "user", "content": "[image] 사용자 명령: '캔을 캔류 통에 넣어줘'. 작업이 완료되었나요?"},
    {"role": "assistant", "content": "{\"status\": \"SUCCESS\", \"reasoning\": \"파란색 캔류 수거함 안에 알루미늄 캔이 들어 있음\"}"}
  ]
}
```

### C-2: Unsloth QLoRA 파인튜닝

```bash
python training/scripts/train_gemma_qlora.py \
  --base-model google/gemma-4-e2b \
  --dataset training/data/qa_recycling_merged.jsonl \
  --output-dir training/outputs/lora_recycling \
  --rank 16 --alpha 32 \
  --epochs 3
```

### C-3: GGUF 변환

```bash
bash training/scripts/convert_lora_to_gguf.sh \
  training/outputs/lora_recycling \
  models/gemma/adapters/adapter_recycling.gguf
```

---

## 학습 파이프라인 — Sim 데이터 생성 (Part B)

### Multi-target 장면 구성

ACT가 언어 임베딩을 무시하고 시각 정보에만 의존하는 **shortcut learning**을
방지하기 위해, 같은 시각 장면에서 **다른 언어 명령에 따라 다른 행동을 하는**
에피소드를 의도적으로 포함한다.

**구성 원칙**:
- 한 장면에 2개 이상의 후보 객체 배치 (예: 캔 1개 + 페트병 1개 + 종이 1개)
- 같은 객체 배치에 대해 다른 타겟 명령으로 여러 에피소드 생성 (예: "캔 집어" / "페트병 집어")
- 카메라 시점도 일부 공유하여 RGB 입력만으로는 정답을 구분할 수 없게 함

`sim/config/multi_target_scenes.yaml`로 장면 템플릿 정의:

```yaml
# sim/config/multi_target_scenes.yaml (예시)
scenes:
  - name: recycling_table_3objects
    objects:
      - {class: can,    color: [red, blue, green],  count: 1}
      - {class: bottle, color: [transparent, green], count: 1}
      - {class: paper,  color: [white],              count: 1}
    target_rotation:
      - "pick up the {color} {class}"
      - "grab the {class} on the table"
multi_target_ratio: 0.6   # 전체 에피소드 중 multi-target 장면 비율
```

### 언어 라벨링 (Gemma + 외부 LLM 혼합)

각 Sim 에피소드에 다양한 자연어 라벨을 부여한다. 두 가지 소스를 혼합:

**소스 1: Gemma 4 E2B 자체 라벨링**
- Sim에서 렌더링한 RGB 이미지를 Gemma 4 E2B에 입력
- 배포 시 사용할 동일 모델로 라벨 생성 → 분포 일치
- `sim/generate_language_labels.py`가 일괄 처리

**소스 2: 외부 LLM Paraphrase**
- 소스 1의 라벨을 외부 LLM(GPT-4 등)에 입력하여 다양한 표현 생성
- 표현 robustness 강화 (배포 시 사용자가 어떻게 말할지 모름)
- 같은 에피소드에 여러 paraphrase를 부여하고 학습 시 무작위 추출

**비율**: 에피소드당 Gemma 라벨 1개 + 외부 LLM paraphrase 2~3개. 학습 시
augmentation으로 무작위 추출.

### 데이터 생성 스크립트

```bash
# 1. ACT 시연 데이터 + multi-target 장면 생성
python sim/generate_act_dataset.py \
  --dr-config sim/config/dr_config.yaml \
  --scenes sim/config/multi_target_scenes.yaml \
  --num-episodes 1000 \
  --output dataset_sim/

# 2. 각 에피소드에 Gemma + 외부 LLM 라벨 부여
python sim/generate_language_labels.py \
  --dataset dataset_sim/ \
  --gemma-server http://localhost:8080 \
  --external-llm gpt-4 \
  --paraphrase-per-episode 3
```

### Domain Randomization 항목

조명·텍스처·카메라 FOV / 질량·마찰·PD 게인 / joint·이미지 노이즈.
depth는 ACT 학습에 사용되지 않으므로 depth 노이즈 DR은 제거됨.

---

## 학습 파이프라인 — ACT Co-training (Part E)

LeRobot 0.4.4의 멀티 데이터셋 가중치 기능을 사용하여 Sim + Real 혼합 학습.

```bash
bash training/scripts/train_act_cotraining.sh
```

스크립트 내부 (의사 코드):
```python
datasets = [
    LeRobotDataset("dataset_sim",  weight=0.7),
    LeRobotDataset("dataset_real", weight=0.3),
]
# observation.language_embedding 채널이 두 데이터셋 모두에 존재해야 함
# 임베딩 차원 D 일치 필수
```

**학습 입력 obs**:
- `observation.state`
- `observation.language_embedding`
- `observation.images.top`

**학습 타겟**:
- `action` (chunk_size=100 시퀀스)

> Sim/Real의 `language_embedding` 차원과 임베딩 모델이 다르면 학습 불가.
> Part D 수집 시점부터 동일 모델 사용 보장.

---

## Jetson 메모리 배분 (Orin Nano 8GB)

| 컴포넌트 | RAM | 비고 |
|----------|-----|------|
| OS + ROS2 | ~1.5 GB | |
| Gemma 4 E2B base (GGUF Q4) | ~3.0 GB | llama-server 상주 |
| **도메인 LoRA (active 1개)** | ~수십 MB | **hot-swap — 동시 적재 금지** |
| 임베딩 모델 (TBD) | ~50–300 MB | 모델 미확정. MiniLM(~80MB) ~ DistilBERT/CLIP(~250MB) |
| ACT Policy | ~0.3 GB | 상주 |
| RGB-D 드라이버 + 버퍼 | ~0.3 GB | |
| KV Cache / 기타 | ~1.0 GB | Gemma 컨텍스트 윈도우 |
| **합계** | **~6.2–6.4 GB** | 도메인 수가 늘어도 합계 불변 |

> Gemma LoRA는 llama-server에 HTTP `/lora-adapter` 요청으로 hot-swap (서브-초 latency).
> 한 태스크가 끝나는 시점(verify_result 직후)에 스왑하면 안정적이다.

---

## 미구현 항목 (TODO)

| 파일 | 미구현 항목 | 선행 조건 |
|------|------------|----------|
| `robot_arm_msgs/` | 패키지 신규 생성 (`SubtaskCommand`, `ExecutionStatus`, `DomainSelection`, `AgentLog` 메시지 + `EmbedText` 서비스) | 메시지/서비스 스키마 확정 |
| `agent_node.py` | 신규 작성. Gemma 함수 호출 루프(`look_scene`/`execute_subtask`/`verify_result`), llama-server HTTP 클라이언트, 사용자 명령→도메인 분류, `verify_result` 내부 depth grasp 검증, `embedding_node` Service 호출, max_iteration=3 | Part C, llama-server 구동, embedding_node |
| `embedding_node.py` | 신규 작성. `/embed_text` Service 제공. 임베딩 모델 로드 및 추론 | 임베딩 모델 확정 |
| `llama_server_manager.py` | 신규 작성. `/domain_selection` 수신 → llama-server `/lora-adapter` HTTP 요청 → `/selected_domain` 발행 | adapter GGUF 준비 |
| `act_node.py` | `_load_policy()`, `_predict_chunk()`, `SubtaskCommand` (language_embedding) 입력 처리, `ExecutionStatus` 발행 | Part E 완료 |
| `camera_node.py` | ~~`PlaceholderAdapter.get_frames()`~~ 구현 완료 | - |
| `record_teleop.py` | 신규 작성. 자연어 라벨 입력 + 임베딩 채널 기록. 라벨링 방식(Option A/B/C) 결정 후 구현 | embedding_node, 라벨링 방식 결정 |
| `setup.py` | `entry_points` 갱신 — 추가: `agent_node`, `embedding_node`, `llama_server_manager` / 제거: `vlm_node`, `yolo_node`, `state_monitor_node` | 신규 노드 작성 |
| `sim/generate_act_dataset.py` | IK 전문가 + DR + multi-target 장면 기반 시연 생성 | SO-101 USD, IK 솔버 결정 |
| `sim/generate_language_labels.py` | 신규 작성. Gemma 자체 라벨링 + 외부 LLM paraphrase 합성 | Gemma 4 E2B 추론 환경 |
| `sim/config/dr_config.yaml` | DR 파라미터 정의 (조명·텍스처·카메라 FOV·질량·마찰·PD 게인·이미지 노이즈) | - |
| `sim/config/multi_target_scenes.yaml` | 장면 템플릿 정의 (다중 객체 배치, 타겟 rotation) | 도메인 객체 셋 결정 |
| `training/scripts/generate_qa_from_pdf.py` | 신규 작성. 도메인 PDF → Cat-A 도메인 지식 Q&A | 도메인 문서 수집 |
| `training/scripts/generate_function_call_qa.py` | 신규 작성. 외부 LLM 합성 Cat-B 함수 호출 시나리오 | 함수 시그니처 확정 |
| `training/scripts/generate_verify_qa.py` | 신규 작성. Sim 렌더링 + 자동 라벨링으로 Cat-C 검증 시나리오 | Sim 환경, ground truth 라벨러 |
| `training/scripts/merge_qa_dataset.py` | 신규 작성. 3종 Q&A를 비율 옵션에 따라 혼합 | Cat-A/B/C 데이터 |
| `training/scripts/train_gemma_qlora.py` | 신규 작성. Unsloth QLoRA 학습 스크립트 | 혼합 Q&A 데이터셋 |
| `training/scripts/convert_lora_to_gguf.sh` | 신규 작성. HF LoRA → GGUF 변환 (`convert_lora_to_gguf.py`) | llama.cpp 빌드 |
| `training/scripts/train_act_cotraining.sh` | LeRobot 멀티 데이터셋 가중치 학습 (Sim+Real) | LeRobot 멀티 데이터셋 가중치 지원 확인, language_embedding 채널 검증 |
| `deploy/config/grasp_roi.yaml` | 엔드이펙터 ROI 정의 (carrera frame 기준) | 캘리브레이션 완료, 엔드이펙터 측정 |

> **제거 예정 노드/스크립트**: `vlm_node.py`, `yolo_node.py`, `state_monitor_node.py`,
> `scripts/record_with_vlm.py` — `agent_node` 함수 호출 툴 및 `record_teleop.py`로 대체.

> **이전 설계 대비 추가 제거 항목**: `agent_node`의 3D 좌표 산출 로직,
> `observation.target_position` 채널, `SubtaskCommand`의 `target_class` /
> `target_position` 필드, depth 노이즈 DR 항목.

---

## 미확정 항목 요약

다음 항목은 향후 결정 후 README 갱신이 필요하다.

| # | 항목 | 영향 범위 |
|---|------|---------|
| 1 | 임베딩 모델 선택 (MiniLM-L6 / DistilBERT / CLIP text encoder / T5-small 등) | `embedding_node`, ACT obs 차원 D, requirements.txt, 메모리 배분 |
| 2 | LoRA Q&A 비율 (균형형 / 함수호출 강화형 / 검증 강화형) | Part C-1 `merge_qa_dataset.py` |
| 3 | Part D 언어 라벨링 방식 (Option A 자유 입력 / B CLI 통일 / C 템플릿+변수) | `record_teleop.py` |
| 4 | Gemma 시각 검증과 depth grasp 검증의 종합 정책 (depth 우선 / Gemma 우선 / 가중 평균) | `agent_node.verify_result` |

---
