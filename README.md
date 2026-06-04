# PAI — 환경 도메인 적응형 VLA 로봇 시스템

RGB 카메라와 **Gemma 4 E2B VLA 에이전트**로 환경 도메인(도서관·분리수거장 등)을 인식하고, 도메인별 **LoRA 어댑터**를 hot-swap하여 SO-101 로봇 팔을 제어한다. 중앙 `agent_node`가 함수 호출(`look_scene` → `execute_subtask` → `verify_result`) 루프로 전체 흐름을 오케스트레이션하며, Gemma는 자연어로 타겟을 묘사하고 별도 `embedding_node`가 이를 임베딩하여 **language-conditioned ACT Policy**의 observation 채널로 주입한다. Gemma 도메인 LoRA는 **PDF→QA→QLoRA→GGUF** 파이프라인으로, ACT Policy는 Isaac Sim 기반 **Sim+Real Co-training**으로 학습한다. Depth 정보는 ACT 학습에는 사용하지 않으며 `verify_result` 단계의 **grasp 성공 검증** 보조 신호로만 활용한다.

---

## 저장소 구조

본 저장소는 **데이터 생성(sim) / 학습(training) / 배포(deploy)** 3가지 목적을
top-level 폴더로 분리하여 관리합니다.

```
PAI/
├── deploy/                                     # [배포/검증] Jetson — 자체 완결 (deploy/만 clone으로 검증 가능)
│   ├── scripts/
│   │   ├── setup_jetson_base.sh                # 기본 환경 설치 (데이터 수집 / 배포)
│   │   └── setup_calibration.sh                # 캘리브레이션 환경 설치
│   ├── config/                                 # Jetson 런타임 설정
│   │   ├── domain_map.yaml                     # 사용자 명령 → 도메인 매핑
│   │   ├── label_vocab.yaml                    # 데이터 수집 / function_call / Sim 공유 vocabulary
│   │   ├── grasp_roi.yaml                      # verify_result용 엔드이펙터 ROI (grasp 검증 — 미래 확장)
│   │   └── destination_roi.yaml                # verify_result용 목적지 ROI (place 검증 — 현재 운영)
│   ├── calibration/                            # 실기 캘리브레이션 결과
│   ├── systemd/
│   │   └── llama-server.service                # llama-server 상시 구동 systemd 유닛
│   ├── sysctl.d/
│   │   └── 99-llama-server-vm.conf             # 커널 VM 튜닝 (llama-server 메모리)
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
│               │   ├── act_node.py             # ACT Policy 추론 (LeRobot pretrained_model 디렉터리)
│               │   ├── robot_controller_node.py# SO-101 팔로워 모터 제어 (LeRobot SDK)
│               │   ├── prompts.py              # 시스템 프롬프트 + ExecStatus enum (SSoT)
│               │   └── utils/                  # home_pose.py / ros_image.py 공통 유틸
│               ├── scripts/
│               │   └── measure_roi.py          # depth ROI 실측 보조 (--mode destination|grasp)
│               ├── launch/
│               ├── urdf/                       # SO-101 URDF + STL 자산 (저장소 포함)
│               │   ├── so101.urdf
│               │   └── assets/                 # *.stl 메쉬 자산
│               ├── resource/
│               ├── package.xml
│               ├── setup.py
│               ├── setup.cfg
│               └── requirements.txt
│
├── sim/                                        # Isaac Sim 자산 + teleop 인프라
│   ├── assets/                                 # URDF→USD 변환 자산
│   │   ├── so101_new_calib/                    # SO-101 USD (Isaac Sim 임포트 결과)
│   │   └── objects/recycling/                  # YCB 객체 래퍼 (생성물 — gitignore)
│   ├── scenes/                                 # 씬 USD
│   │   └── base_scene.usd                      # base 씬
│   ├── config/
│   │   ├── dr_config.yaml                      # Domain Randomization 설정
│   │   └── multi_target_scenes.yaml            # multi-target 장면 구성
│   ├── scene_setup/                            # Isaac Sim Script Editor 씬 저작 스크립트
│   │   ├── setup_sim_camera.py                 # RGB 카메라 셋업 (캘리브레이션→USD)
│   │   ├── setup_recycling_bins.py             # 분리수거 쓰레기통 3개 생성
│   │   ├── setup_objects.py                    # 도메인 객체 래퍼 USD 생성 (recycling=YCB, tableware=SimReady)
│   │   ├── setup_physics_scene.py              # PhysicsScene 솔버 파라미터 튜닝
│   │   ├── setup_so101_gripper_physics.py      # 그리퍼 핑거 collision + PhysicsMaterial
│   │   ├── measure_placement.py                # placement 실측 (table_top_z·bin_exclusion)
│   │   └── probe_reach.py                      # 로봇 가동범위(reach) 실측
│   └── teleop/                                 # Isaac Sim teleop 인프라
│       ├── lerobot_to_rosbridge.py             # 실물 leader → EC2 rosbridge 브리지 (클라우드)
│       ├── lerobot_to_ros2.py                  # 실물 leader → ROS2 직접 publish 브리지 (로컬)
│       ├── sim_cloud_teleop_setting.md         # 클라우드(EC2) Isaac Sim teleop 셋업 가이드
│       └── sim_local_teleop_setting.md         # 로컬 Isaac Sim teleop 셋업 가이드
│
├── training/                                   # [학습: 모방데이터 + LoRA 파인튜닝 + ACT 학습(real-only)]
│   ├── README.md                               # 3개 워크스트림 실행 통합 가이드
│   ├── imitation_data/                         # 모방데이터 수집 (ACT 학습용 시연)
│   │   └── scripts/                            # 수집·라벨링·병합 스크립트
│   │       ├── record_real_teleop.py           # Jetson 실기 teleop 수집 (ROS2)
│   │       ├── record_sim_teleop.py            # Isaac Sim teleop 수집 (직접 API)
│   │       ├── generate_language_labels.py     # Sim 에피소드 언어 라벨링
│   │       ├── postprocess_real_labels.py      # Real 라벨 paraphrase + 임베딩 후처리
│   │       ├── merge_lerobot_datasets.py       # LeRobotDataset 디스크 병합 (--video-codec h264 가속)
│   │       ├── relabel_recycling_dataset.py    # recycling 기존 수집분 재라벨링
│   │       ├── verify_relabel.py               # 재라벨링 결과 검증
│   │       ├── verify_real_dataset.py          # 신규 수집/병합 데이터셋 무결성·임베딩↔task 검증
│   │       ├── regenerate_labels.py            # labels.jsonl 을 데이터셋 task(SSoT)에서 재생성
│   │       └── rename_feature.py               # feature 키 rename (언어 임베딩 → environment_state, ACT ENV 배선)
│   ├── lora_finetuning/                        # Gemma 도메인 LoRA 파인튜닝
│   │   ├── scripts/                            # Q&A 합성·변환·학습·검증 스크립트
│   │   │   ├── generate_function_call_qa.py    # function_call (함수 호출) 시나리오 합성
│   │   │   ├── convert_function_call_to_vision_format.py # function_call → vision chat 포맷 변환
│   │   │   ├── generate_verify_qa.py           # verify (검증) 시나리오 합성 (+scenarios sidecar)
│   │   │   ├── render_verify_images.py         # verify 시나리오 Sim 렌더링 (Cat-C 2-pass)
│   │   │   ├── merge_qa_dataset.py             # function_call / verify 70:30 혼합
│   │   │   ├── llm_client.py                   # OpenAI API 공통 wrapper
│   │   │   ├── train_gemma_qlora.py            # Unsloth QLoRA 학습 (FastVisionModel)
│   │   │   ├── smoke_test_lora.py              # 학습 후 출력 형식 sanity check
│   │   │   └── convert_lora_to_gguf.sh         # GGUF 변환
│   │   ├── requirements.txt                    # LoRA 파인튜닝 의존성
│   │   └── data/                               # Q&A JSONL (도메인별, git 추적)
│   │       ├── recycling/
│   │       └── tableware/
│   └── act/                                    # ACT 학습 (현재 real-only, co-training 유보)
│       └── scripts/                            # 학습·검증 스크립트
│           ├── train_act.sh                    # real-only 도메인별 학습 (현재 운영)
│           ├── train_act_cotraining.sh         # Sim + Real 혼합 학습 (co-training 스켈레톤 — 보존)
│           └── eval_act_offline.py             # 학습된 ACT 오프라인 검증 (action 오차 + 언어 민감도)
│
├── shared/                                     # 3목적 공통 자산 (현재 비어 있음)
├── docs/
├── jetson_release.txt                          # Jetson 실기 환경 덤프 (JetPack/CUDA/L4T 버전)
└── README.md
```

> `deploy/ros2_ws/src/robot_arm_pkg/robot_arm_pkg/` 이중 폴더는 ament_python 패키지의
> 표준 레이아웃이며, 바깥쪽은 ROS2 패키지명 / 안쪽은 Python 모듈 경로입니다.

---

## 아키텍처 핵심 결정 사항

### Language-conditioned VLA 패턴 (RT-2 / OpenVLA / π0 계열)

`agent_node`의 `look_scene` 함수는 Gemma 4 E2B VLM에 RGB 이미지를 입력하여 **자연어 타겟 묘사**(예: "the blue can on the right side of the table")를 출력한다. 이 텍스트는 `embedding_node`에서 벡터로 변환되어 `SubtaskCommand`에 실려 `act_node`로 전달되고, ACT는 이 임베딩을 observation 채널 중 하나로 사용한다. 이전 설계의 3D 좌표 산출 방식은 폐기되었다.

> ⚠️ **언어 임베딩 obs 키는 `observation.environment_state` 여야 한다 (env_state 토큰)**:
> lerobot ACT 는 input feature 를 type 으로 배선하는데(`dataset_to_policy_features`),
> `observation.*` 는 STATE, **정확히 `observation.environment_state` 만 ENV** 로 분류된다.
> ACT 가 언어(384d)를 실제로 쓰려면 ENV(env_state)로 들어가야 한다 — 그래야
> `encoder_env_state_input_proj(384→512)` 토큰이 생긴다. 구 `observation.language_embedding`
> 키는 중복 STATE 로 분류돼 **ACT 가 무시**했다(검증으로 확인). 따라서:
> - 학습 데이터셋은 학습 전 [`rename_feature.py`](training/imitation_data/scripts/rename_feature.py)
>   로 `language_embedding → environment_state` rename.
> - 배포 `act_node` 도 obs dict 를 `observation.environment_state` 키로 구성(`SubtaskCommand.language_embedding` 필드명은 유지).
> 아래 데이터 수집 절의 `observation.language_embedding` 채널은 **수집 시점 원본 키**이며,
> ACT 학습 직전 위 rename 으로 `environment_state` 가 된다.

**이 결정의 결과**:
- `agent_node`는 depth/intrinsics/hand-eye calibration에 의존하지 않는다 (좌표 산출 책임 없음)
- ACT가 RGB + 언어 임베딩으로 perception을 직접 학습 (end-to-end 표현 학습)
- Sim/Real 라벨링 일관성 부담이 줄어든다 (좌표 정확도가 아닌 자연어 일치만 요구)
- depth는 ACT 학습 obs에서 제외되며, `verify_result`의 grasp 검증 보조 신호로만 사용된다

### `target_description` 형식 — 확정

| 항목 | 값 |
|------|----|
| 언어 | **영어 고정** |
| 스키마 | 반구조화 자연어 템플릿 `"the {attribute} {object} {spatial}"` |
| 길이 | 5~15 단어 |
| 필수 구성요소 | `attribute` (색/재질 등 변별 가능 1개), `object` (단일 명사), `spatial` (위치/관계 표현) |
| 어휘 사전 | [`deploy/config/label_vocab.yaml`](deploy/config/label_vocab.yaml) — Sim/Real/배포 분포 일치 강제 |
| 예시 | `"the blue can on the right side of the table"`, `"the transparent plastic next to the paper"` |
| 금지 패턴 | 변별 없는 묘사(`"the can"`), 행동 동사 포함(`"pick up ..."`), 1인칭/감정 표현 |

이 형식은 **Sim 라벨 / Real 라벨 / 배포 시 Gemma 출력** 셋 모두 동일하게 강제된다. LoRA function_call / verify 학습 데이터가 이 분포로 만들어지고, `record_real_teleop.py`는 vocabulary 외 입력을 거부한다.

### 임베딩 모델 — 확정

| 항목 | 값 |
|------|----|
| 모델 | **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** |
| 차원 D | **384** |
| 크기 | ~120MB |
| 언어 | 다국어 (50+ 언어, 영어 코어가 가장 안정) |
| 정규화 | L2 정규화 (코사인 유사도 기반 학습 친화) |
| 강제 범위 | Sim 라벨링 / Real 수집 / ACT 학습 / 배포 전 구간 |

paraphrase 학습된 모델이므로 Sim의 외부 LLM paraphrase 증강([README Sim 데이터 생성](#학습-파이프라인--sim-데이터-생성))과 Real의 [`postprocess_real_labels.py`](training/imitation_data/scripts/postprocess_real_labels.py) 증강이 같은 임베딩 공간 안에서 자연스럽게 일반화된다.

### 자율성 범위

한 태스크 진행 중에는 도메인이 고정되며, `agent_node`는 동일 도메인 내에서 `look_scene → execute_subtask → verify_result` 루프만 반복한다. 자율 도메인 재분류, 동적 함수 생성 등 자율성 확장은 향후 작업으로 유보한다. `verify_result`는 최대 3회까지 반복하며, 그 이상은 사용자 개입 대기 상태로 진입한다.

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
| `ExecutionStatus` | `string subtask_id`, `string status` (SUCCESS/FAIL/TIMEOUT — `robot_arm_pkg.prompts.ExecStatus`), `string failure_reason`, `float32 confidence` | `act_node` → `agent_node` |
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

### 표준 서비스

| 서비스 | 타입 | 제공 노드 | 호출 측 | 동작 |
|--------|------|-----------|---------|------|
| `/go_home` | `std_srvs/Trigger` | `robot_controller_node` | `agent_node` (task 종료 시) | 현재 follower 자세에서 `~/calibration/home_pose.yaml` 자세로 ~2초간 선형 보간 복귀. 진행 중 `/joint_command` 와 `/joint_states` 발행이 잠시 멈춤. |

> **home_pose.yaml 의 단일 출처**: 같은 파일을 데이터 수집(`record_real_teleop.py`)
> 과 배포(`robot_controller_node` `/go_home`)가 공유한다. 새 자세로 바꾸려면
> follower 를 새 자세에 두고 파일을 삭제한 뒤 `record_real_teleop.py` 를 한 번 실행하면
> 자동으로 재생성된다.

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
  ├─ Service Client: /embed_text                        ──▶ [embedding_node]
  └─ Service Client: /go_home (std_srvs/Trigger)        ──▶ [robot_controller_node]
                                                            (task 종료 시 1회 호출)

[embedding_node]
  └─ Service Server: /embed_text (robot_arm_msgs/EmbedText)

[llama_server_manager]
  └─ /selected_domain (std_msgs/String) ──▶ [agent_node] (스왑 완료 통보)

[act_node]
  ├─ /joint_command    (sensor_msgs/JointState)         ──▶ [robot_controller_node]
  └─ /execution_status (robot_arm_msgs/ExecutionStatus) ──▶ [agent_node]

[robot_controller_node] (1000Hz)
  ├─ /joint_states (sensor_msgs/JointState) ──┬──▶ [act_node]
  │                                            └──▶ [agent_node]
  └─ Service Server: /go_home (std_srvs/Trigger)
        └─ ~/calibration/home_pose.yaml 자세로 선형 보간 복귀
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

> Gemma 시각 검증은 의미 수준의 판정(통에 잘 들어갔는가)을 담당하고, depth 검증은
> 현재 운영 기준 결정론적 place(목적지 배치) 성공/실패 신호를 제공한다(아래
> "destination(place) 기준" 블록). 두 신호 종합은 v1 으로 **Gemma 우선 + depth 단방향
> veto**([`agent_node._combine_verify`](deploy/ros2_ws/src/robot_arm_pkg/robot_arm_pkg/agent_node.py))
> 가 구현돼 있으며, depth 우선/가중 등 최종 정책은 실기 verify 정확도 측정 후 확정한다.

> **검증 대상 — 현재 운영은 destination(place) 기준**:
> 실물 ACT 에피소드는 home_pose 시작 → 태스크 → home_pose 복귀로 수집되고, 배포 시에도
> verify 직전 `/go_home` 으로 팔을 home_pose 에 고정한다. 따라서 verify 시점 그리퍼는
> 이미 비어 있어 grasp(집기) 흔적을 볼 수 없다. 그래서 depth 보조 검증은 엔드이펙터가
> 아니라 **목적지(통/목표 영역)** ROI 의 depth 변화로 "물체가 놓였는가"(place)를 본다.
> ROI 정의는 [`deploy/config/destination_roi.yaml`](deploy/config/destination_roi.yaml)
> — 카메라·목적지 고정 전제로 **픽셀 ROI + baseline depth** 로 표현해 카메라 intrinsics
> 에 의존하지 않는다. 위 "Depth grasp 보조 검증" 서술은 grasp 시점 검증을 다시 도입할
> 때를 위한 **미래 정책**으로 남겨둔다(`grasp_roi.yaml`). 구현: `agent_node._depth_place_check`.

---

## 실행 환경

### Jetson

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
│   ├── src/
│   │   ├── robot_arm_msgs/        # 커스텀 메시지 패키지 (신규)
│   │   └── robot_arm_pkg/
│   └── config/
│       └── domain_map.yaml        # deploy/config/domain_map.yaml에서 복사
│
├── models/
│   ├── gemma/
│   │   ├── gemma_4_e2b_q4.gguf   # base 모델, 상시 적재
│   │   └── adapters/              # 도메인별 LoRA, hot-swap
│   │       ├── adapter_recycling.gguf
│   │       └── adapter_tableware.gguf
│   ├── embedding/                 # paraphrase-multilingual-MiniLM-L12-v2 (D=384)
│   │   └── <model_files>          # HF hub 캐시 또는 사전 다운로드
│   └── act/                       # 도메인별 ACT 정책 (단일 act.pth 가정 폐기 → 도메인별 디렉터리)
│       ├── recycling/pretrained_model/  # LeRobot ACT 체크포인트 디렉터리
│       └── tableware/pretrained_model/  #   (config.json + model.safetensors + pre/post processor)
│
├── calibration/
│   ├── camera_intrinsics.yaml
│   ├── camera_robot_transform.yaml
│   ├── destination_roi.yaml       # deploy/config/에서 복사 (place 검증 — 현재 운영)
│   ├── grasp_roi.yaml             # (미래 grasp 검증용 — 선택)
│   └── home_pose.yaml             # SO-101 follower rest 자세. record_real_teleop 첫 실행 시 자동 생성.
│                                  # robot_controller_node /go_home 서비스가 같은 파일을 읽는다.
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
| 주요 작업 | Gemma 도메인 LoRA 학습 (Unsloth QLoRA), ACT 모방학습, Isaac Sim teleop 데이터 수집 |

> Sim teleop 데이터 수집용 Isaac Sim 은 이 외부 PC(로컬) 또는 EC2 클라우드
> 워크스테이션(g6e.4xlarge)에서 구동한다. 환경별 셋업은
> [`sim/teleop/sim_local_teleop_setting.md`](sim/teleop/sim_local_teleop_setting.md) /
> [`sim/teleop/sim_cloud_teleop_setting.md`](sim/teleop/sim_cloud_teleop_setting.md) 참조.

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

## Jetson 메모리 배분 (Orin Nano 8GB)

| 컴포넌트 | RAM | 비고 |
|----------|-----|------|
| OS + ROS2 | ~1.5 GB | |
| Gemma 4 E2B base (GGUF Q4) | ~3.0 GB | llama-server 상주 |
| **도메인 LoRA (active 1개)** | ~수십 MB | **hot-swap — 동시 적재 금지** |
| 임베딩 모델 | ~120 MB | paraphrase-multilingual-MiniLM-L12-v2 (D=384) |
| ACT Policy | ~0.3 GB | 상주 |
| RGB-D 드라이버 + 버퍼 | ~0.3 GB | |
| KV Cache / 기타 | ~1.0 GB | Gemma 컨텍스트 윈도우 |
| **합계** | **~6.2 GB** | 도메인 수가 늘어도 합계 불변 |

> Gemma LoRA는 llama-server에 HTTP `/lora-adapters` 요청으로 hot-swap (서브-초 latency).
> 한 태스크가 끝나는 시점(verify_result 직후)에 스왑하면 안정적이다.

> **GPU 공존 운영 (2026-06-04 실기 확인) — Orin Nano 8GB unified memory**:
> llama-server(GPU, `--n-gpu-layers 99`)와 act_node(GPU, torch)가 같은 메모리를 공유한다.
> 메모리 **총량은 충분**(llama ~2.3G + act ~0.6G + OS ≪ 8G)하지만, torch 기본 caching
> allocator 의 큰 연속 블록 할당이 GPU **단편화**를 일으켜 `NvMap error 12 (ENOMEM)` +
> `NVML ASSERT` 로 cuda 로드가 실패할 수 있다(총량이 충분해도). 다음으로 회피·운영한다:
> - **act_node**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(확장형 세그먼트 할당으로
>   단편화 회피). `act_node.py` 가 setdefault 하므로 빌드본은 자동 적용, 수동 실행 시 prefix
>   로 보강. `-p device:=cuda` 명시 권장(시작 로그의 `device: cpu` 는 lazy 로드 전 기본값).
> - **embedding_node**: `-p device:=cpu` 로 실행해 GPU 를 llama/act 에 양보(임베딩은 이벤트성·
>   경량이고 agent 가 사전 임베딩하므로 CPU 로 충분).
> - **전원/메모리**: 검증 동안 **GUI 끔**(`sudo systemctl set-default multi-user.target`,
>   ~1GB 확보) + **전원 MAXN**(`sudo nvpmodel -m 0 && sudo jetson_clocks`)으로 할당 안정성 확보.
> 위 배분으로 6개 노드 + llama-server 가 8GB 안에 공존함을 실기에서 확인.

---

## Jetson 환경 설치

용도에 따라 두 가지 clone 모드를 권장합니다.

| 워크플로우 | clone 범위 | 필요 스크립트 |
|---|---|---|
| **검증 / 배포만** | `deploy/` 만 (sparse-checkout 등) | `setup_jetson_base.sh` |
| **학습 준비** (캘리브레이션 + 모방 데이터 수집) | PAI 전체 | `setup_jetson_base.sh` + `setup_calibration.sh` |

저장소 루트의 스크립트로 환경을 구성합니다.

| 스크립트 | 대상 작업 |
|----------|-----------|
| `deploy/scripts/setup_jetson_base.sh` | ROS2 + LeRobot 베이스 환경 (양쪽 워크플로우 공통) |
| `deploy/scripts/setup_calibration.sh` | aruco_ros / easy_handeye2 — 학습 준비 시에만 필요 |

```bash
# 1. 기본 환경 설치 (양쪽 워크플로우 공통)
chmod +x deploy/scripts/setup_jetson_base.sh
./deploy/scripts/setup_jetson_base.sh

# 2. 학습 준비 모드일 때만: 캘리브레이션 환경 추가 설치
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
| sentence-transformers | ≥2.2.0 | `embedding_node` — paraphrase-multilingual-MiniLM-L12-v2 (D=384) |

> ⚠️ **Jetson 실기 — torch/numpy/setuptools 는 위 generic 버전이 아니라 고정값 사용**:
> Jetson(JetPack 6.2 / CUDA 12.6)에서는 PyPI generic torch wheel 이 CPU 전용(+cpu)이라
> cuda 불가. **`torch==2.8.0` + `torchvision==0.23.0`** 을 jetson-ai-lab cu126 index
> (`https://pypi.jetson-ai-lab.io/jp6/cu126`)에서 설치하고, **`numpy==1.26.4`** 로 고정한다
> (Jetson torch 는 NumPy 1.x ABI — 2.x 는 `torch.from_numpy` 가 깨짐). `setuptools` 는
> ament_python 빌드 호환을 위해 **59.8.0(<60)** 으로 마지막에 고정한다(lerobot 이
> `setuptools>=71` 을 끌어오므로 설치 순서 주의). 이 모든 처리는
> [`setup_jetson_base.sh`](deploy/scripts/setup_jetson_base.sh) [4] 단계가 수행한다.

---

## 빌드 및 실행

### 빌드 단계 개요

- **개발 PC 확인용**: `cd deploy/ros2_ws && colcon build --packages-select robot_arm_msgs robot_arm_pkg`
- **Jetson 실기**: 저장소의 `deploy/ros2_ws/src/*` · `deploy/config/*` 를 Jetson 의 `~/ros2_ws/src/` · `~/ros2_ws/config/` · `~/calibration/` 로 복사한 뒤 동일하게 `colcon build`.

### 배포 실행 단계 개요

llama-server (Gemma base GGUF, `--jinja`) 를 systemd 로 상시 구동하고, ROS2 노드 6개
(`camera_node`, `llama_server_manager`, `embedding_node`, `agent_node`, `act_node`,
`robot_controller_node`) 를 띄운 뒤 `/user_command` 토픽으로 사용자 명령을 발행합니다.

> 빌드·실행의 자세한 명령은 [`deploy/README.md`](deploy/README.md) 참조.

---

## 전체 파이프라인

```
하드웨어 세팅 (카메라 캘리브레이션)
  │
  ├─▶ Isaac Sim ACT 시연 데이터 수집 (Sim teleop — EC2 또는 로컬)
  │     실물 SO-101 Leader → Isaac Sim follower(디지털 트윈) teleop
  │     + Domain Randomization
  │     multi-target 장면 구성 (같은 시각에 여러 후보 객체 배치)
  │     언어 라벨: Gemma 4 E2B 자체 라벨링 + 외부 LLM paraphrase 혼합
  │     → Sim LeRobotDataset (~500 에피소드)
  │
  ├─▶ Gemma 도메인 LoRA 학습 (외부 PC)
  │     ├─ 도메인별 LLM 합성 Q&A 데이터 생성
  │        2종 혼합 (70:30): 함수 호출 (function_call) / 검증 시나리오 (verify)
  │     ├─ Unsloth QLoRA 파인튜닝 (예: RTX 5070급)
  │     └─ GGUF 변환 → adapter_<domain>.gguf (각 수십 MB)
  │        adapter_recycling.gguf, adapter_tableware.gguf, ...
  │
  ▼
실제 시연 데이터 수집 (Jetson, 축소 운영)
  teleop 기반 record_real_teleop.py (agent_node 미사용)
  → Real LeRobotDataset (50~100 에피소드)
  (observation.state + observation.language_embedding + observation.images.top + action)
  │
  └─▶ ACT 학습 (외부/클라우드 GPU, LeRobot 0.4.4)
        (현 운영: real-only 도메인별 → 도메인별 pretrained_model/.
         아래 co-training 표기는 보존된 원설계)
        Sim + Real 혼합 학습 → 도메인별 pretrained_model/
            │
            ▼
        통합 배포 (Sim 환경 + Jetson 실물 동일 노드 구성)
            llama-server (Gemma 4 E2B base + LoRA hot-swap, systemd 상시)
            + agent_node (오케스트레이션 함수 호출 루프)
            + embedding_node (텍스트 → 벡터 임베딩 서비스)
            + act_node
            │
            ▼
        검증
            Sim rollout / Jetson 실물 모두 동일 노드 구성으로 평가
            (학습에 쓰지 않은 DR 파라미터로 held-out)
```

> Sim 데이터 수집과 Gemma 도메인 LoRA 학습은 독립적으로 병렬 진행 가능. Sim·Real
> 시연 데이터 모두 ACT 학습 데이터 확보가 목적이므로 `agent_node` 없이 실물 SO-101
> Leader teleop 으로 수집한다. Sim 은 Isaac Sim follower(디지털 트윈)를, Real 은
> 실물 follower 를 leader 로 조종한다.
>
> Sim과 Jetson 실물은 **동일 노드 구성**(camera_node, agent_node,
> llama_server_manager, embedding_node, act_node, robot_controller_node)으로
> 운영하며 검증도 같은 파이프라인에서 수행한다. Gemma LoRA는 도메인 1개만
> 적재되고, 사용자 명령 → 도메인 분류 → llama-server `/lora-adapters` HTTP
> 요청으로 hot-swap된다.
>
> ACT는 Sim-to-Real Co-training 방식으로 학습한다. Sim 측 Domain Randomization:
> 조명·텍스처·카메라 FOV, 질량·마찰·PD 게인, joint/이미지 노이즈.
> Sim action space는 실기와 동일하게 position control. Sim 시연을 실물 SO-101
> Leader teleop 으로 수집하므로 Sim·Real 의 action 생성 경로(같은 Leader, 같은
> LeRobot SDK)가 일치하여 두 데이터셋의 행동 분포가 동질화된다.

---

## 카메라-로봇 캘리브레이션

Hand-Eye 캘리브레이션 (Eye-to-Hand 방식): 카메라 고정, SO-101 엔드이펙터에 ArUco 마커 부착.

> 본 설계에서 ACT는 RGB만 사용하고 `agent_node`도 3D 좌표를 산출하지 않으므로,
> 캘리브레이션 정밀도 요구 수준은 이전 설계 대비 낮아진다. 다만 카메라 위치의
> **일관성**(학습 시점과 배포 시점 간)이 중요하므로 캘리브레이션 자체는 여전히 수행한다.
> `verify_result`의 grasp ROI 정의에 카메라-로봇 변환이 사용된다.

### 캘리브레이션 단계 개요

1. **Jetson 사전 설치** — `setup_calibration.sh` 로 aruco_opencv, easy_handeye2 소스 빌드.
2. **URDF 배치** — 저장소 포함 `urdf/so101.urdf` 가 colcon build 시 `share/robot_arm_pkg/urdf/` 로 자동 복사. 엔드이펙터 링크명이 `calibration.launch.py` 의 `robot_effector_frame` 과 일치하는지 확인.
3. **캘리브레이션 launch** — `ros2 launch robot_arm_pkg calibration.launch.py robot_port:=/dev/follower` + rqt EasyHandeye2 GUI.
4. **포즈 샘플링** — 10~15개 포즈에서 Take Sample → Compute → Save.
5. **결과 저장** — `~/calibration/camera_intrinsics.yaml`, `~/calibration/camera_robot_transform.yaml` 로 복사 (+ 저장소 백업).
6. **`grasp_roi.yaml` 작성** — `verify_result` 의 depth grasp 보조 검증용 ROI (`{frame, x/y/z_range, depth_change_threshold}`) 정의. (grasp 시점 검증 — 미래 확장)
7. **`destination_roi.yaml` 작성** — `verify_result` 의 depth place 보조 검증용 목적지 ROI (`{roi_px, baseline_depth, depth_change_threshold}`) 정의. **현재 운영 기본** — verify 시점 home_pose 에서 목적지(통/목표 영역) depth 변화로 배치 확인.

> 자세한 셋업·실행·결과 저장 절차는 [`deploy/calibration/README.md`](deploy/calibration/README.md) 참조.

---

## 데이터 수집

ACT 학습용 시연 데이터를 teleop으로 수집한다. `agent_node` 및 LoRA는 사용하지 않는다.

> **포트 점유 정책**: `record_real_teleop.py` 가 SO-101 follower USB 포트를
> **단독 점유**한다. 동시에 `robot_controller_node` 가 실행 중이면 LeRobot SDK
> 의 포트 open 단계에서 실패하므로, 데이터 수집 동안에는 `robot_controller_node`
> 를 반드시 종료할 것. (스크립트 상단 "운영 / 측정 결정 사항" 블록 [1] 참고)

### 실행 단계 개요

1. **준비** — USB 포트 / `lerobot-calibrate` 완료 / 카메라 캘리브레이션 결과 `~/calibration/` 배치 / 임베딩 모델 배치 / `robot_controller_node` 종료(포트 점유 정책) / `home_pose.yaml` 첫 실행 시 자동 캡처.
2. **teleop 동작 검증** — `lerobot-teleoperate` 로 리더↔팔로워 미러링 + 카메라 표시 사전 확인.
3. **ROS2 노드 기동** — `realsense2_camera` 드라이버 → `camera_node` (/camera/rgb 발행) → `embedding_node` (라벨→임베딩 Service).
4. **teleop 수집** — `python <repo>/training/imitation_data/scripts/record_real_teleop.py ...` (Option C 슬롯 기반 라벨).
5. **paraphrase + 임베딩 후처리** — `python <repo>/training/imitation_data/scripts/postprocess_real_labels.py ...` (외부 PC 또는 Jetson 어디서나 가능).

> 자세한 실행 명령은 [`training/README.md`](training/README.md) "모방데이터 수집" 절 참조.

> **언어 라벨 부여 방법 — Option C (템플릿+변수 슬롯) 확정**:
>
> 작업자는 매 에피소드 시작 시 (object, attribute, spatial) 3개 슬롯을
> [`deploy/config/label_vocab.yaml`](deploy/config/label_vocab.yaml) vocabulary 안에서만
> 선택한다. canonical 라벨은 자동 생성된다:
>
>     "the {attribute} {object} {spatial}"
>     예: "the blue can on the right side of the table"
>
> - 자유 텍스트 입력 금지 (vocabulary 외 입력 거부)
> - 영어 고정 → Sim 라벨 / 배포 시 Gemma 출력과 분포 일치
> - 같은 작업의 라벨이 너무 다르지 않도록 자동 보장
> - 수집 후 [`postprocess_real_labels.py`](training/imitation_data/scripts/postprocess_real_labels.py)로
>   paraphrase 3개 + 임베딩을 일괄 부여 → ACT 학습 시 random augmentation
>
> **Multi-target shortcut 방지 가이드라인** (50–100 에피소드 분포):
> - 같은 객체에 대해 다른 spatial 선택 ≥ 30%
> - 같은 spatial에 대해 다른 객체/attribute ≥ 30%
> - 한 장면에 후보 객체 2개 이상 + 다른 라벨 부여 ≥ 40%

> **임베딩 일관성**: 모든 단계에서 `paraphrase-multilingual-MiniLM-L12-v2`
> (D=384)로 고정. embedding_node / postprocess_real_labels /
> generate_language_labels 셋이 같은 모델을 사용한다. 모델 변경 시 전체
> 데이터셋 재라벨링 필요.

### 데이터셋 구조

| 채널 | shape / dtype | 내용 |
|------|-------|------|
| `observation.state` | (6,) float32 | 팔로워 관절 위치 (SO-101 6축) |
| `observation.language_embedding` | (384,) float32 | `target_description`의 임베딩 벡터 (paraphrase-multilingual-MiniLM-L12-v2) |
| `observation.images.top` | (H,W,3) uint8 | RealSense RGB (`/camera/color/image_raw`) |
| `action` | (6,) float32 | 리더 관절 값 (정답 행동) |

> **언어 임베딩 채널 — 학습 전 `environment_state` 로 rename**: 위 표의
> `observation.language_embedding` 은 **수집 시점 원본 키**다. stock lerobot ACT 는
> 이 키를 STATE 로 분류해 무시하므로, ACT 학습 직전
> [`rename_feature.py`](training/imitation_data/scripts/rename_feature.py) 로
> `observation.environment_state`(ENV) 로 rename 해야 ACT 가 언어를 실제로 사용한다
> (위 "아키텍처 핵심 결정 사항" 절 참조).

> 이전 설계의 `observation.target_position` 채널은 제거되었다. depth는
> `verify_result` 단계에서만 사용되고 ACT 학습 obs에 포함되지 않는다.

> Sim 데이터셋(`dataset_sim`)도 동일 feature 스펙 + 동일 임베딩 모델
> (paraphrase-multilingual-MiniLM-L12-v2, D=384)을 따라야 ACT Co-training이
> 가능하다. 임베딩 차원 D가 384가 아니면 학습이 실패한다.

---

## 학습 파이프라인 — Gemma 도메인 LoRA

### Q&A 데이터 생성 (2종 혼합, 70:30)

도메인별로 **2개 카테고리의 Q&A**를 생성하고 혼합하여 LoRA 학습에 사용한다.
LoRA는 학습 데이터의 분포를 그대로 모방하므로, 배포 시 `agent_node`가 요구하는
2가지 작업 형식(함수 호출 / 시각 검증)을 모두 학습 데이터에 반영해야 한다.

> **domain_knowledge (도메인 지식 QA) 제거 (2026-05-25 freeze)**: 도메인 범위가
> 분리수거(종이·캔·플라스틱 3종)와 식기 정리로 좁아 도메인 지식 QA의 학습 효용이
> 거의 없다. LoRA 의 실제 작업 표면인 함수 호출과 시각 검증만 학습한다.

| 카테고리 | 목적 | 형식 | 생성 방법 |
|---|---|---|---|
| **function_call: 함수 호출 시나리오** | `look_scene`/`execute_subtask` 정확도 | 사용자 명령 → JSON 함수 호출 | 외부 LLM으로 합성 (`generate_function_call_qa.py`) |
| **verify: 검증 시나리오** | `verify_result` 정확도 | 이미지 + 명령 → enum + reasoning | Sim 렌더링 + 자동 라벨링 (`generate_verify_qa.py`) |

#### 비율 (freeze)

| function_call | verify | 근거 |
|-------|-------|------|
| **70%** | **30%** | 한 태스크에 `look_scene` → `execute_subtask` → `verify_result` 가 다단계로 발생 → 함수 호출 표본이 더 자주 필요. `verify_result`는 단일 호출이고 출력이 4-way enum 이라 적은 표본으로도 학습 가능. |

#### function_call 예시 (함수 호출)

```json
{
  "messages": [
    {"role": "system", "content": "사용 가능 함수: look_scene(prompt), execute_subtask(target_description), verify_result(original_command)"},
    {"role": "user", "content": "쓰레기 좀 정리해줘"},
    {"role": "assistant", "content": "{\"function\": \"look_scene\", \"args\": {\"prompt\": \"분리수거할 물체를 찾아주세요\"}}"}
  ]
}
```

#### verify 예시 (검증)

```json
{
  "messages": [
    {"role": "user", "content": "[image] 사용자 명령: '캔을 캔류 통에 넣어줘'. 작업이 완료되었나요?"},
    {"role": "assistant", "content": "{\"status\": \"SUCCESS\", \"reasoning\": \"파란색 캔류 수거함 안에 알루미늄 캔이 들어 있음\"}"}
  ]
}
```

### 학습 단계 개요

1. **Q&A 혼합** — `merge_qa_dataset.py` 로 function_call / verify 를 70:30 으로 병합해 단일 JSONL 생성.
2. **Unsloth QLoRA 파인튜닝** — `train_gemma_qlora.py` (RTX 5070급 GPU, base = `google/gemma-4-e2b`).
3. **GGUF 변환** — `convert_lora_to_gguf.sh` 로 llama.cpp 적재용 `adapter_<domain>.gguf` 산출.

> 자세한 실행 명령은 [`training/README.md`](training/README.md) "Gemma 도메인 LoRA 파인튜닝" 절 참조.
> 양 도메인 학습/검증/GGUF 변환은 2026-06-02 기준 완료. smoke test 결과,
> tableware function_call 마지막 turn cluster gap, verify reasoning 환각
> 정책 등 알려진 이슈는 [`training/README.md`](training/README.md) "2-7. 학습
> 결과 및 알려진 이슈" 와 [`deploy/README.md`](deploy/README.md) "에이전트 함수
> 호출 루프 (agent_node)" 참조.

---

## 학습 파이프라인 — Sim 데이터 수집 (teleop)

Sim 시연 데이터는 **실물 SO-101 Leader teleop**으로 수집한다. 작업자가 실물
Leader를 손으로 조종하면 Isaac Sim 안의 follower(디지털 트윈)가 미러링하고, 그
과정의 관측·행동을 LeRobotDataset으로 녹화한다. 규칙기반 IK 전문가 스크립트
방식은 폐기되었다 — Real 데이터와 동일한 teleop 수집 경로(같은 Leader, 같은
LeRobot SDK)를 써서 Sim·Real 두 데이터셋의 행동 분포를 일치시키는 것이 목적이다.
순수 teleop 전환에 따라 Sim 목표 에피소드 수는 ~500으로 조정되었다.

### Multi-target 장면 구성

ACT가 언어 임베딩을 무시하고 시각 정보에만 의존하는 **shortcut learning**을
방지하기 위해, 같은 시각 장면에서 **다른 언어 명령에 따라 다른 행동을 하는**
에피소드를 의도적으로 포함한다.

**구성 원칙**:
- 한 장면에 2개 이상의 후보 객체 배치 (예: 캔 1개 + 페트병 1개 + 종이 1개)
- 같은 객체 배치에 대해 다른 타겟 명령으로 여러 에피소드 생성 (예: "캔 집어" / "페트병 집어")
- 카메라 시점도 일부 공유하여 RGB 입력만으로는 정답을 구분할 수 없게 함

`sim/config/multi_target_scenes.yaml`로 장면·배치 규칙을 정의한다:

```yaml
# sim/config/multi_target_scenes.yaml (발췌) — domains.<name> 네임스페이스 구조.
# record_sim_teleop 가 --domain 값으로 해당 블록을 선택한다.
domains:
  recycling:
    object_library: sim/assets/objects/recycling   # 래퍼 USD 디렉터리
    placement:                                      # SceneManager 배치 영역·규칙 (실측값)
      spawn_region: {x_range: [-1.5921, 1.5927], y_range: [-0.8126, 0.8132]}
      reach:                                        # 선택 — 가동범위(annular sector)
        center: [-0.5, 0.0]
        r_range: [0.115, 0.35]
        theta_range_deg: [-90.0, 90.0]
      table_top_z: 0.0104
      min_separation: 0.08
      exclusion_zones:                              # 초기 배치 금지 + verify destination
        - {x_range: [-0.4250, -0.2700], y_range: [-0.4050, -0.2050], destination: plastic}
        - {x_range: [-0.3100, -0.1550], y_range: [-0.4050, -0.2050], destination: can}
        - {x_range: [-0.1950, -0.0400], y_range: [-0.4050, -0.2050], destination: paper}
    scenes:                          # can/plastic/paper 대칭 — 총 7개 장면
      - name: recycling_3cat       # 캔+플라스틱+종이 (multi-target)
        objects:
          - {category: can,     count: 1}
          - {category: plastic, count: 1}
          - {category: paper,   count: 1}
      - name: recycling_2can       # 같은 범주 변별 (multi-target)
        objects: [{category: can, count: 2}]
      - name: recycling_single_can # 단일 객체 (single-target)
        objects: [{category: can, count: 1}]
      # 2plastic / 2paper / single_plastic / single_paper (대칭 장면) 생략
    multi_target_ratio: 0.6        # multi-target 장면 목표 비율
  # tableware: { ... }             # plate/fork/spoon 대칭 블록 (동일 구조) 생략
```

`TargetSelector`([D5])가 매 에피소드 이 정의에서 장면·배치 객체·grasp 타겟을
자동 선택한다. multi/single 비율은 `multi_target_ratio`로 수렴시키고, 장면·타겟
카테고리는 누적 사용분이 가장 적은 것을 우선 골라 다양성을 강제한다. grasp 타겟의
자연어 라벨은 실제 배치 좌표에서 `spatial`을 도출해 `"the {object} {spatial}"`로
만들며, `{attribute}`(색)는 Sim 객체가 고정 YCB 메시이므로
`generate_language_labels.py`(Gemma)가 렌더 이미지를 보고 채운다.

### 언어 라벨링 (Gemma + 외부 LLM 혼합)

각 Sim 에피소드에 다양한 자연어 라벨을 부여한다. 두 가지 소스를 혼합:

**소스 1: Gemma 4 E2B 자체 라벨링**
- Sim에서 렌더링한 RGB 이미지를 Gemma 4 E2B에 입력
- 배포 시 사용할 동일 모델로 라벨 생성 → 분포 일치
- `training/imitation_data/scripts/generate_language_labels.py`가 일괄 처리

**소스 2: 외부 LLM Paraphrase**
- 소스 1의 라벨을 외부 LLM(GPT-4 등)에 입력하여 다양한 표현 생성
- 표현 robustness 강화 (배포 시 사용자가 어떻게 말할지 모름)
- 같은 에피소드에 여러 paraphrase를 부여하고 학습 시 무작위 추출

**비율**: 에피소드당 Gemma 라벨 1개 + 외부 LLM paraphrase 2~3개. 학습 시
augmentation으로 무작위 추출.

### 수집 절차

`record_sim_teleop.py`는 Isaac Sim standalone 앱으로 동작하며, **Isaac Sim·
SO-101 Leader·녹화 스크립트가 한 Ubuntu 머신에 있는 로컬 통합 환경**을 전제한다
([D1]). 환경 셋업(Isaac Sim 설치·URDF 임포트·씬 구성)은
[`sim/teleop/sim_local_teleop_setting.md`](sim/teleop/sim_local_teleop_setting.md)
참조. 스크립트는 Leader 를 LeRobot SDK 로 직접 읽고 follower·카메라·관절을 Isaac
Sim API 로 직접 제어하며 (ROS2 미사용 — [D1]·[D4]), 매 에피소드 multi-target 씬
구성 + DR 적용 → 녹화 → LeRobotDataset 저장을 담당한다.

전제: Isaac Sim 의 python 환경에 lerobot 0.4.4 설치
(`~/isaacsim/python.sh -m pip install "lerobot[feetech]==0.4.4"`).

수집은 `~/isaacsim/python.sh training/imitation_data/scripts/record_sim_teleop.py ...` 로
Isaac Sim Python 에서 직접 실행하고, 이후 `python training/imitation_data/scripts/generate_language_labels.py ...`
로 Gemma + 외부 LLM 라벨을 부여합니다.

> 자세한 실행 명령은 [`training/README.md`](training/README.md) "Sim teleop" 절 참조.

> **단위 정합성**: `action`(Leader 관절값)은 Real 수집과 동일하게 LeRobot SDK
> `.pos` 단위로 기록되어 변환이 불필요하다. `observation.state`(Isaac Sim
> follower 관절)는 Isaac Sim API 가 rad 로 주므로 `record_sim_teleop.py`가 SDK
> `.pos` 단위로 변환한다 — [D3] 확정상 plain deg↔rad (affine 불필요). 단위가
> 어긋나면 같은 자세에 다른 라벨이 들어가 co-training 이 망가진다.

> **에피소드 시작 자세**: 매 에피소드 시작 시 follower 를
> [`deploy/calibration/home_pose.yaml`](deploy/calibration/home_pose.yaml)
> (record_real_teleop.py 가 캡처한 실물 follower rest 자세)로 강제한다.
> `record_sim_teleop.py`가 Isaac Sim `set_joint_positions` API 로 직접 고정하며
> (위치 + 속도 0 + drive target 동시 적용) Sim/Real 의 에피소드 시작 자세가
> 일치한다. USD 가동범위를 넘는 관절값은 USD limit 으로 clamp 된다.

### Domain Randomization 항목

`SceneManager.apply_domain_randomization`이 [`sim/config/dr_config.yaml`](sim/config/dr_config.yaml)
기준으로 매 에피소드 `setup_episode_scene` 직후 적용한다. 구현·활성: **조명**
(intensity/color), **카메라 focalLength**, **객체 질량**. jitter 는 기동 시
캡처한 베이스 값(또는 래퍼 USD 의 baked mass)에 곱해 에피소드 누적 drift 를 막는다.
보류(스키마 메모만): 텍스처·마찰·PD 게인·관측 노이즈 — 사유는 `dr_config.yaml` 주석.
depth는 ACT 학습에 사용되지 않으므로 depth 노이즈 DR은 제거됨.

---

## 학습 파이프라인 — ACT Co-training

LeRobot 0.4.4의 멀티 데이터셋 가중치 기능을 사용하여 Sim + Real 혼합 학습.
실행은 `bash training/act/scripts/train_act_cotraining.sh` 한 줄.

> 자세한 실행 명령·하이퍼파라미터는 [`training/README.md`](training/README.md) "ACT Co-training" 절 및 스크립트 헤더 주석 참조.

스크립트 내부 (의사 코드):
```python
datasets = [
    LeRobotDataset("dataset_sim",  weight=0.7),
    LeRobotDataset("dataset_real", weight=0.3),
]
# 언어 임베딩 채널(학습 키: observation.environment_state)이 두 데이터셋 모두에 존재해야 함
# 임베딩 차원 D 일치 필수
```

**학습 입력 obs**:
- `observation.state`
- `observation.environment_state` (=언어 임베딩, ENV 토큰. 수집 키 `language_embedding` 에서 `rename_feature.py` 로 rename — "아키텍처 핵심 결정 사항" 절 참조)
- `observation.images.top`

**학습 타겟**:
- `action` (chunk_size=100 시퀀스)

> Sim/Real의 `language_embedding` 차원과 임베딩 모델이 다르면 학습 불가.
> 데이터 수집 시점부터 동일 모델 사용 보장.

---

## 미구현 항목 (TODO)

아래 항목은 **스켈레톤(인터페이스/CLI/파라미터)은 작성 완료**이고 핵심 로직이
미구현인 상태이다. 각 파일은 `NotImplementedError` 또는 warn 로그로
미구현 영역을 명시한다.

| 파일 | 미구현 핵심 로직 | 선행 조건 |
|------|------------------|----------|
| `training/imitation_data/scripts/postprocess_real_labels.py` | `synthesize_paraphrases()` (외부 LLM), `apply_to_dataset()` (LeRobot metadata) | 외부 LLM 클라이언트, LeRobot 0.4.4 metadata API |
| `training/imitation_data/scripts/record_sim_teleop.py` | (핵심 로직 구현 완료) `SimInterface`·`SceneManager`·`TargetSelector` 의 Isaac Sim 5.1 API 호출은 best-effort — 실기 검증 필요. DR 의 texture/friction/pd_gain/노이즈는 의도적 보류 | Isaac Sim 5.1 환경 검증 |
| `training/imitation_data/scripts/generate_language_labels.py` | Gemma 자체 라벨링 (시스템 프롬프트는 확정), 외부 LLM paraphrase, LeRobot metadata 반영 | llama-server VLM 페이로드, 외부 LLM 클라이언트 |
| `sim/config/dr_config.yaml` | 조명·카메라·질량 범위는 활성 — 보수적 기본값이므로 실기 측정 후 보정 필요. 텍스처·마찰·PD 게인 섹션은 미구현(주석) | 실기 측정, DR 헬퍼 추가 구현 |
| `training/act/scripts/train_act_cotraining.sh` | LeRobot 멀티 데이터셋 가중치 학습 명령 본체 | LeRobot 가중치 API 확인, language_embedding 채널 검증 |
| `deploy/config/grasp_roi.yaml` | 실측 ROI / threshold 값으로 템플릿 갱신 (grasp 시점 검증 — 미래 확장) | 캘리브레이션 완료, 엔드이펙터 측정 |

> **작성 완료 항목** (스켈레톤 + 인터페이스 정합 완료, 본 표에는 포함 안 됨):
> - `robot_arm_msgs/` 패키지 (`SubtaskCommand`, `ExecutionStatus`, `DomainSelection`, `AgentLog` + `EmbedText`)
> - `setup.py` entry_points 갱신 (`agent_node`, `embedding_node`, `llama_server_manager` 추가 / 구 노드 제거)
> - `training/imitation_data/scripts/record_real_teleop.py` (`open_dataset` / `connect_so101` / `record_episode` 본 구현 완료 — 운영 결정 6개는 파일 상단 블록에 명시)
> - `deploy/config/domain_map.yaml` 템플릿
> - `deploy/config/label_vocab.yaml` (데이터 수집 / function_call / Sim 공유 vocabulary)
> - `camera_node.py`
> - `embedding_node.py` (paraphrase-multilingual-MiniLM-L12-v2, D=384 — 모델 확정)
> - `sim/config/multi_target_scenes.yaml` (placement 실측 완료 — spawn_region·table_top_z·exclusion_zones 를 base_scene.usd 에서 측정해 기입, domains.recycling/tableware 네임스페이스 구조)
> - `deploy/calibration/camera_intrinsics.yaml` (RealSense D435 공장 캘리브레이션값), `camera_robot_transform.yaml` (eye-on-base hand-eye 결과), `home_pose.yaml` (follower rest 자세 캡처)
> - `deploy/config/destination_roi.yaml` (place 검증 ROI 실측 완료 — 2026-06-04, recycling 3 bin + tableware 영역, 픽셀 ROI + baseline depth, threshold 0.025)
>
> **Gemma 도메인 LoRA 파이프라인 — 본체 구현 + 양 도메인 실행 완료 (2026-06-02)**:
> - `training/lora_finetuning/scripts/generate_function_call_qa.py` (function_call 합성)
> - `training/lora_finetuning/scripts/convert_function_call_to_vision_format.py` (1회 변환)
> - `training/lora_finetuning/scripts/generate_verify_qa.py` (Cat-C JSONL + scenarios sidecar 출력)
> - `training/lora_finetuning/scripts/render_verify_images.py` (Sim 렌더링 본체, Cat-C 2-pass 분리)
> - `training/lora_finetuning/scripts/merge_qa_dataset.py` (function_call / verify 70:30 oversample)
> - `training/lora_finetuning/scripts/train_gemma_qlora.py` (Unsloth FastVisionModel + chat-template aware loss masking)
> - `training/lora_finetuning/scripts/smoke_test_lora.py` (학습 후 형식 sanity check)
> - `training/lora_finetuning/scripts/convert_lora_to_gguf.sh` (BASE_MODEL HF repo id 자동 다운로드 + 변환 본체)
> - 산출물: `models/gemma/adapters/adapter_{recycling,tableware}.gguf` 양 도메인 GGUF 생성 완료 (gitignore — Jetson 으로 scp 전송 대상)
> - `deploy/ros2_ws/src/robot_arm_pkg/robot_arm_pkg/llama_server_manager.py` `_hot_swap()` 본체 구현 (id 기반 `/lora-adapters` HTTP) + Jetson Orin Nano 8GB 실기 hot-swap 검증 완료 (init deactivate / recycling↔tableware swap / 같은 도메인 재요청 idempotency 통과). 자세한 검증 명령과 결과는 [deploy/README.md](deploy/README.md) "LoRA hot-swap 실기 검증 (2026-06-02)" 참조.
>
> **agent_node 함수 호출 루프 본체 — 구현 완료 (2026-06-03)**:
> - `deploy/ros2_ws/src/robot_arm_pkg/robot_arm_pkg/agent_node.py` `_run_function_loop` / `_call_look_scene` / `_call_execute_subtask` / `_call_verify_result` 본체 구현 (llama-server `/v1/chat/completions` HTTP, `/embed_text`·`/go_home` 서비스, verify_result fallback, attempt 단위 재시도, `WAITING_USER` 종료 상태). 실행 모델은 워커 스레드 + `MultiThreadedExecutor`. 구현 결정 6건은 파일 상단 docstring + [deploy/README.md](deploy/README.md) "구현 결정 사항 (2026-06-03)" 참조.
> - `deploy/ros2_ws/src/robot_arm_pkg/robot_arm_pkg/prompts.py` `LOOK_SCENE_SYSTEM_PROMPT` 추가 (look_scene 그라운딩 전용 — 배포 전용, 학습 분포 없음). `act_node.py` `ExecutionStatus.status` 대문자 enum(`ExecStatus`) 발행으로 교정.
> - depth place 보조 검증(`_depth_place_check`) 구현 완료 — `destination_roi.yaml` 의 픽셀 ROI + baseline depth median 비교(intrinsics 비의존, 단방향 veto). grasp(집기 시점) 검증은 home_pose 복귀 데이터로는 불가 → 미래 확장으로 유보. look_scene 묘사 정확도는 Jetson 실기 검증 필요.
>
> **ACT 정책 학습 — 양 도메인 완료 (2026-06-04, real-only)**:
> - recycling(병합 571ep) / tableware(233ep) 도메인별 `pretrained_model/` 산출 (gitignore — `models/act/<domain>/` 로컬 배치, Jetson scp 대상). 단일 `act.pth` + LoRA hot-swap 가정은 **도메인별 ACT 체크포인트**로 변경. 도메인 전환 시 정책 재로드/재선택은 `act_node` 가 `SubtaskCommand.domain` 으로 수행(아래 act_node 구현 항목 참조).
> - **언어 조건화 배선 수정**: stock lerobot ACT 가 `observation.language_embedding`(중복 STATE)을 무시하던 문제를 발견 → 학습 데이터셋 feature 를 `observation.environment_state`(ENV)로 rename([`rename_feature.py`](training/imitation_data/scripts/rename_feature.py)) 후 재학습해 `encoder_env_state_input_proj(384→512)` 토큰으로 언어를 실제 사용. 오프라인 검증([`eval_act_offline.py`](training/act/scripts/eval_act_offline.py)) [4] 임베딩 swap 민감도로 확인(정식 판정은 Jetson multi-target rollout). 자세한 경위는 메모리/스크립트 docstring 참조.
>
> **act_node ACT 추론 본체 — closed-loop 구현 완료 (2026-06-04, Jetson 실기 검증 대기)**:
> - `deploy/ros2_ws/src/robot_arm_pkg/robot_arm_pkg/act_node.py` `_load_policy`/`_build_obs`/`_timer_callback` 본체 구현. 매 틱 `policy.select_action()`(closed-loop, temporal ensemble) → `horizon_steps` 만큼 실행 후 SUCCESS 발행(=실행 완료; 성공/실패 판정은 `agent_node.verify_result`). 정규화는 검증된 [`eval_act_offline.py`](training/act/scripts/eval_act_offline.py) 패턴(체크포인트 `*normaliz*.safetensors` mean/std 수동 적용 + action 역정규화 — lerobot 0.4.4 는 정규화를 별도 processor 로 처리).
> - **도메인별 재선택**: `model_root`(기본 `~/models/act`)에서 `SubtaskCommand.domain` → `{root}/{domain}/pretrained_model` 자동 재로드. `model_path` 파라미터로 단일 도메인 고정도 가능.
> - **단위 정합**: `/joint_states`(URDF rad + 그리퍼 URDF 스케일) → SDK `.pos`(deg + 0~100) 역변환해 `observation.state` 구성(`robot_controller_node._read_joints` 의 역, 그리퍼 상수 동기 유지). 출력 `/joint_command` 는 `.pos` 그대로라 robot_controller 가 변환 없이 소비. 이미지 해상도는 정책 `config.json` 에서 읽어 resize.
> - 정식 검증은 Jetson 실기 rollout(이미지 해상도/단위/언어 민감도). 그 전 오프라인 sanity 는 [`eval_act_offline.py`](training/act/scripts/eval_act_offline.py).

> **제거 완료 노드/스크립트**: `vlm_node.py`, `yolo_node.py`, `state_monitor_node.py`,
> `scripts/record_with_vlm.py` — `agent_node` 함수 호출 툴 및 `record_real_teleop.py`로 대체됨.

> **이전 설계 대비 추가 제거 항목**: `agent_node`의 3D 좌표 산출 로직,
> `observation.target_position` 채널, `SubtaskCommand`의 `target_class` /
> `target_position` 필드, depth 노이즈 DR 항목, `sim/generate_act_dataset.py`
> (규칙기반 IK 전문가 — `training/imitation_data/scripts/record_sim_teleop.py` teleop 수집으로 대체).

---

## 미확정 항목 요약

다음 항목은 향후 결정 후 README 갱신이 필요하다.

### 공통

| # | 항목 | 영향 범위 |
|---|------|---------|
| 1 | verify_result 종합 판정 — v1 구현됨(`_combine_verify`: Gemma 우선 + depth place 단방향 veto). depth 우선/가중 등 **최종** 정책은 실기 verify 정확도 측정 후 확정 | `agent_node.verify_result` |

### Sim teleop 수집 (`record_sim_teleop.py`) — [D1]~[D9] 확정

설계 결정 [D1]~[D9]는 모두 확정됐다. **확정 내용 전문은
[`record_sim_teleop.py`](training/imitation_data/scripts/record_sim_teleop.py) 파일 헤더 주석으로
이관**했다 (결정과 코드를 한곳에서 관리). 본 문서 곳곳의 `[D1]`~`[D9]` 참조는
그 주석의 항목과 대응한다.

> `record_sim_teleop.py`는 위 [D1]~[D9] 확정안대로 Isaac Sim standalone 앱 +
> 직접 API 설계로 작성 완료. 핵심 로직(`SimInterface`·`SceneManager`·
> `TargetSelector`)이 모두 구현됐고, 남은 것은 Isaac Sim 5.1 API 실기 검증과
> `multi_target_scenes.yaml` placement 실측값 기입이다 (미구현 항목 표 참조).

> **확정 완료** (이전 미확정 항목):
> - 임베딩 모델: `paraphrase-multilingual-MiniLM-L12-v2` (D=384)
> - 데이터 수집 언어 라벨링 방식: Option C (템플릿+변수 슬롯)
> - `target_description` 형식: 영어 반구조화 `"the {attribute} {object} {spatial}"`,
>   5–15 단어, [`deploy/config/label_vocab.yaml`](deploy/config/label_vocab.yaml) vocabulary 강제
> - Sim 데이터 수집 방식: 순수 teleop (규칙기반 IK 전문가 폐기, 목표 ~500 에피소드)

---
