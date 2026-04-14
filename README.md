# PAI — Pick-and-place AI with VLM + ACT Policy

RGB-D 카메라로 물체를 인식하고, VLM이 3D 위치를 추론한 뒤 ACT Policy가 관절 시퀀스를 생성하여 SO-101 로봇 팔을 제어하는 픽앤플레이스 시스템.

---

## 전체 파이프라인

```
Part A  하드웨어 세팅 (카메라 캘리브레이션)
  │
  ├─▶ Part B  Isaac Sim 합성 데이터 생성
  │     │
  │     ▼
  │   Part C  VLM LoRA 파인튜닝 (외부 PC)
  │     │         → vlm_qwen3.5_4b_q4.gguf
  │     │
  │     ▼
  │   Part D  시연 데이터 수집 (Jetson) ← VLM 실시간 구동 필요
  │             record_with_vlm.py 사용
  │             → LeRobot dataset (observation.state + object_pose + action)
  │
  └─▶ Part E  ACT 모방학습 (외부 PC, LeRobot)
                → act_policy.pth
                    │
                    ▼
              Part F  Jetson 통합 배포 (VLM + ACT → 비동기 추론 루프)
                    │
                    ▼
              Part G  Isaac Sim 시뮬레이션 검증 및 비교 실험
```

> Part C 완료 후 Part D 시작 가능 (데이터 수집 중 VLM 실시간 구동 필요)

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
│       └── robot_arm_pkg/
│
├── models/
│   ├── vlm/
│   │   └── vlm_qwen3.5_4b_q4.gguf
│   └── act/
│       └── act_policy.pth
│
├── calibration/
│   ├── camera_intrinsics.yaml
│   └── camera_robot_transform.yaml
│
└── llama.cpp/
    └── llama-server
```

### 외부 PC (학습 / Isaac Sim)

| 항목 | 내용 |
|------|------|
| OS | Ubuntu |
| GPU VRAM | 16GB 이상 |
| CUDA | 12.6 |
| 주요 작업 | VLM LoRA 파인튜닝 (Part C), ACT 모방학습 (Part E), Isaac Sim 시뮬레이션 |

---

## 하드웨어

| 항목 | 내용 |
|------|------|
| 카메라 | Intel RealSense (RGB-D) |
| 로봇 팔 | SO-101 (HuggingFace LeRobot) |
| 리더 팔 | SO-101 Leader (데이터 수집 시 사용) |

---

## Jetson 환경 설치

저장소 루트의 스크립트로 환경을 구성합니다.

| 스크립트 | 대상 작업 |
|----------|-----------|
| `scripts/setup_jetson_base.sh` | 모방 데이터 수집 (Part D), 실제 로봇 구동 (Part F) |
| `scripts/setup_calibration.sh` | 카메라 캘리브레이션 (Part A) — base 설치 후 실행 |

```bash
# 1. 기본 환경 설치 (Part D, F 공통)
chmod +x scripts/setup_jetson_base.sh
./scripts/setup_jetson_base.sh

# 2. 캘리브레이션 환경 추가 설치 (Part A, 필요 시)
chmod +x scripts/setup_calibration.sh
./scripts/setup_calibration.sh
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
| requests | ≥2.28.0 | llama.cpp 서버 HTTP 통신 |
| torch | ≥2.0.0 | ACT Policy 추론, SO-101 모터 제어 |
| lerobot | 0.4.4 | SO-101 SDK, LeRobotDataset, ACTPolicy |
| ultralytics | ≥8.0.0 | YOLOv8n 탐지 (yolo_node) |
| PyYAML | ≥6.0 | camera_robot_transform.yaml 파싱 (yolo_node) |

---

## ROS2 패키지 구조

```
PAI/
├── scripts/
│   ├── setup_jetson_base.sh             # 기본 환경 설치 (Part D, F)
│   └── setup_calibration.sh             # 캘리브레이션 환경 설치 (Part A)
│
└── ros2_ws/
    └── src/
        └── robot_arm_pkg/
            ├── robot_arm_pkg/
            │   ├── camera_node.py           # RGB-D 카메라 입력
            │   ├── vlm_node.py              # VLM 추론 (Qwen3.5-4B GGUF) → /target_class
            │   ├── yolo_node.py             # YOLO 실시간 탐지 → /object_pose (30Hz)
            │   ├── act_node.py              # ACT Policy 추론 (LeRobot .pth)
            │   ├── robot_controller_node.py # SO-101 팔로워 모터 제어 (LeRobot SDK)
            │   └── state_monitor_node.py    # 태스크 성공/실패 판단
            ├── scripts/
            │   └── record_with_vlm.py       # 데이터 수집 스크립트 (Part D)
            ├── package.xml
            ├── setup.py
            └── requirements.txt
```

---

## ROS2 노드 및 토픽

### 배포 단계 노드 구성

| 노드 | 실행 주기 | 역할 |
|------|-----------|------|
| `camera_node` | 30 Hz | RGB-D 카메라 프레임 발행 |
| `vlm_node` | 0.4 Hz | Qwen3.5-4B로 집을 물체 클래스 결정 → `/target_class` |
| `yolo_node` | 30 Hz | 타겟 클래스 필터링 후 3D 좌표 추론 → `/object_pose` |
| `act_node` | 30 Hz | ACT Policy로 관절 시퀀스 예측 |
| `robot_controller_node` | 1000 Hz | SO-101 팔로워 모터 구동 및 관절 상태 피드백 |
| `state_monitor_node` | 10 Hz | 성공/실패 판단, 재시도 트리거 |

### 토픽 구조

```
사용자
  └─ /task_command (std_msgs/String)
        │
        ├──────────────────────────────────────────────────▶ [state_monitor_node]
        │                                                         │  └─ /task_active (std_msgs/Bool)
        │                                                         │        │
        ▼                                                         │        ├──▶ [vlm_node]
[vlm_node] ◀── /reflection_feedback ◀─────────────────────────── │        ├──▶ [yolo_node]
  └─ /target_class (std_msgs/String, 0.4Hz)                       │        └──▶ [act_node]
        │                                                         │
        ▼                                                         │
[camera_node]                                                     │
  ├─ /camera/rgb   (sensor_msgs/Image, 30Hz)                      │
  └─ /camera/depth (sensor_msgs/Image, 30Hz)                      │
        │                                                         │
        ├─────────────────────────────────────────────────────────┤
        ▼                                                         │
[yolo_node] ◀── /target_class                                     │
  └─ /object_pose (geometry_msgs/PoseStamped, base_link, 30Hz)   │
        │                                                         │
        ▼                                                         │
[act_node] ◀── /joint_states ◀────────────────────────────────────┤
  └─ /joint_command (sensor_msgs/JointState, 30Hz)               │
        │                                                         │
        ▼                                                         │
[robot_controller_node]                                           │
  └─ /joint_states (sensor_msgs/JointState, 1000Hz) ─────────────┘
                                                     (state_monitor도 구독)
```

---

## 빌드 및 실행

### 빌드

```bash
cd ros2_ws
colcon build --packages-select robot_arm_pkg
source install/setup.bash
```

### 배포 실행 (Part F)

```bash
# 카메라
ros2 run robot_arm_pkg camera_node

# llama.cpp 서버 (VLM 추론용)
~/llama.cpp/llama-server \
  -m ~/models/vlm/vlm_qwen3.5_4b_q4.gguf \
  --host 0.0.0.0 --port 8080

# VLM 노드 (/target_class 발행)
ros2 run robot_arm_pkg vlm_node \
  --ros-args -p model_path:=~/models/vlm/vlm_qwen3.5_4b_q4.gguf \
             -p server_url:=http://localhost:8080

# YOLO 노드 (/object_pose 30Hz 발행)
ros2 run robot_arm_pkg yolo_node \
  --ros-args -p model_path:=~/models/yolo/yolo_model.pt

# ACT Policy (LeRobot .pth 체크포인트)
ros2 run robot_arm_pkg act_node \
  --ros-args -p model_path:=~/models/act/act_policy.pth \
             -p chunk_size:=100

# 로봇 컨트롤러 (SO-101 팔로워)
ros2 run robot_arm_pkg robot_controller_node \
  --ros-args -p robot_port:=/dev/ttyACM0 \
             -p robot_id:=follower

# 상태 모니터
ros2 run robot_arm_pkg state_monitor_node

# 태스크 명령 발행
ros2 topic pub /task_command std_msgs/String "data: 'pick up the red cube'"
```

---

## 카메라-로봇 캘리브레이션 (Part A)

Hand-Eye 캘리브레이션 (Eye-to-Hand 방식): 카메라 고정, SO-101 엔드이펙터에 ArUco 마커 부착.

### Jetson 사전 설치

```bash
# setup_jetson_base.sh 실행 후 아래 스크립트 실행
./scripts/setup_calibration.sh
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
cd ~/ros2_ws
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
ros2 topic echo /camera/color/camera_info --once | \
  python3 -c "
import sys, yaml, rclpy
# 또는 아래 명령으로 직접 저장:
" 
# 간단한 방법: camera_info 토픽을 YAML로 저장
ros2 topic echo /camera/color/camera_info --once > ~/calibration/camera_intrinsics.yaml
```

---

## 데이터 수집 (Part D)

LeRobot 네이티브 파이프라인을 사용하며, VLM의 `object_pose`를 observation에 포함하여 기록.

### 실행 전 준비

1. Part C 완료 → `vlm_node` 실행 가능 상태
2. SO-101 리더/팔로워 USB 포트 확인
3. LeRobot 캘리브레이션 완료 (`lerobot-calibrate`)

### 실행 순서

```bash
# 1. RealSense 드라이버 실행
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# 2. camera_node 실행 (/camera/rgb, /camera/depth 발행)
ros2 run robot_arm_pkg camera_node

# 3. llama.cpp 서버 실행
~/llama.cpp/llama-server \
  -m ~/models/vlm/vlm_qwen3.5_4b_q4.gguf \
  --host 0.0.0.0 --port 8080

# 4. VLM 노드 실행 (/target_class 발행)
ros2 run robot_arm_pkg vlm_node \
  --ros-args -p server_url:=http://localhost:8080

# 4-1. YOLO 노드 실행 (/object_pose 발행)
ros2 run robot_arm_pkg yolo_node \
  --ros-args -p model_path:=~/models/yolo/yolo_model.pt

# 5. 데이터 수집 스크립트 실행
python ros2_ws/src/robot_arm_pkg/scripts/record_with_vlm.py \
  --robot-port      /dev/ttyACM0 \
  --robot-id        follower \
  --teleop-port     /dev/ttyACM1 \
  --teleop-id       leader \
  --dataset-repo-id myname/my_dataset \
  --single-task     "pick and place" \
  --num-episodes    50
```

### 데이터셋 구조

| 채널 | shape | 내용 |
|------|-------|------|
| `observation.state` | (6,) | 팔로워 관절 위치 (SO-101 6축) |
| `observation.object_pose` | (3,) | YOLO 추론 결과 — robot base_link frame 3D 좌표 (x, y, z) |
| `action` | (6,) | 리더 관절 값 (정답 행동) |

> 카메라 채널(`observation.images.top`)은 Intel RealSense `/camera/color/image_raw` 토픽을 사용

---

## Jetson 메모리 배분 (Orin Nano 8GB)

| 컴포넌트 | RAM |
|----------|-----|
| OS + ROS2 | ~1.5 GB |
| VLM (Qwen3.5-4B GGUF Q4) | ~4.0 GB |
| ACT Policy (PyTorch .pth) | ~0.3 GB |
| RGB-D 드라이버 + 버퍼 | ~0.3 GB |
| KV Cache / 기타 | ~0.5 GB |
| **합계** | **~6.6 GB** |

---

## 미구현 항목 (TODO)

| 파일 | 미구현 항목 | 선행 조건 |
|------|------------|----------|
| `camera_node.py` | ~~`PlaceholderAdapter.get_frames()`~~ 구현 완료 (ROS2 드라이버 릴레이) | - |
| `vlm_node.py` | `_run_inference()` 구현 (llama.cpp 서버 HTTP 통신, 응답 파싱) | Part C 완료 |
| `yolo_node.py` | YOLO 모델 학습/변환 후 `model_path` 파라미터 지정 (코드 구현 완료) | YOLO 모델 준비 |
| `act_node.py` | `_load_policy()`, `_predict_chunk()` | Part E 완료 |
| `state_monitor_node.py` | `_evaluate()` | 성공/실패 기준 확정 |
| `record_with_vlm.py` | ~~카메라 채널 추가~~ 구현 완료 (`observation.images.top`) | - |
