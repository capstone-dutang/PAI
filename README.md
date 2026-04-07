# PAI — Pick-and-place AI with VLM + ACT Policy

RGB-D 카메라로 물체를 인식하고, VLM이 3D 위치를 추론한 뒤 ACT Policy가 관절 시퀀스를 생성하여 SO-101 로봇 팔을 제어하는 픽앤플레이스 시스템.

---

## 실행 환경

### Jetson (배포)

| 항목 | 내용 |
|------|------|
| 하드웨어 | NVIDIA Jetson Orin Nano 8GB |
| JetPack | 6.2.1 |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.6 |
| cuDNN | 9.3.0 |
| TensorRT | 10.3.0 |
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
│       └── act_policy.trt
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
| TensorRT | 10.3.0 (Jetson과 버전 일치 필요) |
| 주요 작업 | VLM LoRA 파인튜닝, ACT 모방학습, Isaac Sim 시뮬레이션 |

---

## 하드웨어

| 항목 | 내용 |
|------|------|
| 카메라 | RGB-D 센서 (미확정) |
| 로봇 팔 | SO-101 (HuggingFace LeRobot) |
| 추론 가속 | NVIDIA TensorRT (Jetson 내장 GPU) |

---

## 소프트웨어 의존성

### ROS2 패키지 (apt / rosdep)

```bash
sudo apt install ros-humble-rclpy \
                 ros-humble-sensor-msgs \
                 ros-humble-std-msgs \
                 ros-humble-geometry-msgs
```

### Python 패키지 (pip)

```bash
pip install -r ros2_ws/src/robot_arm_pkg/requirements.txt
```

| 패키지 | 버전 | 용도 |
|--------|------|------|
| setuptools | ≥58.2.0 | ROS2 Humble 빌드 호환 |
| numpy | ≥1.21.0 | 이미지/텐서 데이터 처리 |
| requests | ≥2.28.0 | llama.cpp 서버 HTTP 통신 |
| tensorrt | — | ACT Policy 추론 (NVIDIA 공식 설치 필요) |
| lerobot | — | SO-101 모터 제어 API |

### TensorRT 별도 설치

TensorRT는 pip만으로 설치되지 않습니다. NVIDIA 공식 가이드를 따라 설치 후 Python 바인딩을 활성화하세요.

> https://docs.nvidia.com/deeplearning/tensorrt/install-guide

---

## ROS2 패키지 구조

```
ros2_ws/
└── src/
    └── robot_arm_pkg/
        ├── robot_arm_pkg/
        │   ├── camera_node.py           # RGB-D 카메라 입력
        │   ├── vlm_node.py              # VLM 추론 (Qwen3.5-4B GGUF)
        │   ├── act_node.py              # ACT Policy 추론 (TensorRT)
        │   ├── robot_controller_node.py # SO-101 모터 제어
        │   └── state_monitor_node.py    # 태스크 성공/실패 판단
        ├── package.xml
        ├── setup.py
        └── requirements.txt
```

---

## ROS2 노드 및 토픽

### 노드 요약

| 노드 | 실행 주기 | 역할 |
|------|-----------|------|
| `camera_node` | 30 Hz | RGB-D 카메라 프레임 발행 |
| `vlm_node` | 0.4 Hz | Qwen3.5-4B로 물체 3D 위치 추론 |
| `act_node` | 30 Hz | ACT Policy로 관절 시퀀스 예측 |
| `robot_controller_node` | 1000 Hz | SO-101 모터 구동 및 관절 상태 피드백 |
| `state_monitor_node` | 10 Hz | 성공/실패 판단, 재시도 트리거 |

### 토픽 구조

```
/task_command  (std_msgs/String)        ← 사용자 입력
      │
      ▼
[camera_node]
  ├─ /camera/rgb   (sensor_msgs/Image, 30Hz)
  └─ /camera/depth (sensor_msgs/Image, 30Hz)
      │
      ▼
[vlm_node] ◀── /reflection_feedback (재시도 요청)
  └─ /object_pose (geometry_msgs/PoseStamped, 0.4Hz)
      │
      ▼
[act_node] ◀── /joint_states
  └─ /joint_command (sensor_msgs/JointState, 30Hz)
      │
      ▼
[robot_controller_node]
  └─ /joint_states (sensor_msgs/JointState, 1000Hz)
      │
      ▼
[state_monitor_node]
  └─ /reflection_feedback (std_msgs/String) ──▶ [vlm_node]
```

---

## 빌드 및 실행

### 빌드

```bash
cd ros2_ws
colcon build --packages-select robot_arm_pkg
source install/setup.bash
```

### 노드 실행

```bash
# 카메라
ros2 run robot_arm_pkg camera_node

# VLM (llama.cpp 서버 주소 및 모델 경로 지정 필요)
ros2 run robot_arm_pkg vlm_node \
  --ros-args -p model_path:=/path/to/vlm.gguf \
             -p server_url:=http://localhost:8080

# ACT Policy (TensorRT 엔진 경로 지정 필요)
ros2 run robot_arm_pkg act_node \
  --ros-args -p model_path:=/path/to/act_policy.trt \
             -p chunk_size:=100

# 로봇 컨트롤러
ros2 run robot_arm_pkg robot_controller_node

# 상태 모니터
ros2 run robot_arm_pkg state_monitor_node

# 태스크 명령 발행 (예시)
ros2 topic pub /task_command std_msgs/String "data: 'pick up the red cube'"
```

---

## Jetson 메모리 배분 (Orin Nano 8GB)

| 컴포넌트 | RAM |
|----------|-----|
| OS + ROS2 | ~1.5 GB |
| VLM (Qwen3.5-4B GGUF Q4) | ~4.0 GB |
| ACT Policy (TensorRT) | ~0.3 GB |
| RGB-D 드라이버 + 버퍼 | ~0.3 GB |
| KV Cache / 기타 | ~0.5 GB |
| **합계** | **~6.6 GB** |

---

## 미구현 항목 (TODO)

| 노드 | 미구현 메서드 | 내용 |
|------|--------------|------|
| `camera_node` | `PlaceholderAdapter.get_frames()` | 실제 RGB-D 드라이버 연동 |
| `vlm_node` | `_run_inference()` | llama.cpp HTTP 추론 구현 |
| `act_node` | `_predict_chunk()` | TensorRT 추론 구현 |
| `robot_controller_node` | `_send_command()`, `_read_joints()` | LeRobot API 연동 |
| `state_monitor_node` | `_evaluate()` | 성공/실패 판단 기준 구현 |
