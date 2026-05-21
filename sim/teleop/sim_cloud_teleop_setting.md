---

# 캡스톤 PAI — 시뮬레이션 텔레오퍼레이션 인프라 구축 정리

## 시스템 전체 그림

```
┌─ 로컬 PC (Windows) ─────────────┐         ┌─ EC2 g6e.4xlarge (Ubuntu 24.04) ──────┐
│                                 │         │                                       │
│  SO-101 Leader (실물)           │         │  Isaac Sim 5.1.0                      │
│    ↓ USB (COM 포트)             │         │   ├─ SO-101 follower (디지털 트윈)    │
│  LeRobot 0.4.4 (conda env)      │         │   ├─ Action Graph 1: /clock publish   │
│    ↓ Python API, 30Hz           │         │   └─ Action Graph 2: Teleop           │
│  lerobot_to_rosbridge.py        │         │       └─ /joint_command → Articulation│
│    ↓ WebSocket                  │         │                                       │
│                                 │         │  ROS2 Humble (RoboStack mamba env)    │
└──────────┼──────────────────────┘         │   ├─ rosbridge_server :9090           │
           │                                │   └─ ros2 CLI 디버깅용                │
           │  ws://EC2_IP:9090              │                                       │
           └────────────────────────────────┼─►                                     │
                                            └───────────────────────────────────────┘
```

---

## Part 1: EC2 인프라

### 1-1. EC2 인스턴스
- **타입**: g6e.4xlarge (NVIDIA L40s 48GB VRAM, 128GB RAM)
- **AMI**: NVIDIA Isaac Sim Development Workstation 2026.1.1
- **OS**: Ubuntu 24.04
- **스토리지**: EBS gp3 512GB
- **리전**: Seoul (ap-northeast-2)
- **비용 정책**: 종료 시 Stop만 (Terminate 절대 X), ~$2.25+/hr

### 1-2. 접속
- **SSH key**: `isaac_sim_key.pem`
- **NICE DCV**: 포트 8443, GUI 접속
- **Security Group**: SSH(22), DCV(8443) — Seongbin IP /32 한정

### 1-3. Isaac Sim
- **설치 위치**: `/opt/IsaacSim/` (심볼릭 링크 `~/IsaacSim/`)
- **버전**: 5.1.0
- **ROS2 Bridge extension**: `isaacsim.ros2.bridge` Enable + Autoload 켬

### 1-4. ROS2 (외부 디버깅용)
- **방식**: RoboStack mamba env (시스템 ROS2 설치 X — Isaac Sim 내장 라이브러리와 충돌 방지)
- **env 이름**: `ros_humble`
- **Python**: 3.11
- **이유**: Jetson Orin Nano의 ROS2 Humble과 버전 통일

### 1-5. 환경 변수 (Isaac Sim과 외부 ROS2 통신용)
- `ROS_DOMAIN_ID=42` (양쪽 동일)
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`

### 1-6. 검증 완료한 것
- ✅ Isaac Sim ↔ 외부 ROS2 토픽 통신 가능 (도메인 ID 42)
- ✅ Action Graph로 `/clock` publish 작동
- ✅ `ros2 topic list`, `ros2 topic echo`로 토픽 흐름 확인 (별도 셸에서)

---

## Part 2: SO-101 URDF 임포트

### 2-1. URDF 다운로드
- **레포**: `https://github.com/TheRobotStudio/SO-ARM100.git`
- **위치**: `~/Documents/SO-ARM100/`
- **사용 파일**: `Simulation/SO101/so101_new_calib.urdf`

### 2-2. URDF Importer 설정 (재임포트 후 확정값)
| 항목 | 값 |
|---|---|
| **Create in Stage** | ⭕ (Referenced Model X — layer 꼬임 방지) |
| **Static Base** | ✅ |
| **Joint Configuration** | Stiffness 모드 (Natural Frequency X) |
| **Drive Type** | Force |
| **모든 joint Target** | Position |
| **Stiffness** | 10000 |
| **Damping** | 100 |
| **Collider Type** | Convex Decomposition |
| **Ignore Mimic** | ☐ |
| **Self Collision** | ☐ |

### 2-3. 임포트 후 추가 패치
- **maxForce** 10 → 1000 (Python 스크립트로 일괄 변경)
  - URDF의 `effort=10`이 그대로 들어와서 Stiffness 10000과 안 맞음
- **prim 경로**: `/so101_new_calib` (World 밖 — 임포트 시 World 미선택)

### 2-4. Joint 6개 (DOF, revolute)
```
shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper
```

### 2-5. Joint limits (USD 변환 후 — degree 단위)
| Joint | Lower | Upper |
|---|---|---|
| shoulder_pan | -110° | 110° |
| shoulder_lift | -100° | 100° |
| elbow_flex | -97° | 97° |
| wrist_flex | -95° | 95° |
| wrist_roll | -157° | 163° |
| gripper | -10° | 100° |

### 2-6. 저장
- **씬 USD**: `/home/ubuntu/Documents/so101_scene_v2.usd` (또는 비슷)
- **저장 위치**: 로컬 디스크 (Omniverse Nucleus 미사용)

---

## Part 3: Action Graph — 텔레오퍼레이션

### 3-1. 기존 그래프 (이전 세션부터 유지)
- `/World/ActionGraph` — on_playback_tick + ros2_publish_clock

### 3-2. 새 그래프 (이번 세션 추가)
- 노드 4개:
  1. **On Playback Tick** (Event)
  2. **ROS2 Context** (`useDomainIDEnvVar` ✅)
  3. **ROS2 Subscribe JointState** (`topicName: /joint_command`, `queueSize: 10`)
  4. **Articulation Controller** (`targetPrim: /so101_new_calib`)

### 3-3. 연결 5개
```
Tick → Subscribe.ExecIn
Context → Subscribe.context
Subscribe.ExecOut → Articulation Controller.ExecIn
Subscribe.jointNames → Articulation Controller.jointNames
Subscribe.positionCommand → Articulation Controller.positionCommand
```

### 3-4. 검증 완료
- ✅ `ros2 topic list`에 `/joint_command` 노출
- ✅ `ros2 topic info`에서 Subscription count 1
- ✅ `ros2 topic pub`로 더미 명령 전송 시 팔 동작

---

## Part 4: 로컬 PC — LeRobot 환경

### 4-1. 폴더 구조 결정
```
C:\한성대4학년1학기\지능시스템캡스톤\PAI\
└── teleop\                          ← 캡스톤 레포 안
    ├── lerobot_to_rosbridge.py      ← 브리지 스크립트
    └── (find_port.py 등 유틸)
```

LeRobot 라이브러리 자체는 PAI 레포 밖 (Miniconda env)에 설치.

### 4-2. Conda env
- **env 이름**: `lerobot`
- **Python**: (Miniconda 설치 시 기본값)
- **위치**: `C:\Users\india\.conda\envs\lerobot\`

### 4-3. 설치 패키지
- **lerobot**: `0.4.4` (Feetech extras 포함 — `pip install "lerobot[feetech]==0.4.4"`)
- **websocket-client**: WebSocket 클라이언트
- **pyserial**: USB 포트 탐지용 (lerobot 의존성으로 자동)

### 4-4. USB 드라이버
- Leader arm controller board 자동 인식 (별도 드라이버 설치 없이)
- COM 포트 확인 완료

### 4-5. LeRobot 0.4.4 API 확정
- **모듈 경로**: `lerobot.teleoperators.so_leader`
- **클래스**: `SO101Leader`, `SO101LeaderConfig` (둘 다 `SOLeader`/`SOLeaderTeleopConfig`의 alias)
- **Config 인자**:
  - `port: str` (예: "COM5")
  - `id: str` (캘리브레이션 식별자)
  - `use_degrees: bool = True` (기본값 — 각도 단위)
- **주요 메서드**: `connect(calibrate=True)`, `get_action()`, `disconnect()`, `calibrate()`
- **`get_action()` 반환**: `{"shoulder_pan.pos": deg, ..., "gripper.pos": 0~100}` (gripper는 RANGE_0_100)

### 4-6. 캘리브레이션
- ✅ 완료
- **저장 위치**: `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<id>.json`

---

## Part 5: 브리지 스크립트

### 5-1. 동작 흐름
1. SO101Leader 연결 (USB)
2. EC2 rosbridge_server에 WebSocket 연결
3. `/joint_command` 토픽 advertise (`sensor_msgs/msg/JointState`)
4. 30Hz 루프:
   - leader.get_action() — 6개 joint를 degree (gripper는 %)로 읽음
   - 5개 joint: degree → rad 변환
   - gripper: 0~100% → Isaac Sim gripper rad으로 linear 매핑 (-10° ~ 100° → rad)
   - WebSocket으로 JSON publish

### 5-2. 단위 정합성
| 단계 | 단위 |
|---|---|
| Leader 출력 | degree (5 joint), 0~100% (gripper) |
| 브리지 변환 후 | rad (모든 6 joint) |
| WebSocket payload | rad |
| rosbridge → ROS2 JointState | rad |
| Isaac Sim Articulation Controller | rad으로 받아서 내부 처리 |

### 5-3. 실행 명령
```cmd
conda activate lerobot
cd C:\한성대4학년1학기\지능시스템캡스톤\PAI\teleop
python lerobot_to_rosbridge.py --port COM<x> --ec2-ip <EC2_IP>
```

---

## Part 6: EC2 추가 설정

### 6-1. Security Group 추가
- **Type**: Custom TCP
- **Port**: 9090
- **Source**: 로컬 PC 공인 IP /32

### 6-2. rosbridge_server 설치
```bash
mamba activate ros_humble
mamba install -c robostack-humble ros-humble-rosbridge-suite -y
```

### 6-3. rosbridge_server 실행
```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

---

## ✅ 최종 검증 완료
**로컬 leader arm 손으로 움직임 → Isaac Sim 안 SO-101이 실시간으로 mirror**

---

## 자주 걸렸던 함정 모음 (다음에 같은 작업 시 참고)

| 함정 | 해결 |
|---|---|
| ROS_DOMAIN_ID 불일치 (Isaac Sim 42 vs Docker 0) | 양쪽 환경 변수 통일 |
| URDF "Referenced Model" 임포트 → drive 변경이 articulation에 안 먹힘 | "Create in Stage"로 임포트 |
| URDF 임포트 시 Stage 트리에서 World 미선택 → prim이 root 바로 아래로 들어감 | 임포트 전 `/World` 클릭, 또는 경로 그대로 사용 |
| URDF effort 값이 maxForce로 들어옴 (10) → Stiffness 10000과 불일치 → 토크 부족 | Python으로 maxForce=1000 일괄 변경 |
| `omniverse:///` 경로로 저장 시도 → 실패 | 로컬 디스크 (`/home/ubuntu/Documents/`)로 저장 |
| LeRobot 0.4.4 import 경로가 0.5.x 가이드와 다름 | `lerobot.teleoperators.so_leader` (so101 아님) |
| `ros2 topic pub`의 publisher 출력 ≠ 수신 확인 | `ros2 topic info`의 Subscription count + 뷰포트로 검증 |
| `set_joint_positions()` (순간이동)과 `apply_action()` (drive target) 차이 | 텔레옵은 drive target 사용 |