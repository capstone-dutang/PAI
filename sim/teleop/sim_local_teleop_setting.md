# 캡스톤 PAI — 로컬 환경 텔레오퍼레이션 매뉴얼

> **시나리오**: Isaac Sim이 설치된 **로컬 PC** 한 대에서, SO-101 Leader(USB 연결)를 Isaac Sim 내 follower(디지털 트윈)와 직접 연결해 텔레오퍼레이션을 수행한다.
>
> **클라우드(EC2) 버전과의 차이**: WebSocket/rosbridge 경유가 불필요. Isaac Sim ROS2 Bridge와 LeRobot이 **같은 머신**에 있으므로 DDS(공유 메모리/UDP)로 바로 통신한다. 단위 변환과 Action Graph 토픽 인터페이스(`/joint_command`)는 동일하게 유지한다.

---

## 시스템 전체 그림

```
┌─ 로컬 PC (Ubuntu 22.04 / 24.04 권장) ──────────────────────────────┐
│                                                                    │
│  ┌─ SO-101 Leader (실물) ──┐                                       │
│  │   USB (예: /dev/ttyACM0) │                                       │
│  └────────────┬─────────────┘                                       │
│               │                                                     │
│  ┌─ conda env: lerobot ─────────────────┐                          │
│  │  LeRobot 0.4.4 (Python API, 30Hz)    │                          │
│  │   ↓                                  │                          │
│  │  lerobot_to_ros2.py                  │                          │
│  │   ↓ rclpy publisher                  │                          │
│  └────────────┬─────────────────────────┘                          │
│               │                                                     │
│               │ /joint_command (sensor_msgs/JointState, rad)        │
│               │ DDS — 같은 머신 내 통신                              │
│               ▼                                                     │
│  ┌─ Isaac Sim 5.1.0 ──────────────────────────────────────┐        │
│  │   ROS2 Bridge extension (isaacsim.ros2.bridge)         │        │
│  │   ├─ Action Graph 1: /clock publish                    │        │
│  │   └─ Action Graph 2: Teleop                            │        │
│  │       └─ /joint_command → Articulation Controller      │        │
│  │           → SO-101 follower (디지털 트윈, 6 DOF)        │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                    │
│  ┌─ conda/mamba env: ros_humble (RoboStack, 디버깅용) ─┐            │
│  │  ros2 topic list / echo / info                     │            │
│  └────────────────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────────────┘
```

**핵심**: rosbridge_server, WebSocket, 포트 9090, EC2 Security Group이 전부 빠진다. 그 외 Isaac Sim 측 설정(URDF 임포트, Action Graph, 단위 정합성)은 EC2 버전과 100% 동일하다.

---

## Part 1: 로컬 환경 사전 요건

### 1-1. 하드웨어
- **GPU**: NVIDIA RTX 30/40 시리즈 이상 권장 (Isaac Sim 5.1.0은 RTX 필수)
- **VRAM**: 최소 8GB, 권장 12GB+
- **RAM**: 32GB 권장
- **USB 포트**: SO-101 Leader 컨트롤러 보드용 1개

### 1-2. OS
- **Ubuntu 22.04 LTS** 또는 **24.04 LTS** 권장
- Windows에서도 동작은 가능하나 ROS2 Humble은 WSL2 또는 Docker 경유 필요 → 본 매뉴얼은 **Ubuntu 네이티브** 기준으로 작성

### 1-3. 소프트웨어 사전 설치
| 항목 | 버전 |
|---|---|
| NVIDIA Driver | 535+ |
| Isaac Sim | 5.1.0 (`/opt/IsaacSim/` 또는 설치 디렉토리) |
| Miniconda / Miniforge | 최신 |
| Git | 최신 |

> 💡 Jetson Orin Nano 환경과의 ROS2 Humble 버전 통일 정책은 그대로 유지한다.

---

## Part 2: Isaac Sim 측 설정 (EC2 버전과 동일)

이 부분은 EC2 매뉴얼과 99% 동일하다. 로컬에서는 NICE DCV 없이 바로 GUI를 띄운다는 것만 다르다.

### 2-1. Isaac Sim 실행 환경 변수

`~/.bashrc`에 추가 (또는 Isaac Sim 실행 셸에서):

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

> ⚠️ **함정 주의**: ROS_DOMAIN_ID는 Isaac Sim, LeRobot 브리지, 디버깅용 ROS2 환경 **모두** 42로 통일해야 한다. (EC2 버전에서 가장 자주 걸린 함정)

### 2-2. ROS2 Bridge extension
- Isaac Sim 실행 후 **Window → Extensions**
- `isaacsim.ros2.bridge` 검색 → **Enable + Autoload** ✅

### 2-3. SO-101 URDF 임포트

#### 2-3-1. URDF 다운로드
```bash
mkdir -p ~/Documents && cd ~/Documents
git clone https://github.com/TheRobotStudio/SO-ARM100.git
# 사용 파일: Simulation/SO101/so101_new_calib.urdf
```

#### 2-3-2. URDF Importer 설정 (확정값)
| 항목 | 값 |
|---|---|
| **Create in Stage** | ⭕ (Referenced Model 사용 X) |
| **Static Base** | ✅ |
| **Joint Configuration** | Stiffness 모드 (Natural Frequency X) |
| **Drive Type** | Force |
| **모든 joint Target** | Position |
| **Stiffness** | 10000 |
| **Damping** | 100 |
| **Collider Type** | Convex Decomposition |
| **Ignore Mimic** | ☐ |
| **Self Collision** | ☐ |

#### 2-3-3. 임포트 후 패치 — maxForce 일괄 변경

URDF의 `effort=10`이 그대로 들어와 Stiffness 10000과 맞지 않는 문제. Isaac Sim Script Editor에서 다음 실행:

```python
import omni.usd
from pxr import UsdPhysics

stage = omni.usd.get_context().get_stage()
ROBOT_PRIM = "/so101_new_calib"  # 임포트 결과 경로에 맞춰 수정

for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if not path_str.startswith(ROBOT_PRIM):
        continue
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        drive.GetMaxForceAttr().Set(1000.0)
        print(f"Patched: {path_str}")
print("Done.")
```

#### 2-3-4. Joint 6개 (DOF, revolute)
```
shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper
```

#### 2-3-5. Joint limits (USD 변환 후 — degree)
| Joint | Lower | Upper |
|---|---|---|
| shoulder_pan | -110° | 110° |
| shoulder_lift | -100° | 100° |
| elbow_flex | -97° | 97° |
| wrist_flex | -95° | 95° |
| wrist_roll | -157° | 163° |
| gripper | -10° | 100° |

#### 2-3-6. 씬 저장
```
File → Save As → ~/Documents/so101_scene_local.usd
```
로컬 디스크에 저장한다 (Omniverse Nucleus 미사용).

### 2-4. Action Graph 구성

#### 2-4-1. /clock publish 그래프 (이전 세션과 동일)
- 경로: `/World/ActionGraph`
- 노드: `On Playback Tick` → `ROS2 Publish Clock`

#### 2-4-2. Teleop 그래프 (신규)

**노드 4개**:
1. **On Playback Tick** (Event)
2. **ROS2 Context** (`useDomainIDEnvVar` ✅)
3. **ROS2 Subscribe JointState**
   - `topicName`: `/joint_command`
   - `queueSize`: `10`
4. **Articulation Controller**
   - `targetPrim`: `/so101_new_calib`

**연결 5개**:
```
Tick → Subscribe.ExecIn
Context → Subscribe.context
Subscribe.ExecOut → Articulation Controller.ExecIn
Subscribe.jointNames → Articulation Controller.jointNames
Subscribe.positionCommand → Articulation Controller.positionCommand
```

### 2-5. Isaac Sim 측 검증 (LeRobot 연결 전)

ROS2 디버깅용 셸을 별도로 띄워서:

```bash
mamba activate ros_humble
export ROS_DOMAIN_ID=42

ros2 topic list                        # /joint_command, /clock 보여야 함
ros2 topic info /joint_command         # Subscription count: 1 (Isaac Sim)

# 더미 명령으로 팔이 움직이는지 확인
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState "{
  name: ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'],
  position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
}"
```
→ 뷰포트에서 gripper가 움직이면 통과.

### 2-6. RGB 카메라 셋업

실기 RealSense 캘리브레이션(`deploy/calibration/camera_intrinsics.yaml` /
`camera_robot_transform.yaml`)으로 Isaac Sim USD 에 `Camera_RGB` prim 을 생성·
배치한다. Sim/Real 카메라의 intrinsics·위치를 일치시키는 단계이며,
`record_sim_teleop.py` 가 구독할 카메라 토픽 발행(README [D4])의 prim 전제다.

1. URDF 임포트(2-3) 완료 + 캘리브레이션 yaml 2개 준비 확인
2. Script Editor 에서 [`setup_sim_camera.py`](../scene_setup/setup_sim_camera.py) 실행
   - 파일 상단 "사용자 설정" 블록의 yaml 경로(로컬: `~/projects/PAI/...`)와
     `SO101_BASE_PRIM` / `CAMERA_PRIM` 경로를 스테이지에 맞게 확인
3. Viewport → Cameras → `Camera_RGB` 로 시점 전환해 화각·위치 검증
4. 셋업 후 씬 USD 재저장 (2-3-6)

---

## Part 3: 디버깅용 ROS2 환경 (RoboStack)

시스템 ROS2를 설치하면 Isaac Sim 내장 ROS2 라이브러리와 충돌하므로, **mamba env**로 격리한다.

### 3-1. Miniforge 설치
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
# 셸 재시작
```

### 3-2. ROS2 Humble env 생성
```bash
mamba create -n ros_humble python=3.11 -y
mamba activate ros_humble

# RoboStack 채널로 ROS2 Humble 설치
mamba install -c conda-forge -c robostack-humble ros-humble-desktop -y
```

### 3-3. 활성화 시 자동 환경 변수
```bash
mamba activate ros_humble
mamba env config vars set ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
mamba deactivate && mamba activate ros_humble
```

> ⚠️ 로컬 환경에서는 **rosbridge_server, websocket-client 설치 불필요**. (EC2 매뉴얼의 Part 6에 해당하는 부분 전체 생략)

---

## Part 4: LeRobot 환경 (로컬 PC, conda)

### 4-1. 폴더 구조
```
~/projects/PAI/                          # 캡스톤 레포 (git clone)
└── teleop/
    ├── lerobot_to_ros2.py               # 로컬 버전 브리지 (rclpy 사용)
    ├── lerobot_to_rosbridge.py          # 클라우드 버전 (참고용 보존)
    └── find_port.py
```

> Windows 경로 (`C:\한성대4학년1학기\지능시스템캡스톤\PAI\`)는 로컬 Ubuntu 환경에서는 `~/projects/PAI/`로 대응. GitHub 미러를 clone한다.

### 4-2. Conda env 생성
```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot

pip install "lerobot[feetech]==0.4.4"
# pyserial은 lerobot 의존성으로 자동 설치됨
```

> 💡 **로컬 환경 핵심 차이**: `websocket-client` **불필요**. 대신 ROS2 메시지를 직접 publish하기 위한 `rclpy`가 필요한데, 이는 **mamba env(ros_humble)**에 들어있다. → 브리지 스크립트는 `ros_humble` env에서 실행한다. (4-5 참조)

### 4-3. USB 권한 (Ubuntu)
```bash
sudo usermod -a -G dialout $USER
# 로그아웃 후 재로그인

# 포트 확인
ls /dev/ttyACM* /dev/ttyUSB*  # /dev/ttyACM0 같은 것이 보여야 함

# LeRobot 내장 유틸로도 확인 가능
python -m lerobot.find_port
```

### 4-4. Leader 캘리브레이션

LeRobot 0.4.4 API:
- **모듈**: `lerobot.teleoperators.so_leader`
- **클래스**: `SO101Leader`, `SO101LeaderConfig`
- **Config 인자**: `port: str`, `id: str`, `use_degrees: bool = True`

```bash
conda activate lerobot
python -c "
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
cfg = SO101LeaderConfig(port='/dev/ttyACM0', id='my_leader', use_degrees=True)
leader = SO101Leader(cfg)
leader.connect(calibrate=True)
leader.calibrate()
leader.disconnect()
"
```

→ 저장 위치: `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json`

### 4-5. 환경 통합 전략 (중요)

로컬 환경에서는 `rclpy`(ROS2 Python)와 `lerobot`을 **같은 env**에서 import해야 한다. 두 가지 옵션:

| 옵션 | 설명 | 추천 여부 |
|---|---|---|
| **A. `ros_humble` env에 lerobot 추가 설치** | `mamba activate ros_humble && pip install "lerobot[feetech]==0.4.4"` | ⭐ 권장 |
| B. `lerobot` env에 rclpy 추가 | `pip install rclpy` (PyPI rclpy는 빌드 문제 多) | 비추천 |

**옵션 A 진행**:
```bash
mamba activate ros_humble
pip install "lerobot[feetech]==0.4.4"

# 검증
python -c "import rclpy; from lerobot.teleoperators.so_leader import SO101Leader; print('OK')"
```

이후 브리지 스크립트는 항상 `ros_humble`에서 실행한다.

---

## Part 5: 브리지 스크립트 — 로컬 버전

### 5-1. 동작 흐름 (EC2 버전과의 차이)

| 단계 | EC2 버전 | 로컬 버전 |
|---|---|---|
| Leader 연결 | USB | USB (동일) |
| 데이터 읽기 | `leader.get_action()` | 동일 |
| 단위 변환 | deg → rad, % → rad | 동일 |
| 전송 | **WebSocket → rosbridge → ROS2** | **rclpy publisher → ROS2 (직접)** |
| 토픽 이름 | `/joint_command` | 동일 |
| 메시지 타입 | `sensor_msgs/msg/JointState` | 동일 |

### 5-2. `lerobot_to_ros2.py`

```python
#!/usr/bin/env python3
"""
SO-101 Leader → Isaac Sim follower 텔레오퍼레이션 브리지 (로컬 버전).
- LeRobot 0.4.4 API로 Leader joint state를 읽어
- ROS2 /joint_command 토픽으로 직접 publish한다.
- 같은 머신 내 Isaac Sim ROS2 Bridge가 이를 받아 follower에 적용한다.

실행:
    mamba activate ros_humble
    export ROS_DOMAIN_ID=42
    python lerobot_to_ros2.py --port /dev/ttyACM0 --id my_leader
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


# Isaac Sim 측 gripper joint limit (degree)
GRIPPER_MIN_DEG = -10.0
GRIPPER_MAX_DEG = 100.0

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def gripper_pct_to_rad(pct: float) -> float:
    """Leader gripper 0~100% → Isaac Sim gripper rad."""
    pct = max(0.0, min(100.0, pct))
    deg = GRIPPER_MIN_DEG + (GRIPPER_MAX_DEG - GRIPPER_MIN_DEG) * (pct / 100.0)
    return math.radians(deg)


class TeleopBridge(Node):
    def __init__(self, leader: SO101Leader, rate_hz: float):
        super().__init__("lerobot_teleop_bridge")
        self.leader = leader
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.rate_hz = rate_hz
        self.timer = self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(f"Teleop bridge started at {rate_hz} Hz")

    def tick(self):
        try:
            action = self.leader.get_action()
        except Exception as e:
            self.get_logger().warn(f"get_action failed: {e}")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_ORDER)

        positions = []
        for j in JOINT_ORDER:
            val = action[f"{j}.pos"]
            if j == "gripper":
                positions.append(gripper_pct_to_rad(val))
            else:
                positions.append(math.radians(val))
        msg.position = positions

        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="e.g. /dev/ttyACM0")
    parser.add_argument("--id", default="my_leader", help="calibration id")
    parser.add_argument("--rate", type=float, default=30.0, help="publish Hz")
    args = parser.parse_args()

    cfg = SO101LeaderConfig(port=args.port, id=args.id, use_degrees=True)
    leader = SO101Leader(cfg)
    leader.connect(calibrate=True)

    rclpy.init()
    node = TeleopBridge(leader, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        leader.disconnect()


if __name__ == "__main__":
    main()
```

### 5-3. 단위 정합성 (변동 없음)

| 단계 | 단위 |
|---|---|
| Leader 출력 | degree (5 joint), 0~100% (gripper) |
| 브리지 변환 후 | rad (모든 6 joint) |
| ROS2 JointState payload | rad |
| Isaac Sim Articulation Controller | rad |

### 5-4. 실행

**터미널 1 — Isaac Sim**:
```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
~/IsaacSim/isaac-sim.sh
# → 씬 열기 → ▶ Play
```

**터미널 2 — 브리지**:
```bash
mamba activate ros_humble
export ROS_DOMAIN_ID=42
cd ~/projects/PAI/teleop
python lerobot_to_ros2.py --port /dev/ttyACM0 --id my_leader
```

**터미널 3 (선택) — 디버깅**:
```bash
mamba activate ros_humble
export ROS_DOMAIN_ID=42
ros2 topic hz /joint_command          # ~30Hz 확인
ros2 topic echo /joint_command --once # 값 확인
```

---

## Part 6: 최종 검증 체크리스트

- [ ] Isaac Sim에서 ▶ Play 후 SO-101 follower가 중력에 떨어지지 않음 (Stiffness 동작 확인)
- [ ] `ros2 topic list`에 `/joint_command`, `/clock` 노출
- [ ] `ros2 topic info /joint_command` — Publisher 1 (브리지), Subscription 1 (Isaac Sim)
- [ ] `ros2 topic hz /joint_command` — ~30Hz
- [ ] **Leader arm을 손으로 움직이면 Isaac Sim follower가 실시간 mirror** ✅

---

## 함정 모음 (EC2 함정 + 로컬 추가분)

| 함정 | 해결 |
|---|---|
| ROS_DOMAIN_ID 불일치 (Isaac Sim vs 브리지 vs 디버깅 셸) | 모든 셸에서 `export ROS_DOMAIN_ID=42` |
| URDF "Referenced Model" 임포트 → drive가 articulation에 안 먹힘 | "Create in Stage"로 임포트 |
| URDF effort=10이 maxForce로 들어와 토크 부족 | Python 스크립트로 maxForce=1000 일괄 변경 |
| LeRobot 0.4.4 import 경로 (0.5.x 가이드와 다름) | `lerobot.teleoperators.so_leader` (so101 아님) |
| `set_joint_positions()` vs `apply_action()` | 텔레옵은 drive target (Articulation Controller) 사용 |
| **(로컬 신규)** 시스템 ROS2 설치 시 Isaac Sim 내장 ROS2와 충돌 | mamba env로 격리 (`ros_humble`) |
| **(로컬 신규)** Ubuntu USB 권한 없음 → `/dev/ttyACM0` 접근 거부 | `usermod -a -G dialout $USER` 후 재로그인 |
| **(로컬 신규)** `lerobot` env에 rclpy 없어 import 실패 | rclpy + lerobot을 같은 env(`ros_humble`)에 설치 |
| **(로컬 신규)** `RMW_IMPLEMENTATION` 미설정 → 셸별 DDS 구현 달라 통신 안 됨 | 모든 셸에서 `rmw_fastrtps_cpp` 통일 |

---

## EC2 버전과 비교한 변경 요약

| 항목 | EC2 버전 | 로컬 버전 |
|---|---|---|
| Isaac Sim 호스트 | EC2 g6e.4xlarge | 로컬 PC |
| 접속 방식 | SSH + NICE DCV (8443) | 직접 GUI |
| LeRobot 호스트 | 로컬 PC (Windows) | 로컬 PC (Ubuntu, 같은 머신) |
| 브리지 통신 | WebSocket → rosbridge → ROS2 | **rclpy → ROS2 (직접)** |
| 추가 의존성 | `websocket-client`, `rosbridge-suite` | `rclpy` (mamba env 내장) |
| Security Group | SSH(22), DCV(8443), **rosbridge(9090)** | 불필요 |
| 비용 | ~$2.25/hr (Stop 정책) | 전기료 |
| 지연 | 네트워크 RTT (서울 리전) + WS 오버헤드 | DDS 로컬 (~ms 미만) |
