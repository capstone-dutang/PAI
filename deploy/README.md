# 배포 가이드 (Jetson)

`deploy/` 하위 ROS2 노드를 Jetson 에서 빌드하고 운영하는 절차입니다.
설계·노드 구성·토픽 그래프는 [`README.md`](../README.md) "ROS2 노드 및 토픽" 절 참조.

## 사전 조건

- `deploy/scripts/setup_jetson_base.sh` 실행 완료 (ROS2 + LeRobot 베이스 환경)
- 모델 배치 — `~/models/gemma/{gemma_4_e2b_q4.gguf, adapters/adapter_<domain>.gguf}`,
  `~/models/embedding/`, `~/models/act/act.pth`
- 캘리브레이션 결과 배치 — `~/calibration/{camera_intrinsics.yaml,
  camera_robot_transform.yaml, grasp_roi.yaml, home_pose.yaml}`
  (캘리브레이션 절차는 [`calibration/README.md`](calibration/README.md) 참조)

## 빌드

저장소 루트에서 (개발 PC 확인용):

```bash
cd deploy/ros2_ws
colcon build --packages-select robot_arm_msgs robot_arm_pkg
source install/setup.bash
```

Jetson 실기 배포 시 다음을 복사한 후 빌드:

```bash
# 1) 저장소의 deploy/ros2_ws/src/* 를 Jetson의 ~/ros2_ws/src/ 로 복사
cp -r <repo>/deploy/ros2_ws/src/* ~/ros2_ws/src/

# 2) 런타임 설정 복사
mkdir -p ~/ros2_ws/config ~/calibration
cp <repo>/deploy/config/domain_map.yaml  ~/ros2_ws/config/domain_map.yaml
cp <repo>/deploy/config/label_vocab.yaml ~/ros2_ws/config/label_vocab.yaml
cp <repo>/deploy/config/grasp_roi.yaml   ~/calibration/grasp_roi.yaml   # 캘리브레이션 후 실측값으로 갱신

# 3) 빌드
cd ~/ros2_ws
colcon build --packages-select robot_arm_msgs robot_arm_pkg
source install/setup.bash
```

## 배포 실행

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
