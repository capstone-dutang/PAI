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

## 에이전트 함수 호출 루프 (agent_node)

### 시퀀스 (freeze)

```
look_scene → execute_subtask → /go_home → verify_result
```

- `_run_function_loop` 가 Gemma 함수 호출 응답을 파싱해 `_call_*()` 핸들러로
  디스패치.
- `execute_subtask` 결과 수신 후 `verify_result` 호출 **직전** 에 `/go_home`
  서비스 1회 호출 — 카메라 시야 안의 팔 자세를 home_pose 로 고정해 verify
  학습 분포 (모두 home_pose 로 렌더) 와 일치 보장.
- 캐노니컬 위치: [agent_node._run_function_loop](ros2_ws/src/robot_arm_pkg/robot_arm_pkg/agent_node.py),
  [prompts.SYSTEM_PROMPT](ros2_ws/src/robot_arm_pkg/robot_arm_pkg/prompts.py).

### verify_result fallback (2026-06-02 추가)

LoRA 학습된 모델의 마지막 turn 응답이 `{"function":"verify_result", ...}` JSON
이 **아닌** 경우 (빈 응답 / 자연어 / 다른 함수) **agent_node 가 verify_result
를 강제 호출** 한다. `args.original_command` 는 `self._user_command` 캐시에서
가져온다 (학습 분포의 stateless 인터페이스와 호환).

배경: tableware function_call 학습이 vocab cluster gap (plate/fork/spoon 3종)
때문에 smoke test 32개 중 7개만 모델이 verify_result 호출을 정상 생성
(recycling 은 8/8 정상). 양 도메인 공통으로 fallback 을 적용해 함수 호출 루프의
**결정론적 종료** 를 보장한다. 자세한 학습/검증 결과는
[training/README.md](../training/README.md) "2-7. 학습 결과 및 알려진 이슈"
참조.

`_run_function_loop` 구현 시 의사코드:

```python
exec_status = self._call_execute_subtask(target)
self._call_go_home()

gemma_resp   = http_chat_completion(messages_with_tool_result(exec_status))
parsed       = try_parse_json(gemma_resp)  # 실패 시 None
is_verify    = (
    isinstance(parsed, dict)
    and parsed.get('function') == 'verify_result'
    and isinstance(parsed.get('args'), dict)
)

if is_verify:
    verify_status = self._call_verify_result(**parsed['args'])
else:
    self.get_logger().warn(
        f'마지막 turn 응답이 verify_result 함수 호출 아님 — 코드 강제 호출. '
        f'gemma_resp={gemma_resp!r}'
    )
    verify_status = self._call_verify_result(
        original_command=self._user_command)
```

### verify reasoning 환각 / status enum 정확도

`train_gemma_qlora.py` 의 `finetune_vision_layers=False` 로 vision encoder 가
base 그대로라서 verify 의 JSON 형식 / status enum 어휘는 정확하지만 실제
이미지 → status 정답률은 낮음 (smoke test 5-6/8 sample 에서 SUCCESS↔FAIL 혼동,
blurred → UNCLEAR 미인식). reasoning 내용은 환각 다수.

**현재 정책**: agent_node 는 `verify_result` 의 `status` enum 만 분기에 사용,
`reasoning` 은 로그용. 잘못된 status 는 다음으로 보완:

- `max_verify_iterations` 재시도 (CLI 인자, 위 실행 예제 참조)
- `_depth_grasp_check` depth 보조 검증 (`agent_node._call_verify_result`
  docstring 의 "depth grasp 보조 검증" 단계)

정확도 개선이 필요해지면 `finetune_vision_layers=True` 재학습 또는 verify QA
augmentation 검토. 자세한 내용은
[training/README.md](../training/README.md) "2-7. 학습 결과 및 알려진 이슈".
