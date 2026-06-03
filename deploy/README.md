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

## LoRA hot-swap 실기 검증 (2026-06-02)

llama-server 의 LoRA 어댑터 hot-swap 인프라 ([llama_server_manager._hot_swap](ros2_ws/src/robot_arm_pkg/robot_arm_pkg/llama_server_manager.py))
를 Jetson Orin Nano 8GB 실기에서 검증 완료. llama.cpp build `b8931-9725a313b`
기준 `/lora-adapters` (복수형) HTTP 엔드포인트가 id 기반으로 동작함을 확인.

### llama-server 시작 명령 (Jetson)

```bash
~/llama.cpp/build/bin/llama-server \
  -m       ~/llama.cpp/gemma4-e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj ~/llama.cpp/gemma4-e2b/mmproj-gemma4-e2b-f16.gguf \
  --no-mmproj-offload \
  --lora   ~/models/gemma/adapters/adapter_recycling.gguf,~/models/gemma/adapters/adapter_tableware.gguf \
  --n-gpu-layers 99 \
  --ctx-size 2048 \
  --batch-size 256 --ubatch-size 256 \
  --jinja --parallel 1 --cache-ram 512 \
  --host 0.0.0.0 --port 8080
```

옵션 메모 (Orin Nano 8GB 메모리 압박 대응):
- `--no-mmproj-offload`: vision + audio projector 를 CPU 에서 실행 (mmproj 가
  vision + audio 통합이라 GPU 만으로는 OOM). verify_result 호출 시 약간 느려짐.
- `--parallel 1` (n_seq_max=1) + `--batch/ubatch 256`: KV cache 와 compute
  buffer 절감. function_call 7-turn + verify 시스템 프롬프트가 ctx 2048 안에
  충분히 들어감.
- 어댑터 두 개를 `--lora` 콤마로 묶음 (CLI 가 다중 `--lora` 를 deprecate 함).
- llama-server 는 등록된 어댑터를 모두 scale=1.0 (활성) 으로 시작하므로
  `llama_server_manager.__init__` 가 즉시 모두 scale=0 으로 deactivate.

### 검증 명령

ROS2 노드 실행:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_arm_pkg llama_server_manager
```

기대 로그:
```
[INFO] llama_server_manager 시작 | server: http://localhost:8080 | adapter_dir: /home/hansungai/models/gemma/adapters/
[INFO] 시작 시 어댑터 deactivate 완료 — registered: ['adapter_recycling.gguf', 'adapter_tableware.gguf']
```

GET 으로 시작 직후 모두 비활성 확인:
```bash
curl -s http://localhost:8080/lora-adapters | python3 -m json.tool
# 둘 다 "scale": 0.0
```

도메인 변경 트리거 (다른 터미널):
```bash
ros2 topic echo /selected_domain &

# recycling 활성화
ros2 topic pub --once /domain_selection robot_arm_msgs/msg/DomainSelection \
  '{domain: "recycling", scale: 1.0}'
# → /selected_domain: 'recycling'
# → GET /lora-adapters: recycling 1.0, tableware 0.0

# tableware 로 swap
ros2 topic pub --once /domain_selection robot_arm_msgs/msg/DomainSelection \
  '{domain: "tableware", scale: 1.0}'
# → GET: recycling 0.0, tableware 1.0

# 같은 도메인 재요청 — idempotency
ros2 topic pub --once /domain_selection robot_arm_msgs/msg/DomainSelection \
  '{domain: "tableware", scale: 1.0}'
# → 노드 로그: "도메인 'tableware' 이미 적재됨 — 스왑 생략, 통보만 발행"
```

### 검증 결과 (2026-06-02 통과)

| 항목 | 결과 |
|---|---|
| 시작 시 인덱스 캐시 구축 (GET `/lora-adapters` → basename → id) | ✅ |
| 시작 시 모두 scale=0 deactivate (POST `/lora-adapters`) | ✅ |
| recycling hot-swap (id 0=1.0, id 1=0.0) + `/selected_domain` 발행 | ✅ |
| tableware hot-swap (id 0=0.0, id 1=1.0) + `/selected_domain` 발행 | ✅ |
| 같은 도메인 재요청 idempotency (HTTP 미발생, 통보만 발행) | ✅ |
| `chat_template_kwargs: {"enable_thinking": false}` 로 `content` 정상 반환 | ✅ |

### 새 도메인 추가 시 운영 주의

`/lora-adapters` POST 는 미리 등록된 어댑터의 scale 변경만 지원 (path 기반
동적 load 미지원). 새 도메인 어댑터를 추가하려면:

1. 새 GGUF 어댑터를 `~/models/gemma/adapters/adapter_<new_domain>.gguf` 로 배치
2. llama-server 의 `--lora` 옵션에 새 path 추가하여 **재시작**
3. [deploy/config/domain_map.yaml](config/domain_map.yaml) 에 매핑 추가

`llama_server_manager` 노드는 재시작된 서버를 `_refresh_adapter_index` 가
자동 재조회하므로 별도 수정 불필요. 자세한 매커니즘은 [llama_server_manager.py](ros2_ws/src/robot_arm_pkg/robot_arm_pkg/llama_server_manager.py)
모듈 docstring 의 "API 형식" 절 참조.
