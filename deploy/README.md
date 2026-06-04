# 배포 가이드 (Jetson)

`deploy/` 하위 ROS2 노드를 Jetson 에서 빌드하고 운영하는 절차입니다.
설계·노드 구성·토픽 그래프는 [`README.md`](../README.md) "ROS2 노드 및 토픽" 절 참조.

## 사전 조건

- `deploy/scripts/setup_jetson_base.sh` 실행 완료 (ROS2 + LeRobot 베이스 환경).
  이 스크립트가 **Jetson(JetPack 6.2 / CUDA 12.6)용 torch/torchvision cu126 wheel**
  (`torch==2.8.0`, `torchvision==0.23.0`, jetson-ai-lab `.io` index)과 **`numpy==1.26.4`**
  (Jetson torch 의 NumPy 1.x ABI)를 설치한다. PyPI generic(+cpu) wheel 은 cuda 불가,
  numpy 2.x 는 `torch.from_numpy` 가 깨지므로 둘 다 필수. setuptools 는 ament_python
  빌드 호환을 위해 59.8.0(<60)으로 마지막에 고정한다(lerobot 이 `setuptools>=71` 을
  끌어오므로 설치 순서 주의 — 자세한 순서·이유는 스크립트 [4] 단계 주석 참조)
- 모델 배치 — `~/models/gemma/{gemma-4-E2B-it-Q4_K_M.gguf, mmproj-gemma4-e2b-f16.gguf,
  adapters/adapter_<domain>.gguf}`, `~/models/embedding/`,
  `~/models/act/<domain>/pretrained_model/` (도메인별 LeRobot ACT 체크포인트 디렉터리
  — 단일 act.pth 아님)
  (모델 파일은 repo `deploy/` 에 없으며 학습·변환 산출물을 수동 배치. llama-server
  바이너리는 `~/llama.cpp/build/bin/` 에 별도 위치)
- 캘리브레이션 결과 배치 — `~/calibration/{camera_intrinsics.yaml,
  camera_robot_transform.yaml, destination_roi.yaml, home_pose.yaml}`
  (`grasp_roi.yaml` 은 미래 grasp 검증용 — 선택. 캘리브레이션 절차는
  [`calibration/README.md`](calibration/README.md) 참조)

## 빌드

저장소 루트에서 (개발 PC 확인용):

```bash
cd deploy/ros2_ws
colcon build --packages-select robot_arm_msgs robot_arm_pkg
source install/setup.bash
```

> ⚠️ **빌드 워크스페이스 — `~/ros2_ws` 로 통일**: `setup_jetson_base.sh` 는 환경
> 설치 편의상 저장소 안(`deploy/ros2_ws`)에서 한 번 빌드하지만, 운영 레이아웃은
> 아래처럼 **`~/ros2_ws`** 로 소스를 복사해 빌드한다(노드 파라미터·systemd 가
> `~/models`·`~/calibration` 홈 경로를 가정하고, 저장소를 깨끗이 유지하기 위함).
> 따라서 `setup_jetson_base.sh` 실행 후에도 아래 단계로 `~/ros2_ws` 를 별도 빌드하고,
> 운영 시 `~/ros2_ws/install/setup.bash` 를 source 한다(`.bashrc` 에 등록 시 저장소-내
> install 과 중복 source 되지 않도록 `~/ros2_ws` 를 나중에 source). 스크립트의 저장소-내
> 빌드와 이 절의 불일치는 알려진 항목으로, 스크립트는 환경설치 전용으로 본다.

Jetson 실기 배포 시 다음을 복사한 후 빌드:

```bash
# 1) 저장소의 deploy/ros2_ws/src/* 를 Jetson의 ~/ros2_ws/src/ 로 복사
#    (~/ros2_ws/src 가 없으면 먼저 생성 — 디렉터리가 없는 채로 cp 하면
#     "target is not a directory" 로 복사가 통째로 실패한다)
mkdir -p ~/ros2_ws/src
cp -r <repo>/deploy/ros2_ws/src/* ~/ros2_ws/src/

# 2) 런타임 설정 복사
mkdir -p ~/ros2_ws/config ~/calibration
cp <repo>/deploy/config/domain_map.yaml  ~/ros2_ws/config/domain_map.yaml
cp <repo>/deploy/config/label_vocab.yaml ~/ros2_ws/config/label_vocab.yaml
cp <repo>/deploy/config/destination_roi.yaml ~/calibration/destination_roi.yaml   # 캘리브레이션 후 실측값으로 갱신 (place 검증)
# grasp_roi.yaml 은 미래 grasp 검증용 — 필요 시 함께 복사

# 3) 빌드
cd ~/ros2_ws
colcon build --packages-select robot_arm_msgs robot_arm_pkg
source install/setup.bash
```

## llama-server 상시 구동 (systemd)

llama-server 는 ROS2 노드들이 HTTP 로 접속하는 상시 백엔드이므로 systemd
서비스로 등록해 **부팅 시 자동 시작 + 장애 시 자동 재시작**되게 한다. 유닛
파일은 [`systemd/llama-server.service`](systemd/llama-server.service) 에 있으며,
2026-06-02 Jetson 실기 검증 명령과 동일한 옵션이다 (옵션 의미는 아래
"LoRA hot-swap 실기 검증" 절의 "옵션 메모" 참조).

### 전제

- 유닛은 사용자 `hansungai` / 홈 `/home/hansungai` 기준 **절대경로**로 작성됨.
  사용자명이 다르면 `User=`, `Group=`, `ExecStart=` 경로를 함께 수정.
- systemd ExecStart 는 `~` 를 확장하지 않으므로 모든 경로가 절대경로여야 함.
- 다음 파일이 유닛에 적힌 위치에 있어야 함:
  - 바이너리 `/home/hansungai/llama.cpp/build/bin/llama-server` (llama.cpp 빌드 위치 — 모델과 별개)
  - base GGUF `/home/hansungai/models/gemma/gemma-4-E2B-it-Q4_K_M.gguf`
  - mmproj `/home/hansungai/models/gemma/mmproj-gemma4-e2b-f16.gguf`
  - 어댑터 `/home/hansungai/models/gemma/adapters/adapter_{recycling,tableware}.gguf`
- 모델은 `~/models/gemma/` 아래로 통일한다 (base·mmproj·adapter). `--lora`/`-m`/`--mmproj`
  는 GGUF 를 읽기만 하므로, 기존 위치(`~/llama.cpp/gemma4-e2b/`)에서 이 경로로 복사한
  뒤 유닛을 실행하면 그대로 동작한다 (바이너리는 옮기지 않음).

### 메모리 단편화 완화 (커널 VM 튜닝, Orin Nano 8GB 권장)

Orin Nano 8GB 의 unified memory 에서는 대용량 연속(higher-order) 할당이 잦은
llama-server 가 메모리 단편화로 로드 실패·OOM 을 겪을 수 있다. 다음 커널 VM
튜닝으로 완화한다. drop-in 파일: [`sysctl.d/99-llama-server-vm.conf`](sysctl.d/99-llama-server-vm.conf)

```bash
# /etc/sysctl.conf 에 직접 append 하지 말고 drop-in 으로 설치 (중복 append 방지)
sudo cp <repo>/deploy/sysctl.d/99-llama-server-vm.conf /etc/sysctl.d/
sudo sysctl --system        # 즉시 적용 (재부팅 시 systemd-sysctl 가 자동 적용)

# 적용 확인
sysctl vm.compaction_proactiveness vm.min_free_kbytes
```

- `vm.compaction_proactiveness=100` — 커널이 백그라운드(kcompactd)에서 더 적극적으로
  메모리 압축, 연속 블록 확보 (기본 20). CPU 를 약간 더 씀.
- `vm.min_free_kbytes=131072` (128MB) — free 예약량을 늘려 압축/할당이 연속 블록을
  유지하기 쉽게 함.

이 값들은 **시스템 전역 커널 설정**이라 systemd 유닛(ExecStart, 프로세스 단위)이
아니라 sysctl 로 적용한다. 부팅 시 `systemd-sysctl.service` 가 `llama-server.service`
보다 먼저 적용하므로 별도 순서 의존성은 없다.

### 설치

```bash
# 1) 유닛 파일 설치
sudo cp <repo>/deploy/systemd/llama-server.service /etc/systemd/system/

# 2) 재로드 + 부팅 자동 시작 등록 + 즉시 시작
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server.service

# 3) 상태 / 로그 확인
systemctl status llama-server.service
journalctl -u llama-server.service -f
```

### 동작 확인

```bash
# 어댑터 두 개가 등록되었는지 (서버 시작 직후엔 scale 1.0)
curl -s http://localhost:8080/lora-adapters | python3 -m json.tool
```

이후 `llama_server_manager` 노드가 시작 시 두 어댑터를 모두 scale=0 으로
deactivate 한다 (아래 "배포 실행" 및 "LoRA hot-swap 실기 검증" 절 참조).

### 운영 명령

```bash
sudo systemctl restart llama-server.service   # --lora 등 옵션 변경/어댑터 추가 후
sudo systemctl stop    llama-server.service
sudo systemctl disable llama-server.service   # 부팅 자동 시작 해제
```

### 트러블슈팅

- **시작 직후 죽고 CUDA 라이브러리 not found**: systemd 는 로그인 셸 환경
  (`~/.bashrc` 의 `LD_LIBRARY_PATH` 등) 을 상속하지 않는다. 대화형 셸에서는
  뜨는데 서비스로는 실패하면 유닛 `[Service]` 에 다음을 추가 후 재로드:
  ```ini
  Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
  ```
- **OOM 으로 재시작 반복** (`journalctl` 에 `Killed`): Orin Nano 8GB 메모리
  압박. 먼저 위 "메모리 단편화 완화" 절의 sysctl 튜닝 적용 여부 확인. 그래도
  부족하면 큰 노드와의 동시 기동을 피하거나 `--ctx-size` / `--cache-ram` 추가 절감.

## 배포 실행

llama-server 는 systemd 서비스로 상시 구동된다 (위 "llama-server 상시 구동
(systemd)" 절). 부팅 시 자동 시작되므로 여기서는 서비스 상태만 확인하고 ROS2
노드를 실행한다.

```bash
# llama-server 가 떠 있는지 확인 (active 여야 함)
systemctl is-active llama-server.service

# RealSense 드라이버 (camera_node 의 입력원 — 반드시 먼저 기동)
#   camera_node 는 이 드라이버의 /camera/camera/{color,aligned_depth_to_color}/image_raw
#   를 받아 /camera/rgb · /camera/depth 로 재발행한다. 드라이버가 없으면 RGB 가 흐르지
#   않아 act_node/agent_node 가 입력 대기로 멈춘다(ExecutionStatus 타임아웃 → 무한 재시도).
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# 카메라 (RealSense → /camera/rgb · /camera/depth 재발행)
ros2 run robot_arm_pkg camera_node

# llama-server LoRA 스왑 매니저
ros2 run robot_arm_pkg llama_server_manager \
  --ros-args -p server_url:=http://localhost:8080 \
             -p adapter_dir:=~/models/gemma/adapters/

# 임베딩 노드 (텍스트 → 벡터 Service) — CPU 로 실행해 GPU 를 llama-server/act 에 양보
#   (Orin Nano 8GB unified memory. 임베딩은 이벤트성·경량이고 agent 가 사전 임베딩하므로
#    CPU 로 충분 — README "Jetson 메모리 배분" 절 참조)
ros2 run robot_arm_pkg embedding_node \
  --ros-args -p model_name:=~/models/embedding/ \
             -p device:=cpu

# 에이전트 (Gemma 함수 호출 오케스트레이션)
ros2 run robot_arm_pkg agent_node \
  --ros-args -p server_url:=http://localhost:8080 \
             -p domain_map:=~/ros2_ws/config/domain_map.yaml \
             -p destination_roi:=~/calibration/destination_roi.yaml \
             -p max_verify_iterations:=3 \
             -p exec_timeout_s:=120.0

# ACT Policy (model_root 에서 SubtaskCommand.domain 으로 도메인별 pretrained_model 자동 재선택)
#   GPU 단편화 회피를 위해 expandable_segments + device:=cuda 명시 (8GB 에서 llama-server
#   와 GPU 공존 — README "Jetson 메모리 배분" 절). act_node.py 가 expandable_segments 를
#   setdefault 하므로 빌드본은 prefix 가 없어도 적용되지만, 수동 실행 시 함께 준다.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
ros2 run robot_arm_pkg act_node \
  --ros-args -p model_root:=~/models/act \
             -p device:=cuda \
             -p horizon_steps:=300
#   단일 도메인만 쓰면 고정 경로 override (도메인 재로드 비활성):
#     -p model_path:=~/models/act/recycling/pretrained_model

# 로봇 컨트롤러 (SO-101 팔로워) — 포트는 환경에 따라 다름 (`ls /dev/ttyACM*` 로 확인)
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

### 구현 결정 사항 (2026-06-03 — 함수 호출 루프 본체 구현)

함수 호출 루프 본체를 구현했다. 결정의 캐노니컬 기록은
[agent_node.py](ros2_ws/src/robot_arm_pkg/robot_arm_pkg/agent_node.py) 모듈
docstring "구현 결정 사항" 블록에 있으며, 여기서는 운영 영향 항목만 navigation 한다
(값 재기재 금지 — drift 방지).

- **실행 모델**: 루프는 전용 워커 스레드에서 실행하고 `main()` 은
  `MultiThreadedExecutor` 로 스핀한다. 구독 콜백은 블로킹하지 않는다.
  `ExecutionStatus` 대기는 `threading.Event`, 서비스(`/embed_text`, `/go_home`)는
  `call_async` + future 대기. exec_status 구독 + 서비스 클라이언트는
  `ReentrantCallbackGroup`.
- **look_scene 그라운딩**: 함수 호출 대화는 **텍스트 전용**(function_call 학습
  분포 = 이미지 없음). 실제 이미지→묘사는
  [prompts.LOOK_SCENE_SYSTEM_PROMPT](ros2_ws/src/robot_arm_pkg/robot_arm_pkg/prompts.py)
  를 system 으로 쓰는 **별도 VLM 호출**(이미지-first). `SYSTEM_PROMPT`(JSON-only)와
  충돌하므로 분리. ⚠️ 전용 학습 분포 없음 — 묘사 정확도는 실기 검증 필요(상수 주석 참조).
- **verify 종합 판정**: Gemma `status` 우선 + depth **단방향 veto**
  (`SUCCESS` 인데 depth place 흔적 없음 → `PARTIAL` 강등으로 재시도 유도).
  depth `None`(미실측/미설정/ depth 없음)이면 Gemma 단독. depth 단독 SUCCESS 승격 없음.
- **재시도 의미**: attempt 단위로 look→execute→go_home→verify **전체를 재실행**한다
  (매 시도가 단일 시퀀스 학습 분포와 일치 — verify 결과를 모델에 다시 먹이지 않음).
  시도 수 ≤ `max_verify_iterations`.
- **종료 상태**: 신규 토픽/메시지 미도입(v1). 초과 시 내부 `WAITING_USER` 상태 +
  AgentLog 종료 마커. 새 `/user_command` 가 들어오면 태스크 포기 후 재시작(루프
  실행 중에만 신규 명령 거부). 전용 `/user_intervention` 토픽은 향후.
- **depth place 검증**: `verify_result` 시 목적지 ROI median depth 변화로 place 보강
  (단방향 veto). [`destination_roi.yaml`](config/destination_roi.yaml) 의 **픽셀 ROI +
  baseline depth** 로 표현해 카메라 intrinsics 에 의존하지 않는다(미터 3D 박스 미사용 —
  [README.md](../README.md) "agent_node 는 intrinsics 비의존" 결정 유지). destination_roi
  미설정/도메인 ROI 없음/depth 없음/유효픽셀 없음 시 `None` → Gemma 단독.
  `_depth_place_check` 주석 참조.
  - grasp(집기 시점) 검증은 home_pose 복귀 데이터로는 불가 — 미래 확장으로 유보
    (`grasp_roi.yaml` + 미래 `_depth_grasp_check`).
- **신규 파라미터**: `exec_timeout_s` — `ExecutionStatus` 대기 타임아웃(기본 120s).
- **의존성**: RGB→base64(PNG) 인코딩에 `python3-pil`(Pillow) 필요(cv_bridge 회피).
- **연관 수정**: `act_node._finish_subtask` 가 `ExecStatus` enum(대문자)을 발행하도록
  교정(이전 소문자 리터럴 → SSoT `prompts.ExecStatus` 불일치 해소). agent_node 도
  수신 status 를 `.upper()` 방어 정규화.

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
- `_depth_place_check` depth 보조 검증 (`agent_node._call_verify_result`
  docstring 의 "depth place 보조 검증" 단계)

정확도 개선이 필요해지면 `finetune_vision_layers=True` 재학습 또는 verify QA
augmentation 검토. 자세한 내용은
[training/README.md](../training/README.md) "2-7. 학습 결과 및 알려진 이슈".

> **실기 1차 관찰 (2026-06-04 Jetson, recycling 1회 — 잠정, 추가 측정 필요)**:
> 통합 루프(look→execute→go_home→verify→재시도→WAITING_USER)가 끝까지 정상 동작했고,
> verify 종합 판정의 신뢰도는 **depth 신호 생존 여부에 좌우**됨을 확인.
> - depth place 가 살아있을 때(`depth=False` 일관)는 Gemma 와 일치해 **정확히 FAIL/PARTIAL**
>   판정 → 재시도 후 WAITING_USER 진입(설계대로).
> - 반면 `depth=None`(ROI 미감지)/`True`(오감지)이면 Gemma 의 환각 SUCCESS 를 단방향
>   veto 가 막지 못해 **오판 SUCCESS** 가 통과(다른 시도에서 관측).
> ⇒ 단기 보완은 **depth 신호 안정화**(destination_roi 정렬·threshold), 근본은 Gemma
>    verify 재학습. depth 우선/가중 등 종합정책 확정은 다회 측정 후로 유보.
> 또한 ACT 자체의 **실제 grasp+place 성공률은 낮음**(real-only·첫 실기)으로,
> `ExecutionStatus=SUCCESS`(=실행 완료)와 verify=FAIL(=실제 미배치)이 공존했다 — 둘은
> 모순이 아니라 정상적 역할 분담이며, ACT 성능은 별도 데이터 보강 영역이다.

## LoRA hot-swap 실기 검증 (2026-06-02)

llama-server 의 LoRA 어댑터 hot-swap 인프라 ([llama_server_manager._hot_swap](ros2_ws/src/robot_arm_pkg/robot_arm_pkg/llama_server_manager.py))
를 Jetson Orin Nano 8GB 실기에서 검증 완료. llama.cpp build `b8931-9725a313b`
기준 `/lora-adapters` (복수형) HTTP 엔드포인트가 id 기반으로 동작함을 확인.

### llama-server 시작 명령 (Jetson)

> 아래는 2026-06-02 검증 당시 그대로의 명령 기록이라 base/mmproj 가
> `~/llama.cpp/gemma4-e2b/` 에 있다. 운영 레이아웃은 base·mmproj 도 `~/models/gemma/`
> 로 통일했으므로(위 "llama-server 상시 구동 (systemd)" 절) 지금 수동 실행 시엔
> `-m`/`--mmproj` 경로를 `~/models/gemma/` 로 바꿔 쓴다.

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
