# 학습 가이드

`training/` 하위 3개 워크스트림의 실행 절차를 한 곳에 모은 가이드입니다.
설계·정책은 [`README.md`](../README.md) 의 관련 절을 참조하세요.

| 워크스트림 | 폴더 | 산출물 |
|---|---|---|
| 모방데이터 수집 | [`imitation_data/`](imitation_data/) | LeRobotDataset (Real teleop + Sim teleop) |
| Gemma 도메인 LoRA 파인튜닝 | [`lora_finetuning/`](lora_finetuning/) | LoRA adapter (.safetensors → GGUF) |
| ACT 학습 (real-only, 도메인별) | [`act/`](act/) | 도메인별 ACT policy 체크포인트 (`pretrained_model/`) |

---

# 1. 모방데이터 수집 (`imitation_data/`)

Real 시연(`record_real_teleop.py`) · Sim 시연(`record_sim_teleop.py`) 수집과
Real 라벨 후처리(`postprocess_real_labels.py`) 절차입니다. 정책·결과 채널 등
핵심 사항은 [`README.md`](../README.md) "데이터 수집" 및 "학습 파이프라인 —
Sim 데이터 수집 (teleop)" 절 참조.

## 1-1. Real teleop (Jetson)

### 실행 전 준비

1. SO-101 리더/팔로워 USB 포트 확인
2. LeRobot 캘리브레이션 완료 (`lerobot-calibrate`)
3. 카메라 캘리브레이션 결과 (`camera_intrinsics.yaml`, `camera_robot_transform.yaml`) 를 `~/calibration/` 에 배치
4. 임베딩 모델 결정 및 배치 (Gemma 도메인 LoRA 학습 / ACT 학습과 동일 모델 사용 필수)
5. `robot_controller_node` 가 떠 있다면 중지 (포트 점유 정책)
6. **home pose 준비**: `~/calibration/home_pose.yaml` 이 없으면 follower 팔을 원하는 rest 자세에 둔 채로 record_real_teleop 첫 실행 — 그 자세가 자동 캡처되어 파일로 저장되고, 이후 에피소드 간 복귀와 배포 시 `/go_home` 서비스에서 같은 파일을 공유합니다. 자세를 바꾸려면 파일 삭제 후 재실행.

### teleoperation 동작 검증

```bash
source ~/venv/lerobot/bin/activate
sudo chmod 666 /dev/follower /dev/leader
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/follower \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/leader \
  --teleop.id=leader \
  --robot.cameras='{"front": {"type": "intelrealsense", "serial_number_or_name": "919122073151", "width": 640, "height": 480, "fps": 15}}' \
  --display_data=true
```

### 실행 순서

```bash
# 1. RealSense 드라이버 실행
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# 2. camera_node 실행 (/camera/rgb 발행)
ros2 run robot_arm_pkg camera_node

# 3. embedding_node 실행 (수집 시 자연어 라벨 → 임베딩 변환에 사용)
ros2 run robot_arm_pkg embedding_node \
  --ros-args -p model_name:=~/models/embedding/

# 4. teleop 데이터 수집 스크립트 실행 (Option C — 슬롯 기반 템플릿)
python <repo>/training/imitation_data/scripts/record_real_teleop.py \
  --robot-port      /dev/ttyACM1 \
  --robot-id        follower \
  --teleop-port     /dev/ttyACM0 \
  --teleop-id       leader \
  --dataset-repo-id myname/dataset_real \
  --domain          recycling \
  --label-vocab     ~/ros2_ws/config/label_vocab.yaml \
  --num-episodes    50

# 5. 수집 종료 후 paraphrase + 임베딩 후처리 (외부 PC 또는 Jetson 어디서나)
python <repo>/training/imitation_data/scripts/postprocess_real_labels.py \
  --dataset                ~/datasets/myname/dataset_real \
  --labels-jsonl           labels_myname_dataset_real.jsonl \
  --external-llm           gpt-4o-mini \
  --paraphrase-per-episode 3
```

## 1-2. Sim teleop (외부 PC 로컬 또는 EC2)

전제: Isaac Sim 의 python 환경에 lerobot 0.4.4 설치
(`~/isaacsim/python.sh -m pip install "lerobot[feetech]==0.4.4"`).

환경 셋업(Isaac Sim 설치·URDF 임포트·씬 구성)은
[`sim/teleop/sim_local_teleop_setting.md`](../sim/teleop/sim_local_teleop_setting.md) /
[`sim/teleop/sim_cloud_teleop_setting.md`](../sim/teleop/sim_cloud_teleop_setting.md) 참조.

```bash
# 1. Sim teleop 시연 데이터 수집 (~500 에피소드) — 반드시 Isaac Sim 의 python 으로 실행
~/isaacsim/python.sh training/imitation_data/scripts/record_sim_teleop.py \
  --scene-usd        sim/scenes/base_scene.usd \
  --teleop-port      /dev/ttyACM0 \
  --dataset-repo-id  myname/dataset_sim \
  --domain           recycling \
  --num-episodes     500

# 2. 각 에피소드에 Gemma + 외부 LLM 라벨 부여
python training/imitation_data/scripts/generate_language_labels.py \
  --dataset dataset_sim/ \
  --gemma-server http://localhost:8080 \
  --external-llm gpt-4o-mini \
  --paraphrase-per-episode 3
```

---

# 2. Gemma 도메인 LoRA 파인튜닝 (`lora_finetuning/`)

Q&A 데이터 생성(`generate_function_call_qa.py` 의 function_call, `generate_verify_qa.py`
의 verify)과 70:30 비율 혼합(`merge_qa_dataset.py`)을 거쳐 단일 JSONL 을 만든
후, 아래 두 단계로 학습·변환합니다.

> **데이터 포맷 — Unsloth FastVisionModel typed-array content (2026-05-25 freeze)**:
> function_call 와 verify 가 같은 FastVisionModel 학습 패스에 들어가도록 모든 샘플의
> `content` 는 `[{"type":"text","text":...}]` (이미지 케이스에서는
> `{"type":"image","image":...}` 가 text 앞에 추가) 형식. tool role 은
> FastVisionModel 미지원이라 `user` role + `"[tool_result] "` prefix 로 변환됨.
> `generate_function_call_qa.py` 는 이미 신 형식으로 직접 생성하며,
> 기존 string-content 데이터는 [`convert_function_call_to_vision_format.py`](lora_finetuning/scripts/convert_function_call_to_vision_format.py)
> 로 1회 변환 가능.

## 2-1. 학습 환경 준비 (외부 PC)

```bash
pip install -r training/lora_finetuning/requirements.txt
export OPENAI_API_KEY=sk-...   # gpt-4o-mini 호출용 (function_call / verify 합성기 공통)
```

## 2-2. function_call 합성 (함수 호출 시나리오)

```bash
# 도메인별로 1회씩 (recycling, tableware)
python training/lora_finetuning/scripts/generate_function_call_qa.py \
  --domain         recycling \
  --label-vocab    deploy/config/label_vocab.yaml \
  --output         training/lora_finetuning/data/recycling/recycling_function_call_qa.jsonl \
  --external-llm   gpt-4o-mini \
  --num-samples    500
```

> 비용 추정: 500 샘플 ≈ $0.02–0.05 (gpt-4o-mini). 진행률·누적 토큰·비용이
> stderr 로 50샘플마다 출력됩니다.

**먼저 dry run (10샘플)** 으로 출력 JSONL 한두 줄 직접 검사 권장:
```bash
python training/lora_finetuning/scripts/generate_function_call_qa.py \
  --domain recycling --output /tmp/function_call_dry.jsonl --num-samples 10
head -n 2 /tmp/function_call_dry.jsonl | python -m json.tool
```

## 2-3. 기존 function_call JSONL 을 vision 포맷으로 변환 (1회용)

이전 string-content + tool role 형식으로 생성된 JSONL 이 있다면 다음으로 변환:

```bash
python training/lora_finetuning/scripts/convert_function_call_to_vision_format.py \
  --input  training/lora_finetuning/data/recycling/function_call_qa_recycling.jsonl \
  --output training/lora_finetuning/data/recycling/function_call_qa_recycling.vision.jsonl

python training/lora_finetuning/scripts/convert_function_call_to_vision_format.py \
  --input  training/lora_finetuning/data/tableware/function_call_qa_tableware.jsonl \
  --output training/lora_finetuning/data/tableware/function_call_qa_tableware.vision.jsonl
```

> 멱등 변환 — 이미 typed-array 인 입력은 그대로 통과. 신규 합성한 JSONL 은
> 이미 vision 포맷이므로 본 단계 불필요.

## 2-4. verify 합성 (검증 시나리오) — 2-pass 구조

LLM 환경(openai) 와 Isaac Sim 환경의 의존성을 분리하기 위해 두 단계로 나뉩니다.

| 단계 | 스크립트 | 환경 | 입력 | 출력 |
|---|---|---|---|---|
| **Step 1** | [`generate_verify_qa.py`](lora_finetuning/scripts/generate_verify_qa.py) | 일반 Python (gpt-4o-mini) | vocab + multi_target_scenes | 학습용 JSONL + `*.scenarios.jsonl` sidecar |
| **Step 2** | [`render_verify_images.py`](lora_finetuning/scripts/render_verify_images.py) | `~/isaacsim/python.sh` (Isaac Sim standalone) | sidecar + base_scene + 자산 | 시나리오별 PNG |

> **freeze 결정의 캐노니컬 위치는 각 스크립트 모듈 docstring 의 매트릭스를
> 참조**합니다 (값 / 분포 / 템플릿 / destination / 렌더 정책 등). 본 README 에
> 재기재하지 않으며, 변경 시 해당 모듈만 수정하면 됩니다.

### 데이터 흐름

```
[Step 1 — LLM 환경]
synthesize_scenario (LLM)            # status / target / target_phrase / cmd
  → generate_reasoning (template)    # status × 5 템플릿, target/dest substitute
  → build_sample                     # vision typed-array messages
  → write JSONL + scenarios sidecar  # PNG 미생성

[Step 2 — Isaac Sim standalone]
load scenarios sidecar
  → SimVerifyRenderer(base_scene, home_pose, camera)
  → for each scenario:
      place target (kinematic) + distractor → physics settle → capture → PNG
```

### 출력 디렉터리 구조

```
training/lora_finetuning/data/recycling/
├── recycling_verify_qa.jsonl                    # 학습용 JSONL (Step 1)
├── recycling_verify_qa.jsonl.scenarios.jsonl    # 시나리오 메타 sidecar (Step 1 → Step 2)
└── verify_images/                               # Step 2 결과
    ├── recycling_00000.png
    ├── recycling_00001.png
    └── ...
```

### Step 1 — JSONL + sidecar 합성 (LLM 환경)

dry run (10 샘플, ≈ $0.001):
```bash
python training/lora_finetuning/scripts/generate_verify_qa.py \
  --domain         recycling \
  --output-jsonl   /tmp/verify_dry.jsonl \
  --image-dir      /tmp/verify_dry_images \
  --num-scenes     10

head -n 1 /tmp/verify_dry.jsonl              | python -m json.tool
head -n 1 /tmp/verify_dry.jsonl.scenarios.jsonl | python -m json.tool
```

본격 합성 (양 도메인, ≈ $0.05):
```bash
for D in recycling tableware; do
  python training/lora_finetuning/scripts/generate_verify_qa.py \
    --domain         $D \
    --output-jsonl   training/lora_finetuning/data/$D/${D}_verify_qa.jsonl \
    --image-dir      training/lora_finetuning/data/$D/verify_images/ \
    --num-scenes     500
done
```

### Step 2 — PNG 렌더링 (Isaac Sim 환경)

Sim 환경 사전 준비:
```bash
~/isaacsim/python.sh -m pip install pillow PyYAML
# (numpy 는 Isaac Sim 기본 포함)
```

렌더 실행:
```bash
for D in recycling tableware; do
  ~/isaacsim/python.sh training/lora_finetuning/scripts/render_verify_images.py \
    --scenarios  training/lora_finetuning/data/$D/${D}_verify_qa.jsonl.scenarios.jsonl \
    --scene-usd  sim/scenes/base_scene.usd \
    --domain     $D \
    --image-dir  training/lora_finetuning/data/$D/verify_images/ \
    --headless
done
```

> 첫 실행 시 디버깅 가능성:
> - Sim 자산 경로(`sim/assets/objects/<domain>/`) 가 비어 있으면 즉시 종료.
>   `setup_objects.py` 를 먼저 실행해 wrapper USD 생성.
> - `home_pose.yaml` 위치/포맷 (record_real_teleop 가 만든 형식과 동일 규약).
> - 객체 pivot 위치는 자산마다 달라 `TARGET_Z_OFFSET` 보정이 필요할 수 있음
>   (render_verify_images.py 상단 상수).
> - 카메라 prim 경로 (`/World/Camera_RGB`) — base_scene 에 따라 다르면
>   `--camera-prim` 으로 override.

## 2-5. Q&A 병합 (70:30, oversample)

```bash
python training/lora_finetuning/scripts/merge_qa_dataset.py \
  --function_call   training/lora_finetuning/data/recycling/recycling_function_call_qa.jsonl \
  --verify          training/lora_finetuning/data/recycling/recycling_verify_qa.jsonl \
  --output          training/lora_finetuning/data/recycling/recycling_merged.jsonl
```

verify 가 비어 있어도 동작 (function_call 만 사용 + 경고). 1차 학습 시 function_call 만으로
시험하려면 빈 파일을 `--verify` 로 넘기면 됩니다:
```bash
: > /tmp/verify_empty.jsonl
python training/lora_finetuning/scripts/merge_qa_dataset.py \
  --function_call training/lora_finetuning/data/recycling/recycling_function_call_qa.jsonl \
  --verify        /tmp/verify_empty.jsonl \
  --output        training/lora_finetuning/data/recycling/recycling_merged.jsonl
```

## 2-6. Unsloth QLoRA 파인튜닝 (FastVisionModel)

```bash
python training/lora_finetuning/scripts/train_gemma_qlora.py \
  --base-model unsloth/gemma-4-E2B-it \
  --dataset    training/lora_finetuning/data/recycling/recycling_merged.jsonl \
  --image-dir  training/lora_finetuning/data/recycling/verify_images \
  --output-dir training/lora_finetuning/outputs/lora_recycling \
  --rank 16 --alpha 32 \
  --epochs 3
```

`train_gemma_qlora.py` 의 핵심 설계 (모듈 docstring 참조):
- **FastVisionModel** + LoRA (rank/alpha 기본 16/32). `finetune_vision_layers=False`
  로 language layers 만 학습 (Unsloth 권장).
- **Custom data collator**: per_device_batch_size=1 강제 (function_call text-only 와
  verify image+text 가 같은 batch 에 섞이는 문제 회피, [unslothai/unsloth#1590](https://github.com/unslothai/unsloth/issues/1590)
  패턴). effective batch 는 `--grad-accum` (기본 4) 로 확보.
- **Image lazy load**: collator 가 sample 의 image string path 를
  `PIL.Image.open(image_dir/path)` 로 변환 — 전체 dataset 를 메모리에 안 올림.
- **선행 조건**: `pip install -r training/lora_finetuning/requirements.txt`
  (Unsloth 는 CUDA/GPU 환경에 따라 [공식 설치 가이드](https://docs.unsloth.ai/)
  를 우선 권장). VRAM ≥16 GB, HuggingFace `unsloth/gemma-4-E2B-it` 액세스.

### 첫 학습 sanity check (function_call 만, ~10 step)

text-only 샘플과 vision collator 호환성을 빠르게 검증할 수 있는 짧은 run:

```bash
python training/lora_finetuning/scripts/train_gemma_qlora.py \
  --base-model unsloth/gemma-4-E2B-it \
  --dataset    training/lora_finetuning/data/recycling/recycling_function_call_qa.vision.jsonl \
  --image-dir  training/lora_finetuning/data/recycling/verify_images \
  --output-dir /tmp/lora_sanity \
  --max-steps  10 \
  --logging-steps 1
```

function_call 만 들어 있어 image 디렉터리는 비어 있어도 됨 (collator 가 image 없는
sample 만 처리). 정상 loss 가 감소하면 본격 학습으로 진행.

### 학습 후 형식 검증 (smoke test)

`smoke_test_lora.py` 가 학습된 LoRA adapter 의 응답을 base + LoRA 로드 후
held-out (실제로는 merged.jsonl 에서 random pick) sample 로 추론해 다음을
검증합니다:

- function_call: 단일 JSON object, `{"function": ..., "args": {...}}` 키 존재,
  function ∈ {look_scene, execute_subtask, verify_result}
- verify: 단일 JSON object, `{"status": ..., "reasoning": ...}` 키 존재,
  status ∈ {SUCCESS, FAIL, PARTIAL, UNCLEAR}

```bash
python training/lora_finetuning/scripts/smoke_test_lora.py \
  --adapter-dir training/lora_finetuning/outputs/lora_recycling \
  --dataset     training/lora_finetuning/data/recycling/recycling_merged.jsonl \
  --image-dir   training/lora_finetuning/data/recycling/verify_images \
  --n-function-call 8 --n-verify 8
```

## 2-7. 학습 결과 및 알려진 이슈 (2026-06-02)

`unsloth/gemma-4-E2B-it` 기반 양 도메인 LoRA 학습 완료 (rank/alpha=16/32, 3 epoch,
loss masking 활성). smoke test 결과:

| 도메인 | function_call | verify (형식) | 종합 |
|---|---|---|---|
| recycling | 8/8 | 8/8 | 16/16 |
| tableware | 7/32 (22%) | 8/8 | — |

### 이슈 1 — tableware function_call 마지막 turn cluster gap

tableware 의 vocab 이 plate/fork/spoon 3종으로 좁아 function_call 학습 분포가
sub-domain 으로 분할됨. 일부 명령 cluster (예: "정리함으로 옮겨줘") 는 마지막
verify_result turn 응답을 모델이 빈 응답으로 끝냄. recycling 은 영향 없음
(can/plastic/paper vocab 이 더 robust). train_loss 종료값 자체는 두 도메인
거의 같음 (recycling 0.0478 / tableware 0.0470).

**해결 정책 (2026-06-02 결정)**: LoRA 추가 학습 / data augmentation 대신
**배포 측 fallback** 으로 대응. agent_node 가 마지막 turn 응답이
`{"function":"verify_result", ...}` 아니면 코드가 verify_result 를 강제 호출.
자세한 내용은 [deploy/README.md](../deploy/README.md) "에이전트 함수 호출 루프 —
verify_result fallback" 참조.

### 이슈 2 — verify reasoning 환각 + status enum 정확도

`train_gemma_qlora.py` 의 `finetune_vision_layers=False` (Unsloth 권장) 로
vision encoder 가 base 그대로 유지됨. 결과:

- **JSON 형식 / status enum 어휘**: 정확 (smoke 8/8)
- **status 정답률**: 낮음 — 5-6/8 sample 에서 SUCCESS↔FAIL 혼동, blurred 이미지를
  UNCLEAR 로 인식 못함
- **reasoning 내용**: 실제 객체 색/위치와 무관한 환각 다수 ("the plastic bottle
  in the plastic bin is not in the plastic bin" 같은 자기모순도 발생)

**현재 정책**: agent_node 가 reasoning 은 로그용으로만 사용하고 status enum 으로
분기. enum 오답은 배포 실 동작에서 `max_verify_iterations` 재시도 + depth place
보조 검증 (`_depth_place_check`) 으로 보완. 정확도 개선이 필요해지면 다음
옵션 검토:

- `finetune_vision_layers=True` — VRAM/시간 비용 큼
- verify QA augmentation (distractor 다양화, status 분포 보정)
- depth place 보조 검증 가중치 상향

## 2-8. GGUF 변환

LoRA 어댑터 (safetensors) 를 llama.cpp 의 GGUF 어댑터로 변환해야 Jetson 의
llama-server 가 `/lora-adapters` HTTP 로 hot-swap 적재할 수 있습니다.

### 1회 환경 준비 (외부 학습 PC) — **변환 전용 venv 권장**

> 학습 env (Unsloth 기반 torch/transformers) 가 깨지지 않도록 **반드시 별도
> venv** 에서 진행하세요. llama.cpp 의 `requirements-convert_lora_to_gguf.txt`
> 는 `torch==2.11.0+cpu`, `transformers==5.5.1`, `numpy==1.26.4` 등을 강제
> 업그레이드합니다. 이를 학습 venv 에 깔면 `torchvision::nms` 미스매치로
> `Gemma4Config` 가 import 안 되어 학습 / smoke test 가 모두 깨집니다.

```bash
# llama.cpp clone (변환 본체)
git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
export LLAMA_CPP_DIR=~/llama.cpp

# 변환 전용 venv
python -m venv ~/venvs/gguf_convert
source ~/venvs/gguf_convert/bin/activate
pip install -r ~/llama.cpp/requirements/requirements-convert_lora_to_gguf.txt
pip install huggingface_hub  # base 모델 자동 다운로드용
```

이후 변환 작업은 항상 `source ~/venvs/gguf_convert/bin/activate` 해서 venv 안에서
실행합니다. 학습으로 돌아갈 때는 `deactivate` 후 학습 venv 활성화.

### 도메인별 변환 실행

```bash
bash training/lora_finetuning/scripts/convert_lora_to_gguf.sh \
  training/lora_finetuning/outputs/lora_recycling \
  models/gemma/adapters/adapter_recycling.gguf

bash training/lora_finetuning/scripts/convert_lora_to_gguf.sh \
  training/lora_finetuning/outputs/lora_tableware \
  models/gemma/adapters/adapter_tableware.gguf
```

`BASE_MODEL` 환경 변수로 base 모델 override 가능 (기본 `unsloth/gemma-4-E2B-it`).
HF repo id 를 그대로 주면 스크립트가 `huggingface_hub.snapshot_download` 로
`$HF_BASE_CACHE/<basename>` (기본 `~/models/hf/gemma-4-E2B-it`) 에 자동 다운로드
후 그 로컬 path 를 llama.cpp 변환기에 전달합니다. 이미 로컬 디렉터리 path 를
주면 그대로 사용.

### 학습 env 가 이미 깨진 경우 (복구)

위 venv 가이드 전에 글로벌로 `pip install -r requirements-convert_lora_to_gguf.txt`
를 실행했다면 학습 env 의 torch/transformers 가 변환용 버전으로 덮여있을 수
있습니다. 학습 / smoke test 가 다음과 같이 실패하면 그 증상:

```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'Gemma4Config'
```

복구 (학습 시 동작했던 버전으로 되돌림):

```bash
# 학습 venv 안에서 (또는 unsloth 도커 컨테이너에서)
pip install --force-reinstall \
  torch==2.10.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
pip install --force-reinstall transformers==5.5.0
```

그 후 변환 작업은 반드시 별도 venv 에서 진행.

### Jetson 배치

생성된 GGUF 파일은 Jetson 의 `~/models/gemma/adapters/` 로 복사:

```
~/models/gemma/adapters/
├── adapter_recycling.gguf
└── adapter_tableware.gguf
```

파일명은 `adapter_<domain>.gguf` 규약 — [domain_map.yaml](../deploy/config/domain_map.yaml)
의 domain 키와 일치해야 [llama_server_manager](../deploy/ros2_ws/src/robot_arm_pkg/robot_arm_pkg/llama_server_manager.py)
가 경로를 build 해 hot-swap 호출.

선행 조건: base 모델 GGUF (`unsloth/gemma-4-E2B-it` quantized) 가 Jetson 의
llama-server 실행 경로에 이미 배치되어 동작 중이어야 합니다. base 모델 동작이
오래 전이라면 어댑터 hot-swap 전에 단독 `llama-server --model <base_gguf>`
실행으로 1회 sanity check 권장.

## 2-9. 데이터 보관 위치

function_call / verify/merged JSONL 은 **도메인별 하위 폴더**로 관리합니다 (git 추적):

```
training/lora_finetuning/data/
├── recycling/
│   ├── recycling_function_call_qa.jsonl
│   ├── recycling_function_call_qa.vision.jsonl
│   ├── recycling_verify_qa.jsonl
│   ├── recycling_verify_qa.jsonl.scenarios.jsonl
│   ├── recycling_merged.jsonl
│   └── verify_images/
└── tableware/
    ├── tableware_function_call_qa.jsonl
    ├── tableware_function_call_qa.vision.jsonl
    ├── tableware_verify_qa.jsonl
    ├── tableware_verify_qa.jsonl.scenarios.jsonl
    ├── tableware_merged.jsonl
    └── verify_images/
```

학습 산출물(LoRA 체크포인트, GGUF) 은 별도 `training/lora_finetuning/outputs/` 또는 `models/`
로 분리하며, 용량이 커서 `.gitignore` 대상입니다.

---

# 3. ACT 학습 (`act/`)

> **설계 변경 (co-training → real-only)**: 원래는 Sim + Real co-training
> ([`train_act_cotraining.sh`](act/scripts/train_act_cotraining.sh)) 으로 설계했으나, Sim teleop
> 수집이 실질적으로 불가능해 **real-only 도메인별 학습**으로 전환했습니다. co-training
> 스켈레톤은 참고용으로 보존하며, **현재 운영 스크립트는 [`train_act.sh`](act/scripts/train_act.sh)**
> 입니다. 단일 `act.pth` + LoRA hot-swap 가정도 **도메인별 ACT 체크포인트**(recycling /
> tableware 각각)로 바뀌었습니다.

LeRobot 0.4.4 `lerobot-train` 으로 도메인별 real 데이터셋을 학습합니다.

```bash
# 도메인은 첫 인자 또는 DOMAIN 환경변수. 나머지 설정은 환경변수로 override.
bash training/act/scripts/train_act.sh recycling
DATASET_ROOT=/data/recycling_merged bash training/act/scripts/train_act.sh recycling
bash training/act/scripts/train_act.sh tableware
```

- 산출물: `${OUTPUT_DIR}/checkpoints/last/pretrained_model/` → Jetson `~/models/act/<domain>/pretrained_model` 로 복사
- recycling 데이터셋은 재라벨([`relabel_recycling_dataset.py`](imitation_data/scripts/relabel_recycling_dataset.py)) + 신규 수집분을 **단일 데이터셋으로 병합**한 뒤 그 root 를 `DATASET_ROOT` 로 지정 (병합 방법은 아래 **3-1** 참조 — 클라우드 학습 환경에서는 `aggregate_datasets` 권장)
- 데이터셋의 언어 임베딩 채널(MiniLM, D=384, L2 정규화)이 배포 `embedding_node` 와 동일 모델이어야 함
- ⚠️ **학습 전 언어 임베딩 키를 `observation.environment_state` 로 rename 필수**
  ([`rename_feature.py`](imitation_data/scripts/rename_feature.py)): stock ACT 는
  `observation.language_embedding`(STATE 중복)을 무시하고, `observation.environment_state`(ENV)
  만 인코더 토큰으로 배선한다. 안 하면 언어 조건화가 동작하지 않음(검증: `eval_act_offline.py` [4] |Δaction|>0).

스크립트 내부의 하이퍼파라미터·선행 조건·환경 변수는 [`train_act.sh`](act/scripts/train_act.sh)
헤더 주석 참조.

## 3-1. recycling 데이터셋 병합 — 클라우드에서는 `aggregate_datasets` 권장

recycling 은 재라벨본(~270) + 신규 수집본을 **단일 데이터셋으로 합쳐** `DATASET_ROOT`
로 지정해야 한다(`lerobot-train` 은 단일 `--dataset.*` 가정). tableware 는 단일
데이터셋이라 병합이 필요 없다.

> ⚠️ **클라우드(RunPod 등) 학습 환경에서는 재인코딩 방식 merge 를 피한다**:
> [`merge_lerobot_datasets.py`](imitation_data/scripts/merge_lerobot_datasets.py) 는
> 프레임을 **디코드 → 재인코딩**한다. vCPU 가 적거나 호스트 CPU 경합이 심한 클라우드
> GPU pod 에서는 **원본 av1 디코드가 병목**이 되어 571 에피소드에 **~수십 시간**이
> 걸린다. 인코딩 코덱을 h264 로 바꿔도(또는 `h264_nvenc`) 병목은 인코딩이 아니라
> **디코드**라 효과가 없고, NVENC 는 RTX 5090(Blackwell) 에서 미지원이다.
> (관련 증상: `torchcodec` 로드 실패 `undefined symbol ... libtorchcodec`, NVENC
> `OpenEncodeSessionEx failed: unsupported device`.)

대신 **lerobot 내장 `aggregate_datasets`** 를 쓴다. 비디오 mp4 를 재인코딩 없이
**stream-copy(concatenate)** 하고 parquet/메타만 합쳐 **수십 초** 만에 끝난다.
임베딩은 그대로 보존되고 task 테이블도 병합·remap 된다.

```bash
# 1) 재인코딩 없는 파일복사 병합 (수십 초)
python - <<'PY'
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
base = Path("training/imitation_data/datasets")
aggregate_datasets(
    repo_ids=["local/recycling_real_relabeled", "local/recycling_real"],
    roots=[base/"recycling_real_relabeled", base/"recycling_real"],
    aggr_repo_id="local/recycling_real_merge",
    aggr_root=base/"recycling_real_merge",
)
print("done")
PY

# 2) labels.jsonl 재생성 (aggregate 는 사이드카 미생성 — 학습엔 불필요하나 검증/정합용)
python training/imitation_data/scripts/regenerate_labels.py \
  --dataset-root training/imitation_data/datasets/recycling_real_merge --domain recycling

# 3) 병합본 검증 (feature/임베딩↔task cos≈1.0 / NaN / 커버리지)
python training/imitation_data/scripts/verify_real_dataset.py \
  --dataset-root training/imitation_data/datasets/recycling_real_merge --domain recycling

# 4) 학습
DATASET_ROOT=training/imitation_data/datasets/recycling_real_merge \
  bash training/act/scripts/train_act.sh recycling
```

> **`aggregate_datasets` 가 없는 lerobot 버전**이면 차선책으로
> `merge_lerobot_datasets.py --video-codec h264`(CPU x264) 를 쓰되, 본 실행 전
> `--limit 5` 로 속도를 먼저 확인한다.
>
> **RTX 5090(Blackwell) 환경 메모**: torch 는 cu128 빌드(예: 2.8.0+cu128), torchcodec
> 은 torch 와 ABI 가 맞는 버전(예: torch 2.8 ↔ torchcodec 0.6.0) 이어야 비디오
> 디코드(학습/병합 공통)가 동작한다. HF 모델 다운로드가 멈추면 `HF_HUB_DISABLE_XET=1`.
