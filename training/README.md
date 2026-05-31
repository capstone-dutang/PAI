# 학습 가이드

`training/` 하위 3개 워크스트림의 실행 절차를 한 곳에 모은 가이드입니다.
설계·정책은 [`README.md`](../README.md) 의 관련 절을 참조하세요.

| 워크스트림 | 폴더 | 산출물 |
|---|---|---|
| 모방데이터 수집 | [`imitation_data/`](imitation_data/) | LeRobotDataset (Real teleop + Sim teleop) |
| Gemma 도메인 LoRA 파인튜닝 | [`lora_finetuning/`](lora_finetuning/) | LoRA adapter (.safetensors → GGUF) |
| ACT Co-training | [`act_cotraining/`](act_cotraining/) | ACT policy 체크포인트 (`act.pth`) |

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
4. 임베딩 모델 결정 및 배치 (Gemma 도메인 LoRA 학습 / ACT Co-training과 동일 모델 사용 필수)
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
  --ros-args -p model_path:=~/models/embedding/

# 4. teleop 데이터 수집 스크립트 실행 (Option C — 슬롯 기반 템플릿)
python <repo>/training/imitation_data/record_real_teleop.py \
  --robot-port      /dev/ttyACM1 \
  --robot-id        follower \
  --teleop-port     /dev/ttyACM0 \
  --teleop-id       leader \
  --dataset-repo-id myname/dataset_real \
  --domain          recycling \
  --label-vocab     ~/ros2_ws/config/label_vocab.yaml \
  --num-episodes    50

# 5. 수집 종료 후 paraphrase + 임베딩 후처리 (외부 PC 또는 Jetson 어디서나)
python <repo>/training/imitation_data/postprocess_real_labels.py \
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
~/isaacsim/python.sh training/imitation_data/record_sim_teleop.py \
  --scene-usd        sim/scenes/base_scene.usd \
  --teleop-port      /dev/ttyACM0 \
  --dataset-repo-id  myname/dataset_sim \
  --domain           recycling \
  --num-episodes     500

# 2. 각 에피소드에 Gemma + 외부 LLM 라벨 부여
python sim/generate_language_labels.py \
  --dataset dataset_sim/ \
  --gemma-server http://localhost:8080 \
  --external-llm gpt-4o-mini \
  --paraphrase-per-episode 3
```

---

# 2. Gemma 도메인 LoRA 파인튜닝 (`lora_finetuning/`)

Q&A 데이터 생성(`generate_function_call_qa.py` 의 Cat-B, `generate_verify_qa.py`
의 Cat-C)과 70:30 비율 혼합(`merge_qa_dataset.py`)을 거쳐 단일 JSONL 을 만든
후, 아래 두 단계로 학습·변환합니다.

> **데이터 포맷 — Unsloth FastVisionModel typed-array content (2026-05-25 freeze)**:
> Cat-B 와 Cat-C 가 같은 FastVisionModel 학습 패스에 들어가도록 모든 샘플의
> `content` 는 `[{"type":"text","text":...}]` (이미지 케이스에서는
> `{"type":"image","image":...}` 가 text 앞에 추가) 형식. tool role 은
> FastVisionModel 미지원이라 `user` role + `"[tool_result] "` prefix 로 변환됨.
> `generate_function_call_qa.py` 는 이미 신 형식으로 직접 생성하며,
> 기존 string-content 데이터는 [`convert_cat_b_to_vision_format.py`](lora_finetuning/convert_cat_b_to_vision_format.py)
> 로 1회 변환 가능.

## 2-1. 학습 환경 준비 (외부 PC)

```bash
pip install -r training/lora_finetuning/requirements.txt
export OPENAI_API_KEY=sk-...   # gpt-4o-mini 호출용 (Cat-B/C 합성기 공통)
```

## 2-2. Cat-B 합성 (함수 호출 시나리오)

```bash
# 도메인별로 1회씩 (recycling, tableware)
python training/lora_finetuning/generate_function_call_qa.py \
  --domain         recycling \
  --label-vocab    deploy/config/label_vocab.yaml \
  --output         training/lora_finetuning/data/recycling/qa_cat_b.jsonl \
  --external-llm   gpt-4o-mini \
  --num-samples    500
```

> 비용 추정: 500 샘플 ≈ $0.02–0.05 (gpt-4o-mini). 진행률·누적 토큰·비용이
> stderr 로 50샘플마다 출력됩니다.

**먼저 dry run (10샘플)** 으로 출력 JSONL 한두 줄 직접 검사 권장:
```bash
python training/lora_finetuning/generate_function_call_qa.py \
  --domain recycling --output /tmp/catB_dry.jsonl --num-samples 10
head -n 2 /tmp/catB_dry.jsonl | python -m json.tool
```

## 2-3. 기존 Cat-B JSONL 을 vision 포맷으로 변환 (1회용)

이전 string-content + tool role 형식으로 생성된 JSONL 이 있다면 다음으로 변환:

```bash
python training/lora_finetuning/convert_cat_b_to_vision_format.py \
  --input  training/lora_finetuning/data/recycling/function_call_qa_recycling.jsonl \
  --output training/lora_finetuning/data/recycling/function_call_qa_recycling.vision.jsonl

python training/lora_finetuning/convert_cat_b_to_vision_format.py \
  --input  training/lora_finetuning/data/tableware/function_call_qa_tableware.jsonl \
  --output training/lora_finetuning/data/tableware/function_call_qa_tableware.vision.jsonl
```

> 멱등 변환 — 이미 typed-array 인 입력은 그대로 통과. 신규 합성한 JSONL 은
> 이미 vision 포맷이므로 본 단계 불필요.

## 2-4. Cat-C 합성 (검증 시나리오)

`generate_verify_qa.py` 가 status 가중 샘플 + 객체 균등 순회 + 템플릿 reasoning
으로 시나리오를 결정론적으로 합성하고, 각 시나리오마다 LLM 1회 (original_command
생성) + Sim 렌더링 1회 (PNG 저장) 를 수행합니다.

> **freeze 결정의 캐노니컬 위치는 [`generate_verify_qa.py`](lora_finetuning/generate_verify_qa.py)
> 모듈 docstring 의 매트릭스를 참조**합니다. 값(분포·템플릿·destination 등)을
> 본 README 에 재기재하지 않으며, 변경 시 모듈의 해당 상수/함수만 수정하면
> 됩니다.

### 데이터 흐름

```
synthesize_scenario (LLM)         # status / target / target_phrase / cmd
   → generate_reasoning (template) # status 별 5 템플릿 중 선택, target/dest substitute
   → build_sample                  # vision typed-array messages
   → write JSONL line              # image path 만 박힘 (실제 PNG 는 별도)
   → render_scenario_image (Sim)   # Isaac Sim 으로 PNG 저장 — 미구현 stub
```

### 출력 디렉터리 구조

```
training/lora_finetuning/data/recycling/
├── qa_cat_c.jsonl
└── verify_images/
    ├── recycling_00000.png
    ├── recycling_00001.png
    └── ...
```

### dry run (구조 검증, Sim 의존 없음)

`--skip-render` 로 PNG 렌더링을 생략하고 JSONL 만 출력 — `original_command`
합성 (LLM) + reasoning 템플릿 + JSONL 스키마 검증용. 10 샘플 기준 ≈ $0.001:

```bash
python training/lora_finetuning/generate_verify_qa.py \
  --domain         recycling \
  --output-jsonl   /tmp/catC_dry.jsonl \
  --image-dir      /tmp/catC_dry_images \
  --num-scenes     10 \
  --skip-render

head -n 1 /tmp/catC_dry.jsonl | python -m json.tool
```

### 본격 합성 (Sim 렌더링 본체 구현 후)

```bash
python training/lora_finetuning/generate_verify_qa.py \
  --domain         recycling \
  --output-jsonl   training/lora_finetuning/data/recycling/qa_cat_c.jsonl \
  --image-dir      training/lora_finetuning/data/recycling/verify_images/ \
  --num-scenes     500

python training/lora_finetuning/generate_verify_qa.py \
  --domain         tableware \
  --output-jsonl   training/lora_finetuning/data/tableware/qa_cat_c.jsonl \
  --image-dir      training/lora_finetuning/data/tableware/verify_images/ \
  --num-scenes     500
```

> `render_scenario_image()` 가 미구현 (`NotImplementedError`) 상태이므로
> `--skip-render` 없이 실행하면 JSONL 은 정상 출력되지만 PNG 가 만들어지지 않아
> 학습 단계의 `PIL.Image.open(path)` 가 실패합니다. **본격 학습 직전에 Sim
> 자산 (`sim/assets/objects/<domain>/`) 준비 + render 본체 구현 + 재실행
> 필수**. Sim 의존부 명세는 `generate_verify_qa.py` 의 `render_scenario_image`
> docstring 참조.

## 2-5. Q&A 병합 (70:30, oversample)

```bash
python training/lora_finetuning/merge_qa_dataset.py \
  --cat-b   training/lora_finetuning/data/recycling/qa_cat_b.jsonl \
  --cat-c   training/lora_finetuning/data/recycling/qa_cat_c.jsonl \
  --output  training/lora_finetuning/data/recycling/qa_merged.jsonl
```

Cat-C 가 비어 있어도 동작 (Cat-B 만 사용 + 경고). 1차 학습 시 Cat-B 만으로
시험하려면 빈 파일을 `--cat-c` 로 넘기면 됩니다:
```bash
: > /tmp/catC_empty.jsonl
python training/lora_finetuning/merge_qa_dataset.py \
  --cat-b training/lora_finetuning/data/recycling/qa_cat_b.jsonl \
  --cat-c /tmp/catC_empty.jsonl \
  --output training/lora_finetuning/data/recycling/qa_merged.jsonl
```

## 2-6. Unsloth QLoRA 파인튜닝 (FastVisionModel)

```bash
python training/lora_finetuning/train_gemma_qlora.py \
  --base-model unsloth/gemma-4-E2B-it \
  --dataset training/lora_finetuning/data/recycling/qa_merged.jsonl \
  --output-dir training/outputs/lora_recycling \
  --rank 16 --alpha 32 \
  --epochs 3
```

> Cat-B(text-only) 와 Cat-C(image+text) 를 한 `FastVisionModel` 패스로 학습.
> Cat-B 는 image 항목이 없는 typed-array text 만 포함되며, `UnslothVisionDataCollator`
> 가 image=None 케이스를 처리하는지는 [unslothai/unsloth#1590](https://github.com/unslothai/unsloth/issues/1590)
> 의 workaround 패턴을 작은 batch 로 사전 검증할 것.

## 2-7. GGUF 변환

```bash
bash training/lora_finetuning/convert_lora_to_gguf.sh \
  training/outputs/lora_recycling \
  models/gemma/adapters/adapter_recycling.gguf
```

## 2-8. 데이터 보관 위치

Cat-B/C/merged JSONL 은 **도메인별 하위 폴더**로 관리합니다 (git 추적):

```
training/lora_finetuning/data/
├── recycling/
│   ├── qa_cat_b.jsonl
│   ├── qa_cat_c.jsonl
│   └── qa_merged.jsonl
└── tableware/
    ├── qa_cat_b.jsonl
    ├── qa_cat_c.jsonl
    └── qa_merged.jsonl
```

학습 산출물(LoRA 체크포인트, GGUF) 은 별도 `training/outputs/` 또는 `models/`
로 분리하며, 용량이 커서 `.gitignore` 대상입니다.

---

# 3. ACT Co-training (`act_cotraining/`)

LeRobot 0.4.4 멀티 데이터셋 가중치 학습으로 Sim + Real 혼합.

```bash
bash training/act_cotraining/train_act_cotraining.sh
```

스크립트 내부의 가중치·하이퍼파라미터·환경 변수는 `train_act_cotraining.sh`
헤더 주석 참조.
