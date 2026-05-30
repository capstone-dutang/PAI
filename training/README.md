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

## 2-3. Cat-C 합성 (검증 시나리오) — Sim 자산 준비 필요

(`generate_verify_qa.py` 본체 구현 후 작성)

## 2-4. Q&A 병합 (70:30, oversample)

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

## 2-5. Unsloth QLoRA 파인튜닝

```bash
python training/lora_finetuning/train_gemma_qlora.py \
  --base-model google/gemma-4-e2b \
  --dataset training/lora_finetuning/data/recycling/qa_merged.jsonl \
  --output-dir training/outputs/lora_recycling \
  --rank 16 --alpha 32 \
  --epochs 3
```

## 2-6. GGUF 변환

```bash
bash training/lora_finetuning/convert_lora_to_gguf.sh \
  training/outputs/lora_recycling \
  models/gemma/adapters/adapter_recycling.gguf
```

## 2-7. 데이터 보관 위치

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
