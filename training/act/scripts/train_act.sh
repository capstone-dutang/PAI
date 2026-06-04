#!/usr/bin/env bash
# =============================================================================
# train_act.sh
# -----------------------------------------------------------------------------
# 목적: ACT Policy 를 **real-only, 도메인별**로 학습한다 (LeRobot 0.4.4 lerobot-train).
#       도메인당 1회 호출 → 도메인별 체크포인트 1개 산출.
#
# 실행 환경: 외부 PC (Ubuntu, GPU VRAM 16GB+)
# 산출물: ${OUTPUT_DIR}/checkpoints/last/pretrained_model/  (디렉터리)
#         → Jetson 배포 시 ~/models/act/<domain>/pretrained_model 로 복사
#         → act_node 의 model_path 가 이 pretrained_model 디렉터리를 가리킨다.
#
# -----------------------------------------------------------------------------
# 설계 변경점 (co-training → real-only) — README 에는 남기지 않고 여기에만 기록
# -----------------------------------------------------------------------------
#   - 원래 설계: Sim + Real **co-training** (train_act_cotraining.sh, 멀티 데이터셋
#     가중치). Sim teleop 수집이 실질적으로 불가능해 **real-only 도메인별 학습**으로
#     전환. co-training 스켈레톤(train_act_cotraining.sh)은 참고용으로 보존한다.
#   - 단일 act.pth + LoRA hot-swap 가정 → **도메인별 ACT 체크포인트**로 변경
#     (recycling / tableware 각각). 도메인 전환은 추후 act_node 의 model 선택/재기동.
#   - ACT 는 language-conditioned 다. ★주의★ config.json input_features 에 키가
#     '포함'되는 것만으론 부족하다 — stock lerobot ACT 는 observation.* 를 STATE 로,
#     정확히 observation.environment_state 만 ENV 로 분류해 인코더 토큰에 배선한다.
#     그래서 언어 임베딩 채널은 학습 전 observation.environment_state 로 rename 해야
#     ACT 가 실제로 사용한다(rename_feature.py). language_embedding 키로 두면 중복
#     STATE 가 되어 ACT 가 무시한다(과거 실수). 임베딩은 MiniLM D=384, L2 정규화,
#     배포 embedding_node 와 동일 모델. 검증: eval_act_offline.py [4] |Δaction|>0.
#
# -----------------------------------------------------------------------------
# 하이퍼파라미터 선택 근거 (의도적 결정)
# -----------------------------------------------------------------------------
#   - chunk_size=100, n_action_steps=100, temporal_ensemble 미사용
#       → 100 스텝을 한 번에 예측해 **open-loop 로 전부 실행 후 재예측**하는 모드.
#         30Hz 면 약 3.3 초마다 재계획이라 반응성이 낮다. 반응성이 필요하면
#         N_ACTION_STEPS 를 줄이거나(예: 25) TEMPORAL_ENSEMBLE_COEFF 를 설정한다.
#         act_node 의 closed-loop select_action 과는 정합 (큐 소진 시 재예측).
#   - paraphrase 증강 미적용: postprocess_real_labels.py 의 paraphrase_pool.jsonl 은
#         사이드카 파일이라 표준 lerobot-train 이 읽지 않는다. 본 학습은 프레임에
#         baked 된 canonical 임베딩만 조건으로 사용한다. paraphrase 를 실제로 쓰려면
#         데이터셋 프레임 임베딩을 무작위 교체하는 별도 훅이 필요(미구현).
#   - 검증: real 은 Sim eval 환경이 없으므로 실기 rollout 으로 평가한다
#         (lerobot-train 의 online eval 미사용).
#
# -----------------------------------------------------------------------------
# 선행 조건
# -----------------------------------------------------------------------------
#   - recycling: 재라벨(relabel_recycling_dataset.py 산출 ~270) + 신규 수집분을
#     **단일 데이터셋**으로 합쳐 두고 그 root 를 DATASET_ROOT 로 지정해야 한다
#     (lerobot-train 단일 --dataset.* 가정). 결합 방식(디스크 병합 / append /
#     멀티 데이터셋)은 데이터 준비 단계에서 확정.
#   - tableware: plate/fork/spoon 갖춰 신규 수집한 단일 데이터셋.
#   - 두 도메인 데이터셋의 feature 스펙(observation.state(6)/environment_state(384,
#     =rename된 언어 임베딩)/images.top(H,W,3)/action(6))과 fps·해상도가 학습 진입
#     전 일관해야 한다.
#
# -----------------------------------------------------------------------------
# 사용법
# -----------------------------------------------------------------------------
#   # 도메인은 첫 인자 또는 DOMAIN 환경변수. 나머지는 환경변수로 override.
#   bash training/act/scripts/train_act.sh recycling
#   DATASET_ROOT=/data/recycling_merged bash training/act/scripts/train_act.sh recycling
#   bash training/act/scripts/train_act.sh tableware
#
#   # 반응성 실험: closed-loop 성격 강화
#   N_ACTION_STEPS=25 bash training/act/scripts/train_act.sh recycling
# =============================================================================

set -euo pipefail

# ── 도메인 ────────────────────────────────────────────────────────────────
DOMAIN="${1:-${DOMAIN:-recycling}}"
if [[ "${DOMAIN}" != "recycling" && "${DOMAIN}" != "tableware" ]]; then
  echo "[train_act] 경고: 알려진 도메인(recycling/tableware)이 아님: '${DOMAIN}'" >&2
fi

# ── 설정 (환경변수 override 가능) ─────────────────────────────────────────
# DATASET_ROOT  : LeRobotDataset 디렉터리 (recycling 은 재라벨+신규 합본).
# DATASET_REPO_ID: lerobot 이 사용할 repo id (push 안 하므로 라벨 역할).
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DOMAIN}_real}"
DATASET_ROOT="${DATASET_ROOT:-training/imitation_data/datasets/${DOMAIN}_real}"
OUTPUT_DIR="${OUTPUT_DIR:-models/act/outputs/${DOMAIN}}"   # models/ 는 gitignore
POLICY_REPO_ID="${POLICY_REPO_ID:-local/act_${DOMAIN}}"

DEVICE="${DEVICE:-cuda}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
N_ACTION_STEPS="${N_ACTION_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-80000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

# temporal ensemble (선택): 값이 있으면 플래그 추가. 비어 있으면 미사용(기존 동작).
# 주의: 정확한 플래그명은 설치된 lerobot 0.4.4 빌드에서 확인 필요.
TEMPORAL_ENSEMBLE_COEFF="${TEMPORAL_ENSEMBLE_COEFF:-}"

# ── 사전 점검 ─────────────────────────────────────────────────────────────
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[train_act] 오류: DATASET_ROOT 가 없음: ${DATASET_ROOT}" >&2
  echo "            recycling 은 재라벨+신규 합본 데이터셋 경로를 지정하세요." >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[train_act] 오류: OUTPUT_DIR 가 이미 존재: ${OUTPUT_DIR}" >&2
  echo "            새 경로를 지정하거나 기존 경로를 정리/resume 처리하세요." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${TEMPORAL_ENSEMBLE_COEFF}" ]]; then
  EXTRA_ARGS+=( "--policy.temporal_ensemble_coeff=${TEMPORAL_ENSEMBLE_COEFF}" )
  echo "[train_act] temporal_ensemble_coeff=${TEMPORAL_ENSEMBLE_COEFF} (플래그명 확인 필요)" >&2
fi

# ── 설정 출력 ─────────────────────────────────────────────────────────────
cat >&2 <<EOF
[train_act] ─ 학습 설정 ──────────────────────────────────
  DOMAIN          = ${DOMAIN}
  DATASET_REPO_ID = ${DATASET_REPO_ID}
  DATASET_ROOT    = ${DATASET_ROOT}
  OUTPUT_DIR      = ${OUTPUT_DIR}
  POLICY_REPO_ID  = ${POLICY_REPO_ID}
  DEVICE          = ${DEVICE}
  chunk_size      = ${CHUNK_SIZE}   n_action_steps = ${N_ACTION_STEPS}
  batch_size      = ${BATCH_SIZE}   steps = ${STEPS}
  save_freq       = ${SAVE_FREQ}   log_freq = ${LOG_FREQ}
  num_workers     = ${NUM_WORKERS}   seed = ${SEED}   wandb = ${WANDB_ENABLE}
───────────────────────────────────────────────────────────
EOF

# ── 학습 ──────────────────────────────────────────────────────────────────
lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --num_workers="${NUM_WORKERS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="act_${DOMAIN}" \
  --seed="${SEED}" \
  --wandb.enable="${WANDB_ENABLE}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[train_act] 완료 — 체크포인트: ${OUTPUT_DIR}/checkpoints/last/pretrained_model/" >&2
