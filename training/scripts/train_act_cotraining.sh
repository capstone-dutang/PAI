#!/usr/bin/env bash
# =============================================================================
# train_act_cotraining.sh
# -----------------------------------------------------------------------------
# 목적: Part E — ACT Policy를 Sim dataset(B-2) + Real dataset(D) 혼합으로
#       Co-training 한다. LeRobot 0.4.4의 멀티 데이터셋 학습 기능을 사용.
#
# 실행 환경: 외부 PC (Ubuntu, GPU VRAM 16GB+)
# 결과물: act_policy.pth → Part F에서 Jetson 배포
#
# 선행 조건:
#   - Part B-2 완료: Sim LeRobotDataset 생성 (~1000 에피소드)
#   - Part D 완료: Real LeRobotDataset 수집 (50~100 에피소드)
#   - 두 데이터셋의 features 스펙 동일:
#       observation.state (6,) / observation.object_pose (3,) /
#       action (6,) / observation.images.top (RGB)
# =============================================================================


# -----------------------------------------------------------------------------
# [1] 환경 변수 (값은 사용자 환경에 맞게 수정 필요 — 현재 미확정)
# -----------------------------------------------------------------------------
# SIM_DATASET_REPO_ID   : Part B-2가 저장한 Sim dataset repo id
# REAL_DATASET_REPO_ID  : Part D가 저장한 Real dataset repo id
# OUTPUT_DIR            : 학습 결과(체크포인트, 로그) 저장 경로
# DEVICE                : cuda:0 등
# SEED                  : 재현성 시드


# -----------------------------------------------------------------------------
# [2] LeRobot 학습 커맨드 (개념 구조)
# -----------------------------------------------------------------------------
# lerobot-train 또는 python -m lerobot.scripts.train 를 사용.
# 멀티 데이터셋 지정 방식:
#   --dataset.repo_id=[${SIM_DATASET_REPO_ID},${REAL_DATASET_REPO_ID}]
#
# 혼합 비율(샘플링 가중치) 고려:
#   - 단순 concat 시 Sim(1000) : Real(100) = 10:1 로 Sim이 우세
#   - Real의 가중치를 상향(예: 1:1 또는 3:1)하여 실기 분포를 충분히 반영
#   - LeRobot 0.4.4의 멀티 dataset weight 인자 확인 필요 (미확정)
#
# 주요 하이퍼파라미터 (ACT):
#   --policy.type=act
#   --policy.chunk_size=100         (Part F act_node와 일치)
#   --policy.n_action_steps=100
#   --batch_size                    : GPU VRAM 기준 결정
#   --training.offline_steps        : 총 스텝 수
#   --training.save_checkpoint_freq : 체크포인트 주기
#   --training.eval_freq            : 검증 주기
#
# 입력 feature:
#   - observation.images.top  (visual encoder)
#   - observation.state       (proprioception)
#   - observation.object_pose (VLM/YOLO 3D 좌표 — Sim은 GT, Real은 추론값)


# -----------------------------------------------------------------------------
# [3] 산출물
# -----------------------------------------------------------------------------
# - ${OUTPUT_DIR}/checkpoints/last.pt
# - ${OUTPUT_DIR}/checkpoints/best.pt
# → Jetson ~/models/act/act_policy.pth 로 복사하여 Part F 배포


# -----------------------------------------------------------------------------
# [4] Part G 검증과의 연계
# -----------------------------------------------------------------------------
# - 학습 완료 후 Part G: Isaac Sim에서 학습 시 사용하지 않은 DR 파라미터로
#   held-out 평가 수행 (Sim dataset과 동일 DR 범위는 train leakage)
# - 별도 dr_config_heldout.yaml 필요 (미작성)


# -----------------------------------------------------------------------------
# TODO (구현 전 확인 필요)
# -----------------------------------------------------------------------------
# [ ] LeRobot 0.4.4가 멀티 데이터셋 + 샘플링 가중치를 공식 지원하는지 확인
#     (미지원 시 dataset concat + WeightedRandomSampler 커스텀 필요)
# [ ] Sim:Real 혼합 비율 결정 (1:1 / 3:1 / 10:1 실험 필요)
# [ ] batch_size, offline_steps, learning rate 확정
# [ ] Real dataset 검증 split 분리 방식