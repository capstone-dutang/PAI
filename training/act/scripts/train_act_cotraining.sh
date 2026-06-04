#!/usr/bin/env bash
# =============================================================================
# train_act_cotraining.sh
# -----------------------------------------------------------------------------
# ⚠️ 상태: **보존된 원설계 스켈레톤 (현 운영 아님)**. 현재 ACT 는 real-only 도메인별
#   train_act.sh 로 학습한다. 아래 본문은 원 co-training 설계 시점 표기라 일부 stale 하며,
#   되살릴 경우 반드시 현재 결정에 맞출 것:
#     (1) 언어 임베딩 obs 키 = observation.environment_state (ENV). observation.language_embedding
#         은 stock lerobot ACT 가 STATE 중복으로 **무시**한다 → 학습 전 rename_feature.py 로 rename.
#         (근거: 메모리 project_act_language_wiring / README "Language-conditioned VLA 패턴" 노트)
#     (2) Sim 목표 ~500 에피소드(순수 teleop 전환, README), 산출물 = 도메인별 pretrained_model/ 디렉터리.
# -----------------------------------------------------------------------------
# 목적: ACT Policy를 Sim dataset + Real dataset 혼합으로
#       Co-training 한다. LeRobot 0.4.4의 멀티 데이터셋 학습 기능을 사용.
#
# 실행 환경: 외부 PC (Ubuntu, GPU VRAM 16GB+)
# 결과물: act.pth → Jetson 배포 (~/models/act/act.pth)
#
# 선행 조건:
#   - Sim LeRobotDataset 생성 완료 (~1000 에피소드)
#                  + training/imitation_data/scripts/generate_language_labels.py로 language_embedding 채움
#   - Real LeRobotDataset 수집 완료 (50~100 에피소드)
#                  + record_real_teleop.py로 language_embedding 포함하여 기록
#   - 두 데이터셋의 features 스펙이 정확히 일치해야 한다 (Co-training 가능 조건):
#       observation.state              (6,)    float32
#       observation.language_embedding (D,)    float32  ← 동일 D 강제
#       observation.images.top         (H,W,3) uint8    (RGB만)
#       action                         (6,)    float32
#   - 임베딩 모델은 데이터 수집 / 배포의 embedding_node와도 동일해야 한다 (README #1).
# =============================================================================


# -----------------------------------------------------------------------------
# [1] 환경 변수 (값은 사용자 환경에 맞게 수정 필요 — 현재 미확정)
# -----------------------------------------------------------------------------
# SIM_DATASET_REPO_ID   : Sim 데이터 생성이 저장한 Sim dataset repo id
# REAL_DATASET_REPO_ID  : 데이터 수집이 저장한 Real dataset repo id
# OUTPUT_DIR            : 학습 결과(체크포인트, 로그) 저장 경로
# DEVICE                : cuda:0 등
# SEED                  : 재현성 시드


# -----------------------------------------------------------------------------
# [2] LeRobot 학습 커맨드 (개념 구조)
# -----------------------------------------------------------------------------
# lerobot-train 또는 python -m lerobot.scripts.train 를 사용.
# 멀티 데이터셋 지정 방식 (의사 코드 — README ACT Co-training 참조):
#   datasets = [
#       LeRobotDataset(SIM_DATASET_REPO_ID,  weight=0.7),
#       LeRobotDataset(REAL_DATASET_REPO_ID, weight=0.3),
#   ]
#
# 혼합 비율(샘플링 가중치) 고려:
#   - 단순 concat 시 Sim(1000) : Real(100) = 10:1 로 Sim이 우세
#   - Real의 가중치를 상향(예: 0.7/0.3 또는 0.5/0.5)하여 실기 분포 반영
#   - LeRobot 0.4.4의 weight 인자 정확한 형태는 운영 시점에 확인 필요 (미확정)
#
# 주요 하이퍼파라미터 (ACT):
#   --policy.type=act
#   --policy.chunk_size=100         (act_node와 일치)
#   --policy.n_action_steps=100
#   --batch_size                    : GPU VRAM 기준 결정
#   --training.offline_steps        : 총 스텝 수
#   --training.save_checkpoint_freq : 체크포인트 주기
#   --training.eval_freq            : 검증 주기
#
# 학습 입력 obs (act_node._predict_chunk()의 obs dict 키와 정확히 일치):
#   - observation.images.top         (visual encoder)
#   - observation.state              (proprioception)
#   - observation.language_embedding (language conditioning, D차원)
#
# 학습 타겟:
#   - action (chunk_size=100 시퀀스)


# -----------------------------------------------------------------------------
# [3] 산출물
# -----------------------------------------------------------------------------
# - ${OUTPUT_DIR}/checkpoints/last.pt
# - ${OUTPUT_DIR}/checkpoints/best.pt
# → Jetson ~/models/act/act.pth 로 복사하여 배포


# -----------------------------------------------------------------------------
# [4] 검증과의 연계
# -----------------------------------------------------------------------------
# - 학습 완료 후 검증: Isaac Sim에서 학습 시 사용하지 않은 DR 파라미터로
#   held-out 평가 수행 (Sim dataset과 동일 DR 범위는 train leakage)
# - 별도 dr_config_heldout.yaml 필요 (미작성)


# -----------------------------------------------------------------------------
# 미구현 / 결정 필요 항목
# -----------------------------------------------------------------------------
# [ ] LeRobot 0.4.4의 멀티 데이터셋 + 샘플링 가중치 공식 지원 여부
#     (미지원 시 dataset concat + WeightedRandomSampler 커스텀 필요)
# [ ] Sim:Real 혼합 비율 결정 (0.7:0.3 / 0.5:0.5 / 0.3:0.7 실험 필요)
# [ ] batch_size, offline_steps, learning rate 확정
# [ ] Real dataset 검증 split 분리 방식
# [ ] features의 language_embedding 차원 D가 두 데이터셋에서 일치하는지를
#     학습 진입 전에 검증하는 단계 추가 (불일치 시 즉시 abort)
