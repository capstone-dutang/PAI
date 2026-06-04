#!/usr/bin/env python3
"""
eval_act_offline.py — 학습된 ACT 정책 오프라인 sanity 검증.

real-only ACT 는 Sim eval 환경이 없어 정식 평가는 실기 rollout 이지만, 그 전에
학습 데이터셋에서 오프라인으로 빠르게 다음을 확인한다:

  [1] 체크포인트/config 정합 (lerobot 불필요)
      - config.json 의 input_features 에 observation.language_embedding 이
        포함됐는지(= language-conditioned 로 학습됐는지), output=action 인지,
        chunk_size 등.
  [2] 정책/데이터셋 로드
  [3] action 예측 오차
      - 에피소드별 reset 후 프레임을 순차 진행하며 select_action(obs) 결과를
        GT action 과 비교(L1). 오차가 action 스케일 대비 작으면 매핑을 학습한 것.
  [4] 언어 조건화 민감도 (가장 중요)
      - 같은 시각 obs 에 '다른 에피소드의 임베딩'을 끼워 넣었을 때 예측 action 이
        바뀌는지. 거의 안 바뀌면 정책이 언어를 무시하는 shortcut 일 수 있음.

학습이 GPU 에서 동시에 돌고 있으면 --device cpu (기본) 로 실행해 간섭을 피한다.

CLI:
  python training/act/scripts/eval_act_offline.py \
    --policy-path  training/act/outputs/recycling/checkpoints/last/pretrained_model \
    --dataset-root training/imitation_data/datasets/recycling_real_merge \
    --device cpu --episodes 5 --max-frames 50
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ACTION_KEY = "action"
# 언어 임베딩 obs 키는 데이터셋마다 다르다(observation.language_embedding 또는,
# ACT 가 ENV 로 인식하게 rename 한 observation.environment_state) → 런타임 auto-detect.


# ──────────────────────────────────────────────────────────────────────────
def episode_bounds(root: Path) -> dict[int, list[int]]:
    """parquet 의 (index, episode_index) 로 에피소드별 전역 frame index 목록 구성.
    lerobot 버전과 무관(직접 읽음). ds[gidx] 가 이 전역 index 로 동작한다."""
    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    tbl = pa.concat_tables([pq.read_table(f, columns=["index", "episode_index"])
                            for f in files])
    idx = tbl["index"].to_numpy()
    ep = tbl["episode_index"].to_numpy()
    order = np.argsort(idx)
    bounds: dict[int, list[int]] = defaultdict(list)
    for gi, e in zip(idx[order].tolist(), ep[order].tolist()):
        bounds[int(e)].append(int(gi))
    return bounds


def phase1_config(policy_path: Path) -> dict | None:
    print("=" * 68)
    print("[1] 체크포인트 / config 점검")
    cfg_p = policy_path / "config.json"
    weights = (list(policy_path.glob("*.safetensors")) + list(policy_path.glob("*.bin")))
    print(f"  경로          : {policy_path}")
    print(f"  config.json   : {'있음' if cfg_p.exists() else '❌ 없음'}")
    print(f"  weight 파일   : {[f.name for f in weights] or '❌ 없음'}")
    if not cfg_p.exists():
        return None
    cfg = json.load(open(cfg_p, encoding="utf-8"))
    inp = list(cfg.get("input_features", {}))
    out = list(cfg.get("output_features", {}))
    print(f"  input_features : {inp}")
    print(f"  output_features: {out}")
    has_lang = any("language_embedding" in k for k in inp)
    print(f"  → language_embedding 입력: "
          f"{'예 ✅ (language-conditioned)' if has_lang else '아니오 ❌ — 언어 무시 학습됨!'}")
    for k in ("type", "chunk_size", "n_action_steps", "n_obs_steps"):
        if k in cfg:
            print(f"  {k} = {cfg[k]}")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="학습된 ACT 정책 오프라인 검증")
    ap.add_argument("--policy-path", required=True,
                    help="checkpoints/last/pretrained_model 디렉터리")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--device", default="cpu",
                    help="cpu(기본, 학습 간섭 회피) 또는 cuda")
    ap.add_argument("--episodes", type=int, default=5, help="표본 에피소드 수")
    ap.add_argument("--max-frames", type=int, default=50,
                    help="에피소드당 평가 프레임 상한(디코드 비용 절감)")
    ap.add_argument("--lang-frames", type=int, default=20,
                    help="언어 민감도 테스트 프레임 수(에피소드당)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    policy_path = Path(args.policy_path).expanduser()
    root = Path(args.dataset_root).expanduser()

    cfg = phase1_config(policy_path)

    # ── [2] 정책 + 데이터셋 로드 ────────────────────────────────────────
    print("=" * 68)
    print("[2] 정책/데이터셋 로드")
    try:
        import torch
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
        except Exception:
            from lerobot.common.policies.act.modeling_act import ACTPolicy  # 구버전
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        policy = ACTPolicy.from_pretrained(str(policy_path))
        policy.to(args.device)
        policy.eval()
        ds = LeRobotDataset(repo_id=f"local/{root.name}", root=str(root))
        print(f"  정책 로드 OK | device={args.device} | dataset frames={len(ds)}")
        obs_keys = [k for k in (cfg.get("input_features", {}) if cfg else {})
                    if k.startswith("observation.")]
        if not obs_keys:  # config 없으면 데이터셋 feature 에서 추론
            obs_keys = [k for k in ds.features if k.startswith("observation.")]
        print(f"  obs 키: {obs_keys}")
        # 언어/env_state 임베딩 키 auto-detect: 이미지도 아니고 state(6)도 아닌 obs.
        # (ACT 는 이 키가 'observation.environment_state' 여야 ENV 토큰으로 사용)
        lang_key = next((k for k in obs_keys
                         if "image" not in k and k != "observation.state"), None)
        print(f"  언어(env_state) 키: {lang_key}")
    except Exception as ex:
        print(f"  ❌ 로드 실패 — {type(ex).__name__}: {ex}")
        print("  (lerobot/torch 환경 또는 정책 경로 확인. [1] 결과만으로도 학습 정합은 확인 가능)")
        return

    bounds = episode_bounds(root)
    all_eps = sorted(bounds)
    pick = [all_eps[i] for i in
            np.linspace(0, len(all_eps) - 1, min(args.episodes, len(all_eps))).astype(int)]

    def obs_batch(item):
        return {k: item[k].unsqueeze(0).to(args.device) for k in obs_keys}

    def gt_action(item):
        a = item[ACTION_KEY]
        return a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)

    # ── [3] action 예측 오차 ────────────────────────────────────────────
    print("=" * 68)
    print(f"[3] action 예측 오차 (표본 에피소드 {pick})")
    try:
        errs, gts = [], []
        for ep in pick:
            policy.reset()
            for gi in bounds[ep][:args.max_frames]:
                item = ds[gi]
                with torch.no_grad():
                    pred = policy.select_action(obs_batch(item))
                pred = pred.squeeze(0).detach().cpu().numpy()
                gt = gt_action(item)
                errs.append(np.abs(pred - gt))
                gts.append(gt)
        errs = np.stack(errs)          # (T, A)
        gts = np.stack(gts)
        l1_joint = errs.mean(0)
        scale = gts.std(0) + 1e-6
        print(f"  프레임 {len(errs)}개 | joint 수 {errs.shape[1]}")
        print(f"  L1 오차 / joint     : {np.round(l1_joint, 3)}")
        print(f"  GT action std/joint : {np.round(gts.std(0), 3)}")
        print(f"  L1 / std (작을수록↑): {np.round(l1_joint / scale, 3)}")
        print(f"  전체 평균 L1 = {errs.mean():.4f} | (참고) GT 평균 |Δstd| 대비 "
              f"{(l1_joint / scale).mean():.2%}")
        print("  ⚠️ 주의: lerobot 0.4.4 는 정규화를 별도 processor 로 처리하는데 본 eval 은")
        print("     미적용이라 [3] 절대값(L1)은 신뢰 불가(과대). **핵심 판정은 아래 [4]** —")
        print("     [4] 는 swap 전/후 모두 동일 조건이라 차이값이 정규화와 무관하게 유효.")
    except Exception as ex:
        print(f"  ❌ 예측 오차 평가 실패 — {type(ex).__name__}: {ex}")

    # ── [4] 언어 조건화 민감도 ──────────────────────────────────────────
    print("=" * 68)
    print("[4] 언어 조건화 민감도 (임베딩 swap 시 action 변화)")
    if lang_key is None:
        print("  ⏭️  언어/env_state 입력 키를 못 찾음 — ACT 가 언어를 안 받는 구성입니다.")
        print("      (observation.environment_state 로 rename 후 재학습했는지 확인)")
        return
    try:
        ep_emb = {ep: ds[bounds[ep][0]][lang_key] for ep in pick}
        deltas = []
        for ep in pick:
            others = [e for e in pick
                      if not torch.allclose(ep_emb[e], ep_emb[ep], atol=1e-4)]
            if not others:
                continue
            for gi in bounds[ep][:args.lang_frames]:
                item = ds[gi]
                base = obs_batch(item)
                policy.reset()
                with torch.no_grad():
                    a_real = policy.select_action(base).squeeze(0).cpu().numpy()
                swap = dict(base)
                swap[lang_key] = ep_emb[random.choice(others)].unsqueeze(0).to(args.device)
                policy.reset()
                with torch.no_grad():
                    a_swap = policy.select_action(swap).squeeze(0).cpu().numpy()
                deltas.append(np.abs(a_real - a_swap))
        if not deltas:
            print("  표본 에피소드 임베딩이 모두 동일 — --episodes 를 늘려 다시 시도.")
            return
        deltas = np.stack(deltas)
        # 비교 기준: 같은 GT action std (위 [3]) — 없으면 자체 계산 생략
        print(f"  테스트 프레임 {len(deltas)}개")
        print(f"  |Δaction| / joint (임베딩 swap): {np.round(deltas.mean(0), 3)}")
        print(f"  전체 평균 |Δaction| = {deltas.mean():.4f}")
        print("  해석: 이 값이 0 에 가까우면 정책이 언어를 무시(shortcut). [3]의 GT std")
        print("        와 비슷하거나 그 이상으로 클수록 언어로 행동을 잘 변별하는 것.")
    except Exception as ex:
        print(f"  ❌ 언어 민감도 평가 실패 — {type(ex).__name__}: {ex}")

    print("=" * 68)
    print("완료. 정식 평가는 Jetson 실기 rollout 으로 수행.")


if __name__ == "__main__":
    main()
