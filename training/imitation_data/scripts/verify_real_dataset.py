#!/usr/bin/env python3
"""
verify_real_dataset.py

record_real_teleop.py 로 새로 수집한 (재라벨이 아닌) LeRobotDataset v3.0 real
데이터셋이 ACT 학습에 문제가 없는지 검증한다.

verify_relabel.py 와의 차이:
  - verify_relabel.py 는 relabel_recycling_dataset.py 산출물 전용(bottle→plastic
    remap, 옛/새 canonical 대비)이라 신규 수집 데이터에는 맞지 않는다.
  - 본 스크립트는 lerobot 패키지 없이 parquet/메타를 직접 읽어 검사하므로 개발
    PC(Windows, lerobot 미설치)에서도 실행 가능하다.

핵심 관점:
  LeRobot 학습이 실제로 사용하는 라벨은 labels.jsonl 이 아니라 데이터셋 내부의
  meta/tasks.parquet(task_index → 문자열) + 프레임의 observation.language_embedding
  이다. 따라서 "에피소드별 task 가 단일한가 / baked 임베딩이 그 task 문자열과
  일치하는가" 가 진짜 정합성 판정이고, labels.jsonl 의 중복·충돌은 보조 로그의
  위생 문제(WARN)로 분리해서 본다.

검사 항목:
  A. info.json   — feature 스펙(state6/lang_emb384/images.top[480,640,3]/action6),
                   fps, codebase_version, episode/frame/task 카운트
  B. 프레임 데이터 — 행수==total_frames, episode_index 0..N-1 완전 커버,
                   index 연속, frame_index/timestamp 정합, state/action/emb NaN·inf,
                   state/action 값 범위
  C. language_embedding — L2 norm≈1, 에피소드 내 broadcast 동일, 전부 0/상수 아님
  D. labels.jsonl — 커버리지, 중복/충돌, vocab 정합, canonical==template
  E. 임베딩↔task — 에피소드별 task_index 단일, baked 임베딩이 task 문자열의
                   재계산 임베딩과 cosine≈1 (= 올바른 라벨이 구워졌는지),
                   task 문자열이 vocab 으로 표현 가능한지
  F. labels↔task — 에피소드 task 문자열이 labels.jsonl 후보 중 하나인지
                   (충돌 ep 이 결국 어떤 라벨로 구워졌는지 규명)
  G. 부가 위생   — videos 파일 수, tmp* 잔존 디렉터리

CLI:
  python training/imitation_data/scripts/verify_real_dataset.py \
    --dataset-root training/imitation_data/datasets/recycling_real
  # --domain 은 생략 시 labels.jsonl 의 domain 필드에서 추론
  # --no-embed 로 임베딩 재계산(모델 로드) 생략 가능
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# 콘솔 인코딩과 무관하게 한글이 깨지지 않도록 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXPECTED = {
    "observation.state": 6,
    "action": 6,
}
# 임베딩 채널 키: 수집 원본은 observation.language_embedding, ACT 학습용으로
# rename(rename_feature.py) 하면 observation.environment_state (ENV). 둘 다 허용.
EMB_CANDIDATES = ("observation.language_embedding", "observation.environment_state")
EMB_DIM = 384
CAMERA_KEY = "observation.images.top"
EXPECTED_IMG = [480, 640, 3]
EXPECTED_FPS = 30


# ──────────────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────────────
class Report:
    def __init__(self):
        self.rows = []  # (level, name, detail)  level ∈ {PASS, FAIL, WARN}

    def ok(self, name, detail=""):
        self.rows.append(("PASS", name, detail))

    def fail(self, name, detail=""):
        self.rows.append(("FAIL", name, detail))

    def warn(self, name, detail=""):
        self.rows.append(("WARN", name, detail))

    def check(self, cond, name, detail_ok="", detail_bad=""):
        if cond:
            self.ok(name, detail_ok)
        else:
            self.fail(name, detail_bad)
        return cond

    def summarize(self):
        n_fail = sum(1 for lvl, _, _ in self.rows if lvl == "FAIL")
        n_warn = sum(1 for lvl, _, _ in self.rows if lvl == "WARN")
        n_pass = sum(1 for lvl, _, _ in self.rows if lvl == "PASS")
        print("=" * 72)
        for lvl, name, detail in self.rows:
            mark = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}[lvl]
            tail = f"  — {detail}" if detail else ""
            print(f"  {mark} [{lvl}] {name}{tail}")
        print("=" * 72)
        print(f"결과: PASS {n_pass} / WARN {n_warn} / FAIL {n_fail}")
        return n_fail, n_warn


def fsl_to_np(table: pa.Table, col: str) -> np.ndarray:
    """list<float>[k] / fixed_size_list<float>[k] 컬럼 → (N, k) float32.

    record_real_teleop 산출물은 fixed_size_list(list_size 보유)지만,
    aggregate_datasets 산출물은 가변 list 로 저장될 수 있어 list_size 가 없다.
    두 경우 모두 지원하도록 폭 k 를 추론한다(행별 길이 동일 가정)."""
    arr = table[col].combine_chunks()
    n = len(arr)
    flat = arr.flatten().to_numpy(zero_copy_only=False)
    k = getattr(arr.type, "list_size", None)
    if not isinstance(k, int) or k <= 0:           # ListType (가변) → 폭 추론
        k = (len(flat) // n) if n else 0
    return flat.reshape(n, k).astype(np.float32)


def load_vocab(path: str, domain: str) -> dict:
    raw = yaml.safe_load(open(path, encoding="utf-8"))
    dom = raw["domains"][domain]
    spatials = list(raw.get("spatial_qualifiers", [])) + list(
        dom.get("spatial_qualifiers", [])
    )
    return {
        "template": raw["template"],
        "objects": {o: set(v["attributes"]) for o, v in dom["objects"].items()},
        "spatials": set(spatials),
    }


def vocab_ok(vocab, obj, attr, spat) -> bool:
    return (
        obj in vocab["objects"]
        and attr in vocab["objects"].get(obj, set())
        and spat in vocab["spatials"]
    )


# ──────────────────────────────────────────────────────────────────────────
# 메인 검증
# ──────────────────────────────────────────────────────────────────────────
def verify(root: str, domain: str | None, vocab_path: str, embed_model: str,
           do_embed: bool) -> int:
    root = os.path.expanduser(root)
    name = Path(root).name
    print("#" * 72)
    print(f"# 데이터셋 검증: {root}")
    print("#" * 72)
    rep = Report()

    # ── A. info.json ────────────────────────────────────────────────────
    info = json.load(open(Path(root) / "meta" / "info.json", encoding="utf-8"))
    feats = info["features"]
    for k, dim in EXPECTED.items():
        shape = feats.get(k, {}).get("shape")
        rep.check(shape == [dim], f"feature {k} shape==[{dim}]",
                  detail_bad=f"실제 {shape}")
    # 임베딩 채널 auto-detect (language_embedding 또는 environment_state)
    emb_key = next((k for k in EMB_CANDIDATES if k in feats), None)
    rep.check(emb_key is not None,
              "임베딩 채널(language_embedding/environment_state) 존재", detail_bad="둘 다 없음")
    if emb_key:
        shape = feats.get(emb_key, {}).get("shape")
        rep.check(shape == [EMB_DIM], f"feature {emb_key} shape==[{EMB_DIM}]",
                  detail_bad=f"실제 {shape}")
        print(f"  임베딩 채널 = {emb_key}")
    img = feats.get(CAMERA_KEY, {}).get("shape")
    rep.check(img == EXPECTED_IMG, f"feature {CAMERA_KEY} shape=={EXPECTED_IMG}",
              detail_bad=f"실제 {img}")
    rep.check(info.get("fps") == EXPECTED_FPS, f"fps=={EXPECTED_FPS}",
              detail_bad=f"실제 {info.get('fps')}")
    cv = info.get("codebase_version")
    (rep.ok if cv == "v3.0" else rep.warn)("codebase_version", cv)
    total_eps = info["total_episodes"]
    total_frames = info["total_frames"]
    total_tasks = info["total_tasks"]
    print(f"  info: episodes={total_eps} frames={total_frames} tasks={total_tasks} "
          f"fps={info.get('fps')} ver={cv}")

    # ── 프레임 parquet 로드 ──────────────────────────────────────────────
    files = sorted(glob.glob(str(Path(root) / "data" / "chunk-*" / "file-*.parquet")))
    rep.check(len(files) > 0, "data parquet 파일 존재", detail_bad="없음")
    tbl = pa.concat_tables([pq.read_table(f) for f in files])
    n = tbl.num_rows
    order = np.argsort(tbl["index"].to_numpy())  # 전역 index 기준 정렬
    ep_idx = tbl["episode_index"].to_numpy()[order]
    g_index = tbl["index"].to_numpy()[order]
    frame_index = tbl["frame_index"].to_numpy()[order]
    ts = tbl["timestamp"].to_numpy()[order]
    task_index = tbl["task_index"].to_numpy()[order]
    state = fsl_to_np(tbl, "observation.state")[order]
    action = fsl_to_np(tbl, "action")[order]
    emb = (fsl_to_np(tbl, emb_key)[order] if emb_key
           else np.zeros((n, EMB_DIM), np.float32))

    # ── B. 프레임 정합 ───────────────────────────────────────────────────
    rep.check(n == total_frames, "행수 == total_frames",
              detail_bad=f"parquet {n} vs info {total_frames}")
    rep.check(np.array_equal(g_index, np.arange(n)),
              "전역 index 0..N-1 연속", detail_bad="index 불연속/중복")

    uniq_eps = np.unique(ep_idx)
    rep.check(np.array_equal(uniq_eps, np.arange(total_eps)),
              f"episode_index 0..{total_eps-1} 완전 커버",
              detail_bad=f"uniq={len(uniq_eps)} (min={uniq_eps.min()} max={uniq_eps.max()})")

    # 에피소드 경계 (정렬된 ep_idx 가정: 같은 ep 가 연속) → 검사도 같이
    # 경계 인덱스
    bounds = defaultdict(list)
    for i, e in enumerate(ep_idx):
        bounds[int(e)].append(i)
    ep_lengths = {e: len(ix) for e, ix in bounds.items()}
    rep.check(all(L > 0 for L in ep_lengths.values()),
              "모든 에피소드 frame 수 > 0",
              detail_bad=str([e for e, L in ep_lengths.items() if L == 0][:5]))
    contiguous = all(ix == list(range(ix[0], ix[0] + len(ix))) for ix in bounds.values())
    rep.check(contiguous, "에피소드 프레임이 연속 블록", detail_bad="에피소드 인터리브됨")

    # frame_index 가 에피소드 내 0..L-1
    bad_fi = [e for e, ix in bounds.items()
              if not np.array_equal(frame_index[ix], np.arange(len(ix)))]
    rep.check(not bad_fi, "frame_index 에피소드내 0..L-1",
              detail_bad=f"위반 ep {bad_fi[:5]}")

    # timestamp: 에피소드내 단조증가 + 간격 ≈ 1/fps
    dt = 1.0 / EXPECTED_FPS
    bad_ts_mono, bad_ts_gap = [], []
    for e, ix in bounds.items():
        t = ts[ix]
        if np.any(np.diff(t) <= 0):
            bad_ts_mono.append(e)
        if len(t) > 1 and np.max(np.abs(np.diff(t) - dt)) > dt * 0.5:
            bad_ts_gap.append(e)
    rep.check(not bad_ts_mono, "timestamp 에피소드내 단조증가",
              detail_bad=f"위반 ep {bad_ts_mono[:5]}")
    (rep.ok if not bad_ts_gap else rep.warn)(
        "timestamp 간격 ≈ 1/fps",
        "" if not bad_ts_gap else f"이탈 ep {bad_ts_gap[:5]} ({len(bad_ts_gap)}개)")

    # NaN/inf
    for label, arr in [("state", state), ("action", action), ("embedding", emb)]:
        bad = int(np.sum(~np.isfinite(arr)))
        rep.check(bad == 0, f"{label} NaN/inf 없음", detail_bad=f"{bad}개")

    # 값 범위 (SO-101 .pos: 관절 deg, gripper 0~100 가정 — 극단치만 WARN)
    smin, smax = state.min(0), state.max(0)
    amin, amax = action.min(0), action.max(0)
    print(f"  state  range/joint: min={np.round(smin,1)} max={np.round(smax,1)}")
    print(f"  action range/joint: min={np.round(amin,1)} max={np.round(amax,1)}")
    extreme = bool(np.any(np.abs(smin) > 400) or np.any(np.abs(smax) > 400)
                   or np.any(np.abs(amin) > 400) or np.any(np.abs(amax) > 400))
    (rep.warn if extreme else rep.ok)(
        "state/action 값 범위 정상(|x|<=400)",
        "극단값 존재 — 단위/캘리브레이션 확인" if extreme else "")

    # ── C. 임베딩 ────────────────────────────────────────────────────────
    norms = np.linalg.norm(emb, axis=1)
    rep.check(np.all(np.abs(norms - 1.0) <= 1e-2), "임베딩 L2 정규화(|emb|≈1)",
              detail_bad=f"norm range [{norms.min():.3f},{norms.max():.3f}]")
    rep.check(float(norms.min()) > 1e-6, "임베딩 all-zero 아님",
              detail_bad="zero 임베딩 존재")
    # 에피소드 내 broadcast (모든 frame 동일)
    bad_bcast = []
    first_emb = {}
    for e, ix in bounds.items():
        block = emb[ix]
        first_emb[e] = block[0]
        if np.max(np.abs(block - block[0])) > 1e-5:
            bad_bcast.append(e)
    rep.check(not bad_bcast, "에피소드내 임베딩 broadcast 동일",
              detail_bad=f"위반 ep {bad_bcast[:5]}")
    # 에피소드 간 다양성
    distinct = len({first_emb[e].tobytes() for e in first_emb})
    rep.check(distinct > 1, "에피소드 간 임베딩 다양성(>1 distinct)",
              detail_bad=f"distinct={distinct}")
    print(f"  임베딩: |emb| in [{norms.min():.4f},{norms.max():.4f}] | "
          f"distinct ep-embeddings={distinct}/{total_eps}")

    # ── tasks.parquet (task_index → 문자열) ─────────────────────────────
    tdf = pq.read_table(Path(root) / "meta" / "tasks.parquet").to_pandas()
    # 문자열이 index 에, 정수가 'task_index' 컬럼에 있음
    idx2task = {int(v): str(k) for k, v in tdf["task_index"].items()}
    rep.check(len(idx2task) == total_tasks, "tasks 수 == total_tasks",
              detail_bad=f"{len(idx2task)} vs {total_tasks}")

    # 에피소드별 task_index 단일성
    bad_multitask = []
    ep_task = {}
    for e, ix in bounds.items():
        tset = set(task_index[ix].tolist())
        if len(tset) != 1:
            bad_multitask.append((e, sorted(tset)))
        ep_task[e] = int(task_index[ix][0])
    rep.check(not bad_multitask, "에피소드별 task_index 단일",
              detail_bad=f"복수 task ep {bad_multitask[:3]}")

    # ── D. labels.jsonl ──────────────────────────────────────────────────
    labels_p = Path(root) / "labels.jsonl"
    labs = [json.loads(l) for l in open(labels_p, encoding="utf-8")] if labels_p.exists() else []
    rep.check(bool(labs), "labels.jsonl 존재", detail_bad=str(labels_p))
    if domain is None:
        doms = Counter(r.get("domain") for r in labs)
        domain = doms.most_common(1)[0][0] if doms else "recycling"
    print(f"  domain = {domain}")
    vocab = load_vocab(vocab_path, domain)

    by_ep_labels = defaultdict(list)  # ep -> [canonical,...] (등장 순서)
    for r in labs:
        by_ep_labels[int(r["episode"])].append(r.get("canonical", ""))
    uniq_lab_eps = set(by_ep_labels)
    rep.check(uniq_lab_eps == set(range(total_eps)),
              "labels 고유 ep == 전체 에피소드",
              detail_bad=f"고유 {len(uniq_lab_eps)} vs {total_eps}")

    dups = {e: len(v) for e, v in by_ep_labels.items() if len(v) > 1}
    conflicts = {e: sorted(set(v)) for e, v in by_ep_labels.items() if len(set(v)) > 1}
    (rep.ok if not dups else rep.warn)(
        "labels.jsonl 에피소드 중복 없음",
        "" if not dups else f"{len(dups)}개 ep 중복(append 추정)")
    (rep.ok if not conflicts else rep.warn)(
        "labels.jsonl 동일 ep 라벨 충돌 없음",
        "" if not conflicts else f"{len(conflicts)}개 ep 충돌 — 보조로그 위생 이슈")

    # vocab / canonical 정합 (labels)
    bad_vocab, bad_canon = [], []
    for r in labs:
        o, a, s, c = r.get("object"), r.get("attribute"), r.get("spatial"), r.get("canonical", "")
        if not vocab_ok(vocab, o, a, s):
            bad_vocab.append(r["episode"])
        if c != vocab["template"].format(attribute=a, object=o, spatial=s):
            bad_canon.append(r["episode"])
    rep.check(not bad_vocab, "labels 가 vocab 으로 표현 가능",
              detail_bad=f"위반 ep {sorted(set(bad_vocab))[:5]}")
    rep.check(not bad_canon, "labels canonical == template",
              detail_bad=f"불일치 ep {sorted(set(bad_canon))[:5]}")

    # 분포 출력
    obj_dist = Counter(r.get("object") for r in labs)
    spat_dist = Counter(r.get("spatial") for r in labs)
    print(f"  labels object 분포 : {dict(obj_dist)}")
    print(f"  labels spatial 분포: {dict(spat_dist)}")

    # task 문자열도 vocab 으로 표현 가능한지 (in-dataset 정답 라벨)
    def parse_canon(c):
        # "the {attribute} {object} {spatial}" 역파싱 — vocab 매칭으로 확인
        for o, attrs in vocab["objects"].items():
            for a in attrs:
                for s in vocab["spatials"]:
                    if c == vocab["template"].format(attribute=a, object=o, spatial=s):
                        return (o, a, s)
        return None
    bad_task_vocab = [ti for ti, c in idx2task.items() if parse_canon(c) is None]
    rep.check(not bad_task_vocab, "tasks 문자열이 vocab canonical",
              detail_bad=f"비vocab task_index {bad_task_vocab[:5]}")

    # ── E. 임베딩 ↔ task 재계산 정합 + F. labels↔task ───────────────────
    if do_embed:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  임베딩 모델 로드: {embed_model} ...")
            model = SentenceTransformer(embed_model)
            # 필요한 모든 문자열(= task 문자열 + labels 후보) 일괄 인코딩
            need = set(idx2task.values())
            for v in by_ep_labels.values():
                need.update(v)
            need.discard("")
            need = sorted(need)
            vecs = model.encode(need, normalize_embeddings=True, convert_to_numpy=True,
                                batch_size=64).astype(np.float32)
            enc = {s: vecs[i] for i, s in enumerate(need)}

            def cos(a, b):
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

            worst = 1.0
            bad_match = []
            for e in range(total_eps):
                task_str = idx2task[ep_task[e]]
                c = cos(first_emb[e], enc[task_str])
                worst = min(worst, c)
                if c <= 0.999:
                    bad_match.append((e, round(c, 4), task_str))
            rep.check(not bad_match,
                      "baked 임베딩 == task 문자열 재계산 (cos>0.999)",
                      detail_bad=f"불일치 {len(bad_match)}개 예시{bad_match[:3]}")
            print(f"  임베딩↔task 최저 cosine = {worst:.4f}")

            # F. 충돌 ep 의 task 문자열이 labels 후보 중 하나인지 + 마지막 라벨인지
            task_not_in_labels = [e for e in range(total_eps)
                                  if idx2task[ep_task[e]] not in by_ep_labels[e]]
            (rep.ok if not task_not_in_labels else rep.warn)(
                "에피소드 task 문자열이 labels 후보에 포함",
                "" if not task_not_in_labels else
                f"{len(task_not_in_labels)}개 ep task 가 labels 에 없음 {task_not_in_labels[:5]}")
            if conflicts:
                task_is_last = sum(1 for e in conflicts
                                   if by_ep_labels[e] and by_ep_labels[e][-1] == idx2task[ep_task[e]])
                print(f"  충돌 ep {len(conflicts)}개 중 'task==labels 마지막행' = {task_is_last}개")
                for e in sorted(conflicts)[:6]:
                    print(f"    ep{e:>4}: task='{idx2task[ep_task[e]]}' | labels={by_ep_labels[e]}")
        except Exception as ex:
            rep.warn("임베딩 재계산", f"건너뜀 — {type(ex).__name__}: {ex}")
    else:
        rep.warn("임베딩 재계산", "--no-embed 로 생략")

    # ── G. 부가 위생 ────────────────────────────────────────────────────
    vids = glob.glob(str(Path(root) / "videos" / CAMERA_KEY / "chunk-*" / "*.mp4"))
    print(f"  videos: {len(vids)}개 mp4")
    rep.check(len(vids) > 0, "비디오 파일 존재", detail_bad="없음")
    tmp_dirs = [p.name for p in Path(root).iterdir()
                if p.is_dir() and p.name.startswith("tmp")]
    (rep.ok if not tmp_dirs else rep.warn)(
        "임시(tmp*) 디렉터리 잔존 없음",
        "" if not tmp_dirs else f"잔존: {tmp_dirs} — 수동 정리 필요")

    n_fail, n_warn = rep.summarize()
    print()
    return n_fail


def main():
    p = argparse.ArgumentParser(description="신규 수집 real LeRobotDataset 검증")
    p.add_argument("--dataset-root", required=True, action="append",
                   help="데이터셋 루트 (반복 지정 가능)")
    p.add_argument("--domain", default=None, help="생략 시 labels.jsonl 에서 추론")
    p.add_argument("--label-vocab", default="deploy/config/label_vocab.yaml")
    p.add_argument("--embed-model",
                   default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--no-embed", action="store_true",
                   help="임베딩 모델 로드/재계산 생략")
    a = p.parse_args()

    total_fail = 0
    for root in a.dataset_root:
        total_fail += verify(root, a.domain, a.label_vocab, a.embed_model,
                             do_embed=not a.no_embed)
    if total_fail:
        print(f"❌ 전체 FAIL {total_fail}건 — 학습 전 조치 필요")
        raise SystemExit(1)
    print("✅ 모든 데이터셋 검증 통과")


if __name__ == "__main__":
    main()
