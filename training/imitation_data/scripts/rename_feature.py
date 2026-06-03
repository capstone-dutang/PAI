#!/usr/bin/env python3
"""
rename_feature.py — LeRobotDataset v3.0 의 observation feature 키를 rename.

용도(기본값): observation.language_embedding → observation.environment_state.

왜 필요한가:
  lerobot ACT 는 입력 feature 를 type 으로 구분해 배선한다
  (dataset_to_policy_features):
    - key == "observation.environment_state"  → FeatureType.ENV
    - key.startswith("observation.")          → FeatureType.STATE
  이 때문에 "observation.language_embedding" 은 STATE 로 분류돼 observation.state(6)
  와 중복됐고, ACT 는 robot_state_feature 하나(observation.state)만 쓰고 **언어
  임베딩을 무시**했다(env_state_feature 가 None → 384 입력 투영 자체가 없음).
  키를 정확히 "observation.environment_state" 로 바꾸면 ENV 로 분류돼 ACT 가
  encoder_env_state_input_proj(384→dim_model) 토큰으로 **언어를 실제로 사용**한다.

rename 대상(전부 처리):
  - meta/info.json                          : features[old] → features[new]
  - meta/stats.json                         : 최상위 key old → new
  - data/chunk-*/file-*.parquet             : 컬럼 old → new
  - meta/episodes/chunk-*/file-*.parquet    : "stats/{old}/*" 컬럼 → "stats/{new}/*"

안전:
  - info.json / stats.json 은 *.bak 백업
  - parquet 은 제자리 재기록(컬럼명만 변경, 행 데이터 동일). 되돌리려면
    --old/--new 를 바꿔 다시 실행.
  - --dry-run 으로 영향만 출력.

CLI:
  python training/imitation_data/scripts/rename_feature.py \
    --dataset-root training/imitation_data/datasets/recycling_real_merge --dry-run
  python training/imitation_data/scripts/rename_feature.py \
    --dataset-root training/imitation_data/datasets/recycling_real_merge
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def remap_columns(cols: list[str], old: str, new: str) -> list[str]:
    """컬럼명에 old 가 들어가면 new 로 치환. 'observation.language_embedding' 과
    'stats/observation.language_embedding/min' 둘 다 처리."""
    return [c.replace(old, new) if old in c else c for c in cols]


def process_parquets(pattern: str, old: str, new: str, dry: bool) -> tuple[int, int]:
    n_files = n_cols = 0
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f)
        newcols = remap_columns(t.column_names, old, new)
        changed = [(a, b) for a, b in zip(t.column_names, newcols) if a != b]
        if not changed:
            continue
        n_files += 1
        n_cols += len(changed)
        print(f"    {Path(f).name}: {len(changed)}개 컬럼 변경"
              + (f"  예) {changed[0][0]} → {changed[0][1]}" if changed else ""))
        if not dry:
            pq.write_table(t.rename_columns(newcols), f)
    return n_files, n_cols


def process_json(path: Path, old: str, new: str, dry: bool) -> int:
    """info.json(features 하위) / stats.json(최상위) 의 key old→new."""
    if not path.exists():
        return 0
    data = json.load(open(path, encoding="utf-8"))
    changed = 0

    def rename_key(d):
        nonlocal changed
        if isinstance(d, dict) and old in d:
            d[new] = d.pop(old)
            changed += 1

    rename_key(data)                                   # stats.json 최상위
    if isinstance(data.get("features"), dict):
        rename_key(data["features"])                   # info.json features
    if changed and not dry:
        shutil.copy2(path, str(path) + ".bak")
        json.dump(data, open(path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    if changed:
        print(f"    {path.name}: key {old} → {new} ({'dry' if dry else 'rewrote, .bak 백업'})")
    return changed


def main():
    ap = argparse.ArgumentParser(description="LeRobotDataset feature 키 rename")
    ap.add_argument("--dataset-root", required=True, action="append")
    ap.add_argument("--old", default="observation.language_embedding")
    ap.add_argument("--new", default="observation.environment_state")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for root_s in a.dataset_root:
        root = Path(root_s).expanduser()
        print("#" * 70)
        print(f"# rename: {a.old} → {a.new}")
        print(f"# dataset: {root}  {'[DRY-RUN]' if a.dry_run else ''}")
        print("#" * 70)
        c_info = process_json(root / "meta" / "info.json", a.old, a.new, a.dry_run)
        c_stats = process_json(root / "meta" / "stats.json", a.old, a.new, a.dry_run)
        print("  data parquet:")
        df, dc = process_parquets(str(root / "data" / "chunk-*" / "file-*.parquet"),
                                  a.old, a.new, a.dry_run)
        print("  episodes parquet:")
        ef, ec = process_parquets(str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet"),
                                  a.old, a.new, a.dry_run)
        print(f"  요약: info={c_info} stats={c_stats} | "
              f"data {df}파일/{dc}컬럼 | episodes {ef}파일/{ec}컬럼")
        if c_info == 0 and dc == 0:
            print("  ⚠️ old 키를 찾지 못함 — 이미 rename됐거나 --old 확인 필요.")
        print()
    if a.dry_run:
        print("dry-run 종료 — 실제 적용하려면 --dry-run 없이 다시 실행.")
    else:
        print("완료. ds 재로드 시 environment_state(ENV)로 분류되는지 확인 후 재학습.")


if __name__ == "__main__":
    main()
