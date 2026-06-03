#!/usr/bin/env python3
"""
regenerate_labels.py

LeRobotDataset v3.0 의 labels.jsonl 을 데이터셋 **자체의 정답 라벨**
(meta/tasks.parquet 의 task 문자열 + 프레임의 task_index)에서 다시 생성한다.

왜 필요한가:
  record_real_teleop.py 는 수집 중 labels.jsonl 을 append 로 기록한다. 같은
  에피소드를 재촬영하면 LeRobot 데이터셋의 parquet/task 는 **최종 테이크로
  덮어쓰여** 정답(SSoT)이 되지만, labels.jsonl 에는 이전 테이크 라벨이 남아
  중복·충돌·desync(에피소드↔라벨 밀림)가 생긴다. (verify_real_dataset.py 의
  WARN 으로 검출됨)

  merge_lerobot_datasets.py 는 task 문자열을 labels.jsonl 에서 가져오므로,
  labels.jsonl 이 어긋나면 병합본 task 메타데이터가 baked 임베딩과 불일치한다.
  (ACT 는 baked 임베딩을 학습하므로 학습 자체엔 영향 없지만, 메타데이터가
  더러워지고 병합본 검증에서 FAIL 로 뜬다.)

  본 스크립트는 데이터셋 안의 task_index→tasks.parquet 문자열(= baked 임베딩과
  cos≈1 로 일치함이 verify_real_dataset.py 로 확인된 정답)을 그대로 사용해
  labels.jsonl 을 **에피소드당 정확히 1줄**로 다시 쓴다. 중복·충돌·desync 가
  한 번에 사라진다. 프레임/임베딩/비디오는 건드리지 않는다.

실행 환경 (중요):
  lerobot 불필요 — numpy / pyarrow / PyYAML 만 있으면 된다. 따라서 데이터셋이
  있는 어느 머신에서나 실행 가능하다 (Windows 개발 PC / Ubuntu 학습 PC 모두).
  **병합(merge_lerobot_datasets.py) 전에**, 병합에 투입할 바로 그 데이터셋
  사본에 대해 실행할 것. (merge/train 은 Ubuntu 학습 PC 에서 도므로, 그 PC 로
  옮긴 뒤 그곳에서 돌리는 것이 가장 깔끔하다.)

안전장치:
  - 덮어쓰기 전 기존 labels.jsonl 을 labels.jsonl.bak 으로 백업 (--no-backup 로 끔)
  - --dry-run: 쓰기 없이 변경 요약/분포만 출력
  - 에피소드별 task_index 가 단일하지 않거나 task 문자열이 vocab 으로
    파싱되지 않으면 중단 (조용한 손상 방지)

CLI:
  # 미리보기 (권장: 먼저 dry-run)
  python training/imitation_data/scripts/regenerate_labels.py \
    --dataset-root training/imitation_data/datasets/recycling_real --dry-run

  # 실제 재생성 (recycling_real 신규본 — desync 있던 대상)
  python training/imitation_data/scripts/regenerate_labels.py \
    --dataset-root training/imitation_data/datasets/recycling_real

  # 여러 개 한 번에 (선택: tableware 도 중복/충돌 정리)
  python training/imitation_data/scripts/regenerate_labels.py \
    --dataset-root training/imitation_data/datasets/recycling_real \
    --dataset-root training/imitation_data/datasets/tableware_real
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_vocab(path: str, domain: str) -> dict:
    raw = yaml.safe_load(open(path, encoding="utf-8"))
    dom = raw["domains"][domain]
    spatials = list(raw.get("spatial_qualifiers", [])) + list(
        dom.get("spatial_qualifiers", [])
    )
    return {
        "template": raw["template"],
        "objects": {o: list(v["attributes"]) for o, v in dom["objects"].items()},
        "spatials": spatials,
    }


def parse_canonical(canon: str, vocab: dict):
    """'the {attribute} {object} {spatial}' → (object, attribute, spatial) | None."""
    tmpl = vocab["template"]
    for o, attrs in vocab["objects"].items():
        for a in attrs:
            for s in vocab["spatials"]:
                if canon == tmpl.format(attribute=a, object=o, spatial=s):
                    return o, a, s
    return None


def infer_domain(root: Path, cli_domain: str | None) -> str:
    if cli_domain:
        return cli_domain
    p = root / "labels.jsonl"
    if p.exists():
        doms = Counter()
        for line in open(p, encoding="utf-8"):
            try:
                doms[json.loads(line).get("domain")] += 1
            except Exception:
                pass
        doms.pop(None, None)
        if doms:
            return doms.most_common(1)[0][0]
    raise SystemExit(
        f"[regen] {root}: domain 을 추론할 수 없음 — --domain 으로 명시하세요.")


def regen_one(root: Path, vocab_path: str, domain: str | None,
              dry_run: bool, backup: bool) -> int:
    print("#" * 70)
    print(f"# labels 재생성: {root}")
    domain = infer_domain(root, domain)
    vocab = load_vocab(vocab_path, domain)
    print(f"#   domain = {domain}")
    print("#" * 70)

    # task_index → task 문자열
    tdf = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
    idx2task = {int(v): str(k) for k, v in tdf["task_index"].items()}

    # 프레임에서 에피소드별 task_index 수집
    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise SystemExit(f"[regen] {root}: data parquet 없음")
    tbl = pa.concat_tables([pq.read_table(f, columns=["episode_index", "task_index"])
                            for f in files])
    ep = tbl["episode_index"].to_numpy()
    ti = tbl["task_index"].to_numpy()

    ep_tasks = defaultdict(set)
    for e, t in zip(ep.tolist(), ti.tolist()):
        ep_tasks[int(e)].add(int(t))

    # 단일성 검증
    multi = {e: sorted(ts) for e, ts in ep_tasks.items() if len(ts) != 1}
    if multi:
        raise SystemExit(
            f"[regen] {root}: 에피소드별 task_index 가 단일하지 않음 — "
            f"{dict(list(multi.items())[:5])} ... 수동 확인 필요.")

    n_eps = len(ep_tasks)
    # 파싱 + 레코드 구성 (에피소드 0..N-1 순서)
    records = []
    bad_parse = []
    obj_c, attr_c, spat_c = Counter(), Counter(), Counter()
    for e in range(n_eps):
        if e not in ep_tasks:
            raise SystemExit(f"[regen] {root}: 에피소드 인덱스 누락 (ep {e})")
        canon = idx2task[next(iter(ep_tasks[e]))]
        parsed = parse_canonical(canon, vocab)
        if parsed is None:
            bad_parse.append((e, canon))
            continue
        o, a, s = parsed
        records.append({
            "object": o, "attribute": a, "spatial": s,
            "canonical": canon, "domain": domain, "episode": e,
        })
        obj_c[o] += 1
        attr_c[(o, a)] += 1
        spat_c[s] += 1
    if bad_parse:
        raise SystemExit(
            f"[regen] {root}: vocab 으로 파싱 불가한 task 문자열 — "
            f"{bad_parse[:5]} ... label_vocab/domain 확인 필요.")

    # 기존 labels.jsonl 과 비교 (중복/충돌/desync 요약)
    labels_p = root / "labels.jsonl"
    old_lines = []
    if labels_p.exists():
        old_lines = [json.loads(l) for l in open(labels_p, encoding="utf-8")]
    old_by_ep = defaultdict(list)
    for r in old_lines:
        old_by_ep[int(r["episode"])].append(r.get("canonical", ""))
    dups = sum(1 for v in old_by_ep.values() if len(v) > 1)
    conflicts = sum(1 for v in old_by_ep.values() if len(set(v)) > 1)
    changed = sum(1 for r in records
                  if not old_by_ep.get(r["episode"])
                  or old_by_ep[r["episode"]][-1] != r["canonical"])

    print(f"  에피소드 수            : {n_eps}")
    print(f"  기존 labels.jsonl 행수 : {len(old_lines)}  (중복 ep {dups} / 충돌 ep {conflicts})")
    print(f"  재생성 후 행수         : {len(records)}  (에피소드당 1줄)")
    print(f"  canonical 이 바뀌는 ep : {changed}개  "
          f"(기존 마지막행과 다른 = desync 교정 대상)")
    print(f"  object 분포 : {dict(obj_c)}")
    print(f"  spatial 분포: {dict(spat_c)}")
    print("  attribute(객체별):")
    for o in vocab["objects"]:
        print(f"    {o:8s} " + str({a: attr_c.get((o, a), 0) for a in vocab["objects"][o]}))

    if dry_run:
        print("  [dry-run] 쓰기 없이 종료.\n")
        return 0

    if backup and labels_p.exists():
        bak = labels_p.with_suffix(".jsonl.bak")
        shutil.copy2(labels_p, bak)
        print(f"  백업: {bak}")
    with open(labels_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✅ 재생성 완료: {labels_p}\n")
    return 0


def main():
    p = argparse.ArgumentParser(
        description="labels.jsonl 을 데이터셋 자체 task(SSoT)에서 재생성")
    p.add_argument("--dataset-root", required=True, action="append")
    p.add_argument("--label-vocab", default="deploy/config/label_vocab.yaml")
    p.add_argument("--domain", default=None, help="생략 시 기존 labels.jsonl 에서 추론")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-backup", action="store_true",
                   help="기존 labels.jsonl 백업(.bak) 생략")
    a = p.parse_args()
    for root in a.dataset_root:
        regen_one(Path(root).expanduser(), a.label_vocab, a.domain,
                  a.dry_run, backup=not a.no_backup)
    if a.dry_run:
        print("dry-run 종료 — 실제 적용하려면 --dry-run 없이 다시 실행하세요.")
    else:
        print("완료. 권장: verify_real_dataset.py 로 재검증 후 병합/학습 진행.")


if __name__ == "__main__":
    main()
