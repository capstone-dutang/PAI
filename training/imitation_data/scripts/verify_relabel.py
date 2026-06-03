"""
verify_relabel.py

relabel_recycling_dataset.py 산출물(재라벨된 LeRobotDataset)이 올바른지 검증한다.

검사 항목:
  A. labels.jsonl 정합
     - 레코드 수 / object·spatial 분포
     - 모든 (object, attribute, spatial) 이 확정 vocab 으로 표현 가능
     - canonical == template.format(...) 일치
     - 'bottle' / 'scissors' 등 제거 대상이 남아 있지 않음
  B. 데이터셋 구조
     - num_episodes == labels 수
     - feature 스펙(state(6)/language_embedding(384)/images.top(H,W,3)/action(6))
     - fps / 총 프레임
  C. 임베딩 정확성 (가장 중요 — 재계산 검증)
     - 표본 에피소드의 저장된 frame 임베딩이 **새 canonical(plastic)** 의
       재계산 임베딩과 cosine ≈ 1.0 인지 (= 재계산이 실제로 반영됐는지)
     - **옛 canonical(bottle)** 임베딩과는 더 낮은 cosine 인지 (= 옛 값이 아님)
     - 에피소드 내 첫/끝 프레임 임베딩 동일(broadcast) + L2 정규화 확인

CLI:
  python training/imitation_data/scripts/verify_relabel.py \
    --dataset-root training/imitation_data/datasets/recycling_real_relabeled \
    --dataset-repo-id myname/dataset_recycling_relabeled \
    --label-vocab deploy/config/label_vocab.yaml \
    --domain recycling \
    --object-remap bottle=plastic \
    --sample 8 \
    --expect-episodes 270
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np

from relabel_recycling_dataset import (
    EXPECTED_EMBED_DIM,
    CAMERA_KEY,
    FEATURE_KEYS,
    _parse_remap,
    _to_np_f32,
    load_vocab,
    load_labels_jsonl,
    load_embed_model,
    open_source,
    episode_frame_indices,
)

# 제거 대상(현재 vocab 에 없어야 하는) 토큰 — 잔존 검사용
FORBIDDEN_TOKENS = ['bottle', 'scissors']


def parse_args():
    p = argparse.ArgumentParser(description='재라벨 데이터셋 검증')
    p.add_argument('--dataset-root',    required=True)
    p.add_argument('--dataset-repo-id', default='')
    p.add_argument('--label-vocab',     default='deploy/config/label_vocab.yaml')
    p.add_argument('--domain',          default='recycling')
    p.add_argument('--embed-model',
                   default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    p.add_argument('--object-remap',    default='bottle=plastic', type=_parse_remap)
    p.add_argument('--sample',          type=int, default=8,
                   help='임베딩 deep-check 표본 에피소드 수')
    p.add_argument('--expect-episodes', type=int, default=0,
                   help='기대 에피소드 수 (0=검사 안 함)')
    a = p.parse_args()
    a.dataset_root = os.path.expanduser(a.dataset_root)
    a.label_vocab  = os.path.expanduser(a.label_vocab)
    if not a.dataset_repo_id:
        a.dataset_repo_id = f'local/{Path(a.dataset_root).name}'
    return a


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def main() -> None:
    args = parse_args()
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = '') -> None:
        results.append((name, ok, detail))

    # ── A. labels.jsonl ──────────────────────────────────────────
    vocab    = load_vocab(args.label_vocab, args.domain)
    labels_p = Path(args.dataset_root) / 'labels.jsonl'
    labels   = load_labels_jsonl(str(labels_p))
    by_ep    = {int(r['episode']): r for r in labels}

    check('labels.jsonl 존재', labels_p.exists(), str(labels_p))
    if args.expect_episodes:
        check(f'labels 수 == {args.expect_episodes}',
              len(labels) == args.expect_episodes, f'실제 {len(labels)}')

    bad_vocab, bad_canon, bad_forbidden = [], [], []
    for r in labels:
        obj, attr, spat = r.get('object'), r.get('attribute'), r.get('spatial')
        canon = r.get('canonical', '')
        if obj not in vocab['objects'] or attr not in vocab['objects'].get(obj, []) \
           or spat not in vocab['spatials']:
            bad_vocab.append(r['episode'])
        if canon != vocab['template'].format(attribute=attr, object=obj, spatial=spat):
            bad_canon.append(r['episode'])
        blob = f'{obj} {attr} {spat} {canon}'
        if any(tok in blob for tok in FORBIDDEN_TOKENS):
            bad_forbidden.append(r['episode'])

    check('모든 라벨이 vocab 으로 표현 가능', not bad_vocab,
          f'위반 ep {bad_vocab[:5]}' if bad_vocab else '')
    check('canonical == template 일치', not bad_canon,
          f'불일치 ep {bad_canon[:5]}' if bad_canon else '')
    check(f'제거 토큰{FORBIDDEN_TOKENS} 잔존 없음', not bad_forbidden,
          f'잔존 ep {bad_forbidden[:5]}' if bad_forbidden else '')

    print('── labels 분포 ──')
    print('  object :', dict(Counter(r.get('object') for r in labels)))
    print('  spatial:', dict(Counter(r.get('spatial') for r in labels)))

    # ── B. 데이터셋 구조 ────────────────────────────────────────
    print(f'── 데이터셋 열기: {args.dataset_root} ──')
    ds        = open_source(args.dataset_repo_id, args.dataset_root)
    ep_frames = episode_frame_indices(ds)
    n_eps     = len(ep_frames)
    n_frames  = len(ds)

    check('num_episodes == labels 수', n_eps == len(labels),
          f'데이터셋 {n_eps} vs labels {len(labels)}')

    for k, want_dim in [('observation.state', 6),
                        ('observation.language_embedding', EXPECTED_EMBED_DIM),
                        ('action', 6)]:
        shape = tuple(ds.features[k]['shape']) if k in ds.features else None
        check(f'feature {k} shape=({want_dim},)', shape == (want_dim,), f'실제 {shape}')
    img_shape = tuple(ds.features[CAMERA_KEY]['shape']) if CAMERA_KEY in ds.features else None
    check('feature images.top 3채널', bool(img_shape) and img_shape[-1] == 3, f'{img_shape}')

    fps = int(getattr(ds, 'fps', None) or ds.meta.fps)
    check('fps == 30', fps == 30, f'실제 {fps}')
    print(f'  num_episodes={n_eps} | total_frames={n_frames} | fps={fps} | img={img_shape}')

    # ── C. 임베딩 정확성 (재계산 반영 검증) ─────────────────────
    print('── 임베딩 deep-check (재계산 반영) ──')
    model     = load_embed_model(args.embed_model)
    inv_remap = {v: k for k, v in args.object_remap.items()}

    ep_sorted = sorted(ep_frames)
    if args.sample > 0 and len(ep_sorted) > args.sample:
        pick = [ep_sorted[i] for i in
                np.linspace(0, len(ep_sorted) - 1, args.sample).astype(int)]
    else:
        pick = ep_sorted

    emb_ok, broadcast_ok, norm_ok = True, True, True
    for ep in pick:
        rec   = by_ep[ep]
        canon = rec['canonical']
        idxs  = ep_frames[ep]
        stored_first = _to_np_f32(ds[int(idxs[0])]['observation.language_embedding'])
        stored_last  = _to_np_f32(ds[int(idxs[-1])]['observation.language_embedding'])

        new_e = model.encode([canon], normalize_embeddings=True,
                             convert_to_numpy=True)[0].astype(np.float32)
        cos_new = _cos(stored_first, new_e)

        # 옛 canonical(remap 역방향: plastic→bottle)과 대비 — 객체가 remap된 경우만 의미
        old_obj = inv_remap.get(rec.get('object'), rec.get('object'))
        contrast = ''
        if old_obj != rec.get('object'):
            old_canon = vocab['template'].format(
                attribute=rec.get('attribute'), object=old_obj, spatial=rec.get('spatial'))
            old_e = model.encode([old_canon], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)
            cos_old = _cos(stored_first, old_e)
            contrast = f' | vs old({old_obj}) cos={cos_old:.4f}'
            if not (cos_new > cos_old):
                emb_ok = False

        cos_fl = _cos(stored_first, stored_last)
        norm   = float(np.linalg.norm(stored_first))
        if cos_new <= 0.999:
            emb_ok = False
        if cos_fl <= 0.999:
            broadcast_ok = False
        if abs(norm - 1.0) > 1e-2:
            norm_ok = False
        print(f'  ep{ep:>4} cos_new={cos_new:.4f}{contrast} | '
              f'first~last={cos_fl:.4f} | |emb|={norm:.4f} | {canon!r}')

    check('저장 임베딩 ≈ 새 canonical 재계산 (cos>0.999)', emb_ok)
    check('새 canonical 이 옛(bottle)보다 가까움', emb_ok)
    check('에피소드 내 임베딩 broadcast 동일', broadcast_ok)
    check('임베딩 L2 정규화(|emb|≈1)', norm_ok)

    # ── 요약 ────────────────────────────────────────────────────
    print('=' * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}{("  — " + detail) if detail and not ok else ""}')
    print('=' * 60)
    print(f'결과: {n_pass}/{len(results)} PASS')
    if n_pass != len(results):
        raise SystemExit(1)
    print('✅ 재라벨 검증 통과')


if __name__ == '__main__':
    main()
