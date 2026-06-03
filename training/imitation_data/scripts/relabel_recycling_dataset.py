"""
relabel_recycling_dataset.py

이미 수집한 recycling Real LeRobotDataset(0.4.4)을 **현재 저장소 vocab**에 맞춰
재라벨한 *새 데이터셋*으로 재기록한다. ACT 를 현재 vocab 으로 재학습하기 위한
전처리이며, LoRA 는 건드리지 않는다 (LoRA 는 이미 현재 vocab 으로 학습됨).

수행 작업:
  1) labels.jsonl 의 에피소드별 (object, attribute, spatial) 을 읽는다.
  2) object remap 적용 (기본 bottle → plastic). attribute/spatial 은 유지.
  3) remap 후 (object, attribute, spatial) 이 **확정 label_vocab.yaml 로 전부
     표현 가능한지** 검증 → 불가하면 그 에피소드는 DROP.
       - 이 규칙으로 "next to the scissors"(scissors 가 현재 vocab 에 없음)
         에피소드가 자동 드롭된다. 나중에 red 등을 vocab 에서 가지치면
         red 라벨 에피소드도 같은 규칙으로 자동 드롭된다.
  4) 유지 에피소드의 새 canonical 을 vocab.template 으로 다시 만들고, 그
     임베딩(paraphrase-multilingual-MiniLM-L12-v2, D=384, L2 정규화)을 다시 계산.
  5) **새 LeRobotDataset** 에 유지 에피소드의 프레임을 그대로 재기록하되,
     observation.language_embedding 을 새 임베딩으로, task 를 새 canonical 로 교체.
     (state/이미지/action 은 원본 유지) → 에피소드 인덱스는 writer 가 0..K-1 로 재부여.
  6) 새 데이터셋 디렉터리에 갱신된 labels.jsonl 기록.

설계 원칙 (지금 작성 / vocab 확정 후 실행):
  - vocab 값은 코드에 박지 않고 **실행 시점에 --label-vocab 에서 읽는다**.
    따라서 vocab 이 확정된 뒤 그대로 실행하면 그 vocab 이 반영된다.
  - 원본 데이터셋은 절대 in-place 수정하지 않는다 (--dst-root 새 디렉터리로 출력).
  - --dry-run: 쓰기 없이 유지/드롭/old→new canonical/임베딩 dim 만 리포트.

주의:
  - 비디오를 디코드→재인코딩하므로 약간의 화질 손실이 있다 (1회성 재라벨이라 허용).
  - 임베딩 모델은 수집/tableware 재수집/배포 embedding_node 와 반드시 동일해야 한다.

CLI 예 (dry-run):
  python training/imitation_data/scripts/relabel_recycling_dataset.py \
    --src-root      training/imitation_data/datasets/recycling_real \
    --src-repo-id   myname/dataset_real \
    --dst-root      training/imitation_data/datasets/recycling_real_relabeled \
    --dst-repo-id   myname/dataset_recycling_relabeled \
    --label-vocab   deploy/config/label_vocab.yaml \
    --domain        recycling \
    --dry-run

  # 실제 실행은 --dry-run 제거 (vocab 확정 후).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


DEFAULT_EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EXPECTED_EMBED_DIM = 384
CAMERA_KEY = 'observation.images.top'
FEATURE_KEYS = [
    'observation.state',
    'observation.language_embedding',
    CAMERA_KEY,
    'action',
]
DEFAULT_OBJECT_REMAP = {'bottle': 'plastic'}


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@dataclass
class Args:
    src_root:     str
    src_repo_id:  str
    dst_root:     str
    dst_repo_id:  str
    labels_jsonl: str
    label_vocab:  str
    domain:       str
    embed_model:  str
    object_remap: Dict[str, str]
    dry_run:      bool
    limit:        int


def _parse_remap(spec: str) -> Dict[str, str]:
    """'bottle=plastic,can=can' → {'bottle':'plastic','can':'can'}."""
    out: Dict[str, str] = {}
    for pair in spec.split(','):
        pair = pair.strip()
        if not pair:
            continue
        if '=' not in pair:
            raise argparse.ArgumentTypeError(
                f'object-remap 항목은 src=dst 형식이어야 함: {pair!r}')
        src, dst = pair.split('=', 1)
        out[src.strip()] = dst.strip()
    return out


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description='recycling Real LeRobotDataset 재라벨 (현재 vocab 정렬)')
    p.add_argument('--src-root',     required=True,
                   help='원본 LeRobotDataset 디렉터리')
    p.add_argument('--src-repo-id',  default='',
                   help='원본 repo_id (미지정 시 src-root 폴더명에서 local/<name> 사용)')
    p.add_argument('--dst-root',     required=True,
                   help='새 데이터셋 디렉터리 (존재하면 에러 — 원본 보호)')
    p.add_argument('--dst-repo-id',  default='',
                   help='새 repo_id (미지정 시 dst-root 폴더명에서 local/<name> 사용)')
    p.add_argument('--labels-jsonl', default='',
                   help='라벨 JSONL (기본: <src-root>/labels.jsonl)')
    p.add_argument('--label-vocab',  default='deploy/config/label_vocab.yaml',
                   help='확정된 label_vocab.yaml (실행 시점에 읽음)')
    p.add_argument('--domain',       default='recycling')
    p.add_argument('--embed-model',  default=DEFAULT_EMBED_MODEL)
    p.add_argument('--object-remap', default='bottle=plastic', type=_parse_remap,
                   help="object 이름 remap (기본 'bottle=plastic')")
    p.add_argument('--dry-run',      action='store_true',
                   help='쓰기 없이 유지/드롭/변경 리포트만')
    p.add_argument('--limit',        type=int, default=0,
                   help='유지 에피소드 중 앞 N개만 처리 (0=전체, 실기 테스트용)')
    a = p.parse_args()

    src_root = os.path.expanduser(a.src_root)
    dst_root = os.path.expanduser(a.dst_root)
    return Args(
        src_root     = src_root,
        src_repo_id  = a.src_repo_id or f'local/{Path(src_root).name}',
        dst_root     = dst_root,
        dst_repo_id  = a.dst_repo_id or f'local/{Path(dst_root).name}',
        labels_jsonl = os.path.expanduser(a.labels_jsonl) if a.labels_jsonl
                       else str(Path(src_root) / 'labels.jsonl'),
        label_vocab  = os.path.expanduser(a.label_vocab),
        domain       = a.domain,
        embed_model  = a.embed_model,
        object_remap = a.object_remap if isinstance(a.object_remap, dict)
                       else DEFAULT_OBJECT_REMAP,
        dry_run      = a.dry_run,
        limit        = a.limit,
    )


# ──────────────────────────────────────────────
# vocab / labels 로드
# ──────────────────────────────────────────────

def load_vocab(path: str, domain: str) -> dict:
    """label_vocab.yaml 로드 + 도메인 추출 (record_real_teleop.load_vocab 규약 동일).

    spatials = 전역 spatial_qualifiers + 도메인별 spatial_qualifiers 병합.
    objects  = {object_name: [attributes...]}.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    template        = raw.get('template')
    global_spatials = raw.get('spatial_qualifiers') or []
    domains         = raw.get('domains') or {}
    if not template or '{attribute}' not in template \
       or '{object}' not in template or '{spatial}' not in template:
        raise ValueError(f'template 누락/형식 오류 ({path}).')
    if domain not in domains:
        raise ValueError(f"도메인 '{domain}' 이 vocab 에 없음. 가능: {list(domains)}")

    block           = domains[domain] or {}
    domain_spatials = block.get('spatial_qualifiers') or []
    obj_block       = block.get('objects') or {}
    objects: Dict[str, List[str]] = {}
    for name, defn in obj_block.items():
        attrs = (defn or {}).get('attributes') or []
        objects[name] = list(attrs)

    return {
        'template':  template,
        'spatials':  list(global_spatials) + list(domain_spatials),
        'objects':   objects,
    }


def load_labels_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ──────────────────────────────────────────────
# 재라벨 계획 (vocab-driven keep/drop)
# ──────────────────────────────────────────────

@dataclass
class EpisodePlan:
    episode:       int      # 원본 episode_index
    old_canonical: str
    new_object:    str
    attribute:     str
    spatial:       str
    new_canonical: str
    keep:          bool
    reason:        str      # 드롭 사유 ('' 이면 유지)


def build_plan(
    labels:       List[dict],
    vocab:        dict,
    object_remap: Dict[str, str],
) -> List[EpisodePlan]:
    """labels → 에피소드별 재라벨/유지 계획.

    keep 조건: object remap 후 (object, attribute, spatial) 이 vocab 으로 전부 표현 가능.
    """
    plans: List[EpisodePlan] = []
    for rec in labels:
        ep        = int(rec['episode'])
        old_obj   = rec.get('object', '')
        attr      = rec.get('attribute', '')
        spat      = rec.get('spatial', '')
        old_canon = rec.get('canonical', '')
        new_obj   = object_remap.get(old_obj, old_obj)

        reasons: List[str] = []
        if new_obj not in vocab['objects']:
            reasons.append(f"object={new_obj!r} 가 vocab 에 없음")
        elif attr not in vocab['objects'][new_obj]:
            reasons.append(f"attribute={attr!r} 가 {new_obj!r} 속성에 없음")
        if spat not in vocab['spatials']:
            reasons.append(f"spatial={spat!r} 가 vocab 에 없음")

        keep = not reasons
        new_canon = (
            vocab['template'].format(attribute=attr, object=new_obj, spatial=spat)
            if keep else ''
        )
        plans.append(EpisodePlan(
            episode=ep, old_canonical=old_canon, new_object=new_obj,
            attribute=attr, spatial=spat, new_canonical=new_canon,
            keep=keep, reason='; '.join(reasons),
        ))
    return plans


def print_report(plans: List[EpisodePlan], object_remap: Dict[str, str]) -> None:
    kept = [p for p in plans if p.keep]
    dropped = [p for p in plans if not p.keep]
    print('=' * 64)
    print(f'총 {len(plans)} 에피소드 | 유지 {len(kept)} | 드롭 {len(dropped)}')
    print(f'object remap: {object_remap}')
    print('-' * 64)
    print('[유지 — old canonical → new canonical] (앞 12개)')
    for p in kept[:12]:
        print(f'  ep{p.episode:>4}  {p.old_canonical!r}\n          → {p.new_canonical!r}')
    if dropped:
        print('-' * 64)
        print('[드롭 사유 집계]')
        for reason, n in Counter(p.reason for p in dropped).most_common():
            print(f'  {n:>4}  {reason}')
    print('-' * 64)
    print('[유지 에피소드 spatial 분포]')
    for s, n in Counter(p.spatial for p in kept).most_common():
        print(f'  {n:>4}  {s!r}')
    print('[유지 에피소드 object 분포]')
    for o, n in Counter(p.new_object for p in kept).most_common():
        print(f'  {n:>4}  {o!r}')
    print('=' * 64)


# ──────────────────────────────────────────────
# 임베딩
# ──────────────────────────────────────────────

def load_embed_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    # sentence-transformers 신버전은 get_embedding_dimension, 구버전은
    # get_sentence_embedding_dimension. 둘 다 지원 (deprecation warning 회피).
    _get_dim = getattr(model, 'get_embedding_dimension', None) \
        or model.get_sentence_embedding_dimension
    dim = int(_get_dim())
    if dim != EXPECTED_EMBED_DIM:
        raise RuntimeError(
            f'임베딩 dim={dim} != {EXPECTED_EMBED_DIM}. 모델이 '
            f'{DEFAULT_EMBED_MODEL} 인지 확인.')
    return model


def embed_canonicals(model, canonicals: List[str]) -> Dict[str, np.ndarray]:
    """unique canonical → L2 정규화 임베딩 (float32 (384,))."""
    uniq = sorted(set(canonicals))
    arr = model.encode(uniq, normalize_embeddings=True, convert_to_numpy=True)
    arr = arr.astype(np.float32)
    return {c: arr[i] for i, c in enumerate(uniq)}


# ──────────────────────────────────────────────
# 텐서/이미지 변환 (LeRobotDataset __getitem__ → add_frame 입력)
# ──────────────────────────────────────────────

def _to_np_f32(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _to_hwc_uint8(img) -> np.ndarray:
    """LeRobot 프레임 이미지 → HWC uint8 RGB (add_frame 입력 규약).

    __getitem__ 은 보통 CHW float[0,1] 텐서를 주므로 HWC uint8 로 환원한다.
    버전에 따라 HWC/uint8 로 올 수도 있어 방어적으로 처리.
    """
    try:
        import torch
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
    except ImportError:
        pass
    img = np.asarray(img)
    # CHW → HWC (채널이 앞에 있고 뒤축이 채널 크기가 아닐 때)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        mx = float(img.max()) if img.size else 1.0
        if mx <= 1.0 + 1e-6:           # [0,1] 정규화 이미지
            img = img * 255.0
        img = np.clip(img, 0, 255).round().astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(img)


# ──────────────────────────────────────────────
# 데이터셋 열기 / 에피소드별 프레임 인덱스
# ──────────────────────────────────────────────

def open_source(src_repo_id: str, src_root: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id=src_repo_id, root=src_root)


def episode_frame_indices(src) -> Dict[int, np.ndarray]:
    """episode_index → 해당 에피소드의 전역 프레임 인덱스 배열 (정렬됨).

    hf_dataset 의 episode_index 컬럼만 읽어 그룹핑 (비디오 디코드 없음).
    """
    if not hasattr(src, 'hf_dataset'):
        raise RuntimeError(
            'LeRobotDataset 에 hf_dataset 속성이 없음 — LeRobot 버전 확인 필요.')
    ep_col = np.asarray(src.hf_dataset['episode_index'])
    out: Dict[int, np.ndarray] = {}
    for ep in np.unique(ep_col):
        out[int(ep)] = np.where(ep_col == ep)[0]
    return out


def build_target_features(src) -> dict:
    """원본 features 에서 FEATURE_KEYS 4개만 추출 (create 입력용)."""
    feats = {}
    for k in FEATURE_KEYS:
        if k not in src.features:
            raise RuntimeError(f'원본에 feature {k!r} 없음 — 호환되지 않는 데이터셋.')
        f = src.features[k]
        feats[k] = {
            'dtype': f['dtype'],
            'shape': tuple(f['shape']),
            'names': f.get('names'),
        }
    return feats


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1) vocab + labels + 계획
    vocab  = load_vocab(args.label_vocab, args.domain)
    labels = load_labels_jsonl(args.labels_jsonl)
    if not labels:
        print(f'[relabel] 라벨 레코드 없음: {args.labels_jsonl}'); return
    plans = build_plan(labels, vocab, args.object_remap)
    print_report(plans, args.object_remap)

    kept = [p for p in plans if p.keep]
    if args.limit > 0:
        kept = kept[:args.limit]
        print(f'[relabel] --limit {args.limit} → 유지 {len(kept)}개만 처리')
    if not kept:
        print('[relabel] 유지할 에피소드가 없음 — 종료.'); return

    # 2) 임베딩 모델 (dim 검증). dry-run 도 한 번 검증.
    print(f'[relabel] 임베딩 모델 로드: {args.embed_model}')
    model = load_embed_model(args.embed_model)
    embed_map = embed_canonicals(model, [p.new_canonical for p in kept])
    print(f'[relabel] 임베딩 계산 완료 — unique canonical {len(embed_map)}개 (dim={EXPECTED_EMBED_DIM})')

    # 3) 원본 열기 + 프레임 인덱스 (dry-run 도 read 경로 검증)
    print(f'[relabel] 원본 열기: repo_id={args.src_repo_id!r} root={args.src_root}')
    src = open_source(args.src_repo_id, args.src_root)
    ep_frames = episode_frame_indices(src)
    n_total_frames = sum(len(ep_frames.get(p.episode, [])) for p in kept)
    missing = [p.episode for p in kept if p.episode not in ep_frames]
    if missing:
        raise RuntimeError(
            f'labels 에는 있으나 데이터셋에 없는 episode_index: {missing[:10]} ...')
    print(f'[relabel] 데이터셋 에피소드 {len(ep_frames)}개 | '
          f'유지 에피소드 프레임 합계 {n_total_frames}')

    if args.dry_run:
        print('[relabel] --dry-run: 쓰기 없이 종료. 위 리포트로 vocab/유지·드롭 확인.')
        return

    # 4) 원본 보호 — dst 가 이미 있으면 중단
    if Path(args.dst_root).exists():
        raise SystemExit(
            f'[relabel] 출력 경로가 이미 존재: {args.dst_root}\n'
            '           원본/기존 산출물 보호를 위해 새 경로를 지정하거나 기존 경로를 삭제하세요.')

    # 5) 새 데이터셋 생성 + 유지 에피소드 재기록
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features    = build_target_features(src)
    fps         = int(getattr(src, 'fps', None) or src.meta.fps)
    robot_type  = getattr(src.meta, 'robot_type', None) or 'so101_follower'
    print(f'[relabel] 새 데이터셋 생성: repo_id={args.dst_repo_id!r} root={args.dst_root} '
          f'fps={fps} robot_type={robot_type}')
    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=fps,
        features=features,
        root=args.dst_root,
        robot_type=robot_type,
        use_videos=True,
    )

    out_labels: List[dict] = []
    for new_ep, p in enumerate(kept):
        embed = embed_map[p.new_canonical]
        idxs  = ep_frames[p.episode]
        for fi in idxs:
            f = src[int(fi)]
            dst.add_frame({
                'observation.state':              _to_np_f32(f['observation.state']),
                'observation.language_embedding': np.asarray(embed, dtype=np.float32),
                CAMERA_KEY:                       _to_hwc_uint8(f[CAMERA_KEY]),
                'action':                         _to_np_f32(f['action']),
                'task':                           p.new_canonical,
            })
        dst.save_episode()
        out_labels.append({
            'episode':   new_ep,
            'object':    p.new_object,
            'attribute': p.attribute,
            'spatial':   p.spatial,
            'canonical': p.new_canonical,
            'domain':    args.domain,
            'src_episode': p.episode,
        })
        print(f'  [{new_ep+1}/{len(kept)}] src ep{p.episode} '
              f'({len(idxs)} frames) → {p.new_canonical!r}')

    # 6) 갱신된 labels.jsonl 기록
    labels_out = Path(args.dst_root) / 'labels.jsonl'
    with open(labels_out, 'w', encoding='utf-8') as fout:
        for rec in out_labels:
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'[relabel] 완료 — 유지 {len(out_labels)} 에피소드 → {args.dst_root}')
    print(f'[relabel] labels.jsonl 기록: {labels_out}')


if __name__ == '__main__':
    main()
