"""
merge_lerobot_datasets.py

2개 이상의 LeRobotDataset(0.4.4)을 **디스크 병합**하여 하나의 새 데이터셋으로
재기록한다. recycling 도메인에서 재라벨된 데이터셋(relabel_recycling_dataset.py
산출 ~270) + 신규 수집 데이터셋을 단일 데이터셋으로 합쳐 ACT 학습
(train_act.sh, lerobot-train 단일 --dataset.*)에 투입하기 위함이다.

재라벨 스크립트와의 차이:
  - 재라벨: object remap + canonical/임베딩 **재계산**.
  - 병합:   각 원본 프레임을 **그대로 보존**한다 (state / images / action /
            observation.language_embedding / task 모두 원본 값 유지). 재계산 없음.
            에피소드 인덱스만 writer 가 0..(total-1) 로 재부여.

전제:
  - 모든 원본의 feature 스펙(observation.state / language_embedding /
    images.top / action)과 fps 가 **정확히 일치**해야 한다. 불일치 시 중단.
  - 각 원본 디렉터리에 labels.jsonl 이 있어야 한다 (에피소드별 canonical = task).
    record_real_teleop.py / relabel_recycling_dataset.py 둘 다 이를 생성한다.

주의:
  - 비디오를 디코드→재인코딩한다. 재라벨 산출물은 이미 1회 재인코딩됐으므로
    병합까지 거치면 그 부분은 총 2회 재인코딩된다 (1회성, 화질 손실 경미).
  - **인코딩 코덱**: 기본 `--video-codec h264`(CPU x264). 원본이 av1 이어도 av1
    재인코딩은 CPU 로 매우 느려(특히 vCPU 적은 클라우드), 빠른 h264 로 다시 굽는다.
    NVIDIA GPU 가 있으면 `--video-codec h264_nvenc` 로 **GPU NVENC 인코딩**(가장
    빠름, ffmpeg 에 nvenc 빌드 필요). 학습은 코덱과 무관(lerobot 이 디코드만 하면
    됨). av1 유지하려면 `--video-codec av1`. 본 실행 전 `--limit 5` 로 속도 확인 권장.
    (유효 코덱명은 lerobot 의 VALID_VIDEO_CODECS — libx264 가 아니라 h264 임)
  - 임베딩/라벨은 원본 보존이므로, 병합 전 각 원본이 **같은 vocab·같은 임베딩
    모델(MiniLM D=384)** 로 만들어졌는지 반드시 확인할 것 (정합은 호출자 책임).

CLI 예 (dry-run):
  python training/imitation_data/scripts/merge_lerobot_datasets.py \
    --src training/imitation_data/datasets/recycling_real_relabeled \
    --src training/imitation_data/datasets/recycling_real_new \
    --dst-root    training/imitation_data/datasets/recycling_real_merged \
    --dst-repo-id myname/dataset_recycling_merged \
    --dry-run

  # 원본의 repo_id 를 명시해야 하면 ROOT::REPO_ID 형식:
  #   --src training/imitation_data/datasets/recycling_real_relabeled::myname/dataset_recycling_relabeled
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 공용 헬퍼 재사용 (같은 디렉터리). 스크립트 실행 시 sys.path[0] 에 포함된다.
from relabel_recycling_dataset import (
    CAMERA_KEY,
    FEATURE_KEYS,
    _to_np_f32,
    _to_hwc_uint8,
    episode_frame_indices,
    build_target_features,
    load_labels_jsonl,
)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@dataclass
class SrcSpec:
    root:    str
    repo_id: str


@dataclass
class Args:
    sources:      List[SrcSpec]
    dst_root:     str
    dst_repo_id:  str
    dry_run:      bool
    limit:        int
    video_codec:  str


def _parse_src(spec: str) -> SrcSpec:
    """'ROOT' 또는 'ROOT::REPO_ID'."""
    if '::' in spec:
        root, repo_id = spec.split('::', 1)
    else:
        root, repo_id = spec, ''
    root = os.path.expanduser(root.strip())
    repo_id = repo_id.strip() or f'local/{Path(root).name}'
    return SrcSpec(root=root, repo_id=repo_id)


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description='LeRobotDataset 디스크 병합 (프레임/라벨 보존)')
    p.add_argument('--src', action='append', required=True, type=_parse_src,
                   help="원본 데이터셋 'ROOT' 또는 'ROOT::REPO_ID' (2회 이상 지정)")
    p.add_argument('--dst-root',    required=True,
                   help='병합 결과 디렉터리 (존재하면 에러 — 원본 보호)')
    p.add_argument('--dst-repo-id', default='',
                   help='병합 결과 repo_id (미지정 시 local/<dst 폴더명>)')
    p.add_argument('--dry-run',     action='store_true',
                   help='쓰기 없이 호환성/카운트만 리포트')
    p.add_argument('--limit',       type=int, default=0,
                   help='병합할 총 에피소드 상한 (0=전체, 실기 테스트용)')
    p.add_argument('--video-codec', default='h264',
                   help="dst 비디오 인코더 (기본 h264=CPU x264). av1(libsvtav1) "
                        "재인코딩이 CPU 로 매우 느려서. 유효값(lerobot 0.4.4): "
                        "h264, h264_nvenc(GPU/5090 권장·가장 빠름), hevc, "
                        "hevc_nvenc, libsvtav1, auto. 'av1'/'libsvtav1' 지정 시 "
                        "패치 안 함(lerobot 기본). 학습엔 코덱 무관.")
    a = p.parse_args()
    if len(a.src) < 2:
        p.error('--src 는 2개 이상 지정해야 함 (병합 대상).')
    dst_root = os.path.expanduser(a.dst_root)
    return Args(
        sources     = a.src,
        dst_root    = dst_root,
        dst_repo_id = a.dst_repo_id or f'local/{Path(dst_root).name}',
        dry_run     = a.dry_run,
        limit       = a.limit,
        video_codec = a.video_codec,
    )


# ──────────────────────────────────────────────
# 데이터셋 열기 / 라벨 / 호환성
# ──────────────────────────────────────────────

def open_ds(repo_id: str, root: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id=repo_id, root=root)


def load_labels_by_episode(root: str) -> Dict[int, dict]:
    """labels.jsonl → {episode_index: record}. 없으면 빈 dict."""
    p = Path(root) / 'labels.jsonl'
    if not p.exists():
        return {}
    return {int(r['episode']): r for r in load_labels_jsonl(str(p))}


def _norm_feat(feat: dict) -> dict:
    """비교용 정규화 (shape→tuple, names→tuple|None)."""
    return {
        'dtype': feat['dtype'],
        'shape': tuple(feat['shape']),
        'names': tuple(feat['names']) if feat.get('names') else None,
    }


def assert_compatible(srcs: list) -> None:
    """모든 원본의 4개 feature 스펙 + fps 일치 검증. 불일치 시 RuntimeError."""
    ref_feat = {k: _norm_feat(build_target_features(srcs[0])[k]) for k in FEATURE_KEYS}
    ref_fps  = int(getattr(srcs[0], 'fps', None) or srcs[0].meta.fps)
    for i, s in enumerate(srcs[1:], start=1):
        feat = {k: _norm_feat(build_target_features(s)[k]) for k in FEATURE_KEYS}
        for k in FEATURE_KEYS:
            if feat[k] != ref_feat[k]:
                raise RuntimeError(
                    f'feature 불일치 — 원본#{i} 의 {k!r}={feat[k]} != '
                    f'원본#0 의 {ref_feat[k]}. (해상도/임베딩 dim/관절 수 확인)')
        fps = int(getattr(s, 'fps', None) or s.meta.fps)
        if fps != ref_fps:
            raise RuntimeError(f'fps 불일치 — 원본#{i} fps={fps} != 원본#0 fps={ref_fps}')


def install_video_codec_override(codec: str) -> None:
    """lerobot 의 비디오 인코더(encode_video_frames)가 항상 지정 코덱을 쓰도록 강제.

    av1(libsvtav1) 재인코딩이 CPU 로 매우 느려, 병합 시 빠른 코덱(libx264)으로
    바꾸기 위함. lerobot 0.4.4 는 인코딩 코덱을 features 메타가 아니라
    encode_video_frames 의 기본값(libsvtav1)에서 가져오므로, 함수 자체를
    monkeypatch 해 vcodec 을 덮어쓴다. codec 가 'av1'/'libsvtav1'/'' 면 패치하지
    않는다(lerobot 기본 사용). 학습은 코덱과 무관(디코드만 하면 됨).
    """
    if not codec or codec in ('av1', 'libsvtav1'):
        return
    import inspect

    def _wrap(orig):
        try:
            has_vcodec = 'vcodec' in inspect.signature(orig).parameters
        except (ValueError, TypeError):
            has_vcodec = True

        def wrapper(*a, **k):
            if has_vcodec:
                k['vcodec'] = codec
            return orig(*a, **k)
        return wrapper

    patched = []
    # 정의 모듈 + 이름으로 임포트해 쓰는 모듈 양쪽을 패치 (임포트 방식 불명확)
    for modname in ('lerobot.datasets.video_utils',
                    'lerobot.datasets.lerobot_dataset'):
        try:
            mod = __import__(modname, fromlist=['encode_video_frames'])
        except Exception:
            continue
        if hasattr(mod, 'encode_video_frames'):
            mod.encode_video_frames = _wrap(mod.encode_video_frames)
            patched.append(modname)
    if patched:
        print(f'[merge] 비디오 코덱 override → {codec!r} (patched: {patched})')
    else:
        print(f'[merge] ⚠️ 코덱 override 실패 — encode_video_frames 를 못 찾음. '
              f'lerobot 기본(av1)으로 인코딩될 수 있음.')


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1) 원본 열기
    print('[merge] 원본 열기')
    srcs = []
    for s in args.sources:
        print(f'  - repo_id={s.repo_id!r} root={s.root}')
        srcs.append(open_ds(s.repo_id, s.root))

    # 2) 호환성 검증
    assert_compatible(srcs)
    fps        = int(getattr(srcs[0], 'fps', None) or srcs[0].meta.fps)
    robot_type = getattr(srcs[0].meta, 'robot_type', None) or 'so101_follower'
    features   = build_target_features(srcs[0])
    # [속도] dst 비디오 코덱 변경 (기본 libx264 — av1 재인코딩이 CPU 로 너무 느림).
    # features 메타와 인코더 양쪽을 맞춰준다.
    if args.video_codec and args.video_codec not in ('av1', 'libsvtav1'):
        if CAMERA_KEY in features and isinstance(features[CAMERA_KEY].get('info'), dict):
            features[CAMERA_KEY]['info']['video.codec'] = args.video_codec
            features[CAMERA_KEY]['info']['video.pix_fmt'] = 'yuv420p'
        install_video_codec_override(args.video_codec)
    print(f'[merge] feature/fps 호환 OK (fps={fps}, robot_type={robot_type})')

    # 3) 에피소드 인덱스 + 라벨 수집 + 리포트
    per_src: List[Tuple[object, Dict[int, np.ndarray], Dict[int, dict], SrcSpec]] = []
    total_eps = 0
    total_frames = 0
    for src, spec in zip(srcs, args.sources):
        ep_frames = episode_frame_indices(src)
        labels    = load_labels_by_episode(spec.root)
        missing   = [ep for ep in ep_frames if ep not in labels]
        if missing:
            raise RuntimeError(
                f'{spec.root}: labels.jsonl 에 episode {missing[:10]} ... 의 라벨 없음. '
                'record_real_teleop / relabel 산출 labels.jsonl 이 있어야 함.')
        n_f = sum(len(v) for v in ep_frames.values())
        total_eps += len(ep_frames)
        total_frames += n_f
        per_src.append((src, ep_frames, labels, spec))
        print(f'  · {spec.root}: 에피소드 {len(ep_frames)} | 프레임 {n_f}')

    print('=' * 60)
    print(f'[merge] 합계 — 에피소드 {total_eps} | 프레임 {total_frames}')
    # 병합 후 domain/spatial 분포 미리보기 (라벨 기반)
    spat = Counter()
    objs = Counter()
    for _, ep_frames, labels, _ in per_src:
        for ep in ep_frames:
            spat[labels[ep].get('spatial')] += 1
            objs[labels[ep].get('object')] += 1
    print('[merge] (병합 후) object 분포:', dict(objs))
    print('[merge] (병합 후) spatial 분포:', dict(spat))
    print('=' * 60)

    if args.dry_run:
        print('[merge] --dry-run: 쓰기 없이 종료.')
        return

    # 4) 원본 보호
    if Path(args.dst_root).exists():
        raise SystemExit(
            f'[merge] 출력 경로가 이미 존재: {args.dst_root}\n'
            '         새 경로를 지정하거나 기존 경로를 삭제하세요.')

    # 5) 새 데이터셋 생성 + 프레임 보존 재기록
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print(f'[merge] 생성: repo_id={args.dst_repo_id!r} root={args.dst_root}')
    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=fps,
        features=features,
        root=args.dst_root,
        robot_type=robot_type,
        use_videos=True,
    )

    out_labels: List[dict] = []
    new_ep = 0
    stop = False
    for src, ep_frames, labels, spec in per_src:
        if stop:
            break
        for ep in sorted(ep_frames):
            if args.limit > 0 and new_ep >= args.limit:
                stop = True
                break
            rec  = labels[ep]
            task = rec['canonical']
            for fi in ep_frames[ep]:
                f = src[int(fi)]
                dst.add_frame({
                    'observation.state':              _to_np_f32(f['observation.state']),
                    'observation.language_embedding': _to_np_f32(f['observation.language_embedding']),
                    CAMERA_KEY:                       _to_hwc_uint8(f[CAMERA_KEY]),
                    'action':                         _to_np_f32(f['action']),
                    'task':                           task,
                })
            dst.save_episode()
            out_labels.append({
                'episode':     new_ep,
                'object':      rec.get('object'),
                'attribute':   rec.get('attribute'),
                'spatial':     rec.get('spatial'),
                'canonical':   task,
                'domain':      rec.get('domain'),
                'src':         spec.root,
                'src_episode': ep,
            })
            print(f'  [{new_ep+1}] {Path(spec.root).name} ep{ep} '
                  f'({len(ep_frames[ep])} frames) → {task!r}')
            new_ep += 1

    labels_out = Path(args.dst_root) / 'labels.jsonl'
    with open(labels_out, 'w', encoding='utf-8') as fout:
        for r in out_labels:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'[merge] 완료 — 병합 에피소드 {len(out_labels)} → {args.dst_root}')
    print(f'[merge] labels.jsonl 기록: {labels_out}')


if __name__ == '__main__':
    main()
