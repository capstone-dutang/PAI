"""
record_teleop.py

Part D — SO-101 teleop 기반 ACT 학습용 시연 데이터 수집 스크립트.

`agent_node` / Gemma LoRA는 사용하지 않는다. SO-101 리더(작업자가 직접 조작)의
관절 값을 SO-101 팔로워에 그대로 미러링하면서, 시점마다 다음을 기록한다:

  observation.state              (6,)   float32  — 팔로워 관절 위치
  observation.language_embedding (384,) float32  — embedding_node가 산출한
                                                    자연어 타겟 묘사 임베딩
                                                    (paraphrase-multilingual-MiniLM-L12-v2)
  observation.images.top         (H,W,3) uint8   — /camera/rgb (RealSense RGB)
  action                         (6,)   float32  — 리더 관절 값 (정답)

언어 라벨링 — Option C (템플릿 + 변수 슬롯) 확정:
  작업자는 (object, attribute, spatial) 3개 슬롯을 deploy/config/label_vocab.yaml
  vocabulary 안에서만 선택한다. canonical 라벨은 자동 생성된다:

      "the {attribute} {object} {spatial}"
      예: "the blue can on the right side of the table"

  자유 텍스트 입력 금지 — 영어 템플릿 고정으로 Sim/Real 분포 일치 보장.
  수집 후 paraphrase는 별도 scripts/postprocess_real_labels.py로 후처리.

전제조건 (README "데이터 수집 (Part D)" 참조):
  1. realsense2_camera 드라이버 실행 (align_depth.enable:=true)
  2. camera_node 실행 (/camera/rgb 발행)
  3. embedding_node 실행 (/embed_text Service,
     paraphrase-multilingual-MiniLM-L12-v2)
  4. SO-101 리더/팔로워 USB 포트 확인 + lerobot-calibrate 완료
  5. deploy/config/label_vocab.yaml 배치

CLI 사용 예:
  python record_teleop.py \\
    --robot-port      /dev/ttyACM0 \\
    --robot-id        follower \\
    --teleop-port     /dev/ttyACM1 \\
    --teleop-id       leader \\
    --dataset-repo-id myname/dataset_real \\
    --domain          recycling \\
    --label-vocab     ~/ros2_ws/config/label_vocab.yaml \\
    --num-episodes    50

미구현 항목 (LeRobot SDK 통합 시 결정 필요):
  - LeRobotDataset 0.4.4 features 스키마에 language_embedding (384,) 추가 호출
  - SO-101 리더/팔로워 연결 (lerobot SDK)
  - 30Hz 동기화 루프 (rclpy 콜백 / LeRobot read 주기)
  - 에피소드 종료 조건 (키 입력 / 타임아웃 / 리더 정지 감지)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image

from robot_arm_msgs.srv import EmbedText


EXPECTED_EMBED_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@dataclass
class RecordArgs:
    robot_port:       str
    robot_id:         str
    teleop_port:      str
    teleop_id:        str
    dataset_repo_id:  str
    domain:           str
    label_vocab:      str
    num_episodes:     int
    fps:              int
    labels_jsonl:     str


def parse_args() -> RecordArgs:
    p = argparse.ArgumentParser(description='Part D — SO-101 teleop 시연 데이터 수집')
    p.add_argument('--robot-port',      required=True, help='SO-101 팔로워 USB 포트')
    p.add_argument('--robot-id',        default='follower', help='팔로워 캘리브레이션 ID')
    p.add_argument('--teleop-port',     required=True, help='SO-101 리더 USB 포트')
    p.add_argument('--teleop-id',       default='leader',   help='리더 캘리브레이션 ID')
    p.add_argument('--dataset-repo-id', required=True,
                   help='LeRobotDataset repo id (예: user/dataset_real)')
    p.add_argument('--domain',          required=True,
                   help='현재 수집할 도메인 (label_vocab.yaml의 domains 키와 일치)')
    p.add_argument('--label-vocab',     default='~/ros2_ws/config/label_vocab.yaml',
                   help='라벨 vocabulary YAML 경로')
    p.add_argument('--num-episodes',    type=int, default=50, help='수집 에피소드 수')
    p.add_argument('--fps',             type=int, default=30, help='기록 주기 (Hz)')
    p.add_argument('--labels-jsonl',    default='',
                   help='canonical 라벨 + 슬롯 메타데이터 기록 JSONL 경로. '
                        '비어 있으면 데이터셋 디렉터리 옆에 labels.jsonl로 자동 생성. '
                        '이 파일은 postprocess_real_labels.py 입력으로 사용된다.')
    a = p.parse_args()
    return RecordArgs(
        robot_port      = a.robot_port,
        robot_id        = a.robot_id,
        teleop_port     = a.teleop_port,
        teleop_id       = a.teleop_id,
        dataset_repo_id = a.dataset_repo_id,
        domain          = a.domain,
        label_vocab     = os.path.expanduser(a.label_vocab),
        num_episodes    = a.num_episodes,
        fps             = a.fps,
        labels_jsonl    = os.path.expanduser(a.labels_jsonl) if a.labels_jsonl else '',
    )


# ──────────────────────────────────────────────
# Vocabulary 로드 / 검증
# ──────────────────────────────────────────────

@dataclass
class LabelVocab:
    template:    str
    spatials:    List[str]
    objects:     Dict[str, List[str]]          # object_name -> attributes
    domain:      str = ''
    raw:         dict = field(default_factory=dict)


def load_vocab(path: str, domain: str) -> LabelVocab:
    """label_vocab.yaml 로드 + 지정 도메인 추출.

    스키마 검증:
      - template, spatial_qualifiers, domains[domain].objects 필수
      - 각 object는 attributes 리스트 필수
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    template = raw.get('template')
    spatials = raw.get('spatial_qualifiers') or []
    domains  = raw.get('domains') or {}

    if not template or '{attribute}' not in template \
       or '{object}' not in template or '{spatial}' not in template:
        raise ValueError(
            f'label_vocab template이 비어 있거나 슬롯 누락 ({path}). '
            f'"the {{attribute}} {{object}} {{spatial}}" 형식 필요.'
        )
    if not spatials:
        raise ValueError(f'spatial_qualifiers가 비어 있음 ({path}).')
    if domain not in domains:
        raise ValueError(
            f"도메인 '{domain}'이 label_vocab에 없음. 가능한 도메인: {list(domains)}"
        )

    obj_block = domains[domain].get('objects') or {}
    objects: Dict[str, List[str]] = {}
    for obj_name, obj_def in obj_block.items():
        attrs = (obj_def or {}).get('attributes') or []
        if not attrs:
            raise ValueError(
                f"객체 '{obj_name}'에 attributes가 비어 있음 (domain={domain})."
            )
        objects[obj_name] = list(attrs)

    if not objects:
        raise ValueError(f"도메인 '{domain}'에 등록된 객체가 없음.")

    return LabelVocab(
        template=template,
        spatials=list(spatials),
        objects=objects,
        domain=domain,
        raw=raw,
    )


# ──────────────────────────────────────────────
# 슬롯 입력 (Option C)
# ──────────────────────────────────────────────

def _pick(prompt: str, choices: List[str]) -> str:
    """번호 또는 값 입력으로 choices 중 하나를 선택.

    자유 텍스트 금지 — choices 안에 없으면 재시도.
    """
    while True:
        print(f"\n{prompt}")
        for i, c in enumerate(choices):
            print(f"  [{i}] {c}")
        raw = input(f"선택 (0-{len(choices)-1} 또는 값 입력): ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"  ! 범위 밖 — 0~{len(choices)-1}")
            continue
        if raw in choices:
            return raw
        print(f"  ! vocabulary 외 입력 거부. 위 목록에서만 선택 가능.")


def prompt_episode_label(
    vocab:       LabelVocab,
    episode_idx: int,
) -> Dict[str, str]:
    """
    Option C 슬롯 입력으로 canonical 라벨 생성.

    Returns:
        {
            'object':    str,
            'attribute': str,
            'spatial':   str,
            'canonical': str,   # 템플릿 채워진 영어 라벨
            'domain':    str,
            'episode':   int,
        }
    """
    print('\n' + '=' * 60)
    print(f'  [Episode {episode_idx}] 라벨 슬롯 입력 (domain={vocab.domain})')
    print('=' * 60)

    obj_names = list(vocab.objects.keys())
    obj   = _pick('1) Object',    obj_names)
    attr  = _pick('2) Attribute', vocab.objects[obj])
    spat  = _pick('3) Spatial',   vocab.spatials)

    canonical = vocab.template.format(attribute=attr, object=obj, spatial=spat)
    print(f"\n  canonical: {canonical}")

    return {
        'object':    obj,
        'attribute': attr,
        'spatial':   spat,
        'canonical': canonical,
        'domain':    vocab.domain,
        'episode':   episode_idx,
    }


# ──────────────────────────────────────────────
# 라벨 메타데이터 JSONL 기록 (postprocess 입력)
# ──────────────────────────────────────────────

def _resolve_labels_jsonl(args: RecordArgs) -> str:
    if args.labels_jsonl:
        path = args.labels_jsonl
    else:
        # 기본: dataset_repo_id 마지막 컴포넌트 옆에 labels.jsonl 생성
        # repo id 형식 "user/name" → ~/.cache/huggingface/lerobot/user/name/labels.jsonl 등
        # 실제 LeRobotDataset 경로 확정 후 보강 필요. 임시로 CWD에 둠.
        name = args.dataset_repo_id.replace('/', '_')
        path = os.path.join(os.getcwd(), f'labels_{name}.jsonl')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def append_label_record(jsonl_path: str, record: dict) -> None:
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ──────────────────────────────────────────────
# ROS2 보조 노드 — 카메라 캐싱 + embedding 서비스 클라이언트
# ──────────────────────────────────────────────

class RecordHelperNode(Node):
    """
    record_teleop가 ROS2와 상호작용하기 위한 헬퍼 노드.
      - /camera/rgb 구독 (BestEffort + KeepLast(1)) — latest-only 캐시
      - /embed_text 서비스 클라이언트
    """

    def __init__(self):
        super().__init__('record_teleop_helper')

        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._latest_rgb: Optional[Image] = None
        self.create_subscription(Image, '/camera/rgb', self._rgb_cb, camera_qos)

        self.embed_client = self.create_client(EmbedText, '/embed_text')

    def _rgb_cb(self, msg: Image) -> None:
        self._latest_rgb = msg

    def latest_rgb(self) -> Optional[Image]:
        return self._latest_rgb

    def call_embed_text(self, text: str, timeout_sec: float = 5.0) -> Optional[List[float]]:
        """
        /embed_text 서비스 호출. 응답의 dim==0 이면 모델 미로드/실패로 간주하고 None 반환.
        """
        if not self.embed_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error('/embed_text 서비스 연결 실패')
            return None

        req = EmbedText.Request()
        req.text = text
        future = self.embed_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        resp = future.result()
        if resp is None or resp.dim == 0:
            self.get_logger().error(
                f'/embed_text 응답 실패 또는 dim=0 (text={text!r})'
            )
            return None
        if resp.dim != EXPECTED_EMBED_DIM:
            self.get_logger().error(
                f'/embed_text dim={resp.dim} 예상값 {EXPECTED_EMBED_DIM}와 불일치. '
                f'embedding_node 모델 확인 필요.'
            )
            return None
        return list(resp.embedding)


# ──────────────────────────────────────────────
# 데이터셋 / 로봇 연결 (LeRobot SDK 미구현)
# ──────────────────────────────────────────────

def open_dataset(args: RecordArgs, embedding_dim: int):
    """
    LeRobotDataset (0.4.4)을 features 스키마와 함께 생성/오픈한다.

    features:
      observation.state              : float32, (6,)
      observation.language_embedding : float32, (embedding_dim,)
      observation.images.top         : video,   (H, W, 3)
      action                         : float32, (6,)

    TODO: LeRobotDataset 0.4.4 API 호출부 구현.
          embedding_dim은 호출 시점에 EXPECTED_EMBED_DIM(=384)으로 고정됨.
    """
    raise NotImplementedError(
        "[record_teleop] open_dataset() 미구현 — LeRobotDataset 0.4.4 통합 작업."
    )


def connect_so101(robot_port: str, robot_id: str, teleop_port: str, teleop_id: str):
    """
    SO-101 팔로워(robot) + 리더(teleop) 연결.

    TODO: LeRobot SDK 호출. robot_controller_node.py와 동일한 방식 사용.
    """
    raise NotImplementedError("[record_teleop] connect_so101() 미구현.")


# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────

def record_episode(
    args:          RecordArgs,
    helper:        RecordHelperNode,
    robot,
    teleop,
    dataset,
    episode_idx:   int,
    label_record:  dict,
    embedding:     List[float],
) -> None:
    """
    1개 에피소드 기록.

    TODO: 30Hz 루프로 다음을 동시 수행
      - teleop.get_action()      → 리더 관절 값 (action)
      - robot.send_action(...)   → 팔로워 미러링
      - robot.get_observation()  → 팔로워 관절 값 (observation.state)
      - helper.latest_rgb()      → 카메라 RGB (observation.images.top)
      - embedding 그대로         → observation.language_embedding (에피소드 내 상수)
      - dataset.add_frame(...)
    """
    raise NotImplementedError(
        "[record_teleop] record_episode() 미구현. "
        "30Hz 동기화 루프 / 에피소드 종료 정책 확정 후 구현하세요."
    )


def main() -> None:
    args = parse_args()

    # 1) Vocabulary 로드 (Option C 슬롯 검증용)
    vocab = load_vocab(args.label_vocab, args.domain)
    labels_jsonl = _resolve_labels_jsonl(args)
    print(f'[record_teleop] vocab loaded ({args.label_vocab}, domain={args.domain})')
    print(f'[record_teleop] label log: {labels_jsonl}')

    rclpy.init()
    helper = RecordHelperNode()

    try:
        # 2) 첫 에피소드 라벨 → 임베딩 차원 검증 (384 강제)
        first_rec   = prompt_episode_label(vocab, episode_idx=0)
        first_embed = helper.call_embed_text(first_rec['canonical'])
        if first_embed is None:
            helper.get_logger().error(
                '첫 에피소드 임베딩 획득 실패. embedding_node 상태 확인 후 재시도.'
            )
            return
        embedding_dim = len(first_embed)
        if embedding_dim != EXPECTED_EMBED_DIM:
            helper.get_logger().error(
                f'embedding_dim={embedding_dim} 예상 {EXPECTED_EMBED_DIM}와 다름. 중단.'
            )
            return
        helper.get_logger().info(f'embedding_dim = {embedding_dim} (확정)')

        # 3) 데이터셋 / 로봇 연결
        dataset = open_dataset(args, embedding_dim)
        robot, teleop = connect_so101(
            args.robot_port, args.robot_id, args.teleop_port, args.teleop_id,
        )

        # 4) 에피소드 루프
        for ep in range(args.num_episodes):
            if ep == 0:
                rec, embed = first_rec, first_embed
            else:
                rec   = prompt_episode_label(vocab, ep)
                embed = helper.call_embed_text(rec['canonical'])
                if embed is None or len(embed) != EXPECTED_EMBED_DIM:
                    helper.get_logger().error(
                        f'에피소드 {ep} 임베딩 실패. 스킵.'
                    )
                    continue

            append_label_record(labels_jsonl, rec)
            record_episode(args, helper, robot, teleop, dataset, ep, rec, embed)

    finally:
        helper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
