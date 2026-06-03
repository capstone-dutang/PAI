"""
llama_server_manager.py

llama-server에 LoRA adapter hot-swap을 요청하고 결과를 통보하는 노드.

agent_node가 도메인 분류를 마치고 /domain_selection을 발행하면, 본 노드가
llama-server의 /lora-adapters HTTP 엔드포인트로 hot-swap 요청을 보내고,
성공 시 /selected_domain (std_msgs/String)을 발행하여 agent_node가 함수 호출
루프를 시작할 수 있게 한다.

llama-server는 systemd 등으로 상시 구동되어 있다고 가정한다 (README 참조).

API 형식 (llama.cpp build b8931-9725a313b 기준, 2026-06-02 Jetson 실기 확정):
  GET /lora-adapters
    → [{"id": int, "path": str, "scale": float,
        "task_name": str, "prompt_prefix": str}, ...]
  POST /lora-adapters
    body: [{"id": int, "scale": float}, ...]
    → 200 OK + {"success": true}

  새 어댑터 등록은 llama-server 의 `--lora <path>[,<path>...]` 옵션 갱신 후
  재시작으로만 가능 (path 기반 동적 load 미지원). 본 노드는 재시작된 서버를
  _refresh_adapter_index 가 자동 재조회.

실기 검증 (2026-06-02 통과):
  Jetson Orin Nano 8GB / llama-server (b8931) / 양 도메인 어댑터 등록 상태에서
  init deactivate / recycling↔tableware swap / 같은 도메인 재요청 idempotency
  모두 정상. 검증 명령과 결과 표는 deploy/README.md "LoRA hot-swap 실기 검증
  (2026-06-02)" 절 참조.

핵심 동작:
  llama-server 시작 시 --lora 옵션으로 등록된 모든 어댑터의 scale 이 1.0 으로
  활성화되어 base + LoRA_recycling + LoRA_tableware 의 weighted sum 상태가
  된다 (함수 호출 분포 망가짐). 본 노드는 __init__ 에서 모든 어댑터의 scale 을
  0.0 으로 deactivate 한 뒤, /domain_selection 수신 시 해당 도메인만 활성화.

환경:
  Python   3.10
  ROS2     Humble (Ubuntu 22.04 LTS)

구독 토픽:
  /domain_selection (robot_arm_msgs/DomainSelection)

발행 토픽:
  /selected_domain  (std_msgs/String)  — hot-swap 성공 시 도메인 이름 발행

파라미터:
  server_url  (str) — llama-server 주소 (기본 http://localhost:8080)
  adapter_dir (str) — adapter_<domain>.gguf 위치 (기본 ~/models/gemma/adapters/)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from robot_arm_msgs.msg import DomainSelection


# llama-server /lora-adapters HTTP 타임아웃 (초). swap 은 서버 내부 메모리
# 조작이라 빠른 작업 — 5초면 충분.
_HTTP_TIMEOUT_SEC = 5.0


class LlamaServerManager(Node):

    def __init__(self):
        super().__init__('llama_server_manager')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('server_url',  'http://localhost:8080')
        self.declare_parameter(
            'adapter_dir',
            os.path.expanduser('~/models/gemma/adapters/'),
        )

        self.server_url:  str = self.get_parameter('server_url').value
        self.adapter_dir: str = self.get_parameter('adapter_dir').value

        # ── 현재 적재된 도메인 (None이면 base 모델만) ──
        self._current_domain: Optional[str] = None

        # llama-server 의 어댑터 id 캐시. basename(adapter_path) → id (int).
        # __init__ 에서 GET /lora-adapters 로 채우고, 미스 시 _refresh 로 재구축.
        self._adapter_id_cache: Dict[str, int] = {}

        # ── 구독 ──────────────────────────────────
        self.create_subscription(
            DomainSelection, '/domain_selection', self._domain_callback, 10
        )

        # ── 발행 ──────────────────────────────────
        self.selected_pub = self.create_publisher(String, '/selected_domain', 10)

        self.get_logger().info(
            f'llama_server_manager 시작 | server: {self.server_url} | '
            f'adapter_dir: {self.adapter_dir}'
        )

        # ── 시작 시 어댑터 인덱스 캐시 + 모두 deactivate ──
        # llama-server 는 --lora 로 등록된 모든 어댑터를 scale=1.0 (활성) 으로
        # 시작한다. base + 두 LoRA 의 weighted sum 은 학습 분포와 어긋나므로
        # 첫 /domain_selection 수신 전까지는 base 만으로 동작시킨다.
        try:
            self._refresh_adapter_index()
            self._deactivate_all()
            self.get_logger().info(
                f'시작 시 어댑터 deactivate 완료 — '
                f'registered: {sorted(self._adapter_id_cache.keys())}'
            )
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f'시작 시 어댑터 초기화 실패 (llama-server 미구동 가능) — '
                f'첫 hot_swap 호출 시 재시도. err={e}'
            )

    # ── 구독 콜백 ─────────────────────────────────

    def _domain_callback(self, msg: DomainSelection) -> None:
        """
        DomainSelection 수신 → hot-swap 시도.

        - 같은 도메인이 이미 적재되어 있으면 스왑은 건너뛰고 즉시 통보만 한다.
          (한 태스크 진행 중에는 도메인이 고정되므로 중복 수신 가능성 있음)
        - 성공 시 /selected_domain 발행으로 agent_node에 통보.
        """
        domain = msg.domain
        scale  = float(msg.scale)

        if domain == self._current_domain:
            self.get_logger().info(
                f'도메인 {domain!r} 이미 적재됨 — 스왑 생략, 통보만 발행'
            )
            self._publish_selected(domain)
            return

        adapter_path = os.path.join(self.adapter_dir, f'adapter_{domain}.gguf')
        self.get_logger().info(
            f'LoRA hot-swap 요청: domain={domain!r} scale={scale} '
            f'adapter={adapter_path}'
        )

        try:
            ok = self._hot_swap(adapter_path, scale)
        except NotImplementedError as e:
            self.get_logger().error(str(e))
            return
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'hot-swap HTTP 실패: {e}')
            return

        if not ok:
            self.get_logger().error(f'hot-swap 거부됨 (domain={domain!r})')
            return

        self._current_domain = domain
        self._publish_selected(domain)

    # ── 발행 헬퍼 ────────────────────────────────

    def _publish_selected(self, domain: str) -> None:
        msg = String()
        msg.data = domain
        self.selected_pub.publish(msg)
        self.get_logger().info(f'/selected_domain 발행: {domain!r}')

    # ── HTTP 호출 ────────────────────────────────

    def _refresh_adapter_index(self) -> List[dict]:
        """
        GET /lora-adapters → basename(path) → id 매핑 재구축 후 응답 list 반환.

        예외:
          requests.RequestException — HTTP 실패
          ValueError — 응답이 list 아님
        """
        r = requests.get(
            f'{self.server_url}/lora-adapters', timeout=_HTTP_TIMEOUT_SEC,
        )
        r.raise_for_status()
        adapters = r.json()
        if not isinstance(adapters, list):
            raise ValueError(
                f'GET /lora-adapters 응답이 list 아님: {adapters!r}'
            )
        self._adapter_id_cache = {
            os.path.basename(a['path']): a['id']
            for a in adapters
            if isinstance(a, dict) and 'path' in a and 'id' in a
        }
        return adapters

    def _deactivate_all(self) -> None:
        """모든 등록 어댑터의 scale 을 0.0 으로 (POST). 캐시가 비었으면 no-op."""
        if not self._adapter_id_cache:
            return
        body = [
            {'id': aid, 'scale': 0.0}
            for aid in self._adapter_id_cache.values()
        ]
        r = requests.post(
            f'{self.server_url}/lora-adapters',
            json=body,
            timeout=_HTTP_TIMEOUT_SEC,
        )
        r.raise_for_status()

    def _hot_swap(self, adapter_path: str, scale: float) -> bool:
        """
        llama-server 에 LoRA adapter hot-swap 요청.

        동작:
          1. basename(adapter_path) 로 캐시에서 target id 조회.
             없으면 _refresh_adapter_index 로 재시도.
          2. POST /lora-adapters body: 등록된 모든 어댑터의 scale 일괄 설정 —
             target 만 `scale`, 나머지는 0.0.
          3. HTTP 200 시 True 반환. 캐시는 다음 호출에서 재사용.

        새 도메인 추가 시 (어댑터 파일만 disk 에 있고 서버 미등록):
          llama-server 의 --lora 옵션을 갱신해 재시작해야 등록됨. 본 메서드는
          미등록 어댑터를 동적 load 하지 않는다 (llama.cpp /lora-adapters API
          제약).

        Returns:
            성공 시 True, 실패 (HTTP 에러 / 어댑터 미등록 / status != 200) 시 False.
        """
        target = os.path.basename(adapter_path)
        if target not in self._adapter_id_cache:
            try:
                self._refresh_adapter_index()
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(
                    f'/lora-adapters 인덱스 재조회 실패: {e}'
                )
                return False
        target_id = self._adapter_id_cache.get(target)
        if target_id is None:
            self.get_logger().error(
                f'어댑터 {target!r} 가 llama-server 에 등록 안 됨. '
                f'서버 시작 명령에 --lora {adapter_path} 가 있는지 확인. '
                f'(registered: {sorted(self._adapter_id_cache.keys())})'
            )
            return False

        body = [
            {'id': aid, 'scale': float(scale) if aid == target_id else 0.0}
            for aid in self._adapter_id_cache.values()
        ]
        try:
            r = requests.post(
                f'{self.server_url}/lora-adapters',
                json=body,
                timeout=_HTTP_TIMEOUT_SEC,
            )
        except requests.RequestException as e:
            self.get_logger().error(f'/lora-adapters POST 실패: {e}')
            return False
        if r.status_code != 200:
            self.get_logger().error(
                f'/lora-adapters POST status={r.status_code} '
                f'body={r.text!r}'
            )
            return False
        return True


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = LlamaServerManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
