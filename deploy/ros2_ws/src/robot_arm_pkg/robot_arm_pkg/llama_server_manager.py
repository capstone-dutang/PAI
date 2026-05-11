"""
llama_server_manager.py

llama-server에 LoRA adapter hot-swap을 요청하고 결과를 통보하는 노드.

agent_node가 도메인 분류를 마치고 /domain_selection을 발행하면, 본 노드가
llama-server의 /lora-adapter HTTP 엔드포인트로 hot-swap 요청을 보내고,
성공 시 /selected_domain (std_msgs/String)을 발행하여 agent_node가 함수 호출
루프를 시작할 수 있게 한다.

llama-server는 systemd 등으로 상시 구동되어 있다고 가정한다 (README 참조).

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

미구현 항목:
  _hot_swap()  — llama-server /lora-adapter 호출. llama.cpp 측 실제 API 형태는
                 빌드 버전에 따라 다를 수 있어 운영 시점에 확정 필요.
"""

from __future__ import annotations

import os
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from robot_arm_msgs.msg import DomainSelection


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

    def _hot_swap(self, adapter_path: str, scale: float) -> bool:
        """
        llama-server에 LoRA adapter hot-swap 요청.

        TODO: llama.cpp 빌드 버전 확정 후 구현
          참고 형태 (llama.cpp 최근 빌드):
            POST {server_url}/lora-adapter
            body: [{"path": adapter_path, "scale": scale}]

          실제 엔드포인트와 페이로드 형식은 빌드 버전에 따라 다르므로
          운영 시점에 llama-server --help 또는 docs/server.md 확인 후 확정.

          예시:
            import requests
            r = requests.post(
                f'{self.server_url}/lora-adapter',
                json=[{'path': adapter_path, 'scale': scale}],
                timeout=5.0,
            )
            return r.status_code == 200

        Returns:
            성공 시 True, 실패 시 False.
        """
        raise NotImplementedError(
            "[llama_server_manager] _hot_swap() 미구현. "
            "llama-server /lora-adapter API 형식 확정 후 구현하세요."
        )


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
