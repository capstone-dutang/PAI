"""
camera_node.py

카메라 드라이버 어댑터 구조로 설계된 카메라 노드.

환경:
  Python   3.10
  ROS2     Humble (Humboldt, 22.04 LTS)

드라이버 확정 시:
  1. BaseCameraAdapter를 상속한 새 어댑터 클래스 구현
  2. CameraNode.__init__() 의 self.adapter = PlaceholderAdapter() 를
     self.adapter = YourAdapter() 로 교체
  3. 어댑터 내부의 get_frames() 구현

발행 토픽:
  /camera/rgb   (sensor_msgs/Image) — 30Hz
  /camera/depth (sensor_msgs/Image) — 30Hz
"""

from __future__ import annotations

from typing import Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


# ──────────────────────────────────────────────
# 어댑터 인터페이스 (모든 드라이버가 구현해야 함)
# ──────────────────────────────────────────────

class BaseCameraAdapter:
    """
    카메라 드라이버 어댑터 인터페이스.
    드라이버 확정 시 이 클래스를 상속하여 get_frames() 를 구현.
    """

    def get_frames(self) -> Tuple[Image, Image]:
        """
        Returns:
            (rgb_msg, depth_msg): sensor_msgs/Image 형식의 튜플
        """
        raise NotImplementedError(
            "get_frames()를 구현해야 합니다. "
            "드라이버 확정 후 BaseCameraAdapter를 상속하여 구현하세요."
        )

    def destroy(self) -> None:
        """노드 종료 시 드라이버 리소스 해제. 필요 시 오버라이드."""
        pass


# ──────────────────────────────────────────────
# 임시 어댑터 (드라이버 확정 전까지 사용)
# 드라이버 확정 시 이 클래스를 실제 어댑터로 교체
# ──────────────────────────────────────────────

class PlaceholderAdapter(BaseCameraAdapter):
    """
    드라이버 미확정 상태의 임시 어댑터.
    get_frames() 호출 시 NotImplementedError 발생.

    드라이버 확정 시 교체 방법:
        class RealSenseAdapter(BaseCameraAdapter):
            def get_frames(self):
                # pyrealsense2 등으로 프레임 취득
                ...
    """

    def get_frames(self) -> Tuple[Image, Image]:
        raise NotImplementedError(
            "[PlaceholderAdapter] 카메라 드라이버가 아직 확정되지 않았습니다. "
            "드라이버 확정 후 PlaceholderAdapter를 실제 어댑터로 교체하세요."
        )


# ──────────────────────────────────────────────
# 카메라 노드
# ──────────────────────────────────────────────

class CameraNode(Node):

    PUBLISH_HZ = 30.0

    def __init__(self):
        super().__init__('camera_node')

        # ── 파라미터 ──────────────────────────────
        # 드라이버 이름: 로깅용. 실제 분기는 어댑터 교체로 처리.
        self.declare_parameter('camera_driver', 'placeholder')
        driver_name = self.get_parameter('camera_driver').value
        self.get_logger().info(f'카메라 드라이버: {driver_name}')

        # ── 어댑터 초기화 ─────────────────────────
        # TODO: 드라이버 확정 시 아래 한 줄을 교체
        #   self.adapter = PlaceholderAdapter()
        #   → self.adapter = RealSenseAdapter()  (예시)
        self.adapter = PlaceholderAdapter()

        # ── 퍼블리셔 ──────────────────────────────
        self.rgb_pub   = self.create_publisher(Image, '/camera/rgb',   10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth', 10)

        # ── 타이머 ────────────────────────────────
        self.create_timer(1.0 / self.PUBLISH_HZ, self._timer_callback)
        self.get_logger().info(f'camera_node 시작 ({self.PUBLISH_HZ}Hz)')

    def _timer_callback(self) -> None:
        try:
            rgb_msg, depth_msg = self.adapter.get_frames()
            self.rgb_pub.publish(rgb_msg)
            self.depth_pub.publish(depth_msg)
        except NotImplementedError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f'프레임 취득 실패: {e}')

    def destroy_node(self) -> None:
        self.adapter.destroy()
        super().destroy_node()


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
