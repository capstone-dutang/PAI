"""
camera_node.py

Intel RealSense ROS2 드라이버(realsense2_camera)가 발행하는 토픽을 구독하여
시스템 내부 토픽(/camera/rgb, /camera/depth)으로 릴레이하는 노드.

전제조건:
  realsense2_camera 드라이버가 align_depth 활성화 상태로 실행 중이어야 함:
    ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

구독 토픽 (realsense2_camera 드라이버 출력):
  /camera/color/image_raw                  (sensor_msgs/Image)
  /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image)

발행 토픽 (시스템 내부):
  /camera/rgb   (sensor_msgs/Image)
  /camera/depth (sensor_msgs/Image)
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters


class CameraNode(Node):

    QUEUE_SIZE = 10
    SYNC_SLOP  = 0.033  # 1프레임(30Hz) 허용 시간 오차 (초)

    def __init__(self):
        super().__init__('camera_node')

        # ── 퍼블리셔 ──────────────────────────────
        self.rgb_pub   = self.create_publisher(Image, '/camera/rgb',   self.QUEUE_SIZE)
        self.depth_pub = self.create_publisher(Image, '/camera/depth', self.QUEUE_SIZE)

        # ── 동기화 구독 (RGB + Depth 타임스탬프 동기화) ──
        rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw'
        )
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/aligned_depth_to_color/image_raw'
        )

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=self.QUEUE_SIZE,
            slop=self.SYNC_SLOP,
        )
        self._sync.registerCallback(self._relay_callback)

        self.get_logger().info(
            'camera_node 시작\n'
            '  구독: /camera/color/image_raw\n'
            '        /camera/aligned_depth_to_color/image_raw\n'
            '  발행: /camera/rgb  /camera/depth'
        )

    def _relay_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)


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
