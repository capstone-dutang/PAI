"""
camera_node.py

Intel RealSense ROS2 드라이버(realsense2_camera)가 발행하는 토픽을 구독하여
시스템 내부 토픽(/camera/rgb, /camera/depth)으로 릴레이하는 노드.

전제조건:
  realsense2_camera 드라이버가 align_depth 활성화 상태로 실행 중이어야 함:
    ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

구독 토픽 (realsense2_camera 드라이버 출력):
  realsense2_camera v4.x 의 default 는 두 단계 namespace ('/camera/camera/...').
  camera_namespace:='' 옵션으로 띄우면 단일 단계 ('/camera/...').
  본 노드는 input_color_topic / input_depth_topic ROS 파라미터로 양쪽 모두
  지원한다. 기본값은 v4.x default (이중 namespace).

발행 토픽 (시스템 내부):
  /camera/rgb   (sensor_msgs/Image)
  /camera/depth (sensor_msgs/Image)

파라미터:
  input_color_topic (str) — realsense color 토픽
                            기본 '/camera/camera/color/image_raw'
                            (camera_namespace:='' 로 launch 한 경우
                             '/camera/color/image_raw' 로 지정)
  input_depth_topic (str) — realsense aligned-depth 토픽
                            기본 '/camera/camera/aligned_depth_to_color/image_raw'
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
import message_filters


class CameraNode(Node):

    QUEUE_SIZE = 10
    SYNC_SLOP  = 0.033  # 1프레임(30Hz) 허용 시간 오차 (초)

    def __init__(self):
        super().__init__('camera_node')

        # ── 파라미터 ──────────────────────────────
        # realsense2_camera v4.x 기본 namespace 는 '/camera/camera/...'.
        # camera_namespace:='' 로 launch 한 경우 '/camera/color/image_raw' 식.
        self.declare_parameter(
            'input_color_topic', '/camera/camera/color/image_raw',
        )
        self.declare_parameter(
            'input_depth_topic', '/camera/camera/aligned_depth_to_color/image_raw',
        )
        input_color = self.get_parameter('input_color_topic').value
        input_depth = self.get_parameter('input_depth_topic').value

        # ── QoS ──────────────────────────────────
        # realsense2_camera 는 카메라 토픽을 BEST_EFFORT 로 발행한다.
        # RELIABLE 구독자는 BEST_EFFORT 퍼블리셔에 연결하지 못해 콜백이 안 돈다
        # ─ 그래서 입력 sub 두 개를 모두 BEST_EFFORT 로 맞춘다. 발행도 동일
        # 정책으로 통일해 act_node / record_real_teleop helper (BestEffort 구독) 와
        # 자연스럽게 결합되도록 한다.
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.QUEUE_SIZE,
        )

        # ── 퍼블리셔 ──────────────────────────────
        self.rgb_pub   = self.create_publisher(Image, '/camera/rgb',   camera_qos)
        self.depth_pub = self.create_publisher(Image, '/camera/depth', camera_qos)

        # ── 동기화 구독 (RGB + Depth 타임스탬프 동기화) ──
        rgb_sub = message_filters.Subscriber(
            self, Image, input_color, qos_profile=camera_qos
        )
        depth_sub = message_filters.Subscriber(
            self, Image, input_depth, qos_profile=camera_qos
        )

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=self.QUEUE_SIZE,
            slop=self.SYNC_SLOP,
        )
        self._sync.registerCallback(self._relay_callback)

        self.get_logger().info(
            'camera_node 시작 (QoS=BestEffort)\n'
            f'  구독: {input_color}\n'
            f'        {input_depth}\n'
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
