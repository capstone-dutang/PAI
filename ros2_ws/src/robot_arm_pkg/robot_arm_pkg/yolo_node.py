"""
yolo_node.py

YOLO 실시간 탐지로 물체 3D 위치를 30Hz로 발행하는 노드.
vlm_node가 발행한 /target_class를 구독하여 탐지 대상 클래스를 필터링한 뒤,
Depth + Camera Intrinsics 로 3D 좌표를 계산하고
camera_robot_transform.yaml 로 robot base_link frame 으로 변환하여 발행.

환경:
  Python   3.10
  ROS2     Humble (Ubuntu 22.04 LTS)
  ultralytics YOLOv8n (또는 TensorRT .engine)

구독 토픽:
  /camera/rgb            (sensor_msgs/Image)         — 30Hz, camera_node 릴레이
  /camera/depth          (sensor_msgs/Image)         — 30Hz, camera_node 릴레이
  /camera/color/camera_info (sensor_msgs/CameraInfo) — 최초 1회, RealSense 드라이버
  /target_class          (std_msgs/String)           — 0.4Hz, vlm_node
  /task_active           (std_msgs/Bool)             — state_monitor_node

발행 토픽:
  /object_pose (geometry_msgs/PoseStamped, frame_id=base_link) — 30Hz

파라미터:
  model_path           (str)   — YOLO 모델 경로 (.pt 또는 TensorRT .engine)
                                  기본: ~/models/yolo/yolo_model.pt
  transform_path       (str)   — camera_robot_transform.yaml 경로
                                  기본: ~/calibration/camera_robot_transform.yaml
  confidence_threshold (float) — YOLO confidence 임계값 (기본 0.5)

처리 흐름:
  YOLO 바운딩박스 (pixel)
    → Depth 3×3 중앙값 샘플링 (미터)
    → 카메라 frame 3D 좌표 (camera intrinsics 적용)
    → robot base_link frame 3D 좌표 (camera_robot_transform.yaml 적용)
    → /object_pose 발행
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
import message_filters


class YOLONode(Node):

    QUEUE_SIZE       = 10
    SYNC_SLOP        = 0.033   # 1프레임(30Hz) 허용 시간 오차 (초)
    DEPTH_PATCH_HALF = 1       # 중심 (2k+1)×(2k+1) 패치 샘플링, k=1 → 3×3

    def __init__(self):
        super().__init__('yolo_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter(
            'model_path',
            os.path.expanduser('~/models/yolo/yolo_model.pt'),
        )
        self.declare_parameter(
            'transform_path',
            os.path.expanduser('~/calibration/camera_robot_transform.yaml'),
        )
        self.declare_parameter('confidence_threshold', 0.5)

        model_path     = self.get_parameter('model_path').value
        transform_path = self.get_parameter('transform_path').value
        self._conf_thr = self.get_parameter('confidence_threshold').value

        # ── YOLO 모델 로드 ────────────────────────
        self._model = self._load_yolo(model_path)

        # ── Camera → Robot 변환 행렬 로드 ─────────
        self._R, self._t = self._load_transform(transform_path)

        # ── Camera Intrinsics (camera_info 수신 전까지 None) ──
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._intrinsics_ready = False

        # ── 탐지 대상 클래스 (target_class 수신 전까지 None) ──
        self._target_class: Optional[str] = None

        # ── 태스크 활성 여부 (state_monitor 제어) ──
        self._task_active: bool = False

        # ── 발행 ──────────────────────────────────
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', self.QUEUE_SIZE)

        # ── 구독: camera_info (최초 1회) ──────────
        self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self._camera_info_callback,
            1,
        )

        # ── 구독: target_class ────────────────────
        self.create_subscription(
            String,
            '/target_class',
            self._target_class_callback,
            10,
        )

        # ── 구독: task_active ─────────────────────
        self.create_subscription(
            Bool,
            '/task_active',
            self._task_active_callback,
            10,
        )

        # ── 구독: RGB + Depth 동기화 ──────────────
        rgb_sub   = message_filters.Subscriber(self, Image, '/camera/rgb')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth')
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=self.QUEUE_SIZE,
            slop=self.SYNC_SLOP,
        )
        self._sync.registerCallback(self._image_callback)

        self.get_logger().info(
            f'yolo_node 시작\n'
            f'  모델: {model_path}\n'
            f'  변환: {transform_path}\n'
            f'  conf_threshold: {self._conf_thr}'
        )

    # ── 초기화 헬퍼 ──────────────────────────────

    def _load_yolo(self, model_path: str):
        """YOLO 모델 로드 (.pt 또는 TensorRT .engine)."""
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            model = YOLO(model_path)
            self.get_logger().info(f'YOLO 모델 로드 완료: {model_path}')
            return model
        except Exception as e:
            self.get_logger().error(f'YOLO 모델 로드 실패: {e}')
            raise

    def _load_transform(self, transform_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        camera_robot_transform.yaml 로드.

        easy_handeye2 저장 포맷 (eye_to_hand):
          transformation:
            x: ...  y: ...  z: ...
            qx: ... qy: ... qz: ... qw: ...

        Returns:
            R (3×3 ndarray): 회전행렬, t (3, ndarray): 이동벡터
            p_base = R @ p_camera + t
        """
        with open(transform_path, 'r') as f:
            data = yaml.safe_load(f)

        # 최상위에 'transformation' 키가 있는 경우와
        # 최상위 키 아래에 중첩된 경우를 모두 처리
        if 'transformation' in data:
            tf = data['transformation']
        else:
            outer = data[next(iter(data))]
            tf = outer.get('transformation', outer)

        t = np.array([
            float(tf['x']),
            float(tf['y']),
            float(tf['z']),
        ], dtype=np.float64)
        R = self._quat_to_rot(
            float(tf['qx']),
            float(tf['qy']),
            float(tf['qz']),
            float(tf['qw']),
        )

        self.get_logger().info(f'변환 행렬 로드 완료: {transform_path}')
        return R, t

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """쿼터니언 → 3×3 회전행렬."""
        return np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

    # ── 구독 콜백 ─────────────────────────────────

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """카메라 내부 파라미터 수신 (최초 1회)."""
        if self._intrinsics_ready:
            return
        # K 행렬: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]
        self._intrinsics_ready = True
        self.get_logger().info(
            f'Camera intrinsics 수신: '
            f'fx={self._fx:.1f} fy={self._fy:.1f} '
            f'cx={self._cx:.1f} cy={self._cy:.1f}'
        )

    def _target_class_callback(self, msg: String) -> None:
        """VLM이 결정한 탐지 대상 클래스 수신."""
        if self._target_class != msg.data:
            self.get_logger().info(
                f'target_class 변경: {self._target_class!r} → {msg.data!r}'
            )
            self._target_class = msg.data

    def _task_active_callback(self, msg: Bool) -> None:
        """태스크 활성 상태 수신. False가 되면 /object_pose 발행 중단."""
        if self._task_active and not msg.data:
            self.get_logger().info('태스크 비활성화 → /object_pose 발행 중단')
        self._task_active = msg.data

    def _image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """RGB + Depth 동기화 수신 → YOLO 탐지 → 3D 변환 → /object_pose 발행."""
        if not self._task_active:
            return

        if not self._intrinsics_ready:
            self.get_logger().warn(
                'Camera intrinsics 미수신, 스킵',
                throttle_duration_sec=5.0,
            )
            return

        if self._target_class is None:
            self.get_logger().warn(
                '/target_class 미수신, 스킵',
                throttle_duration_sec=5.0,
            )
            return

        rgb_np   = self._imgmsg_to_bgr(rgb_msg)
        depth_np = self._imgmsg_to_depth(depth_msg)

        bbox = self._detect(rgb_np, self._target_class)
        if bbox is None:
            return

        center_u, center_v = bbox

        depth_m = self._sample_depth(depth_np, center_u, center_v)
        if depth_m is None or depth_m <= 0.0:
            self.get_logger().warn(
                f'유효 Depth 없음 (u={center_u}, v={center_v}), 스킵',
                throttle_duration_sec=2.0,
            )
            return

        p_base = self._project_to_base(center_u, center_v, depth_m)

        pose_msg = PoseStamped()
        pose_msg.header.stamp    = rgb_msg.header.stamp
        pose_msg.header.frame_id = 'base_link'
        pose_msg.pose.position.x = float(p_base[0])
        pose_msg.pose.position.y = float(p_base[1])
        pose_msg.pose.position.z = float(p_base[2])
        pose_msg.pose.orientation.w = 1.0   # 방향 미사용 (위치만 필요)

        self.pose_pub.publish(pose_msg)

    # ── 처리 로직 ─────────────────────────────────

    def _detect(
        self,
        bgr: np.ndarray,
        target_class: str,
    ) -> Optional[Tuple[int, int]]:
        """
        YOLO로 target_class 탐지.
        여러 인스턴스 중 confidence 최고값 1개 선택.

        Returns:
            (center_u, center_v) 픽셀 좌표. 미탐지 시 None.
        """
        results  = self._model(bgr, verbose=False, conf=self._conf_thr)
        best_conf = -1.0
        best_center: Optional[Tuple[int, int]] = None

        for result in results:
            for box in result.boxes:
                cls_name = result.names[int(box.cls)]
                if cls_name.lower() != target_class.lower():
                    continue
                conf = float(box.conf)
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        return best_center

    def _sample_depth(
        self,
        depth_np: np.ndarray,
        u: int,
        v: int,
    ) -> Optional[float]:
        """
        바운딩박스 중심 주변 3×3 패치의 중앙값 반환 (미터).
        유효 픽셀 (> 0) 만 사용하여 노이즈에 강인하게 처리.
        """
        h, w = depth_np.shape[:2]
        k    = self.DEPTH_PATCH_HALF
        u0, u1 = max(0, u - k), min(w, u + k + 1)
        v0, v1 = max(0, v - k), min(h, v + k + 1)
        patch = depth_np[v0:v1, u0:u1].flatten()
        valid = patch[patch > 0.0]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _project_to_base(self, u: int, v: int, depth_m: float) -> np.ndarray:
        """
        픽셀 (u, v) + depth → 카메라 frame 3D → robot base_link frame 3D.

          카메라 frame:
            x_c = (u - cx) * d / fx
            y_c = (v - cy) * d / fy
            z_c = d

          robot base_link frame:
            p_base = R @ p_camera + t
        """
        x_c = (u - self._cx) * depth_m / self._fx
        y_c = (v - self._cy) * depth_m / self._fy
        z_c = depth_m
        p_camera = np.array([x_c, y_c, z_c], dtype=np.float64)
        return self._R @ p_camera + self._t

    # ── 이미지 변환 유틸 ──────────────────────────

    @staticmethod
    def _imgmsg_to_bgr(msg: Image) -> np.ndarray:
        """sensor_msgs/Image → numpy BGR (cv_bridge 미사용)."""
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            return arr[:, :, ::-1].copy()   # RGB → BGR
        if msg.encoding in ('bgr8',):
            return arr.copy()
        # mono8 또는 기타 단채널 포맷
        return np.stack([arr[:, :, 0]] * 3, axis=-1)

    @staticmethod
    def _imgmsg_to_depth(msg: Image) -> np.ndarray:
        """sensor_msgs/Image depth → numpy float32 (미터 단위)."""
        if msg.encoding == '32FC1':
            return np.frombuffer(
                msg.data, dtype=np.float32,
            ).reshape(msg.height, msg.width).copy()
        if msg.encoding == '16UC1':
            arr = np.frombuffer(
                msg.data, dtype=np.uint16,
            ).reshape(msg.height, msg.width).astype(np.float32)
            return arr / 1000.0   # mm → m
        raise ValueError(f'지원하지 않는 depth encoding: {msg.encoding}')


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = YOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
