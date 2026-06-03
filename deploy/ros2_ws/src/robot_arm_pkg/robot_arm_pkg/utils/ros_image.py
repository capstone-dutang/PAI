"""
robot_arm_pkg.utils.ros_image — sensor_msgs/Image ↔ numpy 변환 유틸.

cv_bridge 는 ROS Humble 의 apt 빌드가 NumPy 1.x ABI 와만 호환되어 NumPy 2.x
환경에서 `_ARRAY_API not found` 로 import/호출이 깨진다. 본 모듈은 NumPy 만
사용해 직접 변환하여 NumPy 1.x / 2.x 양쪽 모두에서 동작한다.

지원 인코딩: RGB 'rgb8'/'bgr8', depth '16UC1'/'32FC1'.
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


def image_msg_to_rgb_array(msg: Image) -> np.ndarray:
    """
    sensor_msgs/Image → (H, W, 3) uint8 RGB numpy.

    bgr8 입력은 채널을 뒤집어 rgb8 로 통일한다.
    호출 측이 안전하게 저장/수정할 수 있도록 contiguous copy 를 반환한다.

    Raises:
        ValueError: 지원하지 않는 encoding 이거나 data 크기가 H*W*3 과 불일치.
    """
    enc = (msg.encoding or '').lower()
    if enc not in ('rgb8', 'bgr8'):
        raise ValueError(
            f'image_msg_to_rgb_array: 지원하지 않는 encoding {msg.encoding!r}. '
            "'rgb8' 또는 'bgr8' 만 지원."
        )

    expected = msg.height * msg.width * 3
    if len(msg.data) != expected:
        raise ValueError(
            f'image_msg_to_rgb_array: data 크기 {len(msg.data)} != '
            f'H*W*3 = {expected} (encoding={msg.encoding}).'
        )

    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if enc == 'bgr8':
        arr = arr[..., ::-1]
    return np.ascontiguousarray(arr)


def depth_msg_to_meters(msg: Image, depth_scale: float = 0.001) -> np.ndarray:
    """
    sensor_msgs/Image(depth) → (H, W) float32, 단위 m.

    16UC1/mono16 (RealSense aligned-depth 기본): uint16 raw × depth_scale.
    32FC1: 이미 m 단위 float (depth_scale 무시).

    cv_bridge 미사용 — NumPy 만으로 직접 변환 (본 모듈 상단 설명 참조).
    byte order 는 msg.is_bigendian 을 따른다.

    Raises:
        ValueError: 지원하지 않는 encoding 이거나 data 크기가 H*W*bytes 와 불일치.
    """
    enc = (msg.encoding or '').lower()
    h, w = msg.height, msg.width

    if enc in ('16uc1', 'mono16'):
        dtype = '>u2' if msg.is_bigendian else '<u2'
        expected = h * w * 2
        if len(msg.data) != expected:
            raise ValueError(
                f'depth_msg_to_meters: data 크기 {len(msg.data)} != H*W*2 = {expected} '
                f'(encoding={msg.encoding}).'
            )
        raw = np.frombuffer(msg.data, dtype=dtype).reshape(h, w)
        return raw.astype(np.float32) * float(depth_scale)

    if enc == '32fc1':
        dtype = '>f4' if msg.is_bigendian else '<f4'
        expected = h * w * 4
        if len(msg.data) != expected:
            raise ValueError(
                f'depth_msg_to_meters: data 크기 {len(msg.data)} != H*W*4 = {expected} '
                f'(encoding={msg.encoding}).'
            )
        return np.frombuffer(msg.data, dtype=dtype).reshape(h, w).astype(np.float32)

    raise ValueError(
        f"depth_msg_to_meters: 지원하지 않는 encoding {msg.encoding!r}. "
        "'16UC1' 또는 '32FC1' 만 지원."
    )
