"""
robot_arm_pkg.utils.ros_image — sensor_msgs/Image ↔ numpy 변환 유틸.

cv_bridge 는 ROS Humble 의 apt 빌드가 NumPy 1.x ABI 와만 호환되어 NumPy 2.x
환경에서 `_ARRAY_API not found` 로 import/호출이 깨진다. 본 모듈은 NumPy 만
사용해 직접 변환하여 NumPy 1.x / 2.x 양쪽 모두에서 동작한다.

지원 인코딩: 'rgb8', 'bgr8'.
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
