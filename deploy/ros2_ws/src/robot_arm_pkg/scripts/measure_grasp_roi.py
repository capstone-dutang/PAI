#!/usr/bin/env python3
"""
measure_grasp_roi — grasp_roi.yaml 실측 보조 스크립트.

`/camera/depth` 를 구독해 ROI(관심 영역) 내 depth 를 카메라 좌표로 역투영하고,
grasp 성공/빈손 두 상태의 depth 변화를 측정해 deploy/config/grasp_roi.yaml 의
{roi.x/y/z_range, depth_change_threshold} 값을 산출한다.

camera_intrinsics.yaml 의 K 로 픽셀을 카메라 좌표로 역투영하므로 결과 ROI 의
frame 은 항상 'camera' 다. base_link 기준이 필요하면 camera_robot_transform 을
별도로 적용할 것 (README 미확정 항목 #1).

────────────────────────────────────────────────────────────────────────────
사용법
────────────────────────────────────────────────────────────────────────────
GUI 모드 (Jetson, DISPLAY 필요 — 캘리브레이션과 동일하게 :1 사용):

    export DISPLAY=:1
    python3 scripts/measure_grasp_roi.py \
        --intrinsics ~/calibration/camera_intrinsics.yaml \
        --out ~/calibration/grasp_roi.yaml

    [윈도우 조작]
      마우스 드래그 : ROI 사각형 지정
      g            : 현재 ROI 를 'grasp 성공'(물체 든 상태) 표본으로 기록
      e            : 현재 ROI 를 '빈손'(배경) 표본으로 기록
      s            : 누적 표본으로 grasp_roi.yaml 계산 → 콘솔 출력 + --out 저장
      r            : ROI / 표본 초기화
      q / ESC      : 종료

    절차: ① 로봇이 물체를 든 verify 자세에서 ROI 드래그 → 'g'
          ② 빈손 상태에서 'e'  ③ 's' 로 저장.

CLI(headless) 모드 — 픽셀 ROI 를 직접 지정, GUI 없이 단발 측정:

    python3 scripts/measure_grasp_roi.py \
        --intrinsics ~/calibration/camera_intrinsics.yaml \
        --roi 280 200 360 280            # u0 v0 u1 v1 (픽셀)

    grasp 성공/빈손 두 번 실행해 출력된 z_median 차이로 threshold 를 잡는다.

────────────────────────────────────────────────────────────────────────────
의존성: rclpy, sensor_msgs, numpy, pyyaml (+ GUI 모드는 opencv-python).
  cv_bridge 는 사용하지 않는다 (NumPy 2.x ABI 비호환 — utils/ros_image.py 참조).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image

try:
    import cv2  # GUI 모드에서만 필요
except ImportError:  # pragma: no cover
    cv2 = None

# 유효 depth 로 인정할 범위 (m). 0(=무효)·과근접·과원거리 픽셀 제거.
Z_VALID_MIN = 0.05
Z_VALID_MAX = 2.0
# ROI 박스 산출 시 outlier 절단용 백분위수.
PCT_LO, PCT_HI = 5.0, 95.0
# 역투영 좌표에 더하는 여유 (m).
BOX_MARGIN = 0.01


# ── depth Image → numpy(m) ────────────────────────────────────────────────

def depth_msg_to_meters(msg: Image, depth_scale: float) -> np.ndarray:
    """
    sensor_msgs/Image(depth) → (H, W) float32, 단위 m.

    16UC1/mono16 (RealSense 기본): uint16 raw × depth_scale.
    32FC1: 이미 m 단위 float.

    Raises:
        ValueError: 지원하지 않는 encoding 또는 data 크기 불일치.
    """
    enc = (msg.encoding or '').lower()
    h, w = msg.height, msg.width

    if enc in ('16uc1', 'mono16'):
        dtype = '>u2' if msg.is_bigendian else '<u2'
        expected = h * w * 2
        if len(msg.data) != expected:
            raise ValueError(
                f'depth data 크기 {len(msg.data)} != H*W*2 = {expected} '
                f'(encoding={msg.encoding}).'
            )
        raw = np.frombuffer(msg.data, dtype=dtype).reshape(h, w)
        return raw.astype(np.float32) * float(depth_scale)

    if enc == '32fc1':
        dtype = '>f4' if msg.is_bigendian else '<f4'
        expected = h * w * 4
        if len(msg.data) != expected:
            raise ValueError(
                f'depth data 크기 {len(msg.data)} != H*W*4 = {expected} '
                f'(encoding={msg.encoding}).'
            )
        return np.frombuffer(msg.data, dtype=dtype).reshape(h, w).astype(np.float32)

    raise ValueError(
        f"depth_msg_to_meters: 지원하지 않는 encoding {msg.encoding!r}. "
        "'16UC1' 또는 '32FC1' 만 지원."
    )


# ── intrinsics 로드 ───────────────────────────────────────────────────────

def load_intrinsics(path: str) -> Tuple[float, float, float, float]:
    """camera_intrinsics.yaml 의 K → (fx, fy, cx, cy)."""
    with open(os.path.expanduser(path), 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    k = d['k']  # [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


# ── ROI 통계 / 역투영 ─────────────────────────────────────────────────────

def compute_roi_stats(
    depth_m: np.ndarray,
    roi_px: Tuple[int, int, int, int],
    intr: Tuple[float, float, float, float],
) -> Optional[dict]:
    """
    ROI 픽셀 박스를 카메라 좌표로 역투영해 박스·depth 통계를 산출.

    역투영: X = (u-cx)·Z/fx,  Y = (v-cy)·Z/fy,  Z = depth.
    무효 depth(0/inf/범위 밖) 제거 후, 좌표 박스는 [PCT_LO, PCT_HI] 백분위수로
    절단해 outlier 영향을 줄인다.

    Returns:
        통계 dict, 또는 유효 픽셀이 없으면 None.
    """
    fx, fy, cx, cy = intr
    h, w = depth_m.shape
    u0, v0, u1, v1 = roi_px
    u0, u1 = max(0, min(u0, u1)), min(w, max(u0, u1))
    v0, v1 = max(0, min(v0, v1)), min(h, max(v0, v1))
    if u1 - u0 < 2 or v1 - v0 < 2:
        return None

    z = depth_m[v0:v1, u0:u1]
    uu, vv = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))

    valid = np.isfinite(z) & (z > Z_VALID_MIN) & (z < Z_VALID_MAX)
    if not np.any(valid):
        return None

    z_v = z[valid]
    x_v = (uu[valid] - cx) * z_v / fx
    y_v = (vv[valid] - cy) * z_v / fy

    def _range(a: np.ndarray) -> Tuple[float, float]:
        lo, hi = np.percentile(a, [PCT_LO, PCT_HI])
        return float(lo - BOX_MARGIN), float(hi + BOX_MARGIN)

    return {
        'roi_px': (u0, v0, u1, v1),
        'n_valid': int(z_v.size),
        'valid_ratio': float(z_v.size / z.size),
        'x_range': list(_range(x_v)),
        'y_range': list(_range(y_v)),
        'z_range': list(_range(z_v)),
        'z_median': float(np.median(z_v)),
        'z_mean': float(np.mean(z_v)),
    }


def build_yaml_text(
    box: dict,
    threshold: float,
    grasp_z: Optional[float],
    empty_z: Optional[float],
) -> str:
    """grasp_roi.yaml 본문 문자열 생성 (템플릿 주석 + 측정 메타 포함)."""
    gz = f'{grasp_z:.4f}' if grasp_z is not None else 'N/A'
    ez = f'{empty_z:.4f}' if empty_z is not None else 'N/A'
    xr, yr, zr = box['x_range'], box['y_range'], box['z_range']
    return (
        "# grasp_roi.yaml — measure_grasp_roi.py 로 실측한 값.\n"
        "# verify_result 의 depth 기반 grasp 보조 검증용 ROI (camera frame).\n"
        "#\n"
        f"# 측정 메타: ROI 픽셀={box['roi_px']}, 유효픽셀={box['n_valid']}\n"
        f"#           grasp z_median={gz} m, empty z_median={ez} m\n"
        "#\n"
        "# 운영 배포: ~/calibration/grasp_roi.yaml 로 복사 후 agent_node 의\n"
        "#           grasp_roi 파라미터가 가리키도록 설정.\n"
        "\n"
        "roi:\n"
        "  frame: camera           # 역투영 기준 — camera 고정\n"
        f"  x_range: [{xr[0]:.4f}, {xr[1]:.4f}]  # m\n"
        f"  y_range: [{yr[0]:.4f}, {yr[1]:.4f}]  # m\n"
        f"  z_range: [{zr[0]:.4f}, {zr[1]:.4f}]  # m\n"
        "\n"
        f"depth_change_threshold: {threshold:.4f}  # m\n"
    )


def _print_box(tag: str, box: dict) -> None:
    print(
        f"[{tag}] 유효픽셀={box['n_valid']} ({box['valid_ratio']*100:.0f}%)  "
        f"x={box['x_range']}  y={box['y_range']}  z={box['z_range']}  "
        f"z_median={box['z_median']:.4f} m"
    )


# ── 구독 노드 ─────────────────────────────────────────────────────────────

class DepthGrabber(Node):
    """`/camera/depth` latest-only 캐시 (agent_node 와 동일 QoS)."""

    def __init__(self, topic: str, depth_scale: float):
        super().__init__('measure_grasp_roi')
        self._depth_scale = depth_scale
        self.latest: Optional[np.ndarray] = None
        self._err_logged = False

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, topic, self._cb, qos)
        self.get_logger().info(f'구독 시작: {topic}')

    def _cb(self, msg: Image) -> None:
        try:
            self.latest = depth_msg_to_meters(msg, self._depth_scale)
        except ValueError as e:
            if not self._err_logged:
                self.get_logger().error(str(e))
                self._err_logged = True


def wait_for_depth(node: DepthGrabber, timeout_s: float = 10.0) -> bool:
    """첫 depth 프레임 수신까지 spin (timeout 초)."""
    end = node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
    while rclpy.ok() and node.latest is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now().nanoseconds > end:
            return False
    return node.latest is not None


# ── CLI(headless) 모드 ────────────────────────────────────────────────────

def run_cli(node: DepthGrabber, roi_px, intr, args) -> int:
    if not wait_for_depth(node):
        print('ERROR: depth 프레임 수신 실패 (토픽/카메라 확인).', file=sys.stderr)
        return 1
    box = compute_roi_stats(node.latest, roi_px, intr)
    if box is None:
        print('ERROR: ROI 내 유효 depth 픽셀이 없습니다.', file=sys.stderr)
        return 1
    _print_box('measure', box)
    print()
    print(build_yaml_text(box, args.threshold, box['z_median'], None))
    if args.out:
        _save(box, args.threshold, box['z_median'], None, args.out)
    return 0


# ── GUI 모드 ──────────────────────────────────────────────────────────────

class _RoiMouse:
    """드래그로 ROI 사각형 지정하는 마우스 상태."""

    def __init__(self):
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self._anchor: Optional[Tuple[int, int]] = None
        self._drag: Optional[Tuple[int, int]] = None

    def on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._anchor = (x, y)
            self._drag = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._anchor is not None:
            self._drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._anchor is not None:
            ax, ay = self._anchor
            self.roi = (min(ax, x), min(ay, y), max(ax, x), max(ay, y))
            self._anchor = self._drag = None

    def live_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if self._anchor and self._drag:
            ax, ay = self._anchor
            dx, dy = self._drag
            return (min(ax, dx), min(ay, dy), max(ax, dx), max(ay, dy))
        return self.roi


def _colorize(depth_m: np.ndarray) -> np.ndarray:
    """depth(m) → 시각화용 BGR (근거리=빨강). 무효=검정."""
    valid = np.isfinite(depth_m) & (depth_m > Z_VALID_MIN) & (depth_m < Z_VALID_MAX)
    norm = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth_m[valid], [2, 98])
        if hi - lo > 1e-6:
            scaled = np.clip((depth_m - lo) / (hi - lo), 0, 1)
            norm = (scaled * 255).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    vis[~valid] = (0, 0, 0)
    return vis


def run_gui(node: DepthGrabber, intr, args) -> int:
    if cv2 is None:
        print('ERROR: GUI 모드는 opencv-python 필요. '
              'pip install opencv-python 또는 --roi 로 CLI 모드 사용.',
              file=sys.stderr)
        return 1
    if not wait_for_depth(node):
        print('ERROR: depth 프레임 수신 실패 (토픽/카메라 확인).', file=sys.stderr)
        return 1

    win = 'measure_grasp_roi'
    cv2.namedWindow(win)
    mouse = _RoiMouse()
    cv2.setMouseCallback(win, mouse.on_mouse)

    grasp_box: Optional[dict] = None
    empty_z: Optional[float] = None
    hud = 'drag=ROI  g=grasp  e=empty  s=save  r=reset  q=quit'

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.005)
        if node.latest is None:
            continue

        vis = _colorize(node.latest)
        rect = mouse.live_rect()
        if rect:
            cv2.rectangle(vis, rect[:2], rect[2:], (255, 255, 255), 1)
        cv2.putText(vis, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        stat = (f'grasp z={grasp_box["z_median"]:.3f}' if grasp_box else 'grasp -')
        stat += f'   empty z={empty_z:.3f}' if empty_z is not None else '   empty -'
        cv2.putText(vis, stat, (6, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            mouse.roi = None
            grasp_box = empty_z = None
            print('[reset]')
        elif key in (ord('g'), ord('e')):
            if not rect:
                print('  ROI 를 먼저 드래그하세요.')
                continue
            box = compute_roi_stats(node.latest, rect, intr)
            if box is None:
                print('  ROI 내 유효 depth 없음.')
                continue
            if key == ord('g'):
                grasp_box = box
                _print_box('grasp', box)
            else:
                empty_z = box['z_median']
                _print_box('empty', box)
        elif key == ord('s'):
            if grasp_box is None:
                print('  먼저 grasp 표본(g)을 기록하세요.')
                continue
            thr = (abs(grasp_box['z_median'] - empty_z) / 2.0
                   if empty_z is not None else args.threshold)
            if empty_z is None:
                print(f'  empty 표본 없음 → threshold 기본값 {thr} 사용.')
            print()
            print(build_yaml_text(grasp_box, thr, grasp_box['z_median'], empty_z))
            if args.out:
                _save(grasp_box, thr, grasp_box['z_median'], empty_z, args.out)

    cv2.destroyAllWindows()
    return 0


def _save(box, threshold, grasp_z, empty_z, out_path) -> None:
    out_path = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(build_yaml_text(box, threshold, grasp_z, empty_z))
    print(f'저장: {out_path}')


# ── main ──────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='grasp_roi.yaml 실측 보조 스크립트')
    p.add_argument('--intrinsics', required=True,
                   help='camera_intrinsics.yaml 경로')
    p.add_argument('--topic', default='/camera/depth', help='depth 토픽')
    p.add_argument('--depth-scale', type=float, default=0.001,
                   help='16UC1 raw→m 스케일 (RealSense 기본 0.001)')
    p.add_argument('--out', default=None, help='grasp_roi.yaml 저장 경로')
    p.add_argument('--threshold', type=float, default=0.02,
                   help='empty 표본 없을 때 사용할 depth_change_threshold 기본값 (m)')
    p.add_argument('--roi', type=int, nargs=4, metavar=('U0', 'V0', 'U1', 'V1'),
                   default=None, help='CLI 모드: 픽셀 ROI 직접 지정 (GUI 생략)')
    args = p.parse_args(argv)

    intr = load_intrinsics(args.intrinsics)
    print(f'intrinsics: fx={intr[0]:.2f} fy={intr[1]:.2f} '
          f'cx={intr[2]:.2f} cy={intr[3]:.2f}')

    rclpy.init()
    node = DepthGrabber(args.topic, args.depth_scale)
    try:
        if args.roi is not None:
            return run_cli(node, tuple(args.roi), intr, args)
        return run_gui(node, intr, args)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())