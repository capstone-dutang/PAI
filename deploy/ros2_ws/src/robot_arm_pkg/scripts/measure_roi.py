#!/usr/bin/env python3
"""
measure_roi — verify_result 의 depth ROI 실측 보조 스크립트 (두 모드).

`/camera/depth` 를 구독해 ROI 의 depth 를 측정한다. 두 검증 정책을 모두 지원한다:

  --mode destination  (현재 운영 기본)
      verify_result 의 depth **place(목적지 배치)** 검증용. 카메라·목적지 고정
      전제로 목적지는 이미지의 고정 픽셀 영역이라, 픽셀 ROI + baseline depth 로
      destination_roi.yaml 을 만든다. intrinsics 불필요.

  --mode grasp        (미래 grasp 검증용)
      verify_result 의 depth **grasp(집기)** 검증용. 엔드이펙터 근처 ROI 를
      camera_intrinsics 의 K 로 카메라 좌표에 역투영해 미터 3D 박스(grasp_roi.yaml)
      를 만든다. intrinsics 필요.

검증 정책 배경: 실물 ACT 에피소드는 home_pose 복귀로 끝나 verify 시점 그리퍼가
비어 있으므로, 현재 운영 검증은 그리퍼가 아니라 목적지(통/목표 영역) depth 변화로
place 를 본다. grasp 시점 검증은 미래 확장이다 (README "grasp 검증 정책" 절).

────────────────────────────────────────────────────────────────────────────
사용법
────────────────────────────────────────────────────────────────────────────
destination 모드 — GUI (Jetson, DISPLAY 필요):

    export DISPLAY=:1
    python3 scripts/measure_roi.py --mode destination --domain recycling \
        --out ~/calibration/destination_roi.yaml

    [윈도우 조작]
      마우스 드래그 : ROI 사각형 지정
      b            : 현재 ROI 를 '빈 목적지' baseline 으로 추가 (목적지 1개 등록)
      o            : 현재 ROI 의 '물체 놓은 후' median 측정 → threshold 힌트 출력
      s            : 등록된 목적지로 destination_roi.yaml 출력 + --out 저장
      r            : 등록 초기화
      q / ESC      : 종료

    절차: ① 빈 목적지 상태에서 각 통/영역을 드래그 → 'b' (여러 개 반복)
          ② (선택) 물체 놓은 상태에서 'o' 로 threshold 힌트 확인  ③ 's'.
    ※ name 은 자동(dest_0, dest_1 …) — 출력 yaml 에서 can_bin/target_area 로 rename.

destination 모드 — CLI(headless), 단일 목적지:

    python3 scripts/measure_roi.py --mode destination --domain recycling \
        --name can_bin --roi 120 200 220 300

grasp 모드 (미래) — GUI:

    export DISPLAY=:1
    python3 scripts/measure_roi.py --mode grasp \
        --intrinsics ~/calibration/camera_intrinsics.yaml \
        --out ~/calibration/grasp_roi.yaml
    #   드래그=ROI → g(물체 든 상태) → e(빈손) → s(저장)

────────────────────────────────────────────────────────────────────────────
의존성: rclpy, sensor_msgs, numpy, pyyaml (+ GUI 모드는 opencv-python).
  cv_bridge 는 사용하지 않는다 (NumPy 2.x ABI 비호환 — utils/ros_image.py 참조).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

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
# grasp 박스 산출 시 outlier 절단용 백분위수.
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


def load_intrinsics(path: str) -> Tuple[float, float, float, float]:
    """camera_intrinsics.yaml 의 K → (fx, fy, cx, cy). grasp 모드 전용."""
    with open(os.path.expanduser(path), 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    k = d['k']  # [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


# ── 공통: ROI median depth (픽셀, intrinsics 비의존) ──────────────────────

def roi_median_depth(
    depth_m: np.ndarray, roi_px: Tuple[int, int, int, int],
) -> Optional[Tuple[float, int, float]]:
    """
    픽셀 ROI 내 유효 depth median (m). 반환 (median, n_valid, valid_ratio).
    유효 픽셀 없거나 ROI 가 너무 작으면 None.
    """
    h, w = depth_m.shape
    x0, y0, x1, y1 = (int(v) for v in roi_px)
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    sub = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(sub) & (sub > Z_VALID_MIN) & (sub < Z_VALID_MAX)
    if not bool(np.any(valid)):
        return None
    return float(np.median(sub[valid])), int(valid.sum()), float(valid.mean())


# ── grasp 모드: 카메라 좌표 역투영 박스 ───────────────────────────────────

def compute_roi_box(
    depth_m: np.ndarray,
    roi_px: Tuple[int, int, int, int],
    intr: Tuple[float, float, float, float],
) -> Optional[dict]:
    """
    grasp 모드 — ROI 픽셀 박스를 카메라 좌표로 역투영해 미터 3D 박스를 산출.

    역투영: X = (u-cx)·Z/fx,  Y = (v-cy)·Z/fy,  Z = depth.
    무효 depth 제거 후, 좌표 박스는 [PCT_LO, PCT_HI] 백분위수로 절단.

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


def build_grasp_yaml(
    box: dict, threshold: float, grasp_z: Optional[float], empty_z: Optional[float],
) -> str:
    """grasp_roi.yaml 본문 문자열 (미터 3D 박스). grasp 모드 전용."""
    gz = f'{grasp_z:.4f}' if grasp_z is not None else 'N/A'
    ez = f'{empty_z:.4f}' if empty_z is not None else 'N/A'
    xr, yr, zr = box['x_range'], box['y_range'], box['z_range']
    return (
        "# grasp_roi.yaml — measure_roi.py (--mode grasp) 로 실측한 값.\n"
        "# verify_result 의 depth grasp 보조 검증용 ROI (camera frame). 미래 확장.\n"
        "#\n"
        f"# 측정 메타: ROI 픽셀={box['roi_px']}, 유효픽셀={box['n_valid']}\n"
        f"#           grasp z_median={gz} m, empty z_median={ez} m\n"
        "\n"
        "roi:\n"
        "  frame: camera           # 역투영 기준 — camera 고정\n"
        f"  x_range: [{xr[0]:.4f}, {xr[1]:.4f}]  # m\n"
        f"  y_range: [{yr[0]:.4f}, {yr[1]:.4f}]  # m\n"
        f"  z_range: [{zr[0]:.4f}, {zr[1]:.4f}]  # m\n"
        "\n"
        f"depth_change_threshold: {threshold:.4f}  # m\n"
    )


def build_destination_yaml(
    domain: str, dests: List[dict], threshold: float, depth_scale: float,
) -> str:
    """destination_roi.yaml 본문 문자열 (픽셀 ROI + baseline). destination 모드."""
    head = (
        "# destination_roi.yaml — measure_roi.py (--mode destination) 로 실측한 값.\n"
        "# verify_result 의 depth place 보조 검증용 목적지 ROI (픽셀, intrinsics 비의존).\n"
        "# name 은 자동(dest_N) — can_bin / target_area 등으로 rename 권장.\n"
        "#\n"
        "# 운영 배포: ~/calibration/destination_roi.yaml 로 복사 후 agent_node 의\n"
        "#           destination_roi 파라미터가 가리키도록 설정.\n"
        "\n"
        f"depth_scale: {depth_scale}\n"
        f"depth_change_threshold: {threshold:.4f}  # m, baseline 보다 이만큼 작아지면 물체 있음\n"
        "destinations:\n"
        f"  {domain}:\n"
    )
    rows = ''
    for d in dests:
        x = d['roi_px']
        rows += (
            f"    - {{name: {d['name']}, "
            f"roi_px: [{x[0]}, {x[1]}, {x[2]}, {x[3]}], "
            f"baseline_depth: {d['baseline_depth']:.4f}}}\n"
        )
    return head + rows


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
        super().__init__('measure_roi')
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


# ── GUI 공통 ──────────────────────────────────────────────────────────────

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


def _gui_preamble(node: DepthGrabber):
    """GUI 공통 사전 점검 + 윈도우/마우스 세팅. 실패 시 (None, None, rc)."""
    if cv2 is None:
        print('ERROR: GUI 모드는 opencv-python 필요. '
              'pip install opencv-python 또는 --roi 로 CLI 모드 사용.',
              file=sys.stderr)
        return None, None, 1
    if not wait_for_depth(node):
        print('ERROR: depth 프레임 수신 실패 (토픽/카메라 확인).', file=sys.stderr)
        return None, None, 1
    win = 'measure_roi'
    cv2.namedWindow(win)
    mouse = _RoiMouse()
    cv2.setMouseCallback(win, mouse.on_mouse)
    return win, mouse, 0


def _draw_hud(vis, hud: str, stat: str) -> None:
    cv2.putText(vis, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, stat, (6, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 255, 0), 1, cv2.LINE_AA)


# ── destination 모드 ──────────────────────────────────────────────────────

def run_dest_cli(node: DepthGrabber, roi_px, args) -> int:
    if not wait_for_depth(node):
        print('ERROR: depth 프레임 수신 실패 (토픽/카메라 확인).', file=sys.stderr)
        return 1
    res = roi_median_depth(node.latest, roi_px)
    if res is None:
        print('ERROR: ROI 내 유효 depth 픽셀이 없습니다.', file=sys.stderr)
        return 1
    med, n, ratio = res
    name = args.name or 'dest_0'
    print(f"[destination] {name} 유효픽셀={n} ({ratio*100:.0f}%) baseline={med:.4f} m")
    dests = [{'name': name, 'roi_px': tuple(int(v) for v in roi_px), 'baseline_depth': med}]
    print()
    txt = build_destination_yaml(args.domain, dests, args.threshold, args.depth_scale)
    print(txt)
    if args.out:
        _save_text(txt, args.out)
    return 0


def run_dest_gui(node: DepthGrabber, args) -> int:
    win, mouse, rc = _gui_preamble(node)
    if rc != 0:
        return rc

    dests: List[dict] = []
    threshold = args.threshold  # 'o' 표본으로 갱신 가능
    hud = 'drag=ROI  b=baseline  o=object(thr)  s=save  r=reset  q=quit'

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.005)
        if node.latest is None:
            continue
        vis = _colorize(node.latest)
        rect = mouse.live_rect()
        if rect:
            cv2.rectangle(vis, rect[:2], rect[2:], (255, 255, 255), 1)
        # 등록된 목적지 박스 표시
        for d in dests:
            x = d['roi_px']
            cv2.rectangle(vis, (x[0], x[1]), (x[2], x[3]), (0, 200, 255), 1)
        _draw_hud(vis, hud, f'dests={len(dests)}  threshold={threshold:.3f}')
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            mouse.roi = None
            dests = []
            threshold = args.threshold
            print('[reset]')
        elif key == ord('b'):
            if not rect:
                print('  ROI 를 먼저 드래그하세요.')
                continue
            res = roi_median_depth(node.latest, rect)
            if res is None:
                print('  ROI 내 유효 depth 없음.')
                continue
            med, n, ratio = res
            name = f'dest_{len(dests)}'
            dests.append({'name': name, 'roi_px': tuple(int(v) for v in rect),
                          'baseline_depth': med})
            print(f"[baseline] {name} 유효픽셀={n} ({ratio*100:.0f}%) baseline={med:.4f} m")
        elif key == ord('o'):
            if not rect:
                print('  ROI 를 먼저 드래그하세요.')
                continue
            res = roi_median_depth(node.latest, rect)
            if res is None:
                print('  ROI 내 유효 depth 없음.')
                continue
            obj_med = res[0]
            if dests:
                base = dests[-1]['baseline_depth']
                hint = abs(base - obj_med) / 2.0
                threshold = hint
                print(f"[object] median={obj_med:.4f} m  (직전 baseline={base:.4f}) "
                      f"-> threshold 힌트={hint:.4f} m")
            else:
                print(f"[object] median={obj_med:.4f} m  (baseline 먼저 'b' 로 등록)")
        elif key == ord('s'):
            if not dests:
                print('  먼저 baseline(b)을 1개 이상 등록하세요.')
                continue
            txt = build_destination_yaml(args.domain, dests, threshold, args.depth_scale)
            print()
            print(txt)
            if args.out:
                _save_text(txt, args.out)

    cv2.destroyAllWindows()
    return 0


# ── grasp 모드 (미래) ─────────────────────────────────────────────────────

def run_grasp_cli(node: DepthGrabber, roi_px, intr, args) -> int:
    if not wait_for_depth(node):
        print('ERROR: depth 프레임 수신 실패 (토픽/카메라 확인).', file=sys.stderr)
        return 1
    box = compute_roi_box(node.latest, roi_px, intr)
    if box is None:
        print('ERROR: ROI 내 유효 depth 픽셀이 없습니다.', file=sys.stderr)
        return 1
    _print_box('measure', box)
    print()
    txt = build_grasp_yaml(box, args.threshold, box['z_median'], None)
    print(txt)
    if args.out:
        _save_text(txt, args.out)
    return 0


def run_grasp_gui(node: DepthGrabber, intr, args) -> int:
    win, mouse, rc = _gui_preamble(node)
    if rc != 0:
        return rc

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
        stat = (f'grasp z={grasp_box["z_median"]:.3f}' if grasp_box else 'grasp -')
        stat += f'   empty z={empty_z:.3f}' if empty_z is not None else '   empty -'
        _draw_hud(vis, hud, stat)
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
            box = compute_roi_box(node.latest, rect, intr)
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
            txt = build_grasp_yaml(grasp_box, thr, grasp_box['z_median'], empty_z)
            print()
            print(txt)
            if args.out:
                _save_text(txt, args.out)

    cv2.destroyAllWindows()
    return 0


# ── 저장 ──────────────────────────────────────────────────────────────────

def _save_text(text: str, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'저장: {out_path}')


# ── main ──────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='verify_result depth ROI 실측 보조 스크립트')
    p.add_argument('--mode', choices=['destination', 'grasp'], default='destination',
                   help='destination(place, 현재 운영) | grasp(집기, 미래)')
    p.add_argument('--intrinsics', default=None,
                   help='camera_intrinsics.yaml 경로 (grasp 모드 필수)')
    p.add_argument('--domain', default='recycling',
                   help='destination 모드: 출력 yaml 의 도메인 키')
    p.add_argument('--name', default=None,
                   help='destination CLI 모드: 단일 목적지 name (기본 dest_0)')
    p.add_argument('--topic', default='/camera/depth', help='depth 토픽')
    p.add_argument('--depth-scale', type=float, default=0.001,
                   help='16UC1 raw→m 스케일 (RealSense 기본 0.001)')
    p.add_argument('--out', default=None, help='출력 yaml 저장 경로')
    p.add_argument('--threshold', type=float, default=0.02,
                   help='depth_change_threshold 기본값 (m). destination=baseline 대비, '
                        'grasp=empty 표본 없을 때 사용.')
    p.add_argument('--roi', type=int, nargs=4, metavar=('U0', 'V0', 'U1', 'V1'),
                   default=None, help='CLI 모드: 픽셀 ROI 직접 지정 (GUI 생략)')
    args = p.parse_args(argv)

    intr = None
    if args.mode == 'grasp':
        if not args.intrinsics:
            print('ERROR: grasp 모드는 --intrinsics 가 필요합니다.', file=sys.stderr)
            return 2
        intr = load_intrinsics(args.intrinsics)
        print(f'intrinsics: fx={intr[0]:.2f} fy={intr[1]:.2f} '
              f'cx={intr[2]:.2f} cy={intr[3]:.2f}')

    rclpy.init()
    node = DepthGrabber(args.topic, args.depth_scale)
    try:
        if args.mode == 'destination':
            if args.roi is not None:
                return run_dest_cli(node, tuple(args.roi), args)
            return run_dest_gui(node, args)
        # grasp
        if args.roi is not None:
            return run_grasp_cli(node, tuple(args.roi), intr, args)
        return run_grasp_gui(node, intr, args)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
