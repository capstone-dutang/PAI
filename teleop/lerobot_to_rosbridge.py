"""SO-101 leader → EC2 rosbridge bridge.

로컬 PC의 SO-101 leader USB에서 joint 위치를 읽어 EC2의 rosbridge_server로
전송. EC2 Isaac Sim의 Action Graph가 /joint_command 토픽을 listen 중.

사용법:
    python lerobot_to_rosbridge.py --port COM5 --ec2-ip 13.124.xxx.xxx
"""
import argparse
import json
import math
import signal
import sys
import time

import websocket
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


# === Isaac Sim 측 토픽/joint 순서 ===
TOPIC_NAME = "/joint_command"
JOINT_ORDER = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# === Gripper 매핑 (Leader 0~100% → Isaac Sim rad) ===
# Isaac Sim SO-101 URDF gripper limit (이전 진단 결과): -10° ~ 100°
GRIPPER_CLOSED_DEG = -10.0   # leader 0%
GRIPPER_OPEN_DEG   = 100.0   # leader 100%


def gripper_map(percent: float) -> float:
    """Leader gripper 0~100% → Isaac Sim gripper degree."""
    percent = max(0.0, min(100.0, percent))
    deg = GRIPPER_CLOSED_DEG + (GRIPPER_OPEN_DEG - GRIPPER_CLOSED_DEG) * (percent / 100.0)
    return math.radians(deg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Leader arm COM port, e.g. COM5")
    parser.add_argument("--ec2-ip", required=True, help="EC2 public IP")
    parser.add_argument("--ec2-port", type=int, default=9090, help="rosbridge_server port")
    parser.add_argument("--id", default="capstone_leader", help="Calibration ID")
    parser.add_argument("--hz", type=float, default=30.0, help="Publish rate (Hz)")
    parser.add_argument("--use-degrees", action="store_true", default=True,
                        help="Read leader in degrees (default True)")
    args = parser.parse_args()

    # === 1. Leader 연결 ===
    print(f"[1/3] Leader 연결 시도: {args.port}")
    config = SO101LeaderConfig(port=args.port, id=args.id, use_degrees=args.use_degrees)
    leader = SO101Leader(config)
    leader.connect()  # 캘리브레이션 파일 없으면 calibrate() 자동 실행
    print(f"      ✅ Leader 연결됨, calibrated={leader.is_calibrated}")

    # === 2. rosbridge WebSocket 연결 ===
    ws_url = f"ws://{args.ec2_ip}:{args.ec2_port}"
    print(f"[2/3] rosbridge 연결 시도: {ws_url}")
    ws = websocket.create_connection(ws_url, timeout=10)
    ws.send(json.dumps({
        "op": "advertise",
        "topic": TOPIC_NAME,
        "type": "sensor_msgs/msg/JointState",
    }))
    print(f"      ✅ {TOPIC_NAME} 토픽 advertise 완료")

    # === 3. 메인 텔레옵 루프 ===
    print(f"[3/3] 텔레옵 시작 ({args.hz}Hz). Ctrl+C로 종료.")
    period = 1.0 / args.hz
    sent_count = 0
    start_time = time.time()

    def cleanup(*_):
        print("\n[!] 종료 신호 수신, 정리 중...")
        try:
            leader.disconnect()
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass
        elapsed = time.time() - start_time
        print(f"   총 {sent_count}개 메시지 전송, {elapsed:.1f}초 ({sent_count/elapsed:.1f} Hz 실측)")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    while True:
        loop_start = time.perf_counter()

        # Leader 읽기
        action = leader.get_action()
        # action = {"shoulder_pan.pos": deg, ..., "gripper.pos": percent}

        # joint 순서대로 변환
        positions = []
        for joint in JOINT_ORDER:
            val = action[f"{joint}.pos"]
            if joint == "gripper":
                positions.append(gripper_map(val))
            else:
                positions.append(math.radians(val))  # degree → rad

        # ROS2 JointState publish
        msg = {
            "op": "publish",
            "topic": TOPIC_NAME,
            "msg": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "name": JOINT_ORDER,
                "position": positions,
                "velocity": [],
                "effort": [],
            },
        }
        ws.send(json.dumps(msg))
        sent_count += 1

        # 디버그 출력 (1초마다 1번)
        if sent_count % int(args.hz) == 0:
            sp = math.degrees(positions[0])
            sl = math.degrees(positions[1])
            print(f"  [{sent_count:5d}] shoulder_pan={sp:+6.1f}°  shoulder_lift={sl:+6.1f}°  "
                  f"gripper={action['gripper.pos']:5.1f}%")

        # Rate 유지
        elapsed = time.perf_counter() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)


if __name__ == "__main__":
    main()