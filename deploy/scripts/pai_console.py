#!/usr/bin/env python3
"""
pai_console.py — PAI 통합 런처 + 대화형 콘솔 (Jetson Orin Nano 운영)

한 번 실행으로 다음을 자동화한다:
  1) 사전 점검 — ~/ros2_ws 빌드 / 모델·캘리브 존재 / llama-server health / robot 포트
  2) 노드 자동 기동 (2026-06-04 실기에서 검증된 순서·GPU 배분 옵션 자동):
       realsense 드라이버 → camera_node → llama_server_manager
       → embedding_node(device:=cpu) → act_node(device:=cuda + expandable_segments)
       → robot_controller_node → agent_node
     각 노드 stdout/stderr 는 ~/ros2_logs/<node>.log 로 분리.
  3) 토픽 흐름 / 처리 절차 실시간 출력 — /selected_domain(LoRA swap),
     /agent_log(look→execute→go_home→verify), /execution_status, /camera/rgb·/joint_states 생존
  4) 대화 루프 — 자연어 명령 입력 → /user_command 발행 → 진행 상황 출력
  5) 안전 종료 — /quit 또는 Ctrl+C 시 기동한 노드를 역순으로 일괄 정리

llama-server 는 systemd 상시 구동(`llama-server.service`) 전제로, 이 콘솔은
health 확인만 한다 (직접 기동/종료하지 않음).

전제 / 실행:
  source /opt/ros/humble/setup.bash
  source ~/ros2_ws/install/setup.bash
  python3 ~/PAI/deploy/scripts/pai_console.py            # 기본값
  python3 ~/PAI/deploy/scripts/pai_console.py --robot-port /dev/ttyACM1 --no-realsense
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

# ── ROS2 의존 (실행 전 환경 source 필요) ──
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import Image, JointState
    from robot_arm_msgs.msg import AgentLog, ExecutionStatus
except Exception as e:  # noqa: BLE001
    sys.stderr.write(
        '[pai_console] ROS2 / robot_arm_msgs import 실패. 먼저 환경을 source 하세요:\n'
        '  source /opt/ros/humble/setup.bash\n'
        '  source ~/ros2_ws/install/setup.bash\n'
        f'원인: {type(e).__name__}: {e}\n'
    )
    sys.exit(1)

HOME = os.path.expanduser('~')

# 임베딩 모델 — 로컬 디렉터리가 없으면 이 HF repo id 로 폴백(~/.cache/huggingface 캐시/Hub)
DEFAULT_EMBED_REPO = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


# ── ANSI 색상 ──
def _c(t, code):
    return f'\033[{code}m{t}\033[0m'


def cyan(t):   return _c(t, '36')
def yellow(t): return _c(t, '33')
def green(t):  return _c(t, '32')
def red(t):    return _c(t, '31')
def gray(t):   return _c(t, '90')
def bold(t):   return _c(t, '1')


def parse_args():
    p = argparse.ArgumentParser(description='PAI 통합 런처 + 대화형 콘솔')
    p.add_argument('--robot-port',  default='/dev/ttyACM0')
    p.add_argument('--robot-id',    default='follower')
    p.add_argument('--model-root',  default=f'{HOME}/models/act')
    p.add_argument('--embed-model', default=f'{HOME}/models/embedding/')
    p.add_argument('--domain-map',  default=f'{HOME}/ros2_ws/config/domain_map.yaml')
    p.add_argument('--dest-roi',    default=f'{HOME}/calibration/destination_roi.yaml')
    p.add_argument('--llama-url',   default='http://localhost:8080')
    p.add_argument('--max-verify',  type=int,   default=3)
    p.add_argument('--exec-timeout', type=float, default=120.0)
    p.add_argument('--horizon',     type=int,   default=300)
    p.add_argument('--log-dir',     default=f'{HOME}/ros2_logs')
    p.add_argument('--no-realsense', action='store_true',
                   help='realsense 드라이버를 이미 별도로 띄웠으면 생략')
    return p.parse_args()


def _resolve_embed_model(a):
    """로컬 임베딩 디렉터리가 있으면 그 경로, 없으면 HF repo id(캐시/Hub)로 폴백."""
    path = os.path.expanduser(a.embed_model)
    return path if os.path.isdir(path) else DEFAULT_EMBED_REPO


def node_specs(a):
    """기동 순서대로 (name, argv, env_extra, settle_sec).

    순서 근거 (2026-06-04 실기 검증): GPU 를 쓰는 act_node 를 llama-server 직후
    가장 먼저 기동해 GPU 를 일찍 확보한다(단편화 마진 + 검증 순서 일치). act 를
    model_root 로 띄우면 모델 로드(GPU 점유)는 첫 subtask 까지 지연(lazy)되지만,
    기동을 앞에 둔다. realsense 는 하드웨어 초기화가 길어 가장 먼저 시작(GPU 무관),
    embedding 은 CPU 라 GPU 경쟁 없음.
    """
    embed_model = _resolve_embed_model(a)
    return [
        ('realsense',
         ['ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
          'align_depth.enable:=true'], {}, 8.0),
        ('act_node',
         ['ros2', 'run', 'robot_arm_pkg', 'act_node', '--ros-args',
          '-p', f'model_root:={a.model_root}',
          '-p', 'device:=cuda',
          '-p', f'horizon_steps:={a.horizon}'],
         {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}, 5.0),
        ('camera_node',
         ['ros2', 'run', 'robot_arm_pkg', 'camera_node'], {}, 3.0),
        ('llama_server_manager',
         ['ros2', 'run', 'robot_arm_pkg', 'llama_server_manager', '--ros-args',
          '-p', f'server_url:={a.llama_url}',
          '-p', f'adapter_dir:={HOME}/models/gemma/adapters/'], {}, 2.0),
        ('embedding_node',
         ['ros2', 'run', 'robot_arm_pkg', 'embedding_node', '--ros-args',
          '-p', f'model_name:={embed_model}',
          '-p', 'device:=cpu'], {}, 6.0),
        ('robot_controller_node',
         ['ros2', 'run', 'robot_arm_pkg', 'robot_controller_node', '--ros-args',
          '-p', f'robot_port:={a.robot_port}',
          '-p', f'robot_id:={a.robot_id}'], {}, 3.0),
        ('agent_node',
         ['ros2', 'run', 'robot_arm_pkg', 'agent_node', '--ros-args',
          '-p', f'server_url:={a.llama_url}',
          '-p', f'domain_map:={a.domain_map}',
          '-p', f'destination_roi:={a.dest_roi}',
          '-p', f'max_verify_iterations:={a.max_verify}',
          '-p', f'exec_timeout_s:={a.exec_timeout}'], {}, 2.0),
    ]


# ── 사전 점검 ──
def _llama_alive(url):
    try:
        with urllib.request.urlopen(f'{url}/health', timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def precheck(a):
    print(bold('── 사전 점검 ──'))
    hard_ok = True

    if os.path.isdir(f'{HOME}/ros2_ws/install'):
        print(green('  ✓ ~/ros2_ws/install'))
    else:
        print(red('  ✗ ~/ros2_ws/install 없음 — colcon build 필요')); hard_ok = False

    for label, path in [('ACT model_root', a.model_root),
                        ('gemma adapters', f'{HOME}/models/gemma/adapters')]:
        if os.path.exists(os.path.expanduser(path)):
            print(green(f'  ✓ {label}: {path}'))
        else:
            print(yellow(f'  ? {label} 없음: {path}'))

    if os.path.isdir(os.path.expanduser(a.embed_model)):
        print(green(f'  ✓ embedding(로컬): {a.embed_model}'))
    else:
        print(yellow(f'  ? embedding 로컬 경로 없음 → HF repo id(캐시/Hub) 폴백: '
                     + DEFAULT_EMBED_REPO))

    if os.path.exists(a.dest_roi):
        print(green('  ✓ destination_roi (verify depth veto 활성)'))
    else:
        print(yellow(f'  ? destination_roi 없음: {a.dest_roi} (depth veto 비활성 → Gemma 단독)'))

    if os.path.exists(a.robot_port):
        print(green(f'  ✓ robot_port: {a.robot_port}'))
    else:
        print(yellow(f'  ? robot_port 없음: {a.robot_port}  ('
                     + gray('ls /dev/ttyACM*') + ' 로 확인)'))

    if _llama_alive(a.llama_url):
        print(green(f'  ✓ llama-server: {a.llama_url} (health ok)'))
    else:
        print(red(f'  ✗ llama-server 응답 없음: {a.llama_url}'))
        print(yellow('     sudo systemctl start llama-server.service 후 재시도 권장'))
        hard_ok = False

    return hard_ok


# ── 노드 기동 ──
def _tail(path, n):
    try:
        with open(path) as f:
            return ' / '.join(l.strip() for l in f.readlines()[-n:])
    except Exception:  # noqa: BLE001
        return ''


def launch_nodes(a):
    os.makedirs(a.log_dir, exist_ok=True)
    procs = []
    print(bold('\n── 노드 기동 ──'))
    for name, argv, env_extra, settle in node_specs(a):
        if name == 'realsense' and a.no_realsense:
            print(gray('  - realsense 생략 (--no-realsense)'))
            continue
        env = {**os.environ, **env_extra}
        logpath = os.path.join(a.log_dir, f'{name}.log')
        logf = open(logpath, 'w')
        extra = (f"  [{', '.join(f'{k}={v}' for k, v in env_extra.items())}]"
                 if env_extra else '')
        print(f'  ▶ {cyan(name)} … 로그: {gray(logpath)}{extra}')
        p = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append([name, p, logf])
        time.sleep(settle)
        if p.poll() is not None:
            print(red(f'    ✗ {name} 기동 직후 종료 (exit {p.returncode})'))
            print('      ' + gray(_tail(logpath, 3)))
        else:
            print(green(f'    ✓ {name} 실행 중 (pid {p.pid})'))
    return procs


# ── 대화형 콘솔 노드 ──
class Console(Node):
    def __init__(self):
        super().__init__('pai_console')
        self.cmd_pub = self.create_publisher(String, '/user_command', 10)
        self.create_subscription(String, '/selected_domain', self._on_domain, 10)
        self.create_subscription(AgentLog, '/agent_log', self._on_agent_log, 10)
        self.create_subscription(ExecutionStatus, '/execution_status', self._on_exec, 10)
        cam_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.last_rgb = 0.0
        self.last_joint = 0.0
        self.create_subscription(Image, '/camera/rgb', self._on_rgb, cam_qos)
        self.create_subscription(JointState, '/joint_states', self._on_joint, 10)

    def send(self, text):
        self.cmd_pub.publish(String(data=text))

    # 토픽 흐름 — 처리 절차를 사람이 읽기 좋게
    def _on_domain(self, m):
        print('\n' + cyan(f'  [LoRA] 도메인 hot-swap → {m.data}'))

    def _on_agent_log(self, m):
        tool = m.tool_called or '(thought)'
        line = f'  [agent] {yellow(tool)}'
        if m.tool_args:
            line += ' ' + gray(m.tool_args)
        if m.tool_result:
            line += f' → {m.tool_result}'
        print('\n' + line)
        if m.thought:
            print(gray(f'          {m.thought}'))

    def _on_exec(self, m):
        col = green if m.status == 'SUCCESS' else yellow
        extra = f' ({m.failure_reason})' if m.failure_reason else ''
        print('\n' + f'  [act] subtask {m.subtask_id[:8]} → {col(m.status)}{extra}')

    def _on_rgb(self, _m):
        self.last_rgb = time.time()

    def _on_joint(self, _m):
        self.last_joint = time.time()


# ── 상태 / 종료 ──
def print_status(procs, a, console):
    print(bold('\n── 노드 상태 ──'))
    for name, p, _ in procs:
        alive = p.poll() is None
        mark = green('●') if alive else red('○')
        tail = f'(pid {p.pid})' if alive else red(f'종료 exit {p.returncode}')
        print(f'  {mark} {name} {tail}')

    now = time.time()

    def fresh(t):
        if t == 0:
            return red('수신 없음')
        dt = now - t
        return (green if dt < 2 else yellow)(f'{dt:.1f}s 전')

    print(bold('  ── 토픽 흐름 ──'))
    print(f'    /camera/rgb   : {fresh(console.last_rgb)}')
    print(f'    /joint_states : {fresh(console.last_joint)}')
    print(f'    llama-server  : '
          + (green('health ok') if _llama_alive(a.llama_url) else red('응답 없음')))


def shutdown(console, procs):
    print(bold('\n── 종료: 노드 정리 ──'))
    try:
        console.destroy_node()
    except Exception:  # noqa: BLE001
        pass
    try:
        rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    # 역순 종료 (agent 먼저 → realsense 마지막). SIGINT → terminate → kill 단계적.
    for _n, p, _l in reversed(procs):
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    time.sleep(2.0)
    for name, p, logf in reversed(procs):
        if p.poll() is None:
            p.terminate()
    time.sleep(1.0)
    for name, p, logf in reversed(procs):
        if p.poll() is None:
            p.kill()
        try:
            logf.close()
        except Exception:  # noqa: BLE001
            pass
        print(gray(f'  - {name} 종료'))
    print(green('정리 완료.'))


_HELP = """
명령:
  <자연어>      사용자 명령 발행 (예: '쓰레기 좀 정리해줘')  → /user_command
  /status       노드 / 토픽 흐름 상태
  /help         도움말
  /quit         전체 종료 (노드 일괄 정리)
"""


def main():
    a = parse_args()
    print(bold(cyan('\n=== PAI 통합 콘솔 (Jetson) ===')))

    if not precheck(a):
        ans = input(yellow('\n일부 점검(✗) 실패. 그래도 기동할까요? [y/N] ')).strip().lower()
        if ans != 'y':
            print('중단합니다.')
            return

    procs = launch_nodes(a)

    rclpy.init()
    console = Console()
    spin = threading.Thread(target=rclpy.spin, args=(console,), daemon=True)
    spin.start()

    print(bold(green('\n준비 완료. 명령을 입력하세요.')))
    print(yellow('⚠️  명령 발행 시 로봇 팔이 실제로 움직입니다 — 비상정지를 준비하세요.'))
    print(gray(_HELP))

    try:
        while True:
            try:
                line = input(bold('\n명령> ')).strip()
            except EOFError:
                break
            if not line:
                continue
            if line in ('/quit', '/exit', 'quit', 'exit'):
                break
            if line == '/status':
                print_status(procs, a, console)
            elif line == '/help':
                print(gray(_HELP))
            else:
                console.send(line)
                print(green(f'→ /user_command 발행: {line!r}')
                      + gray('   (아래로 처리 절차가 출력됩니다)'))
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(console, procs)


if __name__ == '__main__':
    main()
