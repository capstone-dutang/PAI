"""
agent_node.py

Gemma 4 E2B VLA 에이전트의 함수 호출 루프를 오케스트레이션하는 ROS2 노드.

흐름:
  /user_command 수신
    → 도메인 분류 (domain_map.yaml 키워드 매칭)
    → /domain_selection 발행 → llama_server_manager가 LoRA hot-swap
    → /selected_domain 수신 대기
    → (워커 스레드에서) 함수 호출 루프:
         look_scene(prompt)        → Gemma VLM 호출 → 자연어 target_description
         /embed_text 서비스 호출   → language_embedding 획득
         execute_subtask(...)      → SubtaskCommand 발행 → ExecutionStatus 대기
         /go_home (Trigger)        → verify 직전 팔 자세를 home_pose 로 고정
         verify_result(...)        → Gemma 시각 검증 + depth place 보조 검증
            → SUCCESS / FAIL / PARTIAL / UNCLEAR
            → SUCCESS면 종료, 그 외 max_verify_iterations(기본 3)까지 재시도
            → 초과 시 WAITING_USER 상태로 전환

구독 토픽:
  /user_command     (std_msgs/String)
  /camera/rgb       (sensor_msgs/Image, BestEffort+KeepLast(1))
  /camera/depth     (sensor_msgs/Image, BestEffort+KeepLast(1))
  /joint_states     (sensor_msgs/JointState)
  /execution_status (robot_arm_msgs/ExecutionStatus)
  /selected_domain  (std_msgs/String)

발행 토픽:
  /domain_selection (robot_arm_msgs/DomainSelection)
  /subtask_command  (robot_arm_msgs/SubtaskCommand)
  /agent_log        (robot_arm_msgs/AgentLog)

서비스 클라이언트:
  /embed_text       (robot_arm_msgs/EmbedText)
  /go_home          (std_srvs/Trigger)

파라미터:
  server_url             (str)   — llama-server 주소
  domain_map             (str)   — domain_map.yaml 경로
  destination_roi        (str)   — destination_roi.yaml 경로 (verify_result place 검증용)
  max_verify_iterations  (int)   — 검증 재시도 한계 (기본 3)
  exec_timeout_s         (float) — ExecutionStatus 대기 타임아웃 (기본 120s)

═══════════════════════════════════════════════════════════════════════
구현 결정 사항 (2026-06-03 확정)
═══════════════════════════════════════════════════════════════════════
① 실행 모델 — 워커 스레드 + MultiThreadedExecutor
   _run_function_loop 은 구독 콜백(_selected_callback)에서 직접 돌리지 않고
   전용 데몬 스레드에서 실행한다. 콜백은 즉시 반환하므로 executor 스핀 스레드가
   막히지 않는다. 워커 안의 모든 블로킹(HTTP / 서비스 / status 대기)은 executor
   콜백 바깥에서 일어난다.
     - ExecutionStatus 대기: _exec_status_callback 이 threading.Event 를 set,
       _call_execute_subtask 가 Event.wait(timeout) 으로 동기 대기.
     - 서비스(/embed_text, /go_home): 이미 스핀 중이므로 spin_until_future_complete
       사용 불가 → call_async + future.add_done_callback(Event) 패턴으로 대기.
     - exec_status 구독 + 두 서비스 클라이언트는 ReentrantCallbackGroup 에 두어
       워커가 대기하는 동안 응답/콜백이 헤드라인 블로킹 없이 흐르게 한다.
     - main() 은 MultiThreadedExecutor 로 스핀.

② look_scene 호출 — 텍스트 루프 + 별도 VLM 호출 (SYSTEM_PROMPT 미사용)
   함수 호출 루프 대화는 텍스트 전용 (function_call 학습 분포 = 이미지 없음).
   실제 시각 grounding 은 prompts.LOOK_SCENE_SYSTEM_PROMPT 를 쓰는 별도
   chat completion (RGB 이미지 포함). SYSTEM_PROMPT 는 "JSON 으로만 응답" 을
   강제해 자연어 묘사와 충돌하므로 쓰지 않는다.
   ⚠️ 이 호출은 전용 학습 분포가 없다 (prompts.LOOK_SCENE_SYSTEM_PROMPT 주석 참조).

③ verify 종합 판정 — Gemma status 우선 + depth 단방향 veto
   _combine_verify(): depth 가 None(ROI 미설정/미실측/depth 없음)이면 Gemma 단독.
   depth 는 SUCCESS 를 "반박"만 한다 (grasp 흔적 없음 → PARTIAL 강등으로 재시도
   유도). depth 단독으로 SUCCESS 승격은 없다. verify status 정답률이 낮다는
   training/README.md 이슈 2 를 max_verify_iterations 재시도 + depth veto 로 보완.
   reasoning 은 로그용으로만 사용.

   ExecutionStatus.status 대소문자: SSoT(prompts.ExecStatus)·msg 명세는 대문자이나
   act_node 가 소문자로 발행하는 케이스가 있어 본 노드에서 .upper() 방어 정규화.
   (원천 수정 = act_node._finish_subtask 가 ExecStatus enum 사용 — 별도 권장.)

④ 재시도 의미 — 시도(attempt) 단위로 look→execute→go_home→verify 전체를 재실행
   각 시도는 [system, user_command] 에서 새로 시작하므로 매번 function_call 학습
   분포(단일 look→execute→verify 시퀀스)와 일치한다. verify 결과를 모델에 다시
   먹이지 않는다 (verify 이후 turn 은 off-distribution). 시도 수 ≤ max_verify_iterations.
   verify fallback (deploy/README.md): execute 직후 모델 응답이 verify_result 함수
   호출이 아니면 코드가 verify_result 를 강제 호출 (original_command=_user_command).

⑤ _depth_place_check — destination ROI median depth 기반 place 보강 판정
   verify 시점은 /go_home 후 home_pose (그리퍼 비어 있음 — 데이터 수집과 동일
   종료 자세). 따라서 그리퍼 grasp 검증은 불가하며, 검증 대상은 "물체가 목적지
   (통/목표 영역)에 놓였는가" = place 다.
   카메라·목적지 고정 전제로 목적지는 이미지의 고정 픽셀 영역 → destination_roi.yaml
   은 픽셀 ROI + baseline depth 로 표현한다. 미터 3D 박스/카메라 intrinsics 가
   불필요해 README "agent_node 는 intrinsics 비의존" 결정과 충돌하지 않는다.
   판정: 목적지 ROI median 이 baseline(빈 목적지)보다 depth_change_threshold 이상
   작아지면(물체가 쌓여 카메라에 가까움) True. destination_roi 미설정 / 도메인 ROI
   없음 / depth 없음 / 유효픽셀 없음 → None → 결정 ③에 의해 Gemma 단독.

   ▷ grasp(집기 직후 carry 시점) 검증은 home_pose 복귀 데이터로는 불가 — 미래
     확장으로 유보. grasp_roi.yaml + 미래 _depth_grasp_check 가 그 hook
     (README "grasp 검증 정책" 절 참조).

⑥ 종료 상태 — 신규 토픽/메시지 미도입 (v1)
   내부 상태기계 IDLE / RUNNING / WAITING_USER 로 표현. max_verify_iterations
   초과 시 WAITING_USER + AgentLog 종료 마커. 복구: WAITING_USER 에서 새
   /user_command 가 들어오면 태스크 포기 후 재시작 (신규 명령은 RUNNING 중에만
   거부). 전용 /user_intervention 토픽 명세는 향후 작업으로 유보.

미구현(타 노드 의존) 잔여:
  _classify_domain — 단순 substring 매칭 placeholder (운영 중 정책 확정).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import uuid
from typing import Dict, List, Optional

import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from robot_arm_msgs.msg import (
    SubtaskCommand,
    ExecutionStatus,
    DomainSelection,
    AgentLog,
)
from robot_arm_msgs.srv import EmbedText

# 시스템 프롬프트 / 상태 enum SSoT — robot_arm_pkg.prompts
# (학습 측 generate_function_call_qa / generate_verify_qa 도 같은 모듈을 import 한다.
#  단 LOOK_SCENE_SYSTEM_PROMPT 는 배포 전용 — 학습 측은 import 하지 않음.)
from robot_arm_pkg.prompts import (
    SYSTEM_PROMPT, LOOK_SCENE_SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT,
    VerifyStatus, ExecStatus,
)
from robot_arm_pkg.utils.ros_image import image_msg_to_rgb_array, depth_msg_to_meters


# ── HTTP / 서비스 타임아웃 ─────────────────────
# chat completion 은 verify 시 --no-mmproj-offload(CPU projector) 로 느려질 수
# 있어 넉넉히. 서비스(embed)는 빠른 작업, go_home 은 보간(기본 2s) + 여유.
_HTTP_TIMEOUT_SEC    = 120.0
_SERVICE_TIMEOUT_SEC = 10.0
_GO_HOME_TIMEOUT_SEC = 15.0

# 한 시도(attempt) 안에서 모델의 함수 호출 turn 상한 (look + execute + 여유).
# 정상 시퀀스는 2~3 turn. 무한 루프 방지용 안전 상한.
_INNER_MAX_TURNS = 6

# verify_result 가 돌려줄 수 있는 enum 집합 (모델 응답 검증용).
_VALID_VERIFY = {
    VerifyStatus.SUCCESS, VerifyStatus.FAIL,
    VerifyStatus.PARTIAL, VerifyStatus.UNCLEAR,
}

# look_scene 묘사 형식 1차 검증 정규식 ("the ..." 로 시작하는지만 가볍게).
# 전체 vocabulary 멤버십 검증은 후속(label_vocab 경로 파라미터 추가 필요).
_LOOK_SCENE_RE = re.compile(r'^the\s+\S+', re.IGNORECASE)


class AgentNode(Node):

    # 내부 상태기계 (결정 ⑥)
    STATE_IDLE         = 'IDLE'          # 명령 없음 또는 hot-swap 대기 중
    STATE_RUNNING      = 'RUNNING'       # 함수 호출 루프 워커 실행 중
    STATE_WAITING_USER = 'WAITING_USER'  # max 초과 — 사용자 개입 대기

    def __init__(self):
        super().__init__('agent_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('server_url',            'http://localhost:8080')
        self.declare_parameter('domain_map',
                               os.path.expanduser('~/ros2_ws/config/domain_map.yaml'))
        self.declare_parameter('destination_roi',
                               os.path.expanduser('~/calibration/destination_roi.yaml'))
        self.declare_parameter('max_verify_iterations', 3)
        self.declare_parameter('exec_timeout_s',        120.0)

        self.server_url           : str   = self.get_parameter('server_url').value
        self.domain_map_path      : str   = self.get_parameter('domain_map').value
        self.destination_roi_path : str   = self.get_parameter('destination_roi').value
        self.max_verify_iterations: int   = self.get_parameter('max_verify_iterations').value
        self.exec_timeout_s       : float = self.get_parameter('exec_timeout_s').value

        # ── 설정 로드 ─────────────────────────────
        self._domain_map: Dict[str, str] = self._load_domain_map(self.domain_map_path)
        self._destination_roi: Optional[dict] = self._load_destination_roi(self.destination_roi_path)

        # ── 입력 캐시 (latest-only) ───────────────
        self._latest_rgb:    Optional[Image]      = None
        self._latest_depth:  Optional[Image]      = None
        self._latest_joints: Optional[JointState] = None

        # ── 진행 상태 ─────────────────────────────
        # 상태 전이는 _lock 으로 보호 (콜백=executor 스레드 / 루프=워커 스레드).
        self._lock = threading.Lock()
        self._state:              str                       = self.STATE_IDLE
        self._user_command:       Optional[str]             = None
        self._active_domain:      Optional[str]             = None
        self._loop_thread:        Optional[threading.Thread] = None
        self._verify_count:       int                       = 0

        # execute_subtask ↔ ExecutionStatus 동기화.
        self._pending_subtask_id: Optional[str]             = None
        self._last_status:        Optional[ExecutionStatus] = None
        self._status_event = threading.Event()

        # ── QoS ───────────────────────────────────
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 콜백 그룹 (결정 ①) ────────────────────
        # 워커가 call_async future / ExecutionStatus 를 대기하는 동안 해당 콜백이
        # 헤드라인 블로킹 없이 흐르도록 ReentrantCallbackGroup 사용.
        self._cb_group = ReentrantCallbackGroup()

        # ── 구독 ──────────────────────────────────
        self.create_subscription(String,          '/user_command',     self._user_cmd_callback,    10)
        self.create_subscription(Image,           '/camera/rgb',       self._rgb_callback,         camera_qos)
        self.create_subscription(Image,           '/camera/depth',     self._depth_callback,       camera_qos)
        self.create_subscription(JointState,      '/joint_states',     self._joints_callback,      10)
        self.create_subscription(ExecutionStatus, '/execution_status', self._exec_status_callback, 10,
                                 callback_group=self._cb_group)
        self.create_subscription(String,          '/selected_domain',  self._selected_callback,    10)

        # ── 발행 ──────────────────────────────────
        self.domain_pub  = self.create_publisher(DomainSelection, '/domain_selection', 10)
        self.subtask_pub = self.create_publisher(SubtaskCommand,  '/subtask_command',  10)
        self.log_pub     = self.create_publisher(AgentLog,        '/agent_log',        10)

        # ── 서비스 클라이언트 ─────────────────────
        self.embed_client   = self.create_client(EmbedText, '/embed_text', callback_group=self._cb_group)
        self.go_home_client = self.create_client(Trigger,   '/go_home',    callback_group=self._cb_group)

        self.get_logger().info(
            f'agent_node 시작 | server: {self.server_url} | '
            f'domain_map: {self.domain_map_path} | '
            f'max_verify_iterations: {self.max_verify_iterations} | '
            f'exec_timeout_s: {self.exec_timeout_s}'
        )

    # ── 설정 로드 ─────────────────────────────────

    def _load_domain_map(self, path: str) -> Dict[str, str]:
        """
        domain_map.yaml 로드.

        예상 포맷:
          mapping:
            - keywords: ["쓰레기", "분리수거", "재활용"]
              domain : recycling
            - keywords: ["식기", "그릇", "접시"]
              domain : tableware

        반환 형식: {keyword: domain} flat dict (단순 키워드 인덱스).
        파일 없거나 파싱 실패 시 빈 dict 반환.
        """
        try:
            import yaml  # noqa: PLC0415
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.get_logger().warn(
                f'domain_map.yaml 없음: {path} → 도메인 분류 불가 (미구현 placeholder)'
            )
            return {}
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'domain_map.yaml 파싱 실패: {e}')
            return {}

        flat: Dict[str, str] = {}
        for entry in data.get('mapping', []):
            domain = entry.get('domain')
            for kw in entry.get('keywords', []):
                flat[str(kw)] = str(domain)
        return flat

    def _load_destination_roi(self, path: str) -> Optional[dict]:
        """
        destination_roi.yaml 로드. verify_result 의 depth place 검증에 사용.
        파일 없으면 None 반환 → depth place 검증 비활성 (Gemma 단독).
        """
        try:
            import yaml  # noqa: PLC0415
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().warn(
                f'destination_roi.yaml 없음: {path} → depth place 검증 비활성 (Gemma 단독 검증)'
            )
            return None
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'destination_roi.yaml 파싱 실패: {e}')
            return None

    # ── 구독 콜백 ─────────────────────────────────

    def _rgb_callback(self, msg: Image) -> None:
        self._latest_rgb = msg

    def _depth_callback(self, msg: Image) -> None:
        self._latest_depth = msg

    def _joints_callback(self, msg: JointState) -> None:
        self._latest_joints = msg

    def _user_cmd_callback(self, msg: String) -> None:
        """
        외부 트리거. 도메인 분류 → DomainSelection 발행 → /selected_domain 대기.
        함수 호출 루프 진입은 _selected_callback에서.

        결정 ⑥: RUNNING 중에만 신규 명령을 거부한다. IDLE / WAITING_USER 상태면
        새 명령을 수락해 (대기 중이던 태스크를 포기하고) 재시작한다.
        """
        with self._lock:
            if self._state == self.STATE_RUNNING:
                self.get_logger().warn(
                    f'루프 실행 중 — 새 명령 무시: {msg.data!r}'
                )
                return
            self._user_command = msg.data
            self._verify_count = 0
            self._state = self.STATE_IDLE  # hot-swap 대기 단계

        self.get_logger().info(f'/user_command 수신: {msg.data!r}')

        domain = self._classify_domain(msg.data)
        if domain is None:
            self.get_logger().error(
                f'도메인 분류 실패 (매칭되는 키워드 없음): {msg.data!r}'
            )
            self._to_idle()
            return

        sel = DomainSelection()
        sel.domain = domain
        sel.scale  = 1.0
        self.domain_pub.publish(sel)
        self.get_logger().info(f'/domain_selection 발행: domain={domain!r}')

    def _selected_callback(self, msg: String) -> None:
        """
        llama_server_manager가 hot-swap 완료를 통보 → 워커 스레드에서 루프 시작.
        결정 ①: 콜백 자체는 워커를 띄우고 즉시 반환한다.
        """
        with self._lock:
            if self._user_command is None:
                return  # 처리 중인 명령 없음 (외부에서 임의 발행 등)
            if self._state == self.STATE_RUNNING:
                self.get_logger().warn(
                    '이미 루프 실행 중 — /selected_domain 중복 수신 무시'
                )
                return
            self._active_domain = msg.data
            self._state = self.STATE_RUNNING
            self._loop_thread = threading.Thread(
                target=self._run_function_loop,
                name='agent_fn_loop',
                daemon=True,
            )

        self.get_logger().info(
            f'hot-swap 완료 통보 수신: domain={msg.data!r} → 워커 스레드에서 루프 시작'
        )
        self._loop_thread.start()

    def _exec_status_callback(self, msg: ExecutionStatus) -> None:
        """
        act_node가 SubtaskCommand 실행 결과 통보.
        execute_subtask 가 대기 중인 subtask_id 와 일치할 때만 Event 를 깨운다.
        """
        if msg.subtask_id != self._pending_subtask_id:
            return  # 다른 subtask 응답 (아마 stale)

        self._last_status = msg
        self.get_logger().info(
            f'/execution_status 수신: id={msg.subtask_id!r} status={msg.status!r}'
        )
        self._status_event.set()

    # ── 도메인 분류 ───────────────────────────────

    def _classify_domain(self, user_command: str) -> Optional[str]:
        """
        사용자 명령에서 키워드를 찾아 도메인을 결정.

        현재는 _domain_map(flat keyword → domain) 안의 키워드가 명령 문자열에
        포함되는지를 단순 substring 검사로 판단한다. 가장 먼저 매칭되는 항목을
        반환. 매칭 우선순위, 동의어 정규화, 다국어 처리 등은 운영 중 확정.

        매칭 실패 시 None.
        """
        for keyword, domain in self._domain_map.items():
            if keyword in user_command:
                return domain
        return None

    # ── 상태 전이 헬퍼 ────────────────────────────

    def _to_idle(self) -> None:
        with self._lock:
            self._state = self.STATE_IDLE
            self._user_command = None
            self._pending_subtask_id = None

    def _to_waiting_user(self) -> None:
        # _user_command 는 로깅/디버그용으로 유지. 새 명령이 override.
        with self._lock:
            self._state = self.STATE_WAITING_USER
            self._pending_subtask_id = None

    # ── 함수 호출 루프 (워커 스레드) ──────────────

    def _run_function_loop(self) -> None:
        """
        Gemma 함수 호출 루프 본체 (워커 스레드에서 실행 — 결정 ①).

        결정 ④: 시도(attempt) 단위로 look→execute→go_home→verify 전체를 재실행한다.
          각 시도는 [system, user_command] 에서 새로 시작 → 매 시도가 function_call
          학습 분포(단일 시퀀스)와 일치. verify 결과를 모델에 다시 먹이지 않는다.
          시도 수 ≤ max_verify_iterations, SUCCESS 면 즉시 종료.

        결정 ⑥: 초과 시 WAITING_USER 진입 + AgentLog 종료 마커.
        """
        try:
            cmd = self._user_command
            self.get_logger().info(
                f'함수 호출 루프 시작: cmd={cmd!r} domain={self._active_domain!r}'
            )

            final = VerifyStatus.UNCLEAR
            for attempt in range(1, self.max_verify_iterations + 1):
                self._verify_count = attempt
                self.get_logger().info(
                    f'[attempt {attempt}/{self.max_verify_iterations}] '
                    f'look→execute→go_home→verify'
                )
                final = self._run_single_attempt()
                if final == VerifyStatus.SUCCESS:
                    self.get_logger().info('verify SUCCESS — 태스크 완료')
                    self._to_idle()
                    return
                self.get_logger().warn(
                    f'verify={final!r} (attempt {attempt}) — 재시도'
                )

            # max_verify_iterations 초과 → 사용자 개입 대기 (결정 ⑥)
            self.get_logger().error(
                f'max_verify_iterations({self.max_verify_iterations}) 초과 — '
                f'WAITING_USER 진입 (마지막 verify={final!r})'
            )
            self._emit_log(
                thought='',
                tool='',
                args=json.dumps({'command': cmd}, ensure_ascii=False),
                result=f'WAITING_USER:{final}',
            )
            self._to_waiting_user()

        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'함수 호출 루프 예외: {type(e).__name__}: {e}')
            self._to_idle()

    def _run_single_attempt(self) -> str:
        """
        한 시도: 텍스트 전용 함수 호출 대화로 look_scene → execute_subtask 를 거쳐
        verify_result 까지 진행하고 그 status 를 반환한다.

        결정 ②: 대화는 텍스트 전용. look_scene 의 실제 grounding 은 _call_look_scene
                내부의 별도 VLM 호출이 담당하고, 그 결과를 '[tool_result] ' 로 피드한다.
        결정 ④: execute 직후 모델 응답이 verify_result 함수 호출이 아니면 코드가
                verify_result 를 강제 호출 (deploy/README.md verify fallback).
        """
        messages: List[dict] = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': self._user_command},
        ]
        last_target: Optional[str] = None
        exec_seen = False

        for _ in range(_INNER_MAX_TURNS):
            message = self._chat_completion(messages)
            parsed  = self._parse_function_call(message)

            if parsed is None:
                # 함수 호출 JSON 이 아님.
                if exec_seen:
                    # verify fallback (결정 ④)
                    self.get_logger().warn(
                        '마지막 turn 응답이 verify_result 함수 호출 아님 — '
                        f'코드 강제 호출. message={message!r}'
                    )
                    return self._call_verify_result(self._user_command)
                self.get_logger().error(
                    f'함수 호출 파싱 실패 (execute 이전) — 시도 중단. message={message!r}'
                )
                return VerifyStatus.UNCLEAR

            fn   = parsed['function']
            args = parsed['args']
            messages.append(
                {'role': 'assistant',
                 'content': json.dumps(parsed, ensure_ascii=False)}
            )

            if fn == 'look_scene':
                self._emit_log('', fn, json.dumps(args, ensure_ascii=False), '')
                desc = self._call_look_scene(args.get('prompt', ''))
                last_target = desc
                self._feed_tool_result(messages, desc or '')

            elif fn == 'execute_subtask':
                target = args.get('target_description') or last_target or ''
                self._emit_log('', fn,
                               json.dumps({'target_description': target}, ensure_ascii=False), '')
                status = self._call_execute_subtask(target)
                exec_seen = True
                # verify 직전 /go_home 강제 (deploy/README.md freeze 2026-05-25)
                self._call_go_home()
                self._feed_tool_result(messages, status)

            elif fn == 'verify_result':
                return self._call_verify_result(
                    args.get('original_command') or self._user_command
                )

            else:
                self.get_logger().error(f'알 수 없는 함수 호출: {fn!r}')
                if exec_seen:
                    return self._call_verify_result(self._user_command)
                return VerifyStatus.UNCLEAR

        # inner turn 소진 (verify 를 못 봄)
        if exec_seen:
            self.get_logger().warn('inner turn 소진 — verify_result 강제 호출')
            return self._call_verify_result(self._user_command)
        return VerifyStatus.UNCLEAR

    # ── Gemma 함수 호출 핸들러 (개별) ─────────────

    def _call_look_scene(self, prompt: str) -> Optional[str]:
        """
        Gemma VLM에 RGB + prompt 입력 → 자연어 target_description 반환 (결정 ②).

        함수 호출 대화와 분리된 별도 chat completion:
          system : prompts.LOOK_SCENE_SYSTEM_PROMPT (SYSTEM_PROMPT 아님)
          user   : [image (현재 RGB, base64 data URI), text=prompt]  ← image-first
        반환 형식: "the {attribute} {object} {spatial}" (5~15 단어, 영어).

        검증: 템플릿 1차 정규식. 불일치 시 1회 재시도. (전체 vocab 멤버십 검증은 후속.)
        RGB 미수신 시 None 반환 → 호출 측이 빈 tool_result 로 피드.
        """
        if self._latest_rgb is None:
            self.get_logger().error('look_scene: RGB 미수신 — 묘사 불가')
            return None

        try:
            uri = self._rgb_to_data_uri(self._latest_rgb)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'look_scene: RGB 인코딩 실패: {e}')
            return None

        user_content = [
            {'type': 'image_url', 'image_url': {'url': uri}},
            {'type': 'text', 'text': prompt or 'describe the object to grasp'},
        ]
        messages = [
            {'role': 'system', 'content': LOOK_SCENE_SYSTEM_PROMPT},
            {'role': 'user',   'content': user_content},
        ]

        text = ''
        for attempt in range(2):
            message = self._chat_completion(messages, max_tokens=64)
            text = self._message_text(message)
            # 한 줄만 사용 + 따옴표 제거
            text = (text.splitlines()[0] if text else '').strip().strip('"\'').strip()
            if _LOOK_SCENE_RE.match(text):
                return text
            self.get_logger().warn(
                f'look_scene 출력 형식 불일치 (시도 {attempt + 1}): {text!r}'
            )
        # 형식 검증 실패해도 마지막 텍스트를 best-effort 로 반환 (없으면 None)
        return text or None

    def _call_execute_subtask(self, target_description: str) -> str:
        """
        target_description을 임베딩 → SubtaskCommand 발행 → ExecutionStatus 대기.

        결정 ①: _status_event 로 동기 대기 (워커 스레드).
        결정 ③(2c): act_node 가 소문자로 발행하는 케이스 대비 .upper() 방어 정규화.

        반환: ExecStatus 문자열 (SUCCESS / FAIL / TIMEOUT). 대기 타임아웃 시 TIMEOUT.
        """
        if not target_description:
            self.get_logger().error('execute_subtask: target_description 비어있음')

        embedding = self._embed_text(target_description)
        subtask_id = uuid.uuid4().hex

        # 대기 상태 초기화 후 발행 (race 방지: pending 설정 → event clear → publish)
        self._last_status = None
        self._pending_subtask_id = subtask_id
        self._status_event.clear()

        cmd = SubtaskCommand()
        cmd.subtask_id         = subtask_id
        cmd.domain             = self._active_domain or ''
        cmd.target_description = target_description
        cmd.language_embedding = [float(x) for x in embedding]
        self.subtask_pub.publish(cmd)
        self.get_logger().info(
            f'SubtaskCommand 발행: id={subtask_id} target={target_description!r} '
            f'embed_dim={len(embedding)}'
        )

        if not self._status_event.wait(self.exec_timeout_s):
            self.get_logger().error(
                f'ExecutionStatus 대기 타임아웃({self.exec_timeout_s}s) id={subtask_id}'
            )
            self._pending_subtask_id = None
            return ExecStatus.TIMEOUT

        st = self._last_status
        self._pending_subtask_id = None
        if st is None:
            return ExecStatus.TIMEOUT
        return (st.status or '').upper() or ExecStatus.TIMEOUT  # 결정 ③(2c) 방어 정규화

    def _call_go_home(self) -> None:
        """
        /go_home (std_srvs/Trigger) 1회 호출 — verify 직전 팔 자세를 home_pose 로
        고정 (deploy/README.md freeze 2026-05-25). 서비스 부재/실패 시 경고만 하고
        진행한다 (verify 학습 분포(home_pose 단일)와 어긋날 위험은 로그로 남김).
        """
        if not self.go_home_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                '/go_home 서비스 없음 — 건너뜀 (verify 자세 분포 불일치 위험)'
            )
            return

        future = self.go_home_client.call_async(Trigger.Request())
        res = self._wait_future(future, _GO_HOME_TIMEOUT_SEC)
        if res is None:
            self.get_logger().warn('/go_home 응답 타임아웃')
        elif not res.success:
            self.get_logger().warn(f'/go_home 실패: {res.message}')
        else:
            self.get_logger().info(f'/go_home 완료: {res.message}')

    def _call_verify_result(self, original_command: str) -> str:
        """
        verify_result 함수 명세 구현 (결정 ②③).

        Stateless: original_command 인자를 우선 사용 (function_call 학습 분포 일치).
        전제: 호출 직전 /go_home 으로 팔이 home_pose 에 있음 (verify 학습 분포 일치).

        1) Gemma 시각 검증 (별도 chat completion):
             system : prompts.VERIFY_SYSTEM_PROMPT
             user   : [image (현재 RGB), text "original_command: \"...\"..."]  ← image-first
             응답   : {"status": <enum>, "reasoning": <english>}
           (generate_verify_qa.build_sample 의 user_text 와 정확히 일치시킴)
        2) depth place 보조 검증 — _depth_place_check()
        3) 종합 판정 — _combine_verify() (결정 ③: Gemma 우선 + depth 단방향 veto)

        reasoning 은 로그용으로만 사용 (training/README.md 이슈 2).
        반환: VerifyStatus 문자열.
        """
        gemma_status = VerifyStatus.UNCLEAR
        reasoning = ''

        if self._latest_rgb is None:
            self.get_logger().warn('verify_result: RGB 미수신 → UNCLEAR')
        else:
            try:
                uri = self._rgb_to_data_uri(self._latest_rgb)
                user_text = (
                    f'original_command: "{original_command}"\n'
                    f'Output the verification JSON.'
                )
                messages = [
                    {'role': 'system', 'content': VERIFY_SYSTEM_PROMPT},
                    {'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': uri}},
                        {'type': 'text', 'text': user_text},
                    ]},
                ]
                message = self._chat_completion(messages, max_tokens=128)
                obj = self._extract_json(self._message_text(message))
                if isinstance(obj, dict):
                    status = obj.get('status')
                    if status in _VALID_VERIFY:
                        gemma_status = status
                    else:
                        self.get_logger().warn(
                            f'verify status enum 벗어남: {status!r} → UNCLEAR'
                        )
                    reasoning = str(obj.get('reasoning', ''))
                else:
                    self.get_logger().warn(
                        f'verify 응답 JSON 파싱 실패: {self._message_text(message)!r}'
                    )
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'verify_result Gemma 호출 실패: {e}')

        depth = self._depth_place_check()
        final = self._combine_verify(gemma_status, depth)

        self._emit_log(
            thought=reasoning,
            tool='verify_result',
            args=json.dumps({'original_command': original_command}, ensure_ascii=False),
            result=json.dumps(
                {'gemma': gemma_status, 'depth': depth, 'final': final},
                ensure_ascii=False),
        )
        self.get_logger().info(
            f'verify_result: gemma={gemma_status} depth={depth} → final={final}'
        )
        return final

    def _combine_verify(self, gemma_status: str, depth: Optional[bool]) -> str:
        """
        결정 ③: Gemma status 우선 + depth 단방향 veto.
          - depth None  → Gemma 단독 (depth 비활성/미실측).
          - Gemma SUCCESS + depth False → PARTIAL 로 강등 (목적지에 놓인 흔적 없음 → 재시도 유도).
          - 그 외 → Gemma 신뢰. depth 단독 SUCCESS 승격은 없음.
        """
        if depth is None:
            return gemma_status
        if gemma_status == VerifyStatus.SUCCESS and depth is False:
            self.get_logger().warn(
                'verify: Gemma=SUCCESS 이나 depth place 흔적 없음 → PARTIAL 강등'
            )
            return VerifyStatus.PARTIAL
        return gemma_status

    def _depth_place_check(self) -> Optional[bool]:
        """
        destination_roi.yaml 의 목적지 ROI median depth 로 place(목적지 배치) 성공
        흔적을 산출하는 보조 신호 (결정 ⑤).

        verify 시점은 /go_home 후 home_pose (그리퍼 비어 있음 — 데이터 수집과 동일
        종료 자세)이므로 검증 대상은 grasp 가 아니라 place 다. 카메라·목적지 고정
        전제로 목적지를 픽셀 ROI 로 표현 → intrinsics 비의존.

        판정: active_domain 의 목적지 ROI median depth 가 baseline_depth(빈 목적지)
        보다 depth_change_threshold 이상 작아지면(물체가 쌓여 카메라에 가까움) 흔적.

        반환:
          True  — 도메인 목적지 ROI 중 하나라도 place 흔적
          False — 측정된 ROI 모두 변화 없음
          None  — 비활성/판정불가 (yaml 없음 / depth 없음 / 도메인 ROI 없음 /
                  유효 depth 픽셀 없음) → _combine_verify 가 Gemma 단독 (결정 ③).

        grasp(집기 시점) 검증은 home_pose 복귀 데이터로는 불가 — 미래 확장으로 유보
        (README "grasp 검증 정책" 절).
        """
        cfg = self._destination_roi
        if cfg is None or self._latest_depth is None:
            return None

        dests = (cfg.get('destinations') or {}).get(self._active_domain or '')
        if not dests:
            self.get_logger().warn(
                f'depth place 검증: 도메인 {self._active_domain!r} 목적지 ROI 없음 → None',
                throttle_duration_sec=30.0,
            )
            return None

        try:
            depth_m = depth_msg_to_meters(
                self._latest_depth, float(cfg.get('depth_scale', 0.001))
            )
        except ValueError as e:
            self.get_logger().warn(f'depth place 검증: depth 디코딩 실패: {e} → None')
            return None

        threshold = float(cfg.get('depth_change_threshold', 0.02))
        h, w = depth_m.shape
        any_measured = False

        for d in dests:
            roi = d.get('roi_px')
            baseline = d.get('baseline_depth')
            if not roi or baseline is None:
                continue
            med = self._roi_median_depth(depth_m, roi, w, h)
            if med is None:
                continue
            any_measured = True
            if float(baseline) - med >= threshold:
                self.get_logger().info(
                    f"depth place: {d.get('name', '?')} median={med:.3f}m "
                    f"< baseline-thr={float(baseline) - threshold:.3f}m → place 흔적"
                )
                return True

        if not any_measured:
            return None  # 목적지 ROI 에 유효 depth 없음 → 판정 불가
        return False

    @staticmethod
    def _roi_median_depth(depth_m: np.ndarray, roi_px, w: int, h: int) -> Optional[float]:
        """
        픽셀 ROI 내 유효 depth(0/inf/범위 밖 제외) median (m). 유효 픽셀 없으면 None.
        roi_px = [x0, y0, x1, y1] (순서 무관, 이미지 경계로 클램프).
        """
        x0, y0, x1, y1 = (int(v) for v in roi_px)
        x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        sub = depth_m[y0:y1, x0:x1]
        valid = np.isfinite(sub) & (sub > 0.05) & (sub < 5.0)
        if not bool(np.any(valid)):
            return None
        return float(np.median(sub[valid]))

    # ── llama-server HTTP / 파싱 헬퍼 ──────────────

    def _chat_completion(self, messages: List[dict], max_tokens: int = 256) -> Optional[dict]:
        """
        llama-server /v1/chat/completions 호출 → choices[0].message dict 반환.

        chat_template_kwargs.enable_thinking=false 필수 (deploy/README.md 실기 검증):
          <|think|> 비활성. 없으면 함수 호출 JSON 이 content 가 아닌 reasoning_content
          로 우회된다 (_message_text / _parse_function_call 에 fallback 도 둠).

        HTTP 실패 시 None 반환 (호출 측이 파싱 실패로 처리).
        """
        body = {
            'messages': messages,
            'temperature': 0.0,          # 결정론적 (재현성)
            'max_tokens': max_tokens,
            'stream': False,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        try:
            r = requests.post(
                f'{self.server_url}/v1/chat/completions',
                json=body,
                timeout=_HTTP_TIMEOUT_SEC,
            )
            r.raise_for_status()
            data = r.json()
            return data['choices'][0]['message']
        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
            self.get_logger().error(f'/v1/chat/completions 실패: {e}')
            return None

    @staticmethod
    def _message_text(message: Optional[dict]) -> str:
        """
        message 에서 본문 텍스트 추출. content 가 비면 reasoning_content 로 fallback
        (enable_thinking 우회 케이스 방어 — deploy/README.md).
        """
        if not message:
            return ''
        text = (message.get('content') or '').strip()
        if not text:
            text = (message.get('reasoning_content') or '').strip()
        return text

    def _parse_function_call(self, message: Optional[dict]) -> Optional[dict]:
        """
        message → {"function": str, "args": dict} 추출. 실패 시 None.
        content/reasoning_content fallback + 본문 군더더기 JSON 추출 포함.
        """
        obj = self._extract_json(self._message_text(message))
        if (isinstance(obj, dict)
                and isinstance(obj.get('function'), str)
                and isinstance(obj.get('args', {}), dict)):
            return {'function': obj['function'], 'args': obj.get('args', {})}
        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[object]:
        """문자열에서 JSON 객체 파싱. 직접 실패 시 첫 {...} 블록 추출 재시도."""
        if not text:
            return None
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            pass
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _feed_tool_result(messages: List[dict], value: str) -> None:
        """
        tool_result 를 '[tool_result] ' prefix 가 붙은 user 메시지로 추가.
        (function_call 학습 분포 = TOOL_RESULT_PREFIX 와 일치 — generate_function_call_qa.)
        """
        messages.append(
            {'role': 'user', 'content': '[tool_result] ' + str(value)}
        )

    def _rgb_to_data_uri(self, img_msg: Image) -> str:
        """sensor_msgs/Image(RGB) → PNG base64 data URI (OpenAI image_url 형식).

        cv_bridge 는 NumPy 2.x ABI 문제로 회피 (utils.ros_image 와 동일 사유).
        PIL 인코딩 — Jetson 배포 env 에 pillow 필요.
        """
        from PIL import Image as PILImage  # noqa: PLC0415 (lazy: 텍스트 경로 import 보호)

        arr = image_msg_to_rgb_array(img_msg)
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/png;base64,{b64}'

    # ── 서비스 호출 헬퍼 (워커 스레드) ────────────

    def _wait_future(self, future, timeout: float):
        """
        call_async future 를 워커 스레드에서 동기 대기 (결정 ①).
        이미 executor 가 스핀 중이라 spin_until_future_complete 사용 불가 →
        done_callback 으로 Event 를 깨운다.
        """
        ev = threading.Event()
        future.add_done_callback(lambda _f: ev.set())
        if not ev.wait(timeout):
            return None
        try:
            return future.result()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'서비스 future 예외: {e}')
            return None

    def _embed_text(self, text: str) -> List[float]:
        """
        /embed_text 서비스 호출 → language_embedding (실패 시 빈 리스트).
        """
        if not self.embed_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/embed_text 서비스 없음 → 빈 임베딩')
            return []
        req = EmbedText.Request()
        req.text = text
        future = self.embed_client.call_async(req)
        res = self._wait_future(future, _SERVICE_TIMEOUT_SEC)
        if res is None:
            self.get_logger().error('/embed_text 응답 없음/타임아웃 → 빈 임베딩')
            return []
        return list(res.embedding)

    # ── AgentLog 발행 헬퍼 ────────────────────────

    def _emit_log(
        self,
        thought: str,
        tool: str,
        args: str,
        result: str,
    ) -> None:
        msg = AgentLog()
        msg.timestamp   = self.get_clock().now().to_msg().sec.__str__()
        msg.thought     = thought
        msg.tool_called = tool
        msg.tool_args   = args
        msg.tool_result = result
        self.log_pub.publish(msg)


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    # 결정 ①: MultiThreadedExecutor — 워커가 대기하는 동안 콜백/서비스 응답이 흐른다.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
