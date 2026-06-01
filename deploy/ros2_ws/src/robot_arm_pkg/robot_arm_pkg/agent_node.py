"""
agent_node.py

Gemma 4 E2B VLA 에이전트의 함수 호출 루프를 오케스트레이션하는 ROS2 노드.

흐름:
  /user_command 수신
    → 도메인 분류 (domain_map.yaml 키워드 매칭)
    → /domain_selection 발행 → llama_server_manager가 LoRA hot-swap
    → /selected_domain 수신 대기
    → 함수 호출 루프:
         look_scene(prompt)        → Gemma VLM 호출 → 자연어 target_description
         /embed_text 서비스 호출   → language_embedding 획득
         execute_subtask(...)      → SubtaskCommand 발행 → ExecutionStatus 대기
         verify_result(...)        → Gemma 시각 검증 + depth grasp 보조 검증
            → SUCCESS / FAIL / PARTIAL / UNCLEAR
            → SUCCESS면 종료, 그 외 max_verify_iterations(기본 3)까지 재시도
            → 초과 시 사용자 개입 대기 상태로 전환

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

파라미터:
  server_url             (str) — llama-server 주소
  domain_map             (str) — domain_map.yaml 경로
  grasp_roi              (str) — grasp_roi.yaml 경로 (verify_result용)
  max_verify_iterations  (int) — 검증 재시도 한계 (기본 3)

미구현 항목 (구현 시 추가 결정 필요):
  _classify_domain()  — 키워드 매칭은 단순화된 placeholder. 실제 매칭 정책
                        (대소문자/형태소/동의어 처리)은 운영 중 확정.
  _call_look_scene()         — Gemma /v1/chat/completions 함수 호출 페이로드 구성
  _call_execute_subtask()    — SubtaskCommand 발행 + ExecutionStatus 대기 로직
  _call_verify_result()      — Gemma 시각 검증 + depth ROI 검증 종합
  _depth_grasp_check()       — grasp_roi.yaml 기반 depth 변화 측정
  _comb policy (Gemma vs depth) — README 미확정 #1 (depth 우선 / Gemma 우선 / 가중)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

from robot_arm_msgs.msg import (
    SubtaskCommand,
    ExecutionStatus,
    DomainSelection,
    AgentLog,
)
from robot_arm_msgs.srv import EmbedText

# 시스템 프롬프트 / 상태 enum SSoT — robot_arm_pkg.prompts
# (학습 측 generate_function_call_qa 도 같은 모듈을 import 한다)
from robot_arm_pkg.prompts import (  # noqa: F401
    SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT, VerifyStatus, ExecStatus,
)


class AgentNode(Node):

    def __init__(self):
        super().__init__('agent_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('server_url',            'http://localhost:8080')
        self.declare_parameter('domain_map',
                               os.path.expanduser('~/ros2_ws/config/domain_map.yaml'))
        self.declare_parameter('grasp_roi',
                               os.path.expanduser('~/calibration/grasp_roi.yaml'))
        self.declare_parameter('max_verify_iterations', 3)

        self.server_url           : str = self.get_parameter('server_url').value
        self.domain_map_path      : str = self.get_parameter('domain_map').value
        self.grasp_roi_path       : str = self.get_parameter('grasp_roi').value
        self.max_verify_iterations: int = self.get_parameter('max_verify_iterations').value

        # ── 설정 로드 ─────────────────────────────
        self._domain_map: Dict[str, str] = self._load_domain_map(self.domain_map_path)
        self._grasp_roi:  Optional[dict] = self._load_grasp_roi(self.grasp_roi_path)

        # ── 입력 캐시 (latest-only) ───────────────
        self._latest_rgb:    Optional[Image]      = None
        self._latest_depth:  Optional[Image]      = None
        self._latest_joints: Optional[JointState] = None

        # ── 진행 상태 ─────────────────────────────
        # _user_command:       현재 처리 중인 사용자 명령 (None이면 idle)
        # _active_domain:      hot-swap 통보를 받은 후 확정되는 활성 도메인
        # _pending_subtask_id: SubtaskCommand 발행 후 ExecutionStatus 대기 중인 id
        # _last_status:        가장 최근 ExecutionStatus
        # _verify_count:       verify_result 누적 호출 수 (한 사용자 명령 단위)
        self._user_command:       Optional[str]              = None
        self._active_domain:      Optional[str]              = None
        self._pending_subtask_id: Optional[str]              = None
        self._last_status:        Optional[ExecutionStatus]  = None
        self._verify_count:       int                        = 0

        # ── QoS ───────────────────────────────────
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 구독 ──────────────────────────────────
        self.create_subscription(String,          '/user_command',     self._user_cmd_callback,    10)
        self.create_subscription(Image,           '/camera/rgb',       self._rgb_callback,         camera_qos)
        self.create_subscription(Image,           '/camera/depth',     self._depth_callback,       camera_qos)
        self.create_subscription(JointState,      '/joint_states',     self._joints_callback,      10)
        self.create_subscription(ExecutionStatus, '/execution_status', self._exec_status_callback, 10)
        self.create_subscription(String,          '/selected_domain',  self._selected_callback,    10)

        # ── 발행 ──────────────────────────────────
        self.domain_pub  = self.create_publisher(DomainSelection, '/domain_selection', 10)
        self.subtask_pub = self.create_publisher(SubtaskCommand,  '/subtask_command',  10)
        self.log_pub     = self.create_publisher(AgentLog,        '/agent_log',        10)

        # ── 서비스 클라이언트 ─────────────────────
        self.embed_client = self.create_client(EmbedText, '/embed_text')

        self.get_logger().info(
            f'agent_node 시작 | server: {self.server_url} | '
            f'domain_map: {self.domain_map_path} | '
            f'max_verify_iterations: {self.max_verify_iterations}'
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

    def _load_grasp_roi(self, path: str) -> Optional[dict]:
        """
        grasp_roi.yaml 로드. verify_result의 depth 검증에 사용.
        파일 없으면 None 반환 → depth 검증 비활성.
        """
        try:
            import yaml  # noqa: PLC0415
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().warn(
                f'grasp_roi.yaml 없음: {path} → depth grasp 검증 비활성 (Gemma 단독 검증)'
            )
            return None
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'grasp_roi.yaml 파싱 실패: {e}')
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
        """
        if self._user_command is not None:
            self.get_logger().warn(
                f'이미 처리 중인 명령이 있어 새 명령 무시: {msg.data!r}'
            )
            return

        self._user_command = msg.data
        self._verify_count = 0
        self.get_logger().info(f'/user_command 수신: {msg.data!r}')

        domain = self._classify_domain(msg.data)
        if domain is None:
            self.get_logger().error(
                f'도메인 분류 실패 (매칭되는 키워드 없음): {msg.data!r}'
            )
            self._user_command = None
            return

        sel = DomainSelection()
        sel.domain = domain
        sel.scale  = 1.0
        self.domain_pub.publish(sel)
        self.get_logger().info(f'/domain_selection 발행: domain={domain!r}')

    def _selected_callback(self, msg: String) -> None:
        """
        llama_server_manager가 hot-swap 완료를 통보 → 함수 호출 루프 시작.
        """
        if self._user_command is None:
            return  # 처리 중인 명령 없음 (외부에서 임의 발행 등)

        self._active_domain = msg.data
        self.get_logger().info(
            f'hot-swap 완료 통보 수신: domain={msg.data!r} → 함수 호출 루프 시작'
        )
        self._run_function_loop()

    def _exec_status_callback(self, msg: ExecutionStatus) -> None:
        """
        act_node가 SubtaskCommand 실행 결과 통보.
        execute_subtask 호출 측이 대기 중일 때만 의미를 가진다.
        """
        if msg.subtask_id != self._pending_subtask_id:
            return  # 다른 subtask 응답 (아마 stale)

        self._last_status = msg
        self.get_logger().info(
            f'/execution_status 수신: id={msg.subtask_id!r} status={msg.status!r}'
        )
        # 실제 루프 진행은 _run_function_loop의 루프에서 폴링 / 또는 후속 콜백에서 처리.
        # 단일 스레드 spin 환경에서 동기 대기를 구현하려면 별도 callback group이 필요.
        # 여기서는 미구현 영역으로 남기고 _run_function_loop에서 NotImplementedError로 명시한다.

    # ── 도메인 분류 ───────────────────────────────

    def _classify_domain(self, user_command: str) -> Optional[str]:
        """
        사용자 명령에서 키워드를 찾아 도메인을 결정.

        현재는 _domain_map(flat keyword → domain) 안의 키워드가 명령 문자열에
        포함되는지를 단순 substring 검사로 판단한다. 가장 먼저 매칭되는 항목을
        반환. 매칭 우선순위, 동의어 정규화, 다국어 처리 등은 README 미확정.

        매칭 실패 시 None.
        """
        for keyword, domain in self._domain_map.items():
            if keyword in user_command:
                return domain
        return None

    # ── 함수 호출 루프 ────────────────────────────

    def _run_function_loop(self) -> None:
        """
        Gemma 함수 호출 루프 본체.
          look_scene → execute_subtask → (/go_home) → verify_result
          → SUCCESS면 종료, 그 외는 max_verify_iterations까지 재시도.

        verify 직전 /go_home 강제 (2026-05-25 freeze):
          execute_subtask 의 ExecutionStatus 수신 후 verify_result 호출 전에
          /go_home 서비스를 1회 호출한다. 이로써 카메라 시야 안의 팔 자세가
          매번 home_pose 로 고정되어 verify 학습 분포 (모두 home_pose 로 렌더)
          와 일치한다. /go_home 은 약 ~2초 차단성 호출이며, 그 동안 카메라
          프레임은 latest-only 캐시이므로 verify_result 가 받는 RGB 는 복귀
          완료 후의 한 프레임.

        TODO: 구현 시 추가 결정 필요
          1) 동기/비동기 실행 모델
             - 본 노드는 ROS2 콜백에서 HTTP 호출과 ExecutionStatus 대기를
               함께 수행해야 한다. 단일 스레드 spin에서는 차단 호출을 그대로
               쓰면 다른 콜백이 멈춘다.
             - 권장: rclpy MultiThreadedExecutor + ReentrantCallbackGroup으로
               서비스 응답 / status 응답 / Gemma HTTP를 동시 처리 가능하게 함.
               (변경 시 main()도 함께 수정 필요)

          2) Gemma 함수 호출 프로토콜
             - llama-server --jinja 활성 시 OpenAI 호환 함수 호출 형식 사용
             - 시스템 프롬프트에 look_scene/execute_subtask/verify_result 시그니처
               선언 → 모델이 JSON으로 함수 호출 응답
             - 본 메서드는 응답 파싱 후 매칭되는 _call_*() 디스패치
               (function_call 학습 데이터의 형식과 일치해야 함 — README Q&A 데이터 생성)

          3) 종료 처리
             - SUCCESS: self._user_command = None, _active_domain 유지(다음 명령 시 재평가)
             - max 초과: WAITING_USER 상태로 진입. 현재는 단순히 _user_command를
               유지하고 사용자 개입 토픽(예: /user_intervention) 대기 — 토픽
               명세 미확정.
        """
        raise NotImplementedError(
            "[agent_node] _run_function_loop() 미구현. "
            "Gemma 함수 호출 프로토콜과 동기화 모델 확정 후 구현하세요."
        )

    # ── Gemma 함수 호출 핸들러 (개별) ─────────────

    def _call_look_scene(self, prompt: str) -> Optional[str]:
        """
        Gemma VLM에 RGB + prompt 입력 → 자연어 target_description 반환.

        반환 형식 (엄격, LoRA가 이 분포로 학습됨):
          영어 반구조화 템플릿  "the {attribute} {object} {spatial}"
          - 5~15 단어
          - vocabulary: deploy/config/label_vocab.yaml과 일치
          - 예: "the blue can on the right side of the table"

        검증 권장 (구현 시):
          - 정규식으로 템플릿 매칭, 실패 시 1회 재시도
          - vocabulary 외 단어 포함 시 재시도 또는 강제 매핑

        TODO: 구현 시 결정
          - llama-server /v1/chat/completions 페이로드 (이미지 base64 임베드)
          - 시스템 프롬프트는 robot_arm_pkg.prompts.SYSTEM_PROMPT 를 그대로 사용
            (LoRA 학습 분포와 일치 보장 — function_call 합성기도 같은 상수를 import)
          - 응답 파싱(JSON 함수 호출) 또는 그냥 본문 텍스트
          - 최신 RGB 미수신 시 처리 정책 (대기 / 즉시 실패)
        """
        raise NotImplementedError("[agent_node] _call_look_scene() 미구현")

    def _call_execute_subtask(self, target_description: str) -> Optional[ExecutionStatus]:
        """
        target_description을 임베딩 → SubtaskCommand 발행 → ExecutionStatus 수신까지 대기.

        TODO: 구현 시 결정
          - subtask_id 발급 정책 (UUID? 카운터?)
          - ExecutionStatus 대기 타임아웃
          - 임베딩 서비스 실패 시 fallback (재시도 / 빈 임베딩)
        """
        raise NotImplementedError("[agent_node] _call_execute_subtask() 미구현")

    def _call_verify_result(self, original_command: str) -> str:
        """
        verify_result 함수 명세 (README "verify_result 함수 명세") 구현.

        Stateless 인터페이스로 freeze (2026-05-25):
          Gemma 가 함수 호출 시 original_command 를 args 로 전달하며, agent_node
          는 self._user_command 캐시 대신 인자를 우선 사용한다 (function_call 학습 분포
          및 robot_arm_pkg.prompts.SYSTEM_PROMPT 와 일치).

        전제 (2026-05-25 freeze):
          호출 시점에 팔이 home_pose 에 있다 — _run_function_loop 가 verify
          호출 직전에 /go_home 을 보장한다. verify 학습 분포가 home_pose 단일로
          렌더되므로 본 함수가 캡처하는 RGB 도 같은 자세여야 함.

        시각 검증 호출 (Gemma):
          별도 chat completion (function_call 함수 호출 대화와 분리) 로 다음을 보낸다:
            system : robot_arm_pkg.prompts.VERIFY_SYSTEM_PROMPT
            user   : [image (현재 RGB), text "original_command: \\"...\\""]
          응답: {"status": <enum>, "reasoning": <english string>}

        입력:
          - original_command (인자): Gemma 가 전달하는 원본 사용자 명령 문자열
          - 현재 RGB / depth (latest-only 캐시)
          - 직전 subtask 정보 (subtask_id / target_description) — 노드 상태

        처리:
          1) Gemma 시각 검증 (위 별도 호출)
          2) depth grasp 보조 검증 — _depth_grasp_check()
          3) 종합 판정 — README 미확정 #1 (depth 우선 / Gemma 우선 / 가중평균)

        출력: VerifyStatus.SUCCESS | FAIL | PARTIAL | UNCLEAR (문자열)
        """
        raise NotImplementedError("[agent_node] _call_verify_result() 미구현")

    def _depth_grasp_check(self) -> Optional[bool]:
        """
        grasp_roi.yaml로 정의된 ROI 내 depth 변화를 측정하여 grasp 성공 여부 산출.

        TODO: 구현 시 결정
          - ROI frame 변환 (camera | base_link)
          - 이전 시점 depth와의 비교 vs threshold 절대값 비교
          - 노이즈 처리 (median 필터 등)
          - self._grasp_roi == None일 때 None 반환 → verify_result에서 depth 신호 무시
        """
        raise NotImplementedError("[agent_node] _depth_grasp_check() 미구현")

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
