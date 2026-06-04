"""
act_node.py

LeRobot ACT Policy 로 관절 시퀀스를 추론하는 노드 (closed-loop).

이전 설계와 달리 본 노드는 다음 의존을 모두 제거한다:
  - /target_class (vlm_node)
  - /object_pose  (yolo_node)
  - /task_active  (state_monitor_node)
  - observation.object_pose 채널

대신 agent_node가 보낸 SubtaskCommand(language_embedding 포함)를 받아 ACT 추론을
시작하고, 정해진 horizon 만큼 실행한 뒤 ExecutionStatus 를 발행한다.
language_embedding 은 SubtaskCommand 안에 사전 임베딩되어 도착하므로 act_node 는
추론 시점에 임베딩 변환 지연이 없다.

제어 방식 — closed-loop (2026-06-04 결정):
  매 타이머 틱마다 최신 obs(RGB + 관절 + 언어 임베딩)로 policy.select_action() 을
  호출해 1 스텝씩 행동을 발행한다. ACT 의 내부 action queue / temporal ensemble 이
  chunk 를 관리하므로 open-loop 무피드백 재생보다 drift 가 적다. subtask 시작 시
  policy.reset() 으로 queue 를 비운다.

종료(SUCCESS) 의미 — 실행 완료이지 성공 판정이 아님:
  ACT 는 자연스러운 "완료" 신호가 없으므로 horizon_steps 만큼 실행하면 SUCCESS(=실행
  완료) 를 발행한다. 실제 성공/실패 판정은 agent_node 의 verify_result(Gemma + depth
  place 검증)가 담당하며, 실패 시 agent_node 가 look→execute→verify 시퀀스를 재시도한다.

환경:
  Python   3.10
  ROS2     Humble (Ubuntu 22.04 LTS)
  LeRobot  0.4.4 — ACTPolicy.from_pretrained(pretrained_model 디렉터리)

구독 토픽:
  /camera/rgb       (sensor_msgs/Image, BestEffort+KeepLast(1))
  /joint_states     (sensor_msgs/JointState) — robot_controller 피드백
  /subtask_command  (robot_arm_msgs/SubtaskCommand) — agent_node 발행

발행 토픽:
  /joint_command    (sensor_msgs/JointState) — publish_hz (subtask 활성 시)
  /execution_status (robot_arm_msgs/ExecutionStatus) — subtask 종료 시 1회

파라미터:
  model_root     (str)   — 도메인별 체크포인트 루트. 도메인 D 의 정책은
                           {model_root}/{D}/pretrained_model. SubtaskCommand.domain
                           이 바뀌면 해당 도메인 정책으로 재로드. (기본 ~/models/act)
  model_path     (str)   — 단일 도메인 명시 override. 설정 시 이 경로만 사용하고
                           도메인 재로드를 하지 않는다(단일 도메인 배포용). 기본 ''.
  chunk_size     (int)   — 참고용 로그 값. 실제 chunk 는 정책 config 가 정의(기본 100).
  publish_hz     (float) — 발행/추론 주기 (기본 30Hz)
  device         (str)   — '' → cuda 가능 시 cuda, 아니면 cpu. 'cpu'/'cuda' 명시 가능.
  horizon_steps  (int)   — subtask 당 실행 스텝 수. 끝나면 SUCCESS(실행 완료) 발행.
                           에피소드 길이에 맞춰 튜닝(기본 300 ≈ 10s@30Hz).

학습 시 obs 스키마(LeRobotDataset)와 추론 시 obs dict 키는 일치해야 함:
  observation.images.top         — RGB 이미지 텐서 (1,3,H,W) float[0,1] (+ normalizer)
  observation.state              — 관절 상태 텐서 (1,6), SDK .pos 단위(아래 ⚠️)
  observation.environment_state  — 자연어 타겟 묘사 임베딩 (1,384)

  ⚠️ 언어 임베딩 키는 반드시 'observation.environment_state' 여야 한다.
     lerobot ACT 는 input feature 를 type 으로 배선하는데(dataset_to_policy_features),
     'observation.*' 는 STATE, 정확히 'observation.environment_state' 만 ENV 로 분류된다.
     ENV 여야 ACT 가 encoder_env_state_input_proj 토큰으로 언어를 실제 사용한다.
     (구 'observation.language_embedding' 키는 중복 STATE 로 분류돼 ACT 가 무시했음.)
     학습 데이터셋도 이 키로 rename 되어 있어야 한다(rename_feature.py).

  ⚠️ observation.state 단위: robot_controller_node 는 /joint_states 를 URDF 단위
     (회전 5축 라디안 + 그리퍼 URDF 스케일)로 발행하지만, ACT 학습 데이터셋의
     observation.state/action 은 LeRobot SDK .pos 단위(회전 degree, 그리퍼 0~100)다.
     따라서 _joint_state_to_pos() 로 /joint_states → .pos 역변환 후 obs 로 쓴다.
     출력 /joint_command 는 .pos 그대로(=action 단위)라 robot_controller 가 변환 없이 소비.

정규화(normalizer): lerobot 0.4.4 는 정규화를 정책 내부가 아니라 별도 processor 로
  처리한다. 검증된 eval_act_offline.py 와 동일하게, 체크포인트의
  *normaliz*.safetensors 버퍼(mean/std)를 직접 읽어 obs 에 (x-mean)/std 를 적용하고
  출력 action 은 역정규화한다. 안 하면 env_state(L2≈0.05) 가 학습 스케일과 어긋나
  언어 토큰이 죽는다(eval [4] 참조).

상태: closed-loop 본체 구현 완료(real-only, 양 도메인). 정식 검증은 Jetson 실기
  rollout(이미지 해상도/단위/언어 민감도 확인). 오프라인 sanity 는
  training/act/scripts/eval_act_offline.py 참조.
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, JointState

from robot_arm_msgs.msg import SubtaskCommand, ExecutionStatus

# ExecutionStatus.status SSoT — 대문자 enum (agent_node 가 case 변환 없이 Gemma
# tool_result 로 전달, function_call 학습 분포도 대문자). 소문자 리터럴 금지.
from robot_arm_pkg.prompts import ExecStatus

# cv_bridge 비의존 이미지 변환 (NumPy 1.x/2.x 양쪽 호환) — robot_controller/agent 와 동일 유틸.
from robot_arm_pkg.utils.ros_image import image_msg_to_rgb_array

# ── GPU 메모리 단편화 회피 (torch import / CUDA init 전에 설정해야 효력) ──
# 8GB Orin Nano 의 unified memory 에서 llama-server(GPU 상주)와 ACT 가 공존할 때,
# torch 기본 caching allocator 가 큰 연속 블록을 잡아 GPU 단편화가 생기면, 메모리
# 총량이 충분해도(예: llama 2.3G + act 0.6G ≪ 8G) NvMap ENOMEM 으로 cuda 로드가
# 실패한다(2026-06-04 Jetson Orin Nano 실기 확인). expandable_segments 로 확장형
# 세그먼트 할당을 써 이를 회피한다. torch 는 _load_policy 에서 lazy import 되므로
# 모듈 로드 시점의 이 설정이 그 전에 적용된다. 외부에서 PYTORCH_CUDA_ALLOC_CONF 를
# 지정하면 그 값을 존중한다(setdefault).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# SO-101 관절 이름 — robot_controller_node.py 와 동일한 순서 유지
SO101_JOINT_NAMES: List[str] = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]

# 그리퍼 URDF 한계 — /joint_states ↔ SDK .pos(0~100) 역변환용.
# ⚠️ robot_controller_node._GRIPPER_URDF_LOWER/UPPER 와 반드시 동일(SSoT: robot_controller_node).
_GRIPPER_URDF_LOWER = -0.174533
_GRIPPER_URDF_UPPER = 1.74533

ACTION_KEY = 'action'


class ACTNode(Node):

    def __init__(self):
        super().__init__('act_node')

        # ── 파라미터 ──────────────────────────────
        self.declare_parameter('model_root', os.path.expanduser('~/models/act'))
        self.declare_parameter('model_path', '')   # 단일 도메인 override (설정 시 재로드 비활성)
        self.declare_parameter('chunk_size', 100)   # 참고용 — 실제 chunk 는 정책 config
        self.declare_parameter('publish_hz', 30.0)
        self.declare_parameter('device', '')        # '' → cuda 가능 시 cuda, 아니면 cpu
        self.declare_parameter('horizon_steps', 300)

        self.model_root     = os.path.expanduser(self.get_parameter('model_root').value)
        self._model_path_ovr = self.get_parameter('model_path').value
        self.chunk_size     = self.get_parameter('chunk_size').value
        publish_hz          = self.get_parameter('publish_hz').value
        self._device_param  = self.get_parameter('device').value
        self.horizon_steps  = int(self.get_parameter('horizon_steps').value)

        # ── 런타임 핸들 (torch/lerobot 은 _load_policy 에서 lazy import) ──
        self._torch = None
        self.policy = None
        self.device = 'cpu'
        self._norm: dict = {}                 # key -> (mean, std) 텐서
        self._image_key = 'observation.images.top'
        self._state_key = 'observation.state'
        self._env_key   = 'observation.environment_state'
        self._img_hw: Optional[tuple] = None  # (H, W) 정책 입력 해상도 (config 에서)
        self._loaded_domain: Optional[str] = None
        self._loaded_path:   Optional[str] = None

        # ── 입력 캐시 ─────────────────────────────
        self._latest_rgb:    Optional[Image]      = None
        self._latest_joints: Optional[JointState] = None

        # ── 활성 subtask 상태 ─────────────────────
        # _active_cmd: 현재 실행 중인 SubtaskCommand (None이면 idle)
        # _step:       현재 subtask 에서 발행한 스텝 수 (>= horizon_steps 면 종료)
        self._active_cmd: Optional[SubtaskCommand] = None
        self._step:       int                      = 0

        # ── 단일 도메인 override 면 즉시 로드. 아니면 첫 subtask 에서 도메인별 로드. ──
        if self._model_path_ovr:
            self._ensure_policy(domain=None, path=os.path.expanduser(self._model_path_ovr))

        # ── QoS ───────────────────────────────────
        # camera는 BestEffort + KeepLast(1) — 30Hz 스트림에서 latest-only.
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 구독 ──────────────────────────────────
        self.create_subscription(Image,          '/camera/rgb',      self._rgb_callback,     camera_qos)
        self.create_subscription(JointState,     '/joint_states',    self._joints_callback,  10)
        self.create_subscription(SubtaskCommand, '/subtask_command', self._subtask_callback, 10)

        # ── 발행 ──────────────────────────────────
        self.cmd_pub    = self.create_publisher(JointState,      '/joint_command',    10)
        self.status_pub = self.create_publisher(ExecutionStatus, '/execution_status', 10)

        # ── 타이머 ────────────────────────────────
        self.create_timer(1.0 / publish_hz, self._timer_callback)

        self.get_logger().info(
            f'act_node 시작 | model_root: {self.model_root} | '
            f'override: {self._model_path_ovr or "(없음)"} | device: {self.device} | '
            f'chunk_size: {self.chunk_size} | horizon: {self.horizon_steps} | {publish_hz}Hz'
        )

    # ── Policy 로드 / 도메인 선택 ──────────────────

    def _ensure_policy(self, domain: Optional[str], path: Optional[str] = None) -> bool:
        """
        주어진 도메인(또는 명시 path)의 정책이 로드돼 있도록 보장. 필요 시 재로드.

        Returns:
            로드 성공(또는 이미 로드됨) 시 True, 실패 시 False.
        """
        if path is None:
            if not domain:
                self.get_logger().error('[act_node] domain 미지정 + model_path override 없음 — 로드 불가.')
                return False
            path = os.path.join(self.model_root, domain, 'pretrained_model')

        if self.policy is not None and path == self._loaded_path:
            return True  # 이미 같은 정책 로드됨 — 재로드 생략

        policy = self._load_policy(path)
        if policy is None:
            return False
        self.policy         = policy
        self._loaded_path   = path
        self._loaded_domain = domain
        return True

    def _load_policy(self, path: str):
        """
        LeRobot ACT 체크포인트(pretrained_model 디렉터리)를 로드.

        torch/lerobot 미설치 또는 경로/로드 실패 시 None 반환(노드는 계속 동작하며
        timer 가 warn 만 낸다 — 빌드/실행 분리를 위해 import 를 lazy 로 둔다).
        config.json 에서 obs 키/이미지 해상도를 읽고, *normaliz*.safetensors 의
        mean/std 버퍼를 적재한다(eval_act_offline.py 와 동일 패턴).
        """
        if not os.path.isdir(path):
            self.get_logger().error(f'[act_node] 체크포인트 디렉터리 없음: {path}')
            return None
        try:
            import torch
            try:
                from lerobot.policies.act.modeling_act import ACTPolicy
            except Exception:
                from lerobot.common.policies.act.modeling_act import ACTPolicy  # 구버전
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f'[act_node] torch/lerobot import 실패 — 추론 비활성: {type(e).__name__}: {e}'
            )
            return None

        # device 결정
        if self._device_param:
            self.device = self._device_param
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._torch = torch

        # config.json → obs 키 / 이미지 해상도
        self._read_policy_config(path)

        try:
            policy = ACTPolicy.from_pretrained(path)
            policy.to(self.device)
            policy.eval()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(
                f'[act_node] ACTPolicy 로드 실패 ({path}): {type(e).__name__}: {e}'
            )
            return None

        # normalizer 버퍼 (mean/std) — obs 정규화 + action 역정규화에 사용
        self._norm = self._load_norm(path, [self._image_key, self._state_key, self._env_key, ACTION_KEY])

        self.get_logger().info(
            f'[act_node] 정책 로드 OK | {path} | device={self.device} | '
            f'img_hw={self._img_hw} | env_key={self._env_key} | '
            f'normalized={sorted(self._norm.keys())}'
        )
        if not self._norm:
            self.get_logger().warn(
                '[act_node] normalizer 버퍼 없음 — raw obs 사용. 언어 조건화/스케일이 '
                '학습과 어긋날 수 있음(eval_act_offline.py 로 확인 권장).'
            )
        return policy

    def _read_policy_config(self, path: str) -> None:
        """config.json 의 input_features 에서 이미지 키/해상도/env 키를 추출."""
        cfg_p = os.path.join(path, 'config.json')
        if not os.path.exists(cfg_p):
            self.get_logger().warn(
                f'[act_node] config.json 없음({cfg_p}) — 기본 obs 키/카메라 native 해상도 사용.'
            )
            return
        try:
            cfg = json.load(open(cfg_p, encoding='utf-8'))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'[act_node] config.json 파싱 실패 — 기본값 사용: {e}')
            return
        inp = cfg.get('input_features', {}) or {}

        img_keys = [k for k in inp if 'image' in k]
        if img_keys:
            self._image_key = img_keys[0]
        env_keys = [
            k for k in inp
            if k.startswith('observation.') and 'image' not in k and k != 'observation.state'
        ]
        if env_keys:
            self._env_key = env_keys[0]
            if 'environment_state' not in self._env_key:
                self.get_logger().warn(
                    f'[act_node] 언어 obs 키가 {self._env_key!r} — ENV 토큰 배선은 '
                    "'observation.environment_state' 만 동작(rename_feature.py 확인)."
                )

        shp = (inp.get(self._image_key) or {}).get('shape')
        if shp and len(shp) == 3:
            # lerobot 은 보통 [C, H, W]. 첫 축이 채널(1/3)이면 그렇게, 아니면 [H, W, C] 로 해석.
            if int(shp[0]) in (1, 3):
                self._img_hw = (int(shp[1]), int(shp[2]))
            else:
                self._img_hw = (int(shp[0]), int(shp[1]))

    def _load_norm(self, path: str, keys: List[str]) -> dict:
        """*normaliz*.safetensors 에서 key.mean / key.std 버퍼를 적재."""
        norm: dict = {}
        try:
            from safetensors.torch import load_file
            nf = glob.glob(os.path.join(path, '*normaliz*.safetensors'))
            if not nf:
                return norm
            buf = load_file(nf[0])
            for k in keys:
                m, s = buf.get(f'{k}.mean'), buf.get(f'{k}.std')
                if m is not None and s is not None:
                    norm[k] = (m.float().to(self.device), s.float().to(self.device))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'[act_node] normalizer 로드 실패 — raw 사용: {type(e).__name__}: {e}')
        return norm

    # ── 구독 콜백 ─────────────────────────────────

    def _rgb_callback(self, msg: Image) -> None:
        self._latest_rgb = msg

    def _joints_callback(self, msg: JointState) -> None:
        self._latest_joints = msg

    def _subtask_callback(self, msg: SubtaskCommand) -> None:
        """
        agent_node가 보낸 SubtaskCommand 수신 → 새 subtask 활성화.

        같은 subtask_id가 재수신되면 무시한다 (중복 방지).
        도메인이 바뀌면 해당 도메인 정책으로 재로드한다(model_path override 시 생략).
        새 subtask 시작 시 policy.reset() 으로 action queue 를 비운다.
        """
        if self._active_cmd is not None and self._active_cmd.subtask_id == msg.subtask_id:
            return

        self.get_logger().info(
            f'SubtaskCommand 수신: id={msg.subtask_id!r} domain={msg.domain!r} '
            f'target={msg.target_description!r} embed_dim={len(msg.language_embedding)}'
        )

        # 도메인별 정책 보장 (override 모드면 도메인 무시하고 고정 정책 사용)
        if not self._model_path_ovr:
            if not self._ensure_policy(domain=msg.domain):
                self._fail_now(msg, f'정책 로드 실패 (domain={msg.domain!r})')
                return

        if self.policy is None:
            self._fail_now(msg, '정책 미로드 (torch/lerobot/체크포인트 확인)')
            return

        try:
            self.policy.reset()  # action queue / temporal ensemble 초기화
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'[act_node] policy.reset() 경고: {e}')

        self._active_cmd = msg
        self._step       = 0

    # ── 타이머 콜백 (publish_hz) ───────────────────

    def _timer_callback(self) -> None:
        if self._active_cmd is None:
            return  # idle

        if self.policy is None:
            self.get_logger().warn(
                '[act_node] policy 미로드 — 추론 불가.', throttle_duration_sec=5.0
            )
            return

        obs = self._build_obs(self._active_cmd)
        if obs is None:
            # RGB/joint 미수신 — 입력 대기(스텝 미진행).
            self.get_logger().warn(
                '[act_node] RGB/joint_states 미수신, 예측 보류', throttle_duration_sec=2.0
            )
            return

        try:
            with self._torch.no_grad():
                action = self.policy.select_action(obs)
            action = action.squeeze(0).detach().cpu().numpy()
            action = self._unnorm_action(action)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'[act_node] 추론 실패 — {type(e).__name__}: {e}')
            self._finish_subtask(ExecStatus.FAIL, f'inference error: {e}', 0.0)
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = SO101_JOINT_NAMES
        msg.position     = [float(v) for v in action.tolist()]
        self.cmd_pub.publish(msg)
        self._step += 1

        # horizon 도달 → 실행 완료(SUCCESS). 실제 성공 판정은 agent_node.verify_result.
        if self._step >= self.horizon_steps:
            self._finish_subtask(ExecStatus.SUCCESS, '', 0.0)

    # ── obs 구성 ──────────────────────────────────

    def _build_obs(self, cmd: SubtaskCommand) -> Optional[dict]:
        """
        최신 입력으로 정규화된 obs dict 구성. RGB/joint 미수신 시 None.
        키는 학습 config 의 input_features 와 일치해야 한다.
        """
        if self._latest_rgb is None or self._latest_joints is None:
            return None
        torch = self._torch

        # 이미지: (H,W,3) uint8 RGB → (필요 시 resize) → (1,3,H,W) float[0,1] → normalizer
        try:
            arr = image_msg_to_rgb_array(self._latest_rgb)  # (H,W,3) uint8
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'[act_node] 이미지 변환 실패: {e}', throttle_duration_sec=2.0)
            return None
        if self._img_hw is not None and arr.shape[:2] != self._img_hw:
            arr = self._resize_rgb(arr, self._img_hw)
        img = arr.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))                       # CHW
        img_t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(self.device)
        img_t = self._apply_norm(self._image_key, img_t)

        # 관절: /joint_states(rad+urdf) → SDK .pos(deg+0~100) → (1,6) → normalizer
        state = self._joint_state_to_pos(self._latest_joints)    # (6,) float32
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        state_t = self._apply_norm(self._state_key, state_t)

        # 언어 임베딩: 사전 임베딩(D,) → (1,D) → normalizer (ENV 토큰)
        emb = np.asarray(cmd.language_embedding, dtype=np.float32)
        env_t = torch.from_numpy(emb).unsqueeze(0).to(self.device)
        env_t = self._apply_norm(self._env_key, env_t)

        return {
            self._image_key: img_t,
            self._state_key: state_t,
            self._env_key:   env_t,
        }

    def _apply_norm(self, key: str, x):
        """normalizer 버퍼가 있으면 (x-mean)/std 적용."""
        if key in self._norm:
            m, s = self._norm[key]
            return (x - m) / (s + 1e-8)
        return x

    def _unnorm_action(self, a: np.ndarray) -> np.ndarray:
        """예측(정규화 공간) action → 원 단위(.pos)."""
        if ACTION_KEY in self._norm:
            m, s = self._norm[ACTION_KEY]
            return a * s.cpu().numpy() + m.cpu().numpy()
        return a

    @staticmethod
    def _resize_rgb(arr: np.ndarray, hw: tuple) -> np.ndarray:
        """(H,W,3) uint8 → 목표 (H,W) 로 resize. PIL 사용(cv2 비의존)."""
        from PIL import Image as PILImage
        h, w = hw
        im = PILImage.fromarray(arr).resize((w, h))  # PIL 은 (W,H)
        return np.asarray(im, dtype=np.uint8)

    def _joint_state_to_pos(self, js: JointState) -> np.ndarray:
        """
        /joint_states (URDF: 회전 라디안 + 그리퍼 URDF 스케일) → LeRobot SDK .pos
        (회전 degree + 그리퍼 0~100). robot_controller_node._read_joints 의 역변환.

        js.name 이 채워져 있으면 이름으로 매핑하고, 아니면 SO101_JOINT_NAMES 순서로 가정.
        """
        if js.name and len(js.name) == len(js.position):
            by_name = dict(zip(js.name, js.position))
            vals = [float(by_name.get(n, 0.0)) for n in SO101_JOINT_NAMES]
        else:
            vals = [float(v) for v in js.position[:6]]
            vals += [0.0] * (6 - len(vals))

        pos = [math.degrees(v) for v in vals[:5]]                # 회전 5축: rad → deg
        frac = (vals[5] - _GRIPPER_URDF_LOWER) / (_GRIPPER_URDF_UPPER - _GRIPPER_URDF_LOWER)
        pos.append(max(0.0, min(1.0, frac)) * 100.0)             # 그리퍼: URDF → 0~100
        return np.asarray(pos, dtype=np.float32)

    # ── subtask 종료 처리 ─────────────────────────

    def _fail_now(self, cmd: SubtaskCommand, reason: str) -> None:
        """아직 활성화 전 단계에서 즉시 FAIL ExecutionStatus 를 발행."""
        self.get_logger().error(f'[act_node] subtask 거부: {reason}')
        msg = ExecutionStatus()
        msg.subtask_id     = cmd.subtask_id
        msg.status         = ExecStatus.FAIL
        msg.failure_reason = reason
        msg.confidence     = 0.0
        self.status_pub.publish(msg)

    def _finish_subtask(self, status: str, failure_reason: str, confidence: float) -> None:
        """
        ExecutionStatus를 1회 발행하고 idle 상태로 복귀.

        status: ExecStatus.SUCCESS | FAIL | TIMEOUT (대문자 — SSoT prompts.ExecStatus).
                agent_node 가 case 변환 없이 Gemma tool_result 로 전달한다.
                SUCCESS 는 '실행 완료' 이지 '성공 판정' 이 아니다(verify_result 가 판정).
        """
        if self._active_cmd is None:
            return

        msg = ExecutionStatus()
        msg.subtask_id     = self._active_cmd.subtask_id
        msg.status         = status
        msg.failure_reason = failure_reason
        msg.confidence     = float(confidence)
        self.status_pub.publish(msg)

        self.get_logger().info(
            f'ExecutionStatus 발행: id={msg.subtask_id!r} status={status!r} steps={self._step}'
        )

        self._active_cmd = None
        self._step       = 0


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ACTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
