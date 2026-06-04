#!/bin/bash
# =============================================================================
# setup_jetson_base.sh
#
# Jetson 기본 환경 설치 스크립트
# 양쪽 워크플로우 공통 베이스:
#   - 검증/배포 (deploy/만 clone): 실제 로봇 구동
#   - 학습 준비 (PAI 전체 clone): 모방 데이터 수집 (record_real_teleop.py)
#
# 전제 조건:
#   - Jetson Orin Nano, JetPack 6.2 (L4T 36.4.x)
#   - Ubuntu 22.04, CUDA 12.6 설치 완료
#   - ROS2 Humble 설치 완료
#
# 실행 방법:
#   chmod +x deploy/scripts/setup_jetson_base.sh
#   ./deploy/scripts/setup_jetson_base.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS2_WS="$DEPLOY_ROOT/ros2_ws"
REQUIREMENTS="$ROS2_WS/src/robot_arm_pkg/requirements.txt"

echo -e "${GREEN}==============================${NC}"
echo -e "${GREEN} Jetson 기본 환경 설치 시작${NC}"
echo -e "${GREEN}==============================${NC}"

# =============================================================================
# 1. ROS2 apt 저장소 등록
# =============================================================================
echo -e "\n${YELLOW}[1/6] ROS2 apt 저장소 확인${NC}"

if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
    echo "ROS2 apt 저장소 등록 중..."
    sudo apt install -y curl
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu jammy main" | \
        sudo tee /etc/apt/sources.list.d/ros2.list
    sudo apt update
    echo -e "${GREEN}ROS2 apt 저장소 등록 완료${NC}"
else
    echo "ROS2 apt 저장소 이미 등록됨. 스킵."
fi

# =============================================================================
# 2. apt 패키지 설치
# =============================================================================
echo -e "\n${YELLOW}[2/6] apt 패키지 설치${NC}"

sudo apt update
sudo apt install -y \
    python3-opencv \
    libopencv-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-ros-base \
    ros-humble-ros2cli \
    ros-humble-ros2launch \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-message-filters \
    ros-humble-cv-bridge

echo -e "${GREEN}apt 패키지 설치 완료${NC}"

# =============================================================================
# 3. pip 패키지 설치 (학습/추론용 venv 대상)
# =============================================================================
echo -e "\n${YELLOW}[3/7] pip 패키지 설치 (venv)${NC}"

pip install -r "$REQUIREMENTS"

echo -e "${GREEN}pip 패키지 설치 완료${NC}"

# =============================================================================
# 4. ROS 런타임용 user-local Python 패키지 설치
# =============================================================================
# ROS 2 노드는 시스템 Python(/usr/bin/python3)으로 실행되므로 학습용 venv와는
# 별개로 ~/.local site-packages에도 핵심 의존성이 필요하다. venv 활성화 여부와
# 무관하게 시스템 python3를 명시적으로 호출해 user-local로만 설치한다.
#
# ⚠️ 설치 순서가 중요하다 (2026-06-04 Jetson Orin Nano 8GB 실기에서 확인된 함정):
#   (1) pandas/transforms3d/lerobot 을 먼저 설치한다.
#       - transforms3d: easy_handeye2 런타임 의존성.
#       - lerobot[feetech]: SO-101 모터 통신용 Feetech SDK(scservo_sdk)가 lerobot 의
#         feetech extras 에 묶여 있다(없으면 robot_controller_node 가 import 단계에서
#         죽음). lerobot 은 동시에 setuptools>=71 / numpy / torch(CPU) 를 끌어온다.
#   (2) torch/torchvision 을 Jetson(JetPack 6.2 / CUDA 12.6)용 cu126 wheel 로 교체.
#       PyPI generic wheel 은 CPU 전용(+cpu)이라 cuda 불가 → act_node/embedding_node
#       의 GPU 추론이 실패한다. lerobot 0.4.4 의 torch<2.11,>=2.2.1 /
#       torchvision<0.26,>=0.21 범위를 만족하는 2.8.0 / 0.23.0 을 jetson-ai-lab
#       index(.io 도메인 — .dev 는 DNS 실패)에서 받는다.
#   (3) numpy 를 1.26.4 로 고정한다. Jetson torch wheel 은 NumPy 1.x ABI 로 빌드돼
#       numpy 2.x 와 충돌한다("Failed to initialize NumPy" → torch.from_numpy 깨짐,
#       act_node 추론 시 크래시). 반드시 torch 설치 뒤에 1.x 로 내린다.
#   (4) setuptools 를 59.8.0(<60)으로 맨 마지막에 고정한다. ROS 2 humble 의
#       ament_python 은 setuptools 58 기준으로 console_scripts 를 lib/<pkg>/ 로
#       배치하는데, 60+ 에서는 bin/ 으로 가버려 `ros2 run` 이 노드를 못 찾는다.
#       lerobot 이 setuptools>=71 을 요구하지만(설치 시 pip 경고만), 런타임 import 는
#       59.8.0 에서 정상 동작한다. 이 고정을 lerobot 설치보다 앞에 두면 lerobot 이
#       다시 setuptools 80+ 로 올려버리므로 반드시 맨 마지막에 둔다.
echo -e "\n${YELLOW}[4/7] ROS 노드용 user-local 패키지 설치${NC}"

# (1) lerobot + 보조 패키지 (lerobot 이 setuptools/numpy/torch 를 의존성으로 끌어옴)
/usr/bin/python3 -m pip install --user --upgrade \
    pandas \
    transforms3d
/usr/bin/python3 -m pip install --user --upgrade "lerobot[feetech]==0.4.4"

# (2) Jetson(cu126)용 torch/torchvision 으로 교체 (CPU wheel → CUDA wheel)
/usr/bin/python3 -m pip install --user torch==2.8.0 torchvision==0.23.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# (3) numpy 1.x 로 고정 (Jetson torch wheel 의 NumPy 1.x ABI 호환)
/usr/bin/python3 -m pip install --user "numpy==1.26.4"

# (4) ament_python 빌드용 setuptools<60 을 맨 마지막에 고정 (순서 중요 — 위 주석 참조)
/usr/bin/python3 -m pip install --user "setuptools==59.8.0"

echo -e "${GREEN}user-local 패키지 설치 완료${NC}"

# =============================================================================
# 5. torch CUDA 확인
# =============================================================================
echo -e "\n${YELLOW}[5/7] torch CUDA 확인${NC}"

CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo -e "${GREEN}torch CUDA 사용 가능${NC}"
else
    echo -e "${RED}[경고] torch CUDA 사용 불가${NC}"
    echo "NVIDIA Jetson용 PyTorch wheel 설치가 필요할 수 있습니다."
    echo "설치 후 다시 확인하세요: python3 -c \"import torch; print(torch.cuda.is_available())\""
fi

# =============================================================================
# 6. SO-101 USB 권한 설정
# =============================================================================
echo -e "\n${YELLOW}[6/7] SO-101 USB 권한 설정${NC}"

if groups "$USER" | grep -q dialout; then
    echo "dialout 그룹 이미 설정됨. 스킵."
else
    sudo usermod -aG dialout "$USER"
    echo -e "${GREEN}dialout 그룹 추가 완료${NC}"
    echo -e "${YELLOW}[주의] 권한 적용을 위해 재로그인이 필요합니다.${NC}"
fi

# =============================================================================
# 7. robot_arm_pkg 빌드
# =============================================================================
echo -e "\n${YELLOW}[7/7] robot_arm_pkg 빌드${NC}"

source /opt/ros/humble/setup.bash
cd "$ROS2_WS"
colcon build --packages-select robot_arm_pkg
source install/setup.bash

echo -e "${GREEN}robot_arm_pkg 빌드 완료${NC}"

# =============================================================================
# 완료
# =============================================================================
echo -e "\n${GREEN}==============================${NC}"
echo -e "${GREEN} 기본 환경 설치 완료${NC}"
echo -e "${GREEN}==============================${NC}"
echo ""
echo "다음 명령어로 환경을 활성화하세요:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source $ROS2_WS/install/setup.bash"
echo ""
echo "캘리브레이션이 필요하다면 추가로 실행하세요:"
echo "  ./deploy/scripts/setup_calibration.sh"
