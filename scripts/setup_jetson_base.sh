#!/bin/bash
# =============================================================================
# setup_jetson_base.sh
#
# Jetson 기본 환경 설치 스크립트
# 대상 작업: 모방 데이터 수집 (Part D), 실제 로봇 구동 (Part F)
#
# 전제 조건:
#   - Jetson Orin Nano, JetPack 6.2 (L4T 36.4.x)
#   - Ubuntu 22.04, CUDA 12.6 설치 완료
#   - ROS2 Humble 설치 완료
#
# 실행 방법:
#   chmod +x scripts/setup_jetson_base.sh
#   ./scripts/setup_jetson_base.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS2_WS="$REPO_ROOT/ros2_ws"
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
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-message-filters \
    ros-humble-cv-bridge

echo -e "${GREEN}apt 패키지 설치 완료${NC}"

# =============================================================================
# 3. pip 패키지 설치
# =============================================================================
echo -e "\n${YELLOW}[3/6] pip 패키지 설치${NC}"

pip install -r "$REQUIREMENTS"

echo -e "${GREEN}pip 패키지 설치 완료${NC}"

# =============================================================================
# 4. torch CUDA 확인
# =============================================================================
echo -e "\n${YELLOW}[4/6] torch CUDA 확인${NC}"

CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo -e "${GREEN}torch CUDA 사용 가능${NC}"
else
    echo -e "${RED}[경고] torch CUDA 사용 불가${NC}"
    echo "NVIDIA Jetson용 PyTorch wheel 설치가 필요할 수 있습니다."
    echo "설치 후 다시 확인하세요: python3 -c \"import torch; print(torch.cuda.is_available())\""
fi

# =============================================================================
# 5. SO-101 USB 권한 설정
# =============================================================================
echo -e "\n${YELLOW}[5/6] SO-101 USB 권한 설정${NC}"

if groups "$USER" | grep -q dialout; then
    echo "dialout 그룹 이미 설정됨. 스킵."
else
    sudo usermod -aG dialout "$USER"
    echo -e "${GREEN}dialout 그룹 추가 완료${NC}"
    echo -e "${YELLOW}[주의] 권한 적용을 위해 재로그인이 필요합니다.${NC}"
fi

# =============================================================================
# 6. robot_arm_pkg 빌드
# =============================================================================
echo -e "\n${YELLOW}[6/6] robot_arm_pkg 빌드${NC}"

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
echo "  ./scripts/setup_calibration.sh"
