#!/bin/bash
# =============================================================================
# setup_calibration.sh
#
# 캘리브레이션 전용 환경 설치 스크립트
# 대상 작업: 카메라-로봇 Hand-Eye 캘리브레이션 (Part A)
#
# 전제 조건:
#   - setup_jetson_base.sh 실행 완료
#
# 실행 방법:
#   chmod +x scripts/setup_calibration.sh
#   ./scripts/setup_calibration.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS2_WS="$REPO_ROOT/ros2_ws"

echo -e "${GREEN}==============================${NC}"
echo -e "${GREEN} 캘리브레이션 환경 설치 시작${NC}"
echo -e "${GREEN}==============================${NC}"

# =============================================================================
# 전제 조건 확인: setup_jetson_base.sh 실행 여부
# =============================================================================
if [ ! -d "$ROS2_WS/install/robot_arm_pkg" ]; then
    echo -e "${RED}[오류] robot_arm_pkg가 빌드되어 있지 않습니다.${NC}"
    echo "먼저 setup_jetson_base.sh를 실행하세요:"
    echo "  ./scripts/setup_jetson_base.sh"
    exit 1
fi

# =============================================================================
# 1. apt 패키지 설치
# =============================================================================
echo -e "\n${YELLOW}[1/4] apt 패키지 설치${NC}"

sudo apt update
sudo apt install -y \
    ros-humble-robot-state-publisher \
    ros-humble-rqt \
    ros-humble-rqt-gui

echo -e "${GREEN}apt 패키지 설치 완료${NC}"

# =============================================================================
# 2. aruco_opencv 소스 빌드
# =============================================================================
echo -e "\n${YELLOW}[2/4] aruco_opencv 소스 빌드${NC}"

cd "$ROS2_WS/src"

if [ -d "ros_aruco_opencv" ]; then
    echo "ros_aruco_opencv 이미 클론됨. 스킵."
else
    git clone https://github.com/fictionlab/ros_aruco_opencv.git
    echo -e "${GREEN}ros_aruco_opencv 클론 완료${NC}"
fi

# =============================================================================
# 3. easy_handeye2 소스 빌드
# =============================================================================
echo -e "\n${YELLOW}[3/4] easy_handeye2 소스 빌드${NC}"

if [ -d "easy_handeye2" ]; then
    echo "easy_handeye2 이미 클론됨. 스킵."
else
    git clone https://github.com/marcoesposito1988/easy_handeye2.git
    echo -e "${GREEN}easy_handeye2 클론 완료${NC}"
fi

# =============================================================================
# 4. 빌드
# =============================================================================
echo -e "\n${YELLOW}[4/4] 빌드${NC}"

source /opt/ros/humble/setup.bash
cd "$ROS2_WS"

rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select aruco_opencv aruco_opencv_msgs easy_handeye2 robot_arm_pkg
source install/setup.bash

echo -e "${GREEN}빌드 완료${NC}"

# =============================================================================
# 완료
# =============================================================================
echo -e "\n${GREEN}==============================${NC}"
echo -e "${GREEN} 캘리브레이션 환경 설치 완료${NC}"
echo -e "${GREEN}==============================${NC}"
echo ""
echo "다음 명령어로 환경을 활성화하세요:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source $ROS2_WS/install/setup.bash"
echo ""
echo "캘리브레이션 실행:"
echo "  ros2 launch robot_arm_pkg calibration.launch.py marker_id:=0 marker_size:=0.05 robot_port:=/dev/ttyACM0"
