#!/bin/bash
# =============================================================================
# setup_calibration.sh
#
# 캘리브레이션 전용 환경 설치 스크립트
# 대상 작업: 카메라-로봇 Hand-Eye 캘리브레이션
# 사용 시점: 학습 준비 단계 (PAI 전체 clone 권장)
#
# 전제 조건:
#   - setup_jetson_base.sh 실행 완료
#
# 실행 방법:
#   chmod +x deploy/scripts/setup_calibration.sh
#   ./deploy/scripts/setup_calibration.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS2_WS="$DEPLOY_ROOT/ros2_ws"

echo -e "${GREEN}==============================${NC}"
echo -e "${GREEN} 캘리브레이션 환경 설치 시작${NC}"
echo -e "${GREEN}==============================${NC}"

# =============================================================================
# 전제 조건 확인: setup_jetson_base.sh 실행 여부
# =============================================================================
if [ ! -d "$ROS2_WS/install/robot_arm_pkg" ]; then
    echo -e "${RED}[오류] robot_arm_pkg가 빌드되어 있지 않습니다.${NC}"
    echo "먼저 setup_jetson_base.sh를 실행하세요:"
    echo "  ./deploy/scripts/setup_jetson_base.sh"
    exit 1
fi

# =============================================================================
# 1. apt 패키지 설치
# =============================================================================
echo -e "\n${YELLOW}[1/5] apt 패키지 설치${NC}"

sudo apt update
sudo apt install -y \
    python3-rosdep \
    python3-colcon-common-extensions \
    ros-humble-rosidl-runtime-py \
    ros-humble-robot-state-publisher \
    ros-humble-rqt \
    ros-humble-rqt-gui \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins

echo -e "${GREEN}apt 패키지 설치 완료${NC}"

# =============================================================================
# 2. rosdep 초기화
# =============================================================================
echo -e "\n${YELLOW}[2/5] rosdep 초기화${NC}"

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
    echo -e "${GREEN}rosdep init 완료${NC}"
else
    echo "rosdep 이미 초기화됨. 스킵."
fi
rosdep update

# =============================================================================
# 3. aruco_ros 소스 클론
# =============================================================================
# NVIDIA Jetson OpenCV 4.8.0 빌드에 aruco contrib 모듈이 빠져 있어
# fictionlab/ros_aruco_opencv 대신 자체 aruco 라이브러리를 묶은
# pal-robotics/aruco_ros (humble-devel)를 사용한다.
echo -e "\n${YELLOW}[3/5] aruco_ros 소스 클론${NC}"

cd "$ROS2_WS/src"

if [ -d "aruco_ros" ]; then
    echo "aruco_ros 이미 클론됨. 스킵."
else
    git clone -b humble-devel https://github.com/pal-robotics/aruco_ros.git
    echo -e "${GREEN}aruco_ros 클론 완료${NC}"
fi

# =============================================================================
# 4. easy_handeye2 소스 클론
# =============================================================================
echo -e "\n${YELLOW}[4/5] easy_handeye2 소스 클론${NC}"

if [ -d "easy_handeye2" ]; then
    echo "easy_handeye2 이미 클론됨. 스킵."
else
    git clone https://github.com/marcoesposito1988/easy_handeye2.git
    echo -e "${GREEN}easy_handeye2 클론 완료${NC}"
fi

# =============================================================================
# 5. 빌드
# =============================================================================
echo -e "\n${YELLOW}[5/6] 빌드${NC}"

source /opt/ros/humble/setup.bash
export ROS_DISTRO="${ROS_DISTRO:-humble}"
cd "$ROS2_WS"

# transforms3d: rosdep이 venv 환경에서 import 검증에 실패하는 경우가 있어
# 활성 Python 환경에 직접 설치하고 rosdep 키에서는 스킵한다.
pip install transforms3d

rosdep install --from-paths src --ignore-src -r -y \
    --rosdistro "$ROS_DISTRO" \
    --skip-keys "transforms3d"
colcon build --packages-select \
    aruco aruco_msgs aruco_ros \
    easy_handeye2 easy_handeye2_msgs \
    robot_arm_pkg robot_arm_msgs
source install/setup.bash

echo -e "${GREEN}빌드 완료${NC}"

# =============================================================================
# 6. easy_handeye2 entry_point shebang 치환
# =============================================================================
# setuptools 60~79 일부 버전 + colcon(python setup.py install) 조합에서
# entry_point wrapper(rqt_calibrator.py 등)가 placeholder shebang `#!python`
# 으로 생성되어 launch 시 posix_spawn(ENOENT)로 실행 실패하는 문제가 있다.
# requirements.txt가 setuptools<80을 강제하는 한 이 조합은 계속 발생하므로
# 빌드 산출물의 shebang을 자동 치환한다.
echo -e "\n${YELLOW}[6/6] easy_handeye2 shebang 치환${NC}"

HANDEYE_LIB="$ROS2_WS/install/easy_handeye2/lib/easy_handeye2"
if [ -d "$HANDEYE_LIB" ]; then
    fixed=0
    while IFS= read -r -d '' f; do
        if [ "$(head -n 1 "$f" 2>/dev/null)" = "#!python" ]; then
            sed -i '1s|^#!python$|#!/usr/bin/env python3|' "$f"
            fixed=$((fixed + 1))
        fi
    done < <(find "$HANDEYE_LIB" -type f -executable -print0)
    echo -e "${GREEN}shebang 치환 완료 (${fixed}개 파일)${NC}"
else
    echo -e "${YELLOW}easy_handeye2 install 디렉토리 없음. 스킵.${NC}"
fi

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
