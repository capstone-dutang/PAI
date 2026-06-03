# 카메라-로봇 캘리브레이션 실행 가이드

Hand-Eye 캘리브레이션 (Eye-to-Hand 방식): 카메라 고정, SO-101 엔드이펙터에
ArUco 마커 부착. 설계·정밀도 요건은 [`README.md`](../../README.md) "카메라-로봇
캘리브레이션" 절 참조.

## Jetson 사전 설치

```bash
# setup_jetson_base.sh 실행 후 아래 스크립트 실행
./deploy/scripts/setup_calibration.sh
```

aruco_opencv, easy_handeye2 소스 빌드 및 관련 apt 패키지가 자동으로 설치됩니다.

## SO-101 URDF 배치

저장소에 SO-101 URDF와 STL 자산이 포함되어 있습니다:

- `deploy/ros2_ws/src/robot_arm_pkg/urdf/so101.urdf`
- `deploy/ros2_ws/src/robot_arm_pkg/urdf/assets/*.stl`

`setup.py`가 빌드 시 `share/robot_arm_pkg/urdf/`로 복사하도록 활성화되어 있어
별도 외부 복사 없이 그대로 사용 가능합니다.

> URDF의 엔드이펙터 링크 이름을 확인하여 `calibration.launch.py`의
> `robot_effector_frame` 파라미터와 일치시킬 것.

## 캘리브레이션 실행

```bash
# 패키지 빌드 및 소스
cd ~/ros2_ws
colcon build --packages-select robot_arm_pkg
source install/setup.bash

# 캘리브레이션 launch (마커 크기 5cm 기준)
export DISPLAY=:1
ros2 launch robot_arm_pkg calibration.launch.py robot_port:=/dev/follower

# 별도 터미널: rqt GUI에서 EasyHandeye2 플러그인 실행
ros2 run rqt_gui rqt_gui --force-discover

# aruco_tracker의 debug 영상 보기
export DISPLAY=:1
ros2 run rqt_image_view rqt_image_view /aruco_tracker/debug
```

## 캘리브레이션 절차

1. SO-101 엔드이펙터에 ArUco 마커(ID=0, 5cm) 부착
2. rqt → EasyHandeye2 플러그인 열기
3. 로봇을 수동으로 이동 → 카메라가 마커를 인식하면 **Take Sample** 클릭
4. 10~15개 포즈에서 반복 (포즈 간 충분한 회전 변화 필요)
5. **Compute** → 변환 행렬 확인 → **Save**
6. 결과 파일 복사:

```bash
mkdir -p ~/calibration
cp ~/.ros2/easy_handeye2/calibrations/eye_on_base_calibration.calib \
   ~/calibration/camera_robot_transform.yaml

# camera_intrinsics는 RealSense camera_info 토픽에서 추출
ros2 topic echo /camera/color/camera_info --once > ~/calibration/camera_intrinsics.yaml

# 저장소에도 보관 (백업 / 외부 PC Sim 학습에서 동일 intrinsics 사용)
cp ~/calibration/camera_intrinsics.yaml      <repo>/deploy/calibration/
cp ~/calibration/camera_robot_transform.yaml <repo>/deploy/calibration/
```

## `grasp_roi.yaml` 작성

`verify_result`의 depth 기반 grasp 보조 검증에 사용되는 ROI를 정의합니다.
SO-101 엔드이펙터 위치 근처에서 grasp 성공 시 depth 값이 변화하는 영역을 측정하여
설정합니다. 카메라 frame 기준 좌표를 사용합니다.

`scripts/measure_grasp_roi.py` 로 실측을 자동화할 수 있습니다. `/camera/depth` 를
구독해 마우스로 지정한 ROI 를 `camera_intrinsics.yaml` 의 K 로 카메라 좌표에
역투영하고, grasp 성공/빈손 두 상태의 depth 변화로 `depth_change_threshold` 까지
산출합니다.

```bash
# GUI 모드 (Jetson — 캘리브레이션과 동일하게 DISPLAY 필요)
export DISPLAY=:1
python3 scripts/measure_grasp_roi.py \
    --intrinsics ~/calibration/camera_intrinsics.yaml \
    --out ~/calibration/grasp_roi.yaml
#   드래그=ROI 지정 → g(물체 든 상태) → e(빈손) → s(저장)

# headless 모드: 픽셀 ROI 직접 지정, 단발 측정
python3 scripts/measure_grasp_roi.py \
    --intrinsics ~/calibration/camera_intrinsics.yaml --roi 280 200 360 280
```

산출 결과는 아래 형식입니다.

```yaml
# ~/calibration/grasp_roi.yaml (예시)
roi:
  frame: camera        # camera | base_link
  x_range: [-0.05, 0.05]   # m
  y_range: [-0.05, 0.05]
  z_range: [0.20, 0.40]
depth_change_threshold: 0.02  # m, 이 이상의 변화가 있으면 grasp 성공으로 간주
```
