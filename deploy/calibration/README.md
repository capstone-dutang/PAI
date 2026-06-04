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

`scripts/measure_roi.py --mode grasp` 로 실측을 자동화할 수 있습니다. `/camera/depth` 를
구독해 마우스로 지정한 ROI 를 `camera_intrinsics.yaml` 의 K 로 카메라 좌표에
역투영하고, grasp 성공/빈손 두 상태의 depth 변화로 `depth_change_threshold` 까지
산출합니다.

```bash
# GUI 모드 (Jetson — 캘리브레이션과 동일하게 DISPLAY 필요)
export DISPLAY=:1
python3 scripts/measure_roi.py --mode grasp \
    --intrinsics ~/calibration/camera_intrinsics.yaml \
    --out ~/calibration/grasp_roi.yaml
#   드래그=ROI 지정 → g(물체 든 상태) → e(빈손) → s(저장)

# headless 모드: 픽셀 ROI 직접 지정, 단발 측정
python3 scripts/measure_roi.py --mode grasp \
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

## `destination_roi.yaml` 작성 (현재 운영 기본)

`verify_result` 의 depth 기반 **place(목적지 배치) 검증**용 ROI 입니다. 실물 데이터가
home_pose 복귀로 끝나 verify 시점 그리퍼가 비어 있으므로(위 `grasp_roi.yaml` 의 grasp
시점 검증은 미래 확장), 현재 운영 검증은 그리퍼가 아니라 **목적지(통/목표 영역)** 의
depth 변화로 물체가 놓였는지를 봅니다.

카메라·목적지 고정 전제이므로 목적지는 이미지의 고정 픽셀 영역입니다 → **픽셀 ROI +
baseline depth** 로 표현해 카메라 intrinsics 없이 동작합니다(`agent_node._depth_place_check`).

측정 절차 — `measure_roi.py --mode destination` 가 픽셀 ROI 와 baseline 을 직접
출력합니다 (intrinsics 불필요):

```bash
# GUI (Jetson — DISPLAY 필요): 빈 목적지 상태에서 실행
export DISPLAY=:1
python3 scripts/measure_roi.py --mode destination --domain recycling \
    --out ~/calibration/destination_roi.yaml
#   각 통/영역 드래그 → b(baseline 등록, 반복) → (선택) o(threshold 힌트) → s(저장)

# headless: 픽셀 ROI 직접 지정, 단일 목적지
python3 scripts/measure_roi.py --mode destination --domain recycling \
    --name can_bin --roi 120 200 220 300
```

`name` 은 자동(dest_N)이라 출력 yaml 에서 `can_bin`/`target_area` 등으로 rename 합니다.
`depth_change_threshold` 는 'o' 표본(물체 놓은 상태)으로 힌트를 받거나 기본값(0.02)을 씁니다.

```yaml
# ~/calibration/destination_roi.yaml (예시)
depth_scale: 0.001
depth_change_threshold: 0.02   # m, baseline 보다 median 이 이만큼 작아지면 물체 있음
destinations:
  recycling:
    - {name: can_bin,     roi_px: [120, 200, 220, 300], baseline_depth: 0.55}
    - {name: plastic_bin, roi_px: [270, 200, 370, 300], baseline_depth: 0.55}
    - {name: paper_bin,   roi_px: [420, 200, 520, 300], baseline_depth: 0.55}
  tableware:
    - {name: target_area, roi_px: [240, 220, 400, 340], baseline_depth: 0.50}
```
