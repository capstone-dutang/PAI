"""
calibration.launch.py

Part A — 카메라-로봇 Hand-Eye 캘리브레이션 (Eye-to-Hand)
  카메라 고정, SO-101 엔드이펙터에 ArUco 마커 부착

실행 전 준비:
  1. SO-101 팔로워를 USB로 Jetson에 연결
  2. RealSense 카메라를 USB3 포트에 연결
  3. ArUco 마커(ID=0, 0.05m) 출력 후 엔드이펙터에 부착

사용법:
  ros2 launch robot_arm_pkg calibration.launch.py

캘리브레이션 절차:
  1. 브라우저 또는 rqt에서 easy_handeye2 GUI 열기
     → http://localhost:8080 (rqt plugin: EasyHandeye2)
  2. 로봇을 수동으로 10~15개 포즈로 이동 (텔레옵 또는 직접 조작)
  3. 각 포즈에서 'Take Sample' 클릭
  4. 'Compute' → 'Save' 클릭
  5. 결과 파일 확인:
     ~/.ros/easy_handeye2/eye_on_base_calibration.yaml

파라미터 조정:
  marker_id   : 출력한 ArUco 마커 ID (기본 0)
  marker_size : 마커 실제 크기 미터 단위 (기본 0.05 = 5cm)
  robot_port  : SO-101 USB 포트 (기본 /dev/ttyACM0)

TODO — URDF 설정:
  SO-101 URDF를 LeRobot 저장소 또는 HuggingFace에서 받아
  urdf/so101.urdf 경로에 배치할 것.
  엔드이펙터 링크 이름을 확인하여 effector_frame 파라미터 수정.
    참고: https://github.com/huggingface/lerobot 의 URDF 폴더
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():

    # ── Launch Arguments ────────────────────────────────────────────
    marker_id_arg   = DeclareLaunchArgument('marker_id',   default_value='0',
                                            description='ArUco 마커 ID')
    marker_size_arg = DeclareLaunchArgument('marker_size', default_value='0.05',
                                            description='ArUco 마커 크기 (미터)')
    robot_port_arg  = DeclareLaunchArgument('robot_port',  default_value='/dev/ttyACM0',
                                            description='SO-101 USB 포트')

    marker_id   = LaunchConfiguration('marker_id')
    marker_size = LaunchConfiguration('marker_size')
    robot_port  = LaunchConfiguration('robot_port')

    # ── URDF 경로 ────────────────────────────────────────────────────
    # TODO: SO-101 URDF 파일을 urdf/so101.urdf 에 배치 후 아래 경로 사용
    urdf_path = PathJoinSubstitution([
        FindPackageShare('robot_arm_pkg'), 'urdf', 'so101.urdf'
    ])

    # ── 1. RealSense 카메라 드라이버 ────────────────────────────────
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera',
        parameters=[{
            'align_depth.enable': True,
            'pointcloud.enable': False,
        }],
        output='screen',
    )

    # ── 2. ArUco 마커 검출 ──────────────────────────────────────────
    # 패키지: aruco_ros (pal-robotics, humble-devel)
    #   - 자체 aruco C++ 라이브러리 사용 → OpenCV contrib 의존 없음
    #   - TF 발행: camera_color_optical_frame → aruco_marker_frame
    aruco_node = Node(
        package='aruco_ros',
        executable='single',
        name='aruco_single',
        parameters=[{
            'marker_size':        marker_size,
            'marker_id':          marker_id,
            'reference_frame':    'camera_color_optical_frame',
            'camera_frame':       'camera_color_optical_frame',
            'marker_frame':       'aruco_marker_frame',
            'image_is_rectified': True,
            'corner_refinement':  'LINES',
        }],
        remappings=[
            ('/image',       '/camera/color/image_raw'),
            ('/camera_info', '/camera/color/camera_info'),
        ],
        output='screen',
    )

    # ── 3. SO-101 관절 상태 발행 (캘리브레이션용 저속 모드) ────────
    # act_node 없이 관절 상태만 읽어 /joint_states 발행
    robot_controller_node = Node(
        package='robot_arm_pkg',
        executable='robot_controller_node',
        name='robot_controller_node',
        parameters=[{
            'robot_port': robot_port,
            'robot_id':   'follower',
            'publish_hz': 50.0,   # 캘리브레이션 중엔 50Hz로 충분
        }],
        output='screen',
    )

    # ── 4. robot_state_publisher ─────────────────────────────────────
    # /joint_states → TF (base_link ~ gripper_link) 계산
    # TODO: with_urdf 파라미터로 SO-101 URDF 경로 지정
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': open(
                os.path.join(
                    os.path.dirname(__file__), '..', 'urdf', 'so101.urdf'
                )
            ).read()
            if os.path.exists(
                os.path.join(
                    os.path.dirname(__file__), '..', 'urdf', 'so101.urdf'
                )
            )
            else '<!-- TODO: urdf/so101.urdf 파일을 배치하세요 -->',
        }],
        output='screen',
    )

    # ── 5. easy_handeye2 캘리브레이션 (헬퍼 런치 포함) ──────────────
    # easy_handeye2는 단일 노드가 아니라 calibrate.launch.py가
    # handeye_server + rqt_calibrator 등을 한꺼번에 띄우는 구조이므로
    # IncludeLaunchDescription으로 포함시킨다.
    #   calibration_type 값은 easy_handeye2 용어로 'eye_in_hand' 또는
    #   'eye_on_base'. 우리 구성(카메라 고정 + 마커가 로봇 손)은 'eye_on_base'.
    easy_handeye2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('easy_handeye2'),
                'launch',
                'calibrate.launch.py',
            ])
        ),
        launch_arguments={
            'name':                  'eye_on_base_calibration',
            'calibration_type':      'eye_on_base',
            # TODO: SO-101 URDF의 실제 프레임 이름으로 수정
            'robot_base_frame':      'base_link',
            'robot_effector_frame':  'gripper_frame_link',
            'tracking_base_frame':   'camera_color_optical_frame',
            'tracking_marker_frame': 'aruco_marker_frame',
        }.items(),
    )

    return LaunchDescription([
        marker_id_arg,
        marker_size_arg,
        robot_port_arg,
        realsense_node,
        aruco_node,
        robot_controller_node,
        robot_state_publisher_node,
        easy_handeye2_launch,
    ])