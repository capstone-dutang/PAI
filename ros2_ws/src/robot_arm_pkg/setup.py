from setuptools import setup

package_name = 'robot_arm_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/scripts', ['scripts/record_with_vlm.py']),
        ('share/' + package_name + '/launch',  ['launch/calibration.launch.py']),
        # urdf/so101.urdf 파일을 배치한 뒤 아래 주석 해제
        # ('share/' + package_name + '/urdf', ['urdf/so101.urdf']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='todo',
    maintainer_email='todo@todo.com',
    description='ROS2 robot arm package — VLM + ACT Policy',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'camera_node           = robot_arm_pkg.camera_node:main',
            'vlm_node              = robot_arm_pkg.vlm_node:main',
            'act_node              = robot_arm_pkg.act_node:main',
            'robot_controller_node = robot_arm_pkg.robot_controller_node:main',
            'state_monitor_node    = robot_arm_pkg.state_monitor_node:main',
        ],
    },
)
