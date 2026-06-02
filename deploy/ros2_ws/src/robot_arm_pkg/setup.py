from glob import glob

from setuptools import setup

package_name = 'robot_arm_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, f'{package_name}.utils'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',  ['launch/calibration.launch.py']),
        ('share/' + package_name + '/urdf',        ['urdf/so101.urdf']),
        ('share/' + package_name + '/urdf/assets', glob('urdf/assets/*.stl')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seongbin',
    maintainer_email='seongbin55228295@gmail.com',
    description='ROS2 robot arm package — Gemma 4 E2B VLA + ACT Policy',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'camera_node           = robot_arm_pkg.camera_node:main',
            'agent_node            = robot_arm_pkg.agent_node:main',
            'embedding_node        = robot_arm_pkg.embedding_node:main',
            'llama_server_manager  = robot_arm_pkg.llama_server_manager:main',
            'act_node              = robot_arm_pkg.act_node:main',
            'robot_controller_node = robot_arm_pkg.robot_controller_node:main',
        ],
    },
)
