cmake_minimum_required(VERSION 3.8)
project(robot_arm_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(MSG_FILES
  "msg/SubtaskCommand.msg"
  "msg/ExecutionStatus.msg"
  "msg/DomainSelection.msg"
  "msg/AgentLog.msg"
)

set(SRV_FILES
  "srv/EmbedText.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  ${SRV_FILES}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
