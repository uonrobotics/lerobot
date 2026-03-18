#!/bin/bash

# dsr_node: 두산 로봇 실행 스크립트
gnome-terminal --tab \
  --title="dsr_node" \
  -- bash -ic "conda activate dsr; \
              python /home/uon/workspace/test_realsense/scripts/dsr_node.py; \
              exec bash"

sleep 0.5

# d405_node: 리얼센스 카메라 노드
gnome-terminal --tab \
  --title="d405_node" \
  -- bash -ic "conda activate dsr; \
              python /home/uon/workspace/test_realsense/scripts/d405_node.py; \
              exec bash"

sleep 0.5

# Leader arm
gnome-terminal --tab \
  --title="leader_arm" \
  -- bash -ic "conda activate dsr; \
              ros2 launch open_manipulator_bringup omy_l100_leader_ai.launch.py; \
              exec bash"

sleep 0.5

# main: 데이터 수집
gnome-terminal --tab \
  --title="main: lerobot data collector" \
  -- bash -ic "conda activate dsr; \
              python /home/uon/workspace/test_realsense/main.py; \
              exec bash"