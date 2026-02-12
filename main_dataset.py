#!/usr/bin/env python

import os
import time
import json
import numpy as np
import argparse

np.set_printoptions(suppress=True, precision=2)

from pathlib import Path
from pynput import keyboard

# Lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from trajectory_msgs.msg import JointTrajectory

# My code
from communicator.communicator import Communicator
from visualizer.rerun_visualizer import init_rerun, log_rerun_visualization
# from data_convertor.data_coverter import (
#     convert_compressedImage_to_numpy,           # compressedImage -> np
#     convert_jointTrajectory_to_numpy_list,      # jointTrajectory -> np
#     convert_jointState_to_numpy_list            # jointState -> np
# )


# modified by jjh
from data_convertor.data_coverter import DataConverter
from robot.kinematics_solver import KinematicsSolver


# ==============================
# 전역 변수
# ==============================

STATUS = 'idle'                                 # 데이터 수집 상태 (idle, ready, record, save, cancel, done)
IS_RECORDING = False                            # 녹화중 상태
START_TIME = 0                                  # 녹화 시작 시간
RECORDING_TIME = 0                              # 녹화한 시간




# # ==============================
# # Features
# # ==============================
# JOINT_ORDER = [
#     'right_joint1',
#     'right_joint2',
#     'right_joint3',
#     'right_joint4',
#     'right_joint5',
#     'right_joint6',
#     'right_rh_r1_joint'
# ]
# FEATURES = {
#     'observation.images.cam_top': {
#         'dtype': 'video',
#         'shape': (720, 1280, 3),
#         'names': [
#             'height',
#             'width',
#             'channels'
#         ]
#     },
#     'observation.images.cam_wrist': {
#         'dtype': 'video',
#         'shape': (480, 848, 3),
#         'names': [
#             'height',
#             'width',
#             'channels'
#         ]
#     },
#     'observation.state': {
#         'dtype': 'float32',
#         'shape': (7,),
#         'names': JOINT_ORDER
#     },
#     'action': {
#         'dtype': 'float32',
#         'shape': (7,),
#         'names': JOINT_ORDER
#     },
# }


# ==============================
# 키 리스너
# ==============================
KEY_STATUS = {'page_down': False, 'delete': False, 'end': False}
def key_press_callback(key):
    """키보드 눌림 콜백"""
    global STATUS
    # Page Down: 녹화/저장
    if key == keyboard.Key.page_down:
        if not KEY_STATUS['page_down']:
            KEY_STATUS['page_down'] = True
            print(f'[Info ] 키입력: Page Down')

            if STATUS == 'ready':
                STATUS = 'record'

            elif STATUS == 'record':
                STATUS = 'save'

    # End: 데이터 수집 종료
    if key == keyboard.Key.end:
        if not KEY_STATUS['end']:
            KEY_STATUS['end'] = True
            print(f'[Info ] 키입력: End')

            if STATUS == 'ready':
                STATUS = 'done'

    # Delete: 에피소드 취소
    if key == keyboard.Key.delete:
        if not KEY_STATUS['delete']:
            KEY_STATUS['delete'] = True
            print(f'[Info ] 키입력: Delete')

            if STATUS  == 'record':
                STATUS = 'cancel'

def key_release_callback(key):
    """"키보드 릴리즈 콜백"""
    if key == keyboard.Key.page_down:
        KEY_STATUS['page_down'] = False

    if key == keyboard.Key.end:
        KEY_STATUS['end'] = False

    if key == keyboard.Key.delete:
        KEY_STATUS['delete'] = False




def create_lerobot_dataset(repo_id: str) -> LeRobotDataset | None:
    """기존 데이터셋의 피쳐 구성을 확인하고, 사용자 선택에 따라 로드하거나 새로 생성함"""
    dataset_path = ROOT_PATH / repo_id
    info_path = dataset_path / 'meta' / 'info.json'

    should_create_new = True
    dataset = None

    if dataset_path.exists():
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    existing_info = json.load(f)

                existing_features = existing_info.get('features', {})

                # 피쳐 호환성 체크
                is_compatible = True
                for key, expected_val in FEATURES.items():
                    if key not in existing_features:
                        print(f'[Warn ] 필수 피쳐 누락: {key}')
                        is_compatible = False
                        break

                    existing_shape = existing_features[key].get('shape')
                    if existing_shape != list(expected_val['shape']):
                        print(f'[Warn ] 피쳐 형태 불일치 ({key}): {existing_shape} vs {list(expected_val["shape"])}')
                        is_compatible = False
                        break

                if is_compatible:
                    print(f'[Info ] 기존 데이터셋({repo_id})의 피쳐 구성이 현재 설정과 동일합니다.')
                    user_input = input(f'[Query] 기존 데이터셋을 불러올까요? (y: 불러오기 / n: 삭제 후 새로 생성): ').lower()
                    if user_input == 'y':
                        should_create_new = False
                        # 기존 데이터셋 로드
                        dataset = LeRobotDataset(repo_id=repo_id)
                        # 이미지 라이터 시작
                        dataset.start_image_writer(num_processes=4, num_threads=8)
                        print(f'[Info ] 기존 데이터셋을 성공적으로 불러왔습니다. (에피소드 수: {dataset.num_episodes})')
                else:
                    print(f'[Warn ] 기존 데이터셋의 피쳐 구성이 현재 설정과 다릅니다.')
                    user_input = input(f'[Query] 기존 데이터셋을 삭제하고 새로 생성할까요? (y/n): ').lower()
                    if user_input != 'y':
                        print(f'[Info ] 작업을 중단합니다.')
                        return None
            except Exception as e:
                # 404 에러 등 서버 관련 에러가 발생하면 여기서 잡힙니다.
                print(f'[Error] 데이터셋 처리 중 오류 발생: {e}')
                print(f'[Info ] 서버 연결 문제일 수 있습니다. 로컬 데이터를 삭제하고 새로 생성하시겠습니까?')
                user_input = input(f'[Query] 기존 데이터 삭제 후 새로 생성 (y/n): ').lower()
                if user_input != 'y':
                    return None

        if should_create_new:
            import shutil
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                print(f'[Info ] 기존 데이터셋 디렉토리를 제거했습니다: {dataset_path}')

    if should_create_new:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=FPS,
            features=FEATURES,
            robot_type='omy_f3m',
            use_videos=True,
            image_writer_processes=4,
            image_writer_threads=8,
        )
        print(f'[Info ] 새 데이터셋이 생성되었습니다: {dataset_path}')

    return dataset



def set_initialization(args):
    action_type_ = args.action_type

    if action_type_ == "abs_joint":
        observation_order_ = args.joint_names
        action_order_ = args.joint_names

        kinematics_solver_ = None

    elif action_type_ == "delta_pose":
        observation_order_ = [
            'abs_x',
            'abs_y',
            'abs_z',
            'abs_roll',
            'abs_pitch',
            'abs_yaw',
            'gripper'
        ]
        action_order_ = [
            'delta_x',
            'delta_y',
            'delta_z',
            'delta_roll',
            'delta_pitch',
            'delta_yaw',
            'delta_gripper' # right??
        ]

        kinematics_solver_ = KinematicsSolver(path_urdf=args.path_urdf, base_frame=args.base_frame, ee_frame=args.ee_frame)

    else:
        raise RuntimeError("error")

    features_ = {
        'observation.images.cam_top': {
            'dtype': 'video',
            'shape': (720, 1280, 3),
            'names': [
                'height',
                'width',
                'channels'
            ]
        },
        'observation.images.cam_wrist': {
            'dtype': 'video',
            'shape': (480, 848, 3),
            'names': [
                'height',
                'width',
                'channels'
            ]
        },
        'observation.state': {
            'dtype': 'float32',
            'shape': (7,),
            'names': observation_order_
        },
        'action': {
            'dtype': 'float32',
            'shape': (7,),
            'names': action_order_
        },
    }

    return observation_order_, action_order_, features_, kinematics_solver_



def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="arguments_tmp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 기본
    p.add_argument("--root_path", type=str, default=".cache/huggingface/lerobot")
    p.add_argument("--hf_repo_id", type=str, default="user1") # data will be saved in {ROOT_PATH}/{hf_repo_id}/{dataset_name}/
    p.add_argument("--dataset_name", type=str, default="test") # data will be saved in {ROOT_PATH}/{hf_repo_id}/{dataset_name}/
    p.add_argument("--task_description", type=str, default="pick up the zipper bag")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--action_type", type=str, default="delta_pose") # abs_joint, delta_pose, ... (to be updated later)
    # p.add_argument("--lr", type=float, default=3e-4)

    # robot
    p.add_argument("--path_urdf", type=str, default="/home/uon/workspace/lerobot_jjh_dataset/lerobot/robot/omy_f3m.urdf")
    p.add_argument("--base_frame", type=str, default="world")
    p.add_argument("--ee_frame", type=str, default="end_effector_link")
    p.add_argument("--joint_names", type=list, default=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"])

    # ROS2
    p.add_argument("--ROS_DOMAIN_ID", type=int, default=29)

    # topic names
    p.add_argument("--topic_name_cam_top", type=str, default="/camera/cam_top/color/image_raw/compressed")
    p.add_argument("--topic_name_cam_wrist", type=str, default="/camera/cam_wrist/color/image_rect_raw/compressed")
    p.add_argument("--topic_name_leader", type=str, default="/leader/joint_trajectory")
    p.add_argument("--topic_name_follower", type=str, default="/joint_states")
    p.add_argument("--topic_name_leader_pub", type=str, default="/leader/joint_trajectory")

    args = p.parse_args(argv)

    return args



def main():
    args = parse_args()
    
    global IS_RECORDING, START_TIME, RECORDING_TIME, STATUS

    global FPS
    FPS = args.fps

    global ROOT_PATH, PATH_SAVE_DATA
    ROOT_PATH = Path.home() / args.root_path
    PATH_SAVE_DATA = os.path.join(args.hf_repo_id, args.dataset_name)

    global OBSERVATION_ORDER, ACTION_ORDER, FEATURES
    OBSERVATION_ORDER, ACTION_ORDER, FEATURES, kinematics_solver = set_initialization(args)


    dc = DataConverter(args, OBSERVATION_ORDER, ACTION_ORDER, kinematics_solver)

    # Rerun 초기화 ------------------------------
    init_rerun(f'{PATH_SAVE_DATA} Dataset')
    time.sleep(1)
    print()


    # ROS2 설정 ---------------------------------
    if not rclpy.ok():
        rclpy.init()

    # 커뮤니케이터 생성
    communicator = Communicator(args)
    communicator.start() # 쓰레드 시작


    # 키 리스너 설정 ------------------------------
    listener = keyboard.Listener(on_press=key_press_callback, on_release=key_release_callback)
    listener.start()
    print(f'[Info ] 키 리스너 시작:\n'
          f'-- Page Down: 녹화/저장\n'
          f'-- Delete: 에피소드 취소\n'
          f'-- End: 데이터 수집 종료\n'
          f'--------------------------')

    # 데이터셋 생성/불러오기 ------------------------
    dataset = create_lerobot_dataset(PATH_SAVE_DATA)
    if dataset is None:
        return

    # 녹화 준비 완료
    STATUS = 'ready'

    # 메인 녹화 루프
    while True:
        loop_start = time.time()

        # Ros2 토픽 데이터 받기 --------------------
        topic_msg = communicator.get_latest_msgs()

        # 데이터 유효성 검사
        if topic_msg['cam_top'] is None:
            print(f'\r[Warn ] {loop_start:8.6f} 키넥트 카메라 데이터 없음', end='')
            continue

        if topic_msg['cam_wrist'] is None:
            print(f'\r[Warn ] {loop_start:8.6f} 손목 카메라 데이터 없음', end='')
            continue

        if topic_msg['follower'] is None:
            print(f'\r[Warn ] {loop_start:8.6f} 팔로우암 데이터 없음', end='')
            continue

        if topic_msg['leader'] is None:
            print(f'\r[Warn ] {loop_start:8.6f} 리더암 데이터 없음', end='')
            continue



        # # 데이터 변환 -----------------------------
        # img_top = convert_compressedImage_to_numpy(topic_msg['cam_top'])                        # cam_top
        # img_wrist = convert_compressedImage_to_numpy(topic_msg['cam_wrist'])                    # cam_wrist
        # follower_numpy = convert_jointState_to_numpy_list(topic_msg['follower'], OBSERVATION_ORDER, ACTION_ORDER, kinematics_solver)   # state
        # leader_numpy = convert_jointTrajectory_to_numpy_list(topic_msg['leader'], OBSERVATION_ORDER, ACTION_ORDER, kinematics_solver)  # action

        img_top, img_wrist, state_numpy, action_numpy = dc.convert(msg_img1=topic_msg['cam_top'], 
                                                                    msg_img2=topic_msg['cam_wrist'], 
                                                                    msg_joint_states=topic_msg['follower'], 
                                                                    msg_joint_trajectory=topic_msg['leader'])
        
        print("=================================")
        print(f"====state: {state_numpy}")
        print(f"====action: {action_numpy}")



        # 상태 로직 분기 시작 -----------------------

        # Idle 상태
        if STATUS == 'idle':
            pass

        # 데이터 녹화
        elif STATUS == 'record':
            if not IS_RECORDING:
                IS_RECORDING = True
                START_TIME = time.time()

            # 데이터 프레임 생성
            frame_data = {
                'observation.images.cam_top': img_top,
                'observation.images.cam_wrist': img_wrist,
                'observation.state': state_numpy,
                'action': action_numpy,
                'task': args.task_description,
            }
            dataset.add_frame(frame_data)

            # Rerun 시각화
            log_rerun_visualization(
                images={'cam_top': img_top, 'cam_wrist': img_wrist},
                follower_joints=state_numpy,
                leader_joints=action_numpy,
            )

            RECORDING_TIME = time.time() - START_TIME
            print(f'\b\r[Info ] Recording: {RECORDING_TIME:.2f}s', end='\n') # 녹화 시간

        # 에피소드 저장
        elif STATUS == 'save':
            print(f'[Info ] 에피소드 저장중...')
            IS_RECORDING = False
            dataset.save_episode()
            STATUS = 'ready'
            print(f'[Info ] 에피소드 저장 완료')

        # 데이터셋 Finalize
        elif STATUS == 'done':
            IS_RECORDING = False
            dataset.finalize()
            STATUS = 'ready'
            print(f'[Info ] 데이터셋 Finalize 완료')

        # 녹화중인 에피소드 취소
        elif STATUS == 'cancel':
            print(f'[Info ] 에피소드 취소중...')
            IS_RECORDING = False
            dataset.clear_episode_buffer()
            STATUS = 'ready'
            print(f'[Info ] 에피소드 취소 완료')

        # 종료
        elif STATUS == 'exit':
            break


        # FPS 조절
        elapsed = time.time() - loop_start
        sleep_time = max(0, (1.0 / FPS) - elapsed)
        time.sleep(sleep_time)


    print(f'[Info ] 종료')
    communicator.destroy_node()
    listener.stop()
    rclpy.shutdown()


if __name__ == "__main__":
    main()