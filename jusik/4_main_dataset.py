#!/usr/bin/env python

import threading
import time
import rerun as rr
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
from cv_bridge import CvBridge
cb = CvBridge()

from pathlib import Path
from pynput import keyboard
from typing import List


# Lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from trajectory_msgs.msg import JointTrajectory

# My code
from shm import SharedMemory
from comm import Comm
from rerun_visualizer import init_rerun, log_rerun_visualization



# ==============================
# 공유 메모리
# ==============================
fields_config = {
    'status': (20, str),
}
shm = SharedMemory(name='lerorobt-dataset-status-v4', fields_config=fields_config)
shm.set('status', 'idle')



# ==============================
# 키 리스너
# ==============================
KEY_STATUS = {'page_down': False, 'delete': False, 'esc': False, 'end': False}
def key_press_callback(key):
    status = shm.get('status')

    if key == keyboard.Key.page_down:
        if not KEY_STATUS['page_down']:
            KEY_STATUS['page_down'] = True

            if status in ['idle', 'ready']:
                shm.set('status', 'record')
                print(f"[Info ] 키입력: record (현재 상태: {shm.get('status')})")

            elif status == 'record':
                shm.set('status', 'save')
                print(f"[Info ] 키입력: save (현재 상태: {shm.get('status')})")
    if key == keyboard.Key.esc:
        if not KEY_STATUS['esc']:
            KEY_STATUS['esc'] = True
            shm.set('status', 'exit')
            print(f"[Info ] 키입력: exit (현재 상태: {shm.get('status')})")

    if key == keyboard.Key.end:
        if not KEY_STATUS['end']:
            KEY_STATUS['end'] = True
            shm.set('status', 'done')
            print(f"[Info ] 키입력: done (현재 상태: {shm.get('status')})")

    if key == keyboard.Key.delete:
        if not KEY_STATUS['delete']:
            KEY_STATUS['delete'] = True
            shm.set('status', 'cancel')
            print(f'[Info ] 키입력: delete (현재 상태: {shm.get("status")})')


def key_release_callback(key):
    if key == keyboard.Key.page_down:
        KEY_STATUS['page_down'] = False

    if key == keyboard.Key.esc:
        KEY_STATUS['esc'] = False

    if key == keyboard.Key.end:
        KEY_STATUS['end'] = False








# ==============================
# 전역 변수
# ==============================
DEFAULT_SAVE_ROOT_PATH = Path.home() / '.cache/huggingface/lerobot'

NUM_EPISODES = 2                                # 최대 에피소드 개수
FPS = 30                                        # FPS
TASK_DESCRIPTION = "pick up the zipper bag"     # Task Instruction
HF_REPO_ID = "user1/repo1"                      # Repo_io (Dir)
FEATURES = {
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
        'names': [
            'right_joint1',
            'right_joint2',
            'right_joint3',
            'right_joint4',
            'right_joint5',
            'right_joint6',
            'right_rh_r1_joint'
        ]
    },
    'action': {
        'dtype': 'float32',
        'shape': (7,),
        'names': [
            'right_joint1',
            'right_joint2',
            'right_joint3',
            'right_joint4',
            'right_joint5',
            'right_joint6',
            'right_rh_r1_joint'
        ]
    },
}

JOINT_ORDER = [
    'right_joint1',
    'right_joint2',
    'right_joint3',
    'right_joint4',
    'right_joint5',
    'right_joint6',
    'right_rh_r1_joint'
]


IS_RECORDING = False
START_TIME = time.time()
RECORDING_TIME = 0



def main():
    # ------------------------------
    # Rerun 초기화
    # ------------------------------
    init_rerun('Lerobot Data Collection')

    # ------------------------------
    # ROS2 설정
    # ------------------------------
    if not rclpy.ok():
        rclpy.init()

    communicator = Comm()
    communicator.start()

    # ------------------------------
    # 키 리스너 설정
    # ------------------------------
    listener = keyboard.Listener(on_press=key_press_callback, on_release=key_release_callback)
    listener.start()
    print(f'[Info ] 키 리스너 시작:\n -- Page Down: 녹화/저장\n -- ESC: 종료')

    # ------------------------------
    # 데이터셋 생성
    # ------------------------------
    if (DEFAULT_SAVE_ROOT_PATH / HF_REPO_ID).exists():
        import shutil
        shutil.rmtree(DEFAULT_SAVE_ROOT_PATH / HF_REPO_ID)
        print(f'[Info ] 이미 있는 데이터셋 제거함: {DEFAULT_SAVE_ROOT_PATH / HF_REPO_ID}')

    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=FEATURES,
        robot_type='omy_f3m',
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=8,
    )
    print(f'[Info ] 새 데이터셋 생성됨: {DEFAULT_SAVE_ROOT_PATH / HF_REPO_ID}')

    # ------------------------------
    # 컨버터 함수 정의
    # ------------------------------
    def convert_compressedImage_to_cvmat(msg: CompressedImage) -> np.ndarray:
        cv_image = cb.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

        if cv_image.dtype == np.uint16:
            cv_image = cv2.normalize(
                cv_image,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U)
        return cv_image

    def convert_compressedImage_to_numpy(msg: CompressedImage) -> np.ndarray:
        return cv2.cvtColor(convert_compressedImage_to_cvmat(msg), cv2.COLOR_BGR2RGB)

    def convert_jointState_to_numpy_list(msg: JointState, joint_order: List[str]) -> np.ndarray:
        joint_pos_map = dict(zip(msg.name, msg.position))
        ordered_positions = [np.float32(joint_pos_map.get(name, 0.0)) for name in joint_order]
        return np.array(ordered_positions, dtype=np.float32)

    def convert_jointTrajectory_to_numpy_list(msg: JointTrajectory, joint_order: List[str]) -> np.ndarray:
        target_point = msg.points[-1]
        joint_pos_map = dict(zip(msg.joint_names, target_point.positions))
        ordered_positions = [np.float32(joint_pos_map.get(name, 0.0)) for name in joint_order]
        return np.array(ordered_positions, dtype=np.float32)




    while True:
        loop_start = time.time()

        # 1. Ros2 토픽 데이터 받기
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



        # 이미지 변환
        img_top = convert_compressedImage_to_numpy(topic_msg['cam_top'])
        img_wrist = convert_compressedImage_to_numpy(topic_msg['cam_wrist'])

        # 관절 변환
        follower_numpy = convert_jointState_to_numpy_list(topic_msg['follower'], JOINT_ORDER)
        leader_numpy = convert_jointTrajectory_to_numpy_list(topic_msg['leader'], JOINT_ORDER)


        # ------------------------------------------------


        # 상태 로직 분기 시작
        status = shm.get('status')

        # 4. 상태별 로직
        if status == 'idle':
            pass

        elif status == 'record':
            global IS_RECORDING, START_TIME, RECORDING_TIME
            if not IS_RECORDING:
                IS_RECORDING = True
                START_TIME = time.time()

            frame_data = {
                'observation.images.cam_top': img_top,
                'observation.images.cam_wrist': img_wrist,
                'observation.state': follower_numpy,
                'action': leader_numpy,
                'task': TASK_DESCRIPTION,
            }
            dataset.add_frame(frame_data)


            # 3. Rerun 시각화
            log_rerun_visualization(
                images={'cam_top': img_top, 'cam_wrist': img_wrist},
                follower_joints=follower_numpy,
                leader_joints=leader_numpy,
            )

            RECORDING_TIME = time.time() - START_TIME
            print(f'\b\r[Info ] Recording: {RECORDING_TIME:.2f}s', end='\n') # 녹화 시간

        elif status == 'save':
            print(f'[Info ] 에피소드 저장중...')
            IS_RECORDING = False
            dataset.save_episode()
            shm.set('status', 'ready')
            print(f'[Info ] 에피소드 저장 완료')

        elif status == 'done':
            IS_RECORDING = False
            dataset.finalize()
            shm.set('status', 'ready')
            print(f'[Info ] 데이터셋 Finalize 완료')

        elif status == 'cancel':
            print(f'[Info ] 에피소드 취소중...')
            IS_RECORDING = False
            dataset.clear_episode_buffer()
            shm.set('status', 'ready')
            print(f'[Info ] 에피소드 취소 완료')

        elif status == 'exit':
            break

        elapsed = time.time() - loop_start
        sleep_time = max(0, (1.0 / FPS) - elapsed)
        time.sleep(sleep_time)

    print(f'[Info ] 종료')
    Comm.destroy_node()
    listener.stop()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


