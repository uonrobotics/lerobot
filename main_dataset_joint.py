import time
import os
import cv2
import math
import uuid
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pynput import keyboard
import threading
import yaml
from multiprocessing import Process, Manager, Event, Queue, Value
from collections import deque

# ros2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray

# my code
from timer.precision_timer import PrecisionTimer        # 정밀 타이머
from lerobot_dataset.communicator import start_comm_worker   # 커뮤니케이터
from lerobot_dataset.lerobot_dataset import start_dataset_worker
from lerobot_dataset.data_coverter import (             # 데이터 컨버터
    convert_compressedImage_to_numpy,
    convert_jointState_to_numpy_list,
    convert_jointTrajectory_to_numpy_list)


# ==================================================================
# 전역 변수
# ==================================================================
HF_REPO_ID          = "user2/dsr3"      # Repo_io (Dir)
IS_RECORDING        = False
START_TIME          = 0
TASK_DESCRIPTION    = "pick up the object"
STATUS              = 'ready'
TIMER               = PrecisionTimer(30) # 30hz

# 데이터셋 feature
JOINT_ORDER = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
]

FULL_ORDER = JOINT_ORDER + ["gripper"]
FEATURES = {

    'observation.images.cam_top': {
        'dtype': 'video',
        'shape': (720, 630, 3),
        'names': ['height', 'width', 'channels']
    },
    'observation.images.cam_wrist': {
        'dtype': 'video',
        'shape': (480, 848, 3),
        'names': ['height', 'width', 'channels']
    },
    'observation.state': {
        'dtype': 'float32',
        'shape': (7,),
        'names': FULL_ORDER
    },
    'action': {
        'dtype': 'float32',
        'shape': (7,),
        'names': FULL_ORDER
    },
}

DELAY_FRAMES_SIZE = 60
state_buffer = deque(maxlen=DELAY_FRAMES_SIZE + 1) # 버퍼 생성


# ==================================================================
# Rerun
# ==================================================================
def init_rerun():
    """리런 초기화 & 레이아웃 설정 (실시간/지연 데이터 통합 뷰)"""
    rr.init("robot_monitoring", spawn=True)

    # 상단 레이아웃 (카메라 2개)
    top_row = rrb.Horizontal(
        rrb.Spatial2DView(name="Wrist Cam", origin="/camera/wrist"),
        rrb.Spatial2DView(name="Top Cam", origin="/camera/top"),
    )

    # 하단 레이아웃 (플롯 7개: 각 차트마다 실시간과 지연 데이터를 함께 배치)
    bottom_row = rrb.Horizontal(
        # Joint 1
        rrb.TimeSeriesView(
            name='J1 (Action vs State)',
            origin="/",
            contents=["/robot/joint1", "/delayed_robot/joint1"],
            axis_y=rrb.ScalarAxis(range=(-360.0, 360.0))
        ),
        # Joint 2
        rrb.TimeSeriesView(
            name='J2',
            origin="/",
            contents=["/robot/joint2", "/delayed_robot/joint2"],
            axis_y=rrb.ScalarAxis(range=(-360.0, 360.0))
        ),
        # Joint 3
        rrb.TimeSeriesView(
            name='J3',
            origin="/",
            contents=["/robot/joint3", "/delayed_robot/joint3"],
            axis_y=rrb.ScalarAxis(range=(-360.0, 360.0))
        ),
        # Joint 4
        rrb.TimeSeriesView(
            name='J4',
            origin="/",
            contents=["/robot/joint4", "/delayed_robot/joint4"],
            axis_y=rrb.ScalarAxis(range=(-360.0, 360.0))
        ),
        # Joint 5
        rrb.TimeSeriesView(
            name='J5',
            origin="/",
            contents=["/robot/joint5", "/delayed_robot/joint5"],
            axis_y=rrb.ScalarAxis(range=(-360.0, 360.0))
        ),
        # Joint 6
        rrb.TimeSeriesView(
            name='J6',
            origin="/",
            contents=["/robot/joint6", "/delayed_robot/joint6"],
            axis_y=rrb.ScalarAxis(range=(-360.0, 360.0))
        ),
        # Gripper
        rrb.TimeSeriesView(
            name="Grip",
            origin="/",
            contents=["/robot/gripper", "/delayed_robot/gripper"],
            axis_y=rrb.ScalarAxis(range=(-1.0, 1.0))
        ),
    )

    # 전체 레이아웃 (수직 분할)
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            top_row,
            bottom_row,
            row_shares=[2, 1]
        )
    )

    rr.send_blueprint(blueprint)

def log_to_rerun(img_top, img_wrist, joint_data, gripper_val, delayed_data=None):
    """rerun 로깅 - 지연 데이터(State) 포함"""
    # 타임라인 설정
    rr.set_time_seconds("display_time", time.time())

    # 이미지 로깅
    if img_wrist is not None:
        rr.log("/camera/wrist", rr.Image(img_wrist))
    if img_top is not None:
        rr.log("/camera/top", rr.Image(img_top))

    # --- 실시간 조인트 로깅 (현재 Action 값) ---
    if joint_data is not None and len(joint_data) >= 6:
        rr.log("/robot/joint1", rr.Scalars(math.degrees(joint_data[0])))
        rr.log("/robot/joint2", rr.Scalars(math.degrees(joint_data[1])))
        rr.log("/robot/joint3", rr.Scalars(math.degrees(joint_data[2])))
        rr.log("/robot/joint4", rr.Scalars(math.degrees(joint_data[3])))
        rr.log("/robot/joint5", rr.Scalars(math.degrees(joint_data[4])))
        rr.log("/robot/joint6", rr.Scalars(math.degrees(joint_data[5])))

    # --- 지연된 조인트 로깅 (과거 State 값) ---
    # 실시간 그래프와 겹쳐서 비교할 수 있도록 별도 경로로 로깅합니다.
    if delayed_data is not None and len(delayed_data) >= 6:
        rr.log("/delayed_robot/joint1", rr.Scalars(math.degrees(delayed_data[0])))
        rr.log("/delayed_robot/joint2", rr.Scalars(math.degrees(delayed_data[1])))
        rr.log("/delayed_robot/joint3", rr.Scalars(math.degrees(delayed_data[2])))
        rr.log("/delayed_robot/joint4", rr.Scalars(math.degrees(delayed_data[3])))
        rr.log("/delayed_robot/joint5", rr.Scalars(math.degrees(delayed_data[4])))
        rr.log("/delayed_robot/joint6", rr.Scalars(math.degrees(delayed_data[5])))
        rr.log("/delayed_robot/gripper", rr.Scalars(delayed_data[6]))

    # 실시간 그리퍼 로깅
    if gripper_val is not None:
        rr.log("/robot/gripper", rr.Scalars(gripper_val))


# ==================================================================
# 키 리스너
# ==================================================================
KEY_STATUS = {'page_down': False, 'delete': False, 'end': False, 'home': False}
def key_press_callback(key):
    global STATUS

    if key == keyboard.Key.page_down:
        if not KEY_STATUS['page_down']:
            print(f'\n[Info ] 키입력: Page Down')
            KEY_STATUS['page_down'] = True
            if STATUS == 'ready':
                STATUS = 'record'
            elif STATUS == 'record':
                STATUS = 'save'

    if key == keyboard.Key.end:
        if not KEY_STATUS['end']:
            print(f'\n[Info ] 키입력: End')
            KEY_STATUS['end'] = True
            if STATUS == 'ready': STATUS = 'done'

    if key == keyboard.Key.delete:
        if not KEY_STATUS['delete']:
            print(f'\n[Info ] 키입력: Delete')
            KEY_STATUS['delete'] = True
            if STATUS == 'record': STATUS = 'cancel'

    if key == keyboard.Key.home:
        if not KEY_STATUS['home']:
            print(f'\n[Info ] 키입력: Home')
            KEY_STATUS['home'] = True

def key_release_callback(key):
    if key == keyboard.Key.page_down: KEY_STATUS['page_down'] = False
    if key == keyboard.Key.end:       KEY_STATUS['end']       = False
    if key == keyboard.Key.delete:    KEY_STATUS['delete']    = False
    if key == keyboard.Key.home:      KEY_STATUS['home']      = False


# ==================================================================
# Main 함수
# ==================================================================
def main():
    global STATUS, IS_RECORDING, START_TIME, BARCODE_PUBLISHER


    # -----------------------------------------------------------
    # 데이터셋 멀티프로세싱 설정 및 실행
    # ------------------------------------------------------------------
    print("\n[Query] [main] 데이터셋 저장 방식을 선택하세요.")
    ans = input("1: 기존 데이터 이어서 쓰기 (Resume)\n2: 삭제 후 덮어쓰기 (Overwrite)\n선택 (1 또는 2): ")
    dataset_mode = 'resume' if ans == '1' else 'overwrite'
    is_saving_val = Value('i', 0)
    dataset_queue = Queue()
    start_dataset_worker(dataset_queue, HF_REPO_ID, FEATURES, 30, dataset_mode, is_saving_val)

    print("[Info ] [main] 데이터셋 생성 중...")
    time.sleep(1.0) # sleep 3.5s for dataset_proc to init


    # ------------------------------------------------------------------
    # ros2 및 rerun 초기화
    # ------------------------------------------------------------------
    if not rclpy.ok():
        rclpy.init()

    init_rerun()
    time.sleep(0.5)


    # ------------------------------------------------------------------
    # 커뮤니케이터 멀티프로세싱 설정 및 실행
    # ------------------------------------------------------------------
    shared_msgs = Manager().dict({})
    start_comm_worker(shared_msgs, 'config/comm.config.yaml')
    time.sleep(1.0)


    # ------------------------------------------------------------------
    # 키보드 리스너 시작
    # ------------------------------------------------------------------
    listener = keyboard.Listener(on_press=key_press_callback, on_release=key_release_callback)
    listener.start()


    # ------------------------------------------------------------------
    # 지연 데이터 생성
    # -----------------------------------------------------------------
    look_ahead_steps = 15  # 15 fps 지연
    obs_buffer = deque(maxlen=look_ahead_steps + 1)


    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    print('\n')
    print(f'[Info ] [main] 루프 시작 (30Hz) - Multi Processed Mode')
    print(f'-- Page Down: 녹화/저장 | Delete: 녹화 취소 | End: 종료')


    try:
        prev_loop_time = time.time()

        while rclpy.ok():
            # 루프 시간 조절
            TIMER.sleep()
            loop_start = time.time()
            dt = loop_start - prev_loop_time
            prev_loop_time = loop_start

            saving_status = "SAVING" if is_saving_val.value else "IDLE" # SAVE 상태에서는 프로그램을 끄지 마세요

            # --- 데이터 유효성 체크 ---
            msgs = dict(shared_msgs) # 토픽 최신 데이터

            required_keys = ['cam_top', 'cam_wrist', 'follower', 'gripper']
            missing = [k for k in required_keys if k not in msgs or msgs[k] is None]
            is_ready = len(missing) == 0


            # --- 대시보드 하단에 표시할 추가 정보 (상태별 메세지) ---
            info_msg = ""
            if not is_ready:
                info_msg = f"WAITING TOPICS: {', '.join(missing)}"
            elif STATUS == 'record':
                recording_time = time.time() - START_TIME
                info_msg = f"RECORDING: {recording_time:6.2f}s | Task: {TASK_DESCRIPTION}"
            elif is_saving_val.value:
                info_msg = "WRITING TO DISK... PLEASE WAIT"
            else:
                info_msg = "READY TO RECORD (Press PageDown)"

            # --- 대시보드 구성 ---
            print(f"│ [SYSTEM]  Hz: {1.0/dt if dt > 0 else 0:6.1f}  │  dt: {dt:.4f}s  │  Disk: {saving_status:<6}  │  Mode: {STATUS:^7} │ INFO: {info_msg:<66} \n", end="")

            if not is_ready:
                continue


            # --- 데이터 변환 ---
            img_top           = convert_compressedImage_to_numpy(msgs['cam_top'])
            img_wrist         = convert_compressedImage_to_numpy(msgs['cam_wrist'])
            joint_follower    = convert_jointState_to_numpy_list(msgs['follower'], JOINT_ORDER)
            gripper_val       = msgs['gripper'].data
            full_data         = np.concatenate([joint_follower, [gripper_val]]).astype(np.float32)


            # --- 지연 데이터 및 Rerun 로깅 ---
            obs_buffer.append(full_data.copy())
            while len(obs_buffer) < obs_buffer.maxlen:
                obs_buffer.appendleft(full_data.copy())
            delayed_state = obs_buffer[0]


            # --- 상태 머신 로직 (출력 제거 버전) ---
            if STATUS == 'record':
                if not IS_RECORDING:
                    IS_RECORDING = True
                    START_TIME = time.time()

                frame_data = {
                    'observation.images.cam_top'    : img_top,
                    'observation.images.cam_wrist'  : img_wrist,
                    'observation.state'             : delayed_state,
                    'action'                        : full_data,
                    'task'                          : TASK_DESCRIPTION,
                }

                dataset_queue.put({'cmd': 'add_frame', 'data': frame_data})
                log_to_rerun(img_top, img_wrist, joint_follower, gripper_val, delayed_state)

            elif STATUS == 'save':
                IS_RECORDING = False
                dataset_queue.put({'cmd': 'save'})
                STATUS = 'ready'

            elif STATUS == 'cancel':
                IS_RECORDING = False
                dataset_queue.put({'cmd': 'cancel'})
                STATUS = 'ready'

            elif STATUS == 'done':
                IS_RECORDING = False
                dataset_queue.put({'cmd': 'done'})
                STATUS = 'ready'

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 사용자에 의해 종료됨")
    finally:
        # 종료 시 프로세스 정리
        dataset_queue.put({'cmd': 'done'})

        comm_proc.join(timeout=2.0)
        if comm_proc.is_alive():
            comm_proc.terminate()

        dataset_proc.join(timeout=2.0)
        if dataset_proc.is_alive():
            dataset_proc.terminate()

        listener.stop()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()