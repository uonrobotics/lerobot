import os
import argparse
import time
import math
import torch
import numpy as np
from pathlib import Path
from multiprocessing import Process, Manager, Event, Queue
from pynput import keyboard


# ros2
import rclpy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray, Float32
from sensor_msgs.msg import JointState

# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame

# my code
from timer.precision_timer import PrecisionTimer
from lerobot_dataset.communicator import start_comm_worker
from lerobot_dataset.data_coverter import (convert_compressedImage_to_numpy,        # 데이터 컨버터
                                           convert_jointState_to_numpy_list,
                                           convert_jointTrajectory_to_numpy_list)


# ======================================================================
# 전역 변수
# ======================================================================
DEFAULT_SAVE_ROOT_PATH = Path.home() / '.cache/huggingface/lerobot'
TIMER                  = PrecisionTimer(30) # 30hz
MODEL_PATH             = ('/nas/MIN_JU_SIK/dataset/user2/dsr1_train/checkpoints/180000/pretrained_model') # local
DATASET_PATH           = '/nas/MIN_JU_SIK/dataset/user2/dsr1' # local
# MODEL_PATH             = ('/nas/MIN_JU_SIK/dataset/user2/dsr2_train/checkpoints/180000/pretrained_model') # local
# DATASET_PATH           = '/nas/MIN_JU_SIK/dataset/user2/dsr2' # local
TASK_DESCRIPTION       = "pick up the objec"
ROBOT_TYPE             = "omy_f3m"

JOINT_ORDER = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
]



# ==================================================================
# comm worker
# ==================================================================
def communication_process(shared_msgs, config_path, init_event, send_queue):
    """별도 프로세스에서 실행될 ROS2 노드 루틴"""
    if not rclpy.ok():
        rclpy.init()

    node = Communicator(config_path, shared_data=shared_msgs, send_queue=send_queue)

    if not node.init():
        if rclpy.ok(): rclpy.shutdown()
        return

    init_event.set() # 초기화 완료 알림

    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"Comm Process Error: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



# ==================================================================
# 키 리스너
# ==================================================================
OBSERVATION_STAGE = 0 # 녹화 시작하는 순간 stage: 0

KEY_STATUS = {'space': False, 'up': False, 'down':False}
def key_press_callback(key):
    global OBSERVATION_STAGE

    if key == keyboard.Key.space:
        if not KEY_STATUS['space']:
            KEY_STATUS['space'] = True
            OBSERVATION_STAGE = 0
            print(f'[Info ] Observation Stage: {OBSERVATION_STAGE}')

    if key == keyboard.Key.up:
        if not KEY_STATUS['up']:
            KEY_STATUS['up'] = True
            OBSERVATION_STAGE = OBSERVATION_STAGE +1; # sacpebar: stage +1
            print(f'[Info ] Observation Stage: {OBSERVATION_STAGE}')

    if key == keyboard.Key.down:
        if not KEY_STATUS['down']:
            KEY_STATUS['down'] = True
            OBSERVATION_STAGE = OBSERVATION_STAGE -1

def key_release_callback(key):
    if key == keyboard.Key.space: KEY_STATUS['space'] = False
    if key == keyboard.Key.up: KEY_STATUS['up'] = False
    if key == keyboard.Key.down: KEY_STATUS['down'] = False


















def main():
    # ------------------------------------------------------------------
    # 파라미터 파싱
    # python dsr_node.py --config config/comm.config.yaml
    #  ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/comm.config.yaml')
    parser.add_argument('--config', type=str, default=default_config)
    args, _ = parser.parse_known_args()


    # ------------------------------------------------------------------
    # ros2 초기화
    # ------------------------------------------------------------------
    if not rclpy.ok():
        rclpy.init()


    # ------------------------------------------------------------------
    # 키보드 리스너 시작
    # ------------------------------------------------------------------
    listener = keyboard.Listener(on_press=key_press_callback, on_release=key_release_callback)
    listener.start()


    # ------------------------------------------------------------------
    # 통신 프로세스 시작
    # ------------------------------------------------------------------
    shared_msgs = Manager().dict({})
    start_comm_worker(shared_msgs, args.config)
    time.sleep(1.0)


    # ------------------------------------------------------------------
    # ACT 모델 불러오기
    # ------------------------------------------------------------------
    device = torch.device("cuda")
    model = ACTPolicy.from_pretrained(MODEL_PATH)

    dataset_metadata = LeRobotDatasetMetadata(DATASET_PATH)
    preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)
    print(f'ACT 모델 불러옴')

    # ==================================================================




    # 인퍼런스 노드
    inference_node = rclpy.create_node('inference_pub_node')
    # 퍼블리셔 선언
    infer_joint_pub = inference_node.create_publisher(JointState, '/infer_joint_states', 10)
    gripper_pub = inference_node.create_publisher(Float32, '/infer_gripper', 10)

    print("[Info] Inference 모듈 시작")

    try:
        while True:
            TIMER.sleep()

            # --- 토픽 유효성 체크 ---
            msgs = dict(shared_msgs) # 토픽 최신 데이터

            required_keys = ['cam_top', 'cam_wrist', 'follower', 'gripper']
            missing = [k for k in required_keys if k not in msgs or msgs[k] is None]
            is_ready = len(missing) == 0

            if not is_ready:
                print(f"토픽 데이터 대기중: {missing}")
                continue


            # --- 데이터 변환 ---
            img_top           = convert_compressedImage_to_numpy(msgs['cam_top'])               # cam top
            img_wrist         = convert_compressedImage_to_numpy(msgs['cam_wrist'])             # cam wrist
            joint_follower    = convert_jointState_to_numpy_list(msgs['follower'], JOINT_ORDER) # 조인트
            gripper_val       = msgs['gripper'].data                                            # 그리퍼


            # --- 입력 데이터 구성 ---
            obs = {
                'cam_top'   : img_top,
                "cam_wrist" : img_wrist,
                'joint1'    : joint_follower[0],
                'joint2'    : joint_follower[1],
                'joint3'    : joint_follower[2],
                'joint4'    : joint_follower[3],
                'joint5'    : joint_follower[4],
                'joint6'    : joint_follower[5],
                'gripper': gripper_val,
            }


            # -- 프레임 빌드 ---
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
            )
            obs_frame['task'] = TASK_DESCRIPTION    # build_inference_frame시 누락됨
            obs_frame['robot_type'] = ROBOT_TYPE    # build_inference_frame시 누락됨

            # --- 데이터 전처리 ---
            # obs = preprocess(obs_frame) # 데이터를 Nomailize함 (학습시 데이터가 Nomalize하지 않아서 사용 않함)


            # --- 모델 추론 ---
            action = model.select_action(obs_frame)
            action = postprocess(action)

            # --- 출력 데이터 추출 ---
            action_np = action.squeeze().cpu().numpy()


            # 1. JointState 메시지 생성 (Arm 관절 6개)
            joint_msg = JointState()
            joint_msg.header.stamp = inference_node.get_clock().now().to_msg()
            joint_msg.name = JOINT_ORDER

            # action_np의 앞선 6개 값을 관절값으로 할당
            joint_msg.position = action_np[:6].tolist()

            # 2. Gripper 메시지 생성 (마지막 1개 값)
            gripper_msg = Float32()
            gripper_msg.data = float(action_np[6])
            # print(f'gripper_msg: {gripper_msg.data}')

            # 3. Publish
            infer_joint_pub.publish(joint_msg)
            gripper_pub.publish(gripper_msg)


    except KeyboardInterrupt:
        print("종료 중...")

if __name__ == '__main__':
    main()