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

# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame

# my code
from lerobot_dataset.communicator import Communicator
from timer.precision_timer import PrecisionTimer
from lerobot_dataset.communicator import Communicator
from lerobot_dataset.data_coverter import (convert_compressedImage_to_numpy,        # 데이터 컨버터
                                           convert_jointState_to_numpy_list,
                                           convert_jointTrajectory_to_numpy_list)


# ======================================================================
# 전역 변수
# ======================================================================
DEFAULT_SAVE_ROOT_PATH = Path.home() / '.cache/huggingface/lerobot'
TIMER                  = PrecisionTimer(30) # 30hz
MODEL_PATH             = ('/nas/MIN_JU_SIK/dataset/user1/dsr1_train/checkpoints/200000/pretrained_model') # local
DATASET_PATH           = '/nas/MIN_JU_SIK/dataset/user1/dsr1' # local
TASK_DESCRIPTION       = "scan the barcode and return object"
ROBOT_TYPE             = "omy_f3m"

JOINT_ORDER_R = [
    'robot1/joint1',
    'robot1/joint2',
    'robot1/joint3',
    'robot1/joint4',
    'robot1/joint5',
    'robot1/joint6',
    'robot1/rh_r1_joint'
]

JOINT_ORDER_L = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
]

TF_NAMES = [f"tf_{r}{c}" for r in range(4) for c in range(4)]
# 최종 17개 이름: TF(16개) + Gripper(1개)
STATE_NAMES = TF_NAMES + ['gripper']



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









TF_PUB = None





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
    # 멀티프로세싱 자원 준비
    # ------------------------------------------------------------------
    manager = Manager()
    shared_msgs = manager.dict({
        'cam_wrist'       : None,
        'follower'        : None,
        'tcp_tf'          : None,
        'gripper'         : None
    })
    send_queue = Queue() # 전송용 큐
    init_event = Event()


    # ------------------------------------------------------------------
    # 통신 프로세스 시작
    # ------------------------------------------------------------------
    comm_proc = Process(
        target=communication_process,
        args=(shared_msgs, args.config, init_event, send_queue)
    )
    comm_proc.daemon = True
    comm_proc.start()

    print("[Info] Communicator 초기화 대기 중...")
    if not init_event.wait(timeout=5.0):
        print("[Error] 초기화 실패")
        comm_proc.terminate()
        return


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
    tf_pub = inference_node.create_publisher(Float64MultiArray, '/infer_tf_matrix', 10)
    gripper_pub = inference_node.create_publisher(Float32, '/infer_gripper', 10)

    print("[Info] Inference 모듈 시작")

    try:
        while True:
            TIMER.sleep()

            # --- 토픽 유효성 체크 ---
            missing_keys = [k for k, v in shared_msgs.items() if v is None]
            if missing_keys:
                print(f"[Warn ] 토픽 데이터 없음: {missing_keys}")
                continue


            # --- 최신 데이터 ---
            msgs = dict(shared_msgs)


            # --- 데이터 변환 ---
            # 이미지
            img_wrist         = convert_compressedImage_to_numpy(msgs['cam_wrist'])

            # 팔로워 조인트 (사용 않함)
            # joint_follower    = convert_jointState_to_numpy_list(msgs['follower'], JOINT_ORDER)

            # TCP tf
            tf_flatten = np.array(msgs['tcp_tf'].data, dtype=np.float32).flatten()

            # 그리퍼
            gripper_val = msgs['gripper'].data

            #  TF(16)와 Gripper(1) 결합 -> (17,)
            state_tf_with_gripper = np.concatenate([tf_flatten, [gripper_val]]).astype(np.float32)


            # --- 입력 데이터 구성 ---
            obs = {
                "cam_wrist" : img_wrist,
                TF_NAMES[0]:  tf_flatten[0],
                TF_NAMES[1]:  tf_flatten[1],
                TF_NAMES[2]:  tf_flatten[2],
                TF_NAMES[3]:  tf_flatten[3],
                TF_NAMES[4]:  tf_flatten[4],
                TF_NAMES[5]:  tf_flatten[5],
                TF_NAMES[6]:  tf_flatten[6],
                TF_NAMES[7]:  tf_flatten[7],
                TF_NAMES[8]:  tf_flatten[8],
                TF_NAMES[9]:  tf_flatten[9],
                TF_NAMES[10]: tf_flatten[10],
                TF_NAMES[11]: tf_flatten[11],
                TF_NAMES[12]: tf_flatten[12],
                TF_NAMES[13]: tf_flatten[13],
                TF_NAMES[14]: tf_flatten[14],
                TF_NAMES[15]: tf_flatten[15],
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

            # 1. TCP Matrix (16개) 퍼블리시
            tf_msg = Float64MultiArray()
            tf_msg.data = action_np[:16].tolist()
            tf_pub.publish(tf_msg)

            # 2. Gripper (1개) 퍼블리시
            gripper_msg = Float32()
            gripper_msg.data = float(action_np[16])
            gripper_pub.publish(gripper_msg)

            # --- 보기 좋은 출력 ---
            action_matrix = action_np[:16].reshape(4, 4)
            action_gripper = action_np[16]

            print("\n" + "="*45)
            print(f" [Model Inference Result] ")
            print("-" * 45)
            print("  Predicted TCP Matrix (4x4):")
            for row in action_matrix:
                # 각 행을 소수점 4자리까지 정렬하여 출력
                print(f"  [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}, {row[3]:7.4f}]")

            print("-" * 45)
            print(f"  Predicted Gripper: {action_gripper:7.4f}")
            print("="*45)


            # --- 전송 ---
            # send_queue.put(('robot1', joint_msg_L ))
            # send_queue.put(('robot2', joint_msg))

    except KeyboardInterrupt:
        print("종료 중...")

if __name__ == '__main__':
    main()