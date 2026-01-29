import torch
import time
import numpy as np
np.set_printoptions(suppress=True, precision=2)

from pathlib import Path

# Lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame

# ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from trajectory_msgs.msg import JointTrajectory

# my code
from communicator import Communicator
from data_convertor.data_coverter import (
    convert_compressedImage_to_numpy,           # compressedImage -> np
    # jointTrajectory -> np
    convert_jointState_to_numpy_list            # jointState -> np
)


FPS = 30

DEFAULT_SAVE_ROOT_PATH = Path.home() / '.cache/huggingface/lerobot'
JOINT_ORDER = [
    'right_joint1',
    'right_joint2',
    'right_joint3',
    'right_joint4',
    'right_joint5',
    'right_joint6',
    'right_rh_r1_joint'
]


MODEL_PATH = DEFAULT_SAVE_ROOT_PATH / "outputs/train_user1_repo1/checkpoints/last/pretrained_model/" # local
DATASET_PATH = DEFAULT_SAVE_ROOT_PATH / "user1/repo1" # local

def main():
    # ros2 설정 -------------------------------
    if not rclpy.ok():
        rclpy.init()

    # 통신 노드 시작
    communicator = Communicator()
    communicator.start()


    # ACT 모델 불러오기 -------------------------
    device = torch.device("cuda")
    model = ACTPolicy.from_pretrained(MODEL_PATH)


    # 데이터셋 불러오기 -------------------------
    dataset_metadata = LeRobotDatasetMetadata(DATASET_PATH)
    preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)



    while True:
        loop_start = time.time()

        # 토픽 데이터 받기 ---------------------
        topic_msg = communicator.get_latest_msgs()

        # 데이터 유효성 검사
        if topic_msg['cam_top'] is None:
            print(f'\r[Warn ] {loop_start:8.6f} 키넥트 카메라 데이터 없음')
            continue

        if topic_msg['cam_wrist'] is None:
            print(f'\r[Warn ] {loop_start:8.6f} 손목 카메라 데이터 없음')
            continue

        if topic_msg['follower'] is None:
            print(f'\r[Warn ] {loop_start:8.6f} 팔로우암 데이터 없음')
            continue



        # 데이터 변환 -------------------------
        img_top = convert_compressedImage_to_numpy(topic_msg['cam_top'])                        # cam_top
        img_wrist = convert_compressedImage_to_numpy(topic_msg['cam_wrist'])                    # cam_wrist
        follower_numpy = convert_jointState_to_numpy_list(topic_msg['follower'], JOINT_ORDER)   # state



        # Inference 데이터 설정 ---------------
        obs = {
            "cam_top": img_top,
            "cam_wrist": img_wrist,
            "right_joint1": follower_numpy[0],
            "right_joint2": follower_numpy[1],
            "right_joint3": follower_numpy[2],
            "right_joint4": follower_numpy[3],
            "right_joint5": follower_numpy[4],
            "right_joint6": follower_numpy[5],
            "right_rh_r1_joint": follower_numpy[6],
        }

        # 프레임 빌드
        obs_frame = build_inference_frame(
            observation=obs, ds_features=dataset_metadata.features, device=device
        )
        obs_frame['task'] = "pick up the zipper bag"    # build_inference_frame시 누락됨
        obs_frame['robot_type'] = 'omy_f3m'             # build_inference_frame시 누락됨

        # 데이터 전처리
        # obs = preprocess(obs_frame) # 데이터를 Nomailize함 (학습시 데이터가 Nomalize하지 않아서 사용 않함)

        # Model predict
        action = model.select_action(obs_frame)
        action = postprocess(action)

        # 로봇 동작
        joint = action.squeeze().tolist()
        # communicator.action_publish(joint)

        # FPS 제한
        elapsed = time.time() - loop_start
        sleep_time = max(0, (1.0 / FPS) - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()