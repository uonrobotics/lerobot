import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action


import sys
sys.path.append("/home/uon/ochansol/isaac_code/isaac_chansol")
from socket_utils.vla_socket import vla_server
from typing import Any, Dict, Callable, Optional, Set, Tuple
from typing_extensions import override
import numpy as np


# pre_trained_path = "/home/uon/ochansol/lerobot/chansol/model_train/weights/isaac_omy_put_food_act_adamw_090000steps_16bs"
# pre_trained_path = "/nas/AI_Checkpoints/VLA/act/act_demo_apple_pepper_1+2_batch16/checkpoints/last/pretrained_model"
# pre_trained_path = "/home/uon/ochansol/lerobot/chansol/model_train/weights/isaac_omy_put_apple_act_adamw_090000steps_16bs"

# dataset_root_path = "/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking"
# dataset_root_path = "/nas/Dataset/VLA/UON/omy_f3m_demo_apple_pepper_1+2"

# pre_trained_path = "/nas/AI_Checkpoints/VLA/act/test_1_adamw_140000steps_10bs"
# dataset_root_path = "/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking_auto_shift4"

# pre_trained_path = "/nas/AI_Checkpoints/VLA/act/Isaacsim_OMY_apple_picking_auto/shift4_filtered_gradcam/adamw_290000steps_16bs"
# dataset_root_path = "/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking_auto_shift4_filtered"

# pre_trained_path = "/nas/AI_Checkpoints/VLA/act/Isaacsim_OMY_apple_picking_auto_augmented_shift1_merged/shift1_gradcam/adamw_040000steps_16bs"
# dataset_root_path = "/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking_auto_augmented_shift1_merged"

pre_trained_path = "/nas/AI_Checkpoints/VLA/act/Isaacsim_OMY_apple_picking_auto_fixed_box/shift1_filtered_colorjitter_gradcam/adamw_070000steps_16bs"
dataset_root_path = "/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking_auto_fixed_box"


dataset_id = "user1/repo1"


model = ACTPolicy.from_pretrained(pretrained_name_or_path=pre_trained_path)
device = torch.device("cuda")  # or "cuda" or "cpu" or "mps"
# This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
dataset_metadata = LeRobotDatasetMetadata(
    repo_id=dataset_id,
    root=dataset_root_path,
)
preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)


def infer_fn(images, obss, action_type) -> Dict[str, Any]:

    obs = {
        'cam_top': images['full'],
        'cam_wrist': images['wrist'],
        "joint1":obss["joint_state"][0],
        "joint2":obss["joint_state"][1],
        "joint3":obss["joint_state"][2],
        "joint4":obss["joint_state"][3],
        "joint5":obss["joint_state"][4],
        "joint6":obss["joint_state"][5],
        "rh_r1_joint":obss["joint_state"][6],

    }
    print(obs["joint1"], obs["joint2"], obs["joint3"], obs["joint4"], obs["joint5"], obs["joint6"], obs["rh_r1_joint"])
    obs_frame = build_inference_frame(
    observation=obs, ds_features=dataset_metadata.features, device=device
    )

    obs = preprocess(obs_frame)
    action = model.select_action(obs)
    action = postprocess(action)

    action = make_robot_action(action, dataset_metadata.features)
    action = np.array([i for i in action.values()])

    return action



cfg = vla_server.VLAServerConfig(host="0.0.0.0", port=1823, decode_jpeg=True)
server = vla_server.VLARpcServer(cfg, infer_fn=infer_fn)


server.start_forever()
