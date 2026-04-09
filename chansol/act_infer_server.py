import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act_vit.modeling_act_vit import ACTViTPolicy

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action


import sys
sys.path.append("/home/cubox/workspace/isaac_chansol")
from socket_utils.vla_socket import vla_server
from typing import Any, Dict, Callable, Optional, Set, Tuple
from typing_extensions import override
import numpy as np


pre_trained_path = "/nas/AI_Checkpoints/VLA/act_vit/Isaacsim/0401_act_vit_aug_fullft_trans_vit_freeze/checkpoints/150000/pretrained_model"
dataset_root_path = "/nas/Dataset/VLA/UON/Isaacsim/OMY_apple_picking/auto_fixed_place_aug"

dataset_id = "user1/repo1"


model = ACTViTPolicy.from_pretrained(pretrained_name_or_path=pre_trained_path)
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
