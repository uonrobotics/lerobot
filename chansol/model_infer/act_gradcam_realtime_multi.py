#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import fields
from io import BytesIO
from multiprocessing import Event, Manager, Process, Queue
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rclpy
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame

from ..model_train.act_gradcam import (
    compute_action_aware_gradcam,
    compute_cnn_only_gradcam,
    compute_joint_observation_saliency,
)


DUALARM_ROOT = Path("/nas/Dataset/dualarm")
if str(DUALARM_ROOT) not in sys.path:
    sys.path.append(str(DUALARM_ROOT))

from lerobot_dataset.communicator import Communicator  # type: ignore  # noqa: E402
from lerobot_dataset.data_coverter import (  # type: ignore  # noqa: E402
    convert_compressedImage_to_numpy,
    convert_jointState_to_numpy_list,
)


# ============================================================
# User Settings
# ============================================================

REPO_ID = "user1/repo1"
DEVICE = torch.device("cuda")

TASK_DESCRIPTION = "pick the object"
ROBOT_TYPE = "omy_f3m"
OBSERVATION_STAGE = 0

MODEL_STAGE_CONFIGS = {
    0: {
        "name": "pick",
        "checkpoint_path": Path("/nas/Dataset/Dataset_dualarm_ckpts/0313_act_dual_pick_b8/300000/pretrained_model"),
        "dataset_root": Path("/nas/Dataset/dualarm_data/0313_each_arm/20260313_dualarm_pick"),
    },
    1: {
        "name": "barcode",
        "checkpoint_path": Path("/nas/Dataset/Dataset_dualarm_ckpts/0313_act_dual_barcode_b8/300000/pretrained_model"),
        "dataset_root": Path("/nas/Dataset/dualarm_data/0313_each_arm/20260313_dualarm_barcode"),
    },
    2: {
        "name": "remove",
        "checkpoint_path": Path("/nas/Dataset/dualarm_ckpts/0316_act_dual_remove_b8/300000/pretrained_model"),
        "dataset_root": Path("/nas/Dataset/dualarm_data/0313_each_arm/20260316_dualarm_remove_total"),
    },
}

COMM_CONFIG_PATH = DUALARM_ROOT / "config/comm.config.yaml"
CAM_NUM = None
ACTION_TIMESTEP = 0
ACTION_DIM = None
REFRESH_HZ = 4.0
DOWNSCALE = 0.9

JOINT_ORDER_R = [
    "robot1/joint1",
    "robot1/joint2",
    "robot1/joint3",
    "robot1/joint4",
    "robot1/joint5",
    "robot1/joint6",
    "robot1/rh_r1_joint",
]

JOINT_ORDER_L = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "rh_r1_joint",
]


def communication_process(shared_msgs, config_path: str, init_event: Event, send_queue: Queue) -> None:
    if not rclpy.ok():
        rclpy.init()

    node = Communicator(config_path, shared_data=shared_msgs, send_queue=send_queue)
    if not node.init():
        if rclpy.ok():
            rclpy.shutdown()
        return

    init_event.set()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def load_act_policy_compat(checkpoint_path: Path) -> ACTPolicy:
    config_path = checkpoint_path / "config.json"
    raw_config = json.loads(config_path.read_text())
    allowed_fields = {field.name for field in fields(ACTConfig)} | {"type"}
    sanitized_config = {k: v for k, v in raw_config.items() if k in allowed_fields}

    removed_keys = sorted(set(raw_config) - set(sanitized_config))
    if not removed_keys:
        return ACTPolicy.from_pretrained(pretrained_name_or_path=checkpoint_path)

    print(f"[Info] Ignoring unsupported config keys for ACTConfig: {removed_keys}")

    with TemporaryDirectory() as tmpdir:
        tmp_config_path = Path(tmpdir) / "config.json"
        tmp_config_path.write_text(json.dumps(sanitized_config, indent=4))
        config = ACTConfig.from_pretrained(tmpdir)
        return ACTPolicy.from_pretrained(pretrained_name_or_path=checkpoint_path, config=config)


def build_live_observation(msgs: dict[str, Any], observation_stage: int) -> dict[str, Any]:
    img_top = convert_compressedImage_to_numpy(msgs["cam_top"])
    img_wrist_l = convert_compressedImage_to_numpy(msgs["cam_wristL"])
    img_wrist_r = convert_compressedImage_to_numpy(msgs["cam_wristR"])

    follower_r = convert_jointState_to_numpy_list(msgs["followerR"], JOINT_ORDER_R)
    follower_l = convert_jointState_to_numpy_list(msgs["followerL"], JOINT_ORDER_L)

    scan_pulse_msg = msgs.get("scan_pulse")
    scan_done_msg = msgs.get("scan_done")
    scan_pulse = float(scan_pulse_msg.data) if scan_pulse_msg is not None else 0.0
    scan_done = float(scan_done_msg.data) if scan_done_msg is not None else 0.0

    return {
        "cam_top": img_top,
        "cam_wristL": img_wrist_l,
        "cam_wristR": img_wrist_r,
        "robot1/joint1": follower_r[0],
        "robot1/joint2": follower_r[1],
        "robot1/joint3": follower_r[2],
        "robot1/joint4": follower_r[3],
        "robot1/joint5": follower_r[4],
        "robot1/joint6": follower_r[5],
        "robot1/rh_r1_joint": follower_r[6],
        "joint1": follower_l[0],
        "joint2": follower_l[1],
        "joint3": follower_l[2],
        "joint4": follower_l[3],
        "joint5": follower_l[4],
        "joint6": follower_l[5],
        "rh_r1_joint": follower_l[6],
        "observation_stage": observation_stage,
        "scan_pulse": scan_pulse,
        "scan_done": scan_done,
    }


def build_raw_sample_for_gradcam(obs_frame: dict[str, Any]) -> dict[str, Any]:
    raw_sample: dict[str, Any] = {}
    for key, value in obs_frame.items():
        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] == 1:
                raw_sample[key] = value[0]
            else:
                raw_sample[key] = value
        else:
            raw_sample[key] = value
    return raw_sample


def _camera_title(camera_key: str) -> str:
    return camera_key.split(".")[-1].replace("_", " ").title()


def compose_multi_camera_frame(
    camera_results: list[dict[str, np.ndarray | float | str]],
    joint_names: list[str],
    joint_saliency: np.ndarray,
    joint_score: float,
) -> np.ndarray:
    n_cameras = len(camera_results)
    fig = plt.figure(figsize=(8, max(4.8, 2.5 * n_cameras + 1.0)), dpi=120)
    gs = fig.add_gridspec(n_cameras + 1, 2, height_ratios=[1.0] * n_cameras + [0.32])
    fig.suptitle("Real-time ACT Grad-CAM", fontsize=10)

    for row_idx, camera_result in enumerate(camera_results):
        action_ax = fig.add_subplot(gs[row_idx, 0])
        cnn_ax = fig.add_subplot(gs[row_idx, 1])
        camera_label = _camera_title(str(camera_result["camera_key"]))

        action_ax.imshow(camera_result["action_overlay"])
        action_ax.set_title(
            f"{camera_label} Action Grad-CAM\nscore={float(camera_result['action_score']):.3f}",
            fontsize=9,
        )
        action_ax.axis("off")

        cnn_ax.imshow(camera_result["cnn_overlay"])
        cnn_ax.set_title(
            f"{camera_label} CNN-only Grad-CAM\nscore={float(camera_result['cnn_score']):.3f}",
            fontsize=9,
        )
        cnn_ax.axis("off")

    joint_ax = fig.add_subplot(gs[n_cameras, :])
    x = np.arange(len(joint_names))
    joint_ax.bar(x, joint_saliency, color="tomato", edgecolor="black", linewidth=0.6)
    joint_ax.set_title(f"Joint Observation Saliency\nscore={joint_score:.3f}", fontsize=9)
    joint_ax.set_ylim(0.0, 1.0)
    joint_ax.set_ylabel("Importance", fontsize=8)
    joint_ax.set_xticks(x)
    joint_ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)

    image = Image.open(buffer).convert("RGB")
    if DOWNSCALE != 1.0:
        new_size = (
            max(1, int(image.width * DOWNSCALE)),
            max(1, int(image.height * DOWNSCALE)),
        )
        image = image.resize(new_size, Image.Resampling.BILINEAR)
    return np.asarray(image)


def render_gradcam_frame(
    policy: ACTPolicy,
    preprocess,
    ds_features: dict[str, dict],
    joint_names: list[str],
    active_camera_count: int,
    obs: dict[str, Any],
) -> np.ndarray:
    obs_frame = build_inference_frame(
        observation=obs,
        ds_features=ds_features,
        device=DEVICE,
        task=TASK_DESCRIPTION,
        robot_type=ROBOT_TYPE,
    )
    processed_batch = preprocess(obs_frame)
    raw_sample = build_raw_sample_for_gradcam(obs_frame)

    camera_results: list[dict[str, np.ndarray | float | str]] = []
    for camera_index in range(active_camera_count):
        action_cam = compute_action_aware_gradcam(
            policy=policy,
            processed_batch=processed_batch,
            raw_sample=raw_sample,
            camera_index=camera_index,
            action_timestep=ACTION_TIMESTEP,
            action_dim=ACTION_DIM,
        )
        cnn_cam = compute_cnn_only_gradcam(
            policy=policy,
            processed_batch=processed_batch,
            raw_sample=raw_sample,
            camera_index=camera_index,
        )
        camera_results.append(
            {
                "camera_key": action_cam.camera_key,
                "action_overlay": action_cam.overlay,
                "cnn_overlay": cnn_cam.overlay,
                "action_score": action_cam.score,
                "cnn_score": cnn_cam.score,
            }
        )

    joint_saliency = compute_joint_observation_saliency(
        policy=policy,
        processed_batch=processed_batch,
        action_timestep=ACTION_TIMESTEP,
        action_dim=ACTION_DIM,
    )

    frame = compose_multi_camera_frame(
        camera_results=camera_results,
        joint_names=joint_names,
        joint_saliency=joint_saliency.normalized_saliency,
        joint_score=joint_saliency.score,
    )

    del processed_batch, raw_sample, camera_results, joint_saliency
    return frame


def build_stage_bundle(stage: int) -> dict[str, Any]:
    if stage not in MODEL_STAGE_CONFIGS:
        raise KeyError(f"Unsupported observation stage: {stage}")

    stage_cfg = MODEL_STAGE_CONFIGS[stage]
    checkpoint_path = Path(stage_cfg["checkpoint_path"])
    dataset_root = Path(stage_cfg["dataset_root"])

    policy = load_act_policy_compat(checkpoint_path)
    policy.to(DEVICE)
    policy.eval()
    policy.reset()

    dataset_metadata = LeRobotDatasetMetadata(repo_id=REPO_ID, root=dataset_root)
    preprocess, _ = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)

    image_feature_keys = list(policy.config.image_features)
    active_camera_count = len(image_feature_keys) if CAM_NUM is None else min(CAM_NUM, len(image_feature_keys))
    if active_camera_count <= 0:
        raise ValueError("CAM_NUM must be at least 1 or None.")

    joint_names = list(dataset_metadata.features["observation.state"]["names"])

    return {
        "stage": stage,
        "stage_name": stage_cfg["name"],
        "checkpoint_path": checkpoint_path,
        "dataset_root": dataset_root,
        "policy": policy,
        "dataset_metadata": dataset_metadata,
        "preprocess": preprocess,
        "joint_names": joint_names,
        "active_camera_count": active_camera_count,
    }


def main() -> None:
    manager = Manager()
    shared_msgs = manager.dict(
        {
            "cam_top": None,
            "cam_wristR": None,
            "cam_wristL": None,
            "followerR": None,
            "followerL": None,
            "scan_pulse": None,
            "scan_done": None,
        }
    )
    send_queue = Queue()
    init_event = Event()

    comm_proc = Process(
        target=communication_process,
        args=(shared_msgs, str(COMM_CONFIG_PATH), init_event, send_queue),
    )
    comm_proc.daemon = True
    comm_proc.start()

    print("[Info] Waiting for communicator...")
    if not init_event.wait(timeout=5.0):
        comm_proc.terminate()
        raise RuntimeError("Failed to initialize communicator.")

    active_stage = OBSERVATION_STAGE
    stage_bundle = build_stage_bundle(active_stage)

    plt.ion()
    display_fig, display_ax = plt.subplots(figsize=(10, 8))
    display_artist = None
    display_ax.axis("off")

    def on_key(event) -> None:
        nonlocal active_stage, stage_bundle
        if event.key not in {"0", "1", "2"}:
            return
        requested_stage = int(event.key)
        if requested_stage == active_stage:
            return
        print(f"[Info] Switching stage {active_stage} -> {requested_stage}")
        active_stage = requested_stage
        stage_bundle = build_stage_bundle(active_stage)

    display_fig.canvas.mpl_connect("key_press_event", on_key)

    try:
        while True:
            missing_keys = [key for key, value in dict(shared_msgs).items() if key in {"cam_top", "cam_wristR", "cam_wristL", "followerR", "followerL"} and value is None]
            if missing_keys:
                print(f"[Warn] Missing topic data: {missing_keys}")
                time.sleep(1.0 / REFRESH_HZ)
                continue

            obs = build_live_observation(dict(shared_msgs), active_stage)
            frame = render_gradcam_frame(
                policy=stage_bundle["policy"],
                preprocess=stage_bundle["preprocess"],
                ds_features=stage_bundle["dataset_metadata"].features,
                joint_names=stage_bundle["joint_names"],
                active_camera_count=stage_bundle["active_camera_count"],
                obs=obs,
            )

            if display_artist is None:
                display_artist = display_ax.imshow(frame)
            else:
                display_artist.set_data(frame)
            display_ax.set_title(
                f"stage={active_stage} ({stage_bundle['stage_name']}) | "
                f"{stage_bundle['checkpoint_path'].parent.name} | "
                f"timestep={ACTION_TIMESTEP} | dim={ACTION_DIM}",
                fontsize=10,
            )
            display_fig.canvas.draw_idle()
            plt.pause(0.001)

            del obs, frame
            time.sleep(1.0 / REFRESH_HZ)

    except KeyboardInterrupt:
        print("Stopping real-time Grad-CAM viewer...")
    finally:
        plt.close(display_fig)
        if comm_proc.is_alive():
            comm_proc.terminate()
            comm_proc.join(timeout=1.0)


if __name__ == "__main__":
    main()
