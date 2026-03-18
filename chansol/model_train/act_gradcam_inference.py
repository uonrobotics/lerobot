#!/usr/bin/env python3

from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from act_gradcam import (
    build_single_sample_batch,
    compute_action_aware_gradcam,
    compute_cnn_only_gradcam,
    compute_joint_observation_saliency,
)

from tqdm import tqdm

# ============================================================
# User Settings
# ============================================================

CHECKPOINT_PATH = Path("/nas/Dataset/dualarm_ckpts/0316_act_dual_remove_b8/240000/pretrained_model")
DATASET_ROOT = Path("/nas/Dataset/dualarm_data/0313_each_arm/20260316_dualarm_remove_total")
REPO_ID = "user1/repo1"
OUTPUT_DIR = Path("/home/uon/ochansol/lerobot/chansol/model_train/grad_cam_out")
DEVICE = torch.device("cuda")

# Episode selection
EPISODE_INDEX = 70

# GIF settings
FRAME_STRIDE = 1
MAX_FRAMES = None  # e.g. 100
GIF_FPS = 8
DOWNSCALE = 0.9
CAM_NUM = None  # None이면 dataset/policy에 있는 모든 카메라 사용

# Grad-CAM target settings
ACTION_TIMESTEP = None  # None이면 chunk 전체 action을 종합해서 시각화
ACTION_DIM = None  # None이면 선택된 timestep 또는 chunk 전체의 action vector 절대값 합 기준


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def build_dataset_and_preprocessor(policy: ACTPolicy):
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )
    preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)

    delta_timestamps = {
        "action": make_delta_timestamps(policy.config.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        key: make_delta_timestamps(policy.config.observation_delta_indices, dataset_metadata.fps)
        for key in policy.config.image_features
    }

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )
    return dataset, dataset_metadata, preprocessor


def get_episode_row_indices(dataset: LeRobotDataset, episode_index: int) -> list[int]:
    episode_column = dataset.hf_dataset["episode_index"]
    frame_column = dataset.hf_dataset["frame_index"]

    rows: list[tuple[int, int]] = []
    for row_idx, (ep_idx, frm_idx) in enumerate(zip(episode_column, frame_column, strict=False)):
        ep_val = int(ep_idx.item()) if hasattr(ep_idx, "item") else int(ep_idx)
        frm_val = int(frm_idx.item()) if hasattr(frm_idx, "item") else int(frm_idx)
        if ep_val == episode_index:
            rows.append((frm_val, row_idx))

    if not rows:
        raise IndexError(f"Could not find any frames for episode_index={episode_index}.")

    rows.sort(key=lambda x: x[0])
    ordered_indices = [row_idx for _, row_idx in rows]

    if FRAME_STRIDE > 1:
        ordered_indices = ordered_indices[::FRAME_STRIDE]
    if MAX_FRAMES is not None:
        ordered_indices = ordered_indices[:MAX_FRAMES]

    return ordered_indices


def _camera_title(camera_key: str) -> str:
    return camera_key.split(".")[-1].replace("_", " ").title()


def compose_multi_camera_frame(
    camera_results: list[dict[str, np.ndarray | float | str]],
    episode_index: int,
    frame_index: int,
    joint_names: list[str],
    joint_saliency: np.ndarray,
    joint_score: float,
) -> Image.Image:
    n_cameras = len(camera_results)
    fig = plt.figure(figsize=(8, max(4.8, 2.5 * n_cameras + 1.0)), dpi=120)
    gs = fig.add_gridspec(n_cameras + 1, 2, height_ratios=[1.0] * n_cameras + [0.32])
    fig.suptitle(f"Episode {episode_index} | Frame {frame_index}", fontsize=10)

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

    return image


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    policy = ACTPolicy.from_pretrained(pretrained_name_or_path=CHECKPOINT_PATH)
    policy.to(DEVICE)
    policy.eval()

    dataset, dataset_metadata, preprocessor = build_dataset_and_preprocessor(policy)
    row_indices = get_episode_row_indices(dataset, EPISODE_INDEX)
    joint_names = list(dataset_metadata.features["observation.state"]["names"])
    image_feature_keys = list(policy.config.image_features)
    active_camera_count = len(image_feature_keys) if CAM_NUM is None else min(CAM_NUM, len(image_feature_keys))
    if active_camera_count <= 0:
        raise ValueError("CAM_NUM must be at least 1 or None.")

    gif_frames: list[Image.Image] = []
    for row_idx in tqdm(row_indices, desc="Processing frames"):
        sample = dataset[row_idx]
        batch = build_single_sample_batch(sample, preprocessor)
        frame_index = int(sample["frame_index"].item()) if hasattr(sample["frame_index"], "item") else int(sample["frame_index"])

        camera_results: list[dict[str, np.ndarray | float | str]] = []
        for camera_index in range(active_camera_count):
            action_cam = compute_action_aware_gradcam(
                policy=policy,
                processed_batch=batch,
                raw_sample=sample,
                camera_index=camera_index,
                action_timestep=ACTION_TIMESTEP,
                action_dim=ACTION_DIM,
            )
            cnn_cam = compute_cnn_only_gradcam(
                policy=policy,
                processed_batch=batch,
                raw_sample=sample,
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
            processed_batch=batch,
            action_timestep=ACTION_TIMESTEP,
            action_dim=ACTION_DIM,
        )

        composed = compose_multi_camera_frame(
            camera_results=camera_results,
            episode_index=EPISODE_INDEX,
            frame_index=frame_index,
            joint_names=joint_names,
            joint_saliency=joint_saliency.normalized_saliency,
            joint_score=joint_saliency.score,
        )
        gif_frames.append(composed)

        del camera_results, joint_saliency

    if not gif_frames:
        raise RuntimeError("No frames were generated for the GIF.")

    output_path = OUTPUT_DIR / f"episode_{EPISODE_INDEX:04d}_action_gradcam.gif"
    duration_ms = int(1000 / max(1, GIF_FPS))
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
    )

    print(f"Saved GIF: {output_path}")
    print(f"Frames used: {len(gif_frames)}")
    print(f"Frame stride: {FRAME_STRIDE}")


if __name__ == "__main__":
    main()
