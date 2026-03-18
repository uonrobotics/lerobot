"""Train ACT and periodically save Grad-CAM images for selected episode/frame samples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from act_gradcam import (
    build_single_sample_batch,
    compute_action_aware_gradcam,
    compute_cnn_only_gradcam,
    compute_joint_observation_saliency,
)


wandb.init(
    project="Isaacsim_OMY_apple_picking_auto",
    name="shift4_filtered_gradcam",
    resume="allow",
    config={
        "lr": 1e-5,
        "batch_size": 16,
        "model": "act",
        "dataset_path": "/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking_auto_shift4_filtered",
        "training_steps": int(100e4),
    },
)


# ============================================================
# User Settings
# ============================================================

DATASET_ID = "user1/repo1"
DATASET_ROOT_PATH = wandb.config.dataset_path
PRE_CHECKPOINT_PATH = ""
OUTPUT_ROOT = Path("/nas/AI_Checkpoints/VLA/act")
OPTIM_NAME = "adamw"
SCHEDULER_NAME = "cosine_decay_with_warmup"
DEVICE = torch.device("cuda")
LOG_FREQ = 20
SAVE_STEP = 10000

# Grad-CAM save settings
GRADCAM_EVERY_STEPS = 500
GRADCAM_OUTPUT_DIRNAME = "gradcam_snapshots"
GRADCAM_ACTION_TIMESTEP = None
GRADCAM_ACTION_DIM = None

# Pick exactly the samples you want to visualize.
# One image will be saved per tuple, so 4 tuples means 4 images every trigger step.
GRADCAM_TARGETS: list[tuple[int, int]] = [
    (0, 40),
    (1, 55),
    (2, 44),
    (3, 34),

    (0, 107),
    (1, 113),
    (2, 86),
    (3, 77),

    (0, 144),
    (1, 147),
    (2, 120),
    (3, 145),
]


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def build_gradcam_sample_lookup(dataset: LeRobotDataset) -> dict[tuple[int, int], int]:
    episode_column = dataset.hf_dataset["episode_index"]
    frame_column = dataset.hf_dataset["frame_index"]
    lookup: dict[tuple[int, int], int] = {}
    for row_idx, (ep_idx, frm_idx) in enumerate(zip(episode_column, frame_column, strict=False)):
        ep_val = int(ep_idx.item()) if hasattr(ep_idx, "item") else int(ep_idx)
        frm_val = int(frm_idx.item()) if hasattr(frm_idx, "item") else int(frm_idx)
        lookup[(ep_val, frm_val)] = row_idx
    return lookup


def compose_gradcam_snapshot(
    action_results,
    cnn_results,
    joint_names: list[str],
    joint_saliency: np.ndarray,
    joint_score: float,
    episode_index: int,
    frame_index: int,
) -> plt.Figure:
    n_cameras = len(action_results)
    fig = plt.figure(figsize=(8, max(5.5, 2.8 * n_cameras)), dpi=120)
    gs = fig.add_gridspec(n_cameras + 1, 2, height_ratios=[1.0] * n_cameras + [0.35])
    fig.suptitle(f"Episode {episode_index} | Frame {frame_index}", fontsize=11)

    for row_idx, (action_result, cnn_result) in enumerate(zip(action_results, cnn_results, strict=False)):
        action_ax = fig.add_subplot(gs[row_idx, 0])
        cnn_ax = fig.add_subplot(gs[row_idx, 1])
        camera_label = action_result.camera_key.split(".")[-1]

        action_ax.imshow(action_result.overlay)
        action_ax.set_title(
            f"{camera_label} Action Grad-CAM\nscore={action_result.score:.3f}",
            fontsize=9,
        )
        action_ax.axis("off")

        cnn_ax.imshow(cnn_result.overlay)
        cnn_ax.set_title(
            f"{camera_label} CNN-only Grad-CAM\nscore={cnn_result.score:.3f}",
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
    return fig


def save_periodic_gradcams(
    policy: ACTPolicy,
    preprocessor,
    dataset: LeRobotDataset,
    dataset_metadata: LeRobotDatasetMetadata,
    sample_lookup: dict[tuple[int, int], int],
    output_dir: Path,
    step: int,
) -> None:
    was_training = policy.training
    policy.eval()
    policy.zero_grad(set_to_none=True)

    joint_names = list(dataset_metadata.features["observation.state"]["names"])
    image_feature_keys = list(policy.config.image_features)
    wandb_episode_images: dict[str, list[wandb.Image]] = {}

    for episode_index, frame_index in GRADCAM_TARGETS:
        key = (episode_index, frame_index)
        if key not in sample_lookup:
            print(f"[gradcam] skip missing sample episode={episode_index}, frame={frame_index}")
            continue

        sample = dataset[sample_lookup[key]]
        batch = build_single_sample_batch(sample, preprocessor)

        action_results = []
        cnn_results = []
        for camera_index in range(len(image_feature_keys)):
            action_results.append(
                compute_action_aware_gradcam(
                    policy=policy,
                    processed_batch=batch,
                    raw_sample=sample,
                    camera_index=camera_index,
                    action_timestep=GRADCAM_ACTION_TIMESTEP,
                    action_dim=GRADCAM_ACTION_DIM,
                )
            )
            cnn_results.append(
                compute_cnn_only_gradcam(
                    policy=policy,
                    processed_batch=batch,
                    raw_sample=sample,
                    camera_index=camera_index,
                )
            )

        joint_result = compute_joint_observation_saliency(
            policy=policy,
            processed_batch=batch,
            action_timestep=GRADCAM_ACTION_TIMESTEP,
            action_dim=GRADCAM_ACTION_DIM,
        )

        fig = compose_gradcam_snapshot(
            action_results=action_results,
            cnn_results=cnn_results,
            joint_names=joint_names,
            joint_saliency=joint_result.normalized_saliency,
            joint_score=joint_result.score,
            episode_index=episode_index,
            frame_index=frame_index,
        )

        save_path = output_dir / (
            f"step_{step:07d}_episode_{episode_index:04d}_frame_{frame_index:04d}.png"
        )
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        wandb_key = f"gradcam/ep_{episode_index:04d}_frame_{frame_index:04d}"
        wandb_image = wandb.Image(
            fig,
            caption=f"step={step}, episode={episode_index}, frame={frame_index}",
        )
        episode_key = f"gradcam/episode_{episode_index:04d}"
        wandb_episode_images.setdefault(episode_key, []).append(wandb_image)
        plt.close(fig)

        del action_results, cnn_results, joint_result, batch, sample
        policy.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if wandb_episode_images:
        wandb.log(
            {
                "train/step": step,
                **wandb_episode_images,
            },
            step=step,
        )

    if was_training:
        policy.train()


def main():
    output_directory = OUTPUT_ROOT / wandb.run.project / wandb.run.name
    output_directory.mkdir(parents=True, exist_ok=True)
    gradcam_output_dir = output_directory / GRADCAM_OUTPUT_DIRNAME
    gradcam_output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = wandb.config.batch_size
    training_steps = wandb.config.training_steps

    dataset_metadata = LeRobotDatasetMetadata(
        repo_id=DATASET_ID,
        root=DATASET_ROOT_PATH,
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        optimizer_lr=wandb.config.lr,
    )

    if PRE_CHECKPOINT_PATH:
        policy = ACTPolicy(cfg).from_pretrained(pretrained_name_or_path=PRE_CHECKPOINT_PATH)
    else:
        policy = ACTPolicy(cfg)

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(DEVICE)

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        key: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for key in cfg.image_features
    }

    dataset = LeRobotDataset(
        DATASET_ID,
        root=DATASET_ROOT_PATH,
        delta_timestamps=delta_timestamps,
    )
    gradcam_sample_lookup = build_gradcam_sample_lookup(dataset)

    if OPTIM_NAME == "adamw":
        optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    elif OPTIM_NAME == "sgd":
        from lerobot.optim.optimizers import SGDConfig

        optimizer = SGDConfig(
            lr=cfg.optimizer_lr,
            momentum=0.9,
            weight_decay=cfg.optimizer_weight_decay,
        ).build(policy.parameters())
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIM_NAME}")

    scheduler = None
    if SCHEDULER_NAME == "cosine_decay_with_warmup":
        scheduler_cfg = CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=min(1000, max(1, training_steps // 100)),
            num_decay_steps=training_steps,
            peak_lr=cfg.optimizer_lr,
            decay_lr=cfg.optimizer_lr * 0.1,
        )
        scheduler = scheduler_cfg.build(optimizer, training_steps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=DEVICE.type != "cpu",
        drop_last=True,
    )

    step = 0
    done = False
    while not done:
        pbar = tqdm(dataloader, desc="Training", unit="batch")
        for batch in pbar:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.3f}", step=step, lr=current_lr)

            if step % LOG_FREQ == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/step": step,
                        "train/lr": current_lr,
                    }
                )

            if step > 0 and step % GRADCAM_EVERY_STEPS == 0:
                save_periodic_gradcams(
                    policy=policy,
                    preprocessor=preprocessor,
                    dataset=dataset,
                    dataset_metadata=dataset_metadata,
                    sample_lookup=gradcam_sample_lookup,
                    output_dir=gradcam_output_dir,
                    step=step,
                )

            if step % SAVE_STEP == 0 and step != 0:
                output_path = output_directory / f"{OPTIM_NAME}_{step:06d}steps_{batch_size}bs"
                output_path.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(output_path)
                preprocessor.save_pretrained(output_path)
                postprocessor.save_pretrained(output_path)

            step += 1
            if step >= training_steps:
                done = True
                break

    wandb.finish()


if __name__ == "__main__":
    main()
