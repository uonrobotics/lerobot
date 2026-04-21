#!/usr/bin/env python3

import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "lerobot").exists():
            return candidate

    raise RuntimeError(
        f"Could not find the lerobot repository root from {current}. "
        "Expected to find both pyproject.toml and src/lerobot."
    )


REPO_ROOT = _find_repo_root()
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.utils.import_utils import register_third_party_plugins


# ============================================================
# User Settings
# ============================================================
# PRETRAINED_PATH = Path(
#     "/nas/AI_Checkpoints/VLA/groot/Isaacsim/OMY_apple_picking/auto_random_place_aug/checkpoints/045000/pretrained_model"
# )
# DATASET_ROOT = Path("/nas/Dataset/VLA/UON/Isaacsim/OMY_apple_picking/auto_random_place_aug")
PRETRAINED_PATH = Path(
    "/nas/AI_Checkpoints/VLA/groot/0419_3obj/0418_scan_orange_b4_gpu2_150k/045000/pretrained_model"
)
DATASET_ROOT = Path("/nas/Dataset/VLA/0417_3obj/0417_merged_add_scan_stride2")
DATASET_ID = "user1/repo1"
DEVICE = "cuda"
TOKENIZER_ASSETS_REPO = None
TASK_OVERRIDE = None
PRINT_CHUNK = False
NUM_EPISODES_TO_SAMPLE = 5
NUM_FRAMES_PER_EPISODE = 5
RANDOM_SEED = 42
EVAL_DIR = Path(__file__).resolve().parent / "eval"


def _patch_groot_resize_wrapper() -> None:
    from lerobot.policies.groot import processor_groot
    from lerobot.processor.core import TransitionKey

    if getattr(processor_groot.GrootPackInputsStep, "_chansol_resize_patched", False):
        return

    original_call = processor_groot.GrootPackInputsStep.__call__

    def wrapped_call(self, transition):
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        img_keys = sorted([k for k in obs if k.startswith("observation.images.")])
        if not img_keys and "observation.image" in obs:
            img_keys = ["observation.image"]

        image_tensors = [obs[k] for k in img_keys if isinstance(obs.get(k), torch.Tensor)]
        if len(image_tensors) > 1:
            spatial_shapes = {tuple(img.shape[-2:]) for img in image_tensors}
            if len(spatial_shapes) > 1:
                target_h = min(img.shape[-2] for img in image_tensors)
                target_w = min(img.shape[-1] for img in image_tensors)
                resized_obs = dict(obs)
                for key in img_keys:
                    image = obs.get(key)
                    if not isinstance(image, torch.Tensor) or tuple(image.shape[-2:]) == (target_h, target_w):
                        continue

                    resized = F.interpolate(
                        image.float(),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if image.dtype != resized.dtype:
                        resized = resized.to(image.dtype)
                    resized_obs[key] = resized

                transition = dict(transition)
                transition[TransitionKey.OBSERVATION] = resized_obs

        return original_call(self, transition)

    processor_groot.GrootPackInputsStep.__call__ = wrapped_call
    processor_groot.GrootPackInputsStep._chansol_resize_patched = True


def _patch_groot_processor_compat_wrapper() -> None:
    from shutil import copytree

    from lerobot.policies.groot import processor_groot
    from lerobot.policies.groot.groot_n1 import DEFAULT_VENDOR_EAGLE_PATH
    from lerobot.policies.groot.utils import ensure_eagle_cache_ready

    if getattr(processor_groot, "_chansol_processor_compat_patched", False):
        return

    original_build = processor_groot._build_eagle_processor

    def compat_build(tokenizer_assets_repo=processor_groot.DEFAULT_TOKENIZER_ASSETS_REPO):
        vendor_dir = Path(DEFAULT_VENDOR_EAGLE_PATH)
        assets_path = Path(tokenizer_assets_repo).expanduser()
        use_local_assets = assets_path.is_absolute() or assets_path.exists()
        cache_dir = (
            assets_path.resolve()
            if use_local_assets
            else processor_groot.HF_LEROBOT_HOME / tokenizer_assets_repo
        )

        if use_local_assets:
            copytree(vendor_dir, cache_dir, dirs_exist_ok=True)
        else:
            ensure_eagle_cache_ready(
                vendor_dir=vendor_dir,
                cache_dir=cache_dir,
                assets_repo=tokenizer_assets_repo,
            )

        proc = original_build(tokenizer_assets_repo=str(cache_dir) if use_local_assets else tokenizer_assets_repo)
        image_processor = getattr(proc, "image_processor", None)
        if image_processor is not None and not hasattr(image_processor, "_prepare_image_like_inputs"):
            image_processor.__class__._prepare_image_like_inputs = image_processor.__class__._prepare_input_images
        proc.tokenizer.padding_side = "left"
        return proc

    processor_groot._build_eagle_processor = compat_build
    processor_groot._chansol_processor_compat_patched = True


def image_tensor_to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {image.shape}")

    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        max_value = float(np.nanmax(image))
        min_value = float(np.nanmin(image))
        if min_value >= 0.0 and max_value <= 1.0:
            image = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
        else:
            image = np.clip(np.round(image), 0, 255).astype(np.uint8)

    return image


def build_raw_observation(sample: dict[str, Any], ds_meta: LeRobotDatasetMetadata) -> dict[str, Any]:
    raw_observation: dict[str, Any] = {}

    for key in ds_meta.features:
        if not key.startswith("observation.images."):
            continue
        camera_name = key.removeprefix("observation.images.")
        raw_observation[camera_name] = image_tensor_to_hwc_uint8(sample[key])

    state_feature = ds_meta.features.get("observation.state")
    if state_feature and state_feature.get("names"):
        state_values = sample["observation.state"]
        if isinstance(state_values, torch.Tensor):
            state_values = state_values.detach().cpu().numpy()
        state_values = np.asarray(state_values, dtype=np.float32)

        for idx, name in enumerate(state_feature["names"]):
            raw_observation[name] = float(state_values[idx])

    return raw_observation


def tensor_to_list(value: torch.Tensor | np.ndarray) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def get_episode_rows(dataset: LeRobotDataset) -> dict[int, list[tuple[int, int]]]:
    episode_column = dataset.hf_dataset["episode_index"]
    frame_column = dataset.hf_dataset["frame_index"]

    rows_by_episode: dict[int, list[tuple[int, int]]] = {}
    for row_idx, (ep_idx, frm_idx) in enumerate(zip(episode_column, frame_column, strict=False)):
        episode_index = int(ep_idx.item()) if hasattr(ep_idx, "item") else int(ep_idx)
        frame_index = int(frm_idx.item()) if hasattr(frm_idx, "item") else int(frm_idx)
        rows_by_episode.setdefault(episode_index, []).append((frame_index, row_idx))

    for episode_index in rows_by_episode:
        rows_by_episode[episode_index].sort(key=lambda item: item[0])

    return rows_by_episode


def postprocess_action_chunk(postprocess, raw_chunk: torch.Tensor) -> torch.Tensor:
    processed_steps = []
    for step_idx in range(raw_chunk.shape[1]):
        step_action = raw_chunk[:, step_idx, :]
        processed_step = postprocess(step_action)
        processed_steps.append(processed_step)
    return torch.stack(processed_steps, dim=1)


def compute_chunk_metrics(pred_chunk: np.ndarray, gt_chunk: np.ndarray) -> dict[str, Any]:
    diff = pred_chunk - gt_chunk
    abs_diff = np.abs(diff)
    sq_diff = diff**2

    return {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt(sq_diff.mean())),
        "max_abs_error": float(abs_diff.max()),
        "l2_error": float(np.linalg.norm(diff)),
        "per_step_mae": abs_diff.mean(axis=1).tolist(),
        "per_dim_mae": abs_diff.mean(axis=0).tolist(),
    }


def main() -> None:
    register_third_party_plugins()
    _patch_groot_resize_wrapper()
    _patch_groot_processor_compat_wrapper()

    device = torch.device(DEVICE)
    ds_meta = LeRobotDatasetMetadata(repo_id=DATASET_ID, root=DATASET_ROOT)
    dataset = LeRobotDataset(repo_id=DATASET_ID, root=DATASET_ROOT)
    rows_by_episode = get_episode_rows(dataset)

    model = GrootPolicy.from_pretrained(pretrained_name_or_path=PRETRAINED_PATH)
    model.to(device)
    model.eval()
    model.config.device = str(device)
    if TOKENIZER_ASSETS_REPO:
        model.config.tokenizer_assets_repo = TOKENIZER_ASSETS_REPO

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        dataset_stats=ds_meta.stats,
    )

    config_chunk_size = int(model.config.n_action_steps)
    eval_chunk_size = min(config_chunk_size, 16)
    rng = np.random.default_rng(RANDOM_SEED)

    valid_episode_indices = [
        episode_index
        for episode_index, rows in rows_by_episode.items()
        if len(rows) >= eval_chunk_size
    ]
    if not valid_episode_indices:
        raise RuntimeError(f"No episodes have at least eval_chunk_size={eval_chunk_size} frames.")

    num_episodes = min(NUM_EPISODES_TO_SAMPLE, len(valid_episode_indices))
    sampled_episodes = sorted(rng.choice(valid_episode_indices, size=num_episodes, replace=False).tolist())

    evaluations: list[dict[str, Any]] = []
    for episode_index in sampled_episodes:
        rows = rows_by_episode[episode_index]
        max_start = len(rows) - eval_chunk_size
        candidate_starts = np.arange(max_start + 1)
        num_frames = min(NUM_FRAMES_PER_EPISODE, len(candidate_starts))
        sampled_starts = sorted(rng.choice(candidate_starts, size=num_frames, replace=False).tolist())

        for start_offset in sampled_starts:
            start_frame_index, start_row_idx = rows[start_offset]
            sample = dataset[start_row_idx]
            raw_observation = build_raw_observation(sample, ds_meta)
            task = TASK_OVERRIDE if TASK_OVERRIDE is not None else sample["task"]
            obs_frame = build_inference_frame(
                observation=raw_observation,
                task=task,
                ds_features=ds_meta.features,
                device=device,
            )

            with torch.inference_mode():
                batch = preprocess(obs_frame)
                raw_chunk = model.predict_action_chunk(batch)
                pred_chunk = postprocess_action_chunk(postprocess, raw_chunk)

            pred_chunk_np = pred_chunk.squeeze(0).detach().cpu().float().numpy()
            pred_chunk_len = int(pred_chunk_np.shape[0])
            gt_row_indices = [row_idx for _, row_idx in rows[start_offset : start_offset + pred_chunk_len]]
            gt_frame_indices = [frame_idx for frame_idx, _ in rows[start_offset : start_offset + pred_chunk_len]]
            gt_chunk_np = np.stack(
                [
                    np.asarray(dataset.hf_dataset[row_idx]["action"], dtype=np.float32)
                    for row_idx in gt_row_indices
                ],
                axis=0,
            )

            metrics = compute_chunk_metrics(pred_chunk_np, gt_chunk_np)
            eval_item = {
                "episode_index": int(episode_index),
                "start_frame_index": int(start_frame_index),
                "start_row_index": int(start_row_idx),
                "task": task,
                "config_chunk_size": int(config_chunk_size),
                "evaluated_chunk_size": int(pred_chunk_len),
                "gt_row_indices": [int(row_idx) for row_idx in gt_row_indices],
                "gt_frame_indices": [int(frame_idx) for frame_idx in gt_frame_indices],
                "metrics": metrics,
                "predicted_first_action_named": make_robot_action(pred_chunk[:, 0, :], ds_meta.features),
                "ground_truth_first_action_named": {
                    name: float(value)
                    for name, value in zip(
                        ds_meta.features["action"]["names"],
                        gt_chunk_np[0].tolist(),
                        strict=False,
                    )
                },
            }
            if PRINT_CHUNK:
                eval_item["predicted_action_chunk"] = pred_chunk_np.tolist()
                eval_item["ground_truth_action_chunk"] = gt_chunk_np.tolist()
                eval_item["raw_action_chunk"] = raw_chunk.detach().cpu().float().numpy().tolist()

            evaluations.append(eval_item)

    summary = {
        "num_samples": len(evaluations),
        "mean_mae": float(np.mean([item["metrics"]["mae"] for item in evaluations])),
        "mean_rmse": float(np.mean([item["metrics"]["rmse"] for item in evaluations])),
        "mean_l2_error": float(np.mean([item["metrics"]["l2_error"] for item in evaluations])),
        "max_l2_error": float(np.max([item["metrics"]["l2_error"] for item in evaluations])),
    }

    result = {
        "dataset_root": str(DATASET_ROOT.resolve()),
        "dataset_id": DATASET_ID,
        "pretrained_path": str(PRETRAINED_PATH.resolve()),
        "device": DEVICE,
        "random_seed": int(RANDOM_SEED),
        "num_episodes_sampled": int(num_episodes),
        "num_frames_per_episode": int(NUM_FRAMES_PER_EPISODE),
        "config_chunk_size": int(config_chunk_size),
        "evaluated_chunk_size": int(eval_chunk_size),
        "sampled_episodes": sampled_episodes,
        "summary": summary,
        "evaluations": evaluations,
    }

    print(json.dumps(result["summary"], indent=2))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = EVAL_DIR / f"groot_chunk_eval_{timestamp}.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"Saved eval result to: {save_path}")


if __name__ == "__main__":
    main()
