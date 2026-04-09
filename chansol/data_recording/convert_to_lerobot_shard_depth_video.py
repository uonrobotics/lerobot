#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import configs.Robotis_OMY_isaac_depth as cfg
from get_data import DataAggregator
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw episodes into one LeRobot shard dataset while storing depth as "
            "compressed pseudo-video frames (uint8 RGB grayscale). Existing scripts are left untouched."
        )
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the raw recorded data.")
    parser.add_argument("--save_path", type=str, required=True, help="Output shard directory.")
    parser.add_argument("--shift", type=int, default=1, help="Time step shift for leader action.")
    parser.add_argument("--start_num", type=int, default=None, help="Inclusive start index in episode_exist_list.")
    parser.add_argument("--end_num", type=int, default=None, help="Exclusive end index in episode_exist_list.")
    parser.add_argument("--episodes_per_shard", type=int, default=None, help="Optional shard size.")
    parser.add_argument("--shard_index", type=int, default=None, help="Optional shard number.")
    parser.add_argument("--image_writer_processes", type=int, default=4, help="LeRobot image writer processes.")
    parser.add_argument("--image_writer_threads", type=int, default=8, help="LeRobot image writer threads.")
    parser.add_argument(
        "--depth-min",
        type=float,
        default=None,
        help="Global minimum depth value used for uint8 quantization. If omitted, estimated from the dataset slice.",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Global maximum depth value used for uint8 quantization. If omitted, estimated from the dataset slice.",
    )
    parser.add_argument(
        "--depth-low-percentile",
        type=float,
        default=1.0,
        help="Lower percentile used when estimating depth-min automatically.",
    )
    parser.add_argument(
        "--depth-high-percentile",
        type=float,
        default=99.0,
        help="Upper percentile used when estimating depth-max automatically.",
    )
    parser.add_argument(
        "--depth-estimation-samples-per-episode",
        type=int,
        default=16,
        help="Number of depth frames sampled per episode when estimating quantization range.",
    )
    return parser.parse_args()


def resolve_episode_range(
    total_episodes: int,
    start_num: int | None,
    end_num: int | None,
    episodes_per_shard: int | None,
    shard_index: int | None,
) -> tuple[int, int]:
    if episodes_per_shard is not None or shard_index is not None:
        if episodes_per_shard is None or shard_index is None:
            raise ValueError("--episodes_per_shard and --shard_index must be provided together.")
        if episodes_per_shard <= 0:
            raise ValueError("--episodes_per_shard must be > 0.")
        if shard_index < 0:
            raise ValueError("--shard_index must be >= 0.")
        start_num = shard_index * episodes_per_shard
        end_num = min(total_episodes, start_num + episodes_per_shard)
    else:
        start_num = 0 if start_num is None else start_num
        end_num = total_episodes if end_num is None or end_num < 0 else end_num

    start_num = max(0, start_num)
    end_num = min(total_episodes, end_num)

    if start_num >= end_num:
        raise ValueError(
            f"Resolved empty episode range: start_num={start_num}, end_num={end_num}, total={total_episodes}"
        )
    return start_num, end_num


def resize_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if depth.shape == target_hw:
        return depth.astype(np.float32, copy=False)

    tensor = torch.from_numpy(depth.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return resized[0, 0].cpu().numpy()


def encode_depth_as_video_frame(
    depth: np.ndarray,
    *,
    depth_min: float,
    depth_max: float,
    target_hw: tuple[int, int],
) -> np.ndarray:
    depth = resize_depth(depth, target_hw)
    depth = np.nan_to_num(depth, nan=depth_max, posinf=depth_max, neginf=depth_min)
    depth = np.clip(depth, depth_min, depth_max)

    scale = depth_max - depth_min
    if scale <= 0:
        raise ValueError(f"Invalid depth range: depth_min={depth_min}, depth_max={depth_max}")

    normalized = (depth - depth_min) / scale
    quantized = np.round(normalized * 255.0).astype(np.uint8)
    return np.repeat(quantized[..., None], 3, axis=2)


def estimate_depth_range(
    data: DataAggregator,
    episode_ids: list[int],
    *,
    low_percentile: float,
    high_percentile: float,
    samples_per_episode: int,
) -> tuple[float, float]:
    samples: list[np.ndarray] = []

    for ep_num in tqdm(episode_ids, desc="Estimating depth range"):
        data.setup(ep_num)
        total = data.total_data_num
        if total <= 0:
            continue

        sample_count = min(samples_per_episode, total)
        frame_indices = np.linspace(0, total - 1, sample_count, dtype=int)
        for idx in frame_indices:
            data.step_idx = int(idx)
            for depth in (data.get_depth_top(), data.get_depth_wrist()):
                flat = np.asarray(depth, dtype=np.float32).reshape(-1)
                finite = flat[np.isfinite(flat)]
                if finite.size > 0:
                    samples.append(finite)

    if not samples:
        raise RuntimeError("Could not estimate depth range because no finite depth values were found.")

    merged = np.concatenate(samples, axis=0)
    depth_min = float(np.percentile(merged, low_percentile))
    depth_max = float(np.percentile(merged, high_percentile))

    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_min >= depth_max:
        raise RuntimeError(
            f"Estimated invalid depth range: depth_min={depth_min}, depth_max={depth_max}"
        )
    return depth_min, depth_max


def build_features(
    *,
    depth_top_shape: tuple[int, int],
    depth_wrist_shape: tuple[int, int],
) -> dict:
    features = {key: value.copy() for key, value in cfg.FEATURES.items()}

    # LeRobot video/image features currently require 3D visual shapes.
    # Keep the original observation.depth.* keys, but store the encoded depth video
    # as pseudo-RGB grayscale frames so the dataset can be written as mp4.
    features["observation.depth.cam_top"] = {
        "dtype": "video",
        "shape": (*depth_top_shape, 3),
        "names": ["height", "width", "channels"],
    }
    features["observation.depth.cam_wrist"] = {
        "dtype": "video",
        "shape": (*depth_wrist_shape, 3),
        "names": ["height", "width", "channels"],
    }
    return features


def save_depth_encoding_metadata(
    save_root: Path,
    *,
    depth_min: float,
    depth_max: float,
    top_original_shape: tuple[int, int],
    wrist_original_shape: tuple[int, int],
    top_encoded_shape: tuple[int, int],
    wrist_encoded_shape: tuple[int, int],
) -> None:
    metadata = {
        "encoding": {
            "type": "uint8_grayscale_triplicated_video",
            "reconstruction": "depth ~= frame[..., 0] / 255 * (depth_max - depth_min) + depth_min",
            "depth_min": depth_min,
            "depth_max": depth_max,
        },
        "features": {
            "observation.depth.cam_top": {
                "original_shape_hw": list(top_original_shape),
                "encoded_shape_hw": list(top_encoded_shape),
            },
            "observation.depth.cam_wrist": {
                "original_shape_hw": list(wrist_original_shape),
                "encoded_shape_hw": list(wrist_encoded_shape),
            },
        },
    }
    with (save_root / "depth_video_encoding.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()

    save_root = Path(args.save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    data = DataAggregator(args.data_root)
    total_episodes = len(data.episode_exist_list)
    start_num, end_num = resolve_episode_range(
        total_episodes=total_episodes,
        start_num=args.start_num,
        end_num=args.end_num,
        episodes_per_shard=args.episodes_per_shard,
        shard_index=args.shard_index,
    )

    selected_episode_ids = [int(ep) for ep in data.episode_exist_list[start_num:end_num]]
    if not selected_episode_ids:
        raise RuntimeError("No episodes selected.")

    data.setup(selected_episode_ids[0])
    sample_rgb_top = data.get_image_top()
    sample_rgb_wrist = data.get_image_wrist()
    sample_depth_top = data.get_depth_top()
    sample_depth_wrist = data.get_depth_wrist()

    rgb_top_shape = tuple(cfg.FEATURES["observation.images.cam_top"]["shape"])
    rgb_wrist_shape = tuple(cfg.FEATURES["observation.images.cam_wrist"]["shape"])
    top_depth_hw = (
        rgb_top_shape[0],
        rgb_top_shape[1],
    )
    wrist_depth_hw = (
        rgb_wrist_shape[0],
        rgb_wrist_shape[1],
    )

    if args.depth_min is None or args.depth_max is None:
        depth_min, depth_max = estimate_depth_range(
            data,
            selected_episode_ids,
            low_percentile=args.depth_low_percentile,
            high_percentile=args.depth_high_percentile,
            samples_per_episode=args.depth_estimation_samples_per_episode,
        )
    else:
        depth_min = float(args.depth_min)
        depth_max = float(args.depth_max)

    print(
        f"[Info] Creating depth-video shard at {save_root}\n"
        f"       episode slice index range: [{start_num}, {end_num})\n"
        f"       selected episodes: {len(selected_episode_ids)}\n"
        f"       depth quantization range: [{depth_min:.6f}, {depth_max:.6f}]\n"
        f"       encoded depth sizes: top={top_depth_hw}, wrist={wrist_depth_hw}"
    )

    dataset = LeRobotDataset.create(
        repo_id=cfg.HF_REPO_ID,
        fps=cfg.FPS,
        features=build_features(
            depth_top_shape=top_depth_hw,
            depth_wrist_shape=wrist_depth_hw,
        ),
        root=save_root,
        robot_type="omy_f3m",
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    save_depth_encoding_metadata(
        save_root,
        depth_min=depth_min,
        depth_max=depth_max,
        top_original_shape=tuple(sample_depth_top.shape),
        wrist_original_shape=tuple(sample_depth_wrist.shape),
        top_encoded_shape=top_depth_hw,
        wrist_encoded_shape=wrist_depth_hw,
    )

    for ep_num in selected_episode_ids:
        data.setup(episode_num=ep_num)
        for _ in tqdm(range(data.total_data_num), desc=f"Episode {ep_num} Recording"):
            img_top = data.get_image_top()
            img_wrist = data.get_image_wrist()
            depth_top = encode_depth_as_video_frame(
                data.get_depth_top(),
                depth_min=depth_min,
                depth_max=depth_max,
                target_hw=top_depth_hw,
            )
            depth_wrist = encode_depth_as_video_frame(
                data.get_depth_wrist(),
                depth_min=depth_min,
                depth_max=depth_max,
                target_hw=wrist_depth_hw,
            )

            follower_numpy, time_stamp = data.get_follower_action(dtype=np.float32)
            leader_numpy = data.get_leader_action(shift=args.shift, dtype=np.float32)
            data.step_idx += 1

            frame_data = {
                "observation.images.cam_top": img_top,
                "observation.images.cam_wrist": img_wrist,
                "observation.depth.cam_top": depth_top,
                "observation.depth.cam_wrist": depth_wrist,
                "observation.state": follower_numpy,
                "action": leader_numpy,
                "task": cfg.TASK_DESCRIPTION,
                "timestamp": time_stamp,
            }
            dataset.add_frame(frame_data)

        print(f"[Info] Saving episode {ep_num} into shard...")
        dataset.save_episode()

    dataset.finalize()
    print(f"[Info] Depth-video shard complete: {save_root}")


if __name__ == "__main__":
    main()
