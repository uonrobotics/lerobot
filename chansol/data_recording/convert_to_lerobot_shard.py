#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# import configs.Robotis_OMY_isaac as cfg
import configs.Robotis_OMY_isaac_depth as cfg
from get_data import DataAggregator
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert only a slice of raw episodes into one independent LeRobot shard dataset. "
            "Run this on multiple PCs with different episode ranges, then merge the shards later."
        )
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the raw recorded data.")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Output directory for this shard dataset. Must be unique per PC/shard.",
    )
    parser.add_argument("--shift", type=int, default=1, help="Time step shift for leader action.")
    parser.add_argument("--start_num", type=int, default=None, help="Inclusive start index in episode_exist_list.")
    parser.add_argument("--end_num", type=int, default=None, help="Exclusive end index in episode_exist_list.")
    parser.add_argument(
        "--episodes_per_shard",
        type=int,
        default=None,
        help="Optional shard size. Used together with --shard_index.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Optional shard number. Used together with --episodes_per_shard.",
    )
    parser.add_argument(
        "--image_writer_processes",
        type=int,
        default=4,
        help="LeRobot image writer process count.",
    )
    parser.add_argument(
        "--image_writer_threads",
        type=int,
        default=8,
        help="LeRobot image writer thread count.",
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

    print(
        f"[Info] Creating shard dataset at {save_root}\n"
        f"       episode slice index range: [{start_num}, {end_num})\n"
        f"       selected episodes: {len(selected_episode_ids)}"
    )

    dataset = LeRobotDataset.create(
        repo_id=cfg.HF_REPO_ID,
        fps=cfg.FPS,
        features=cfg.FEATURES,
        root=save_root,
        robot_type="omy_f3m",
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    for ep_num in selected_episode_ids:
        data.setup(episode_num=ep_num)
        for _ in tqdm(range(data.total_data_num), desc=f"Episode {ep_num} Recording"):
            img_top = data.get_image_top()
            img_wrist = data.get_image_wrist()
            depth_top = data.get_depth_top()
            depth_wrist = data.get_depth_wrist()

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
    print(f"[Info] Shard complete: {save_root}")


if __name__ == "__main__":
    main()
