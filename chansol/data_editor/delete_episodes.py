#!/usr/bin/env python3

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    DEFAULT_TASKS_PATH,
    INFO_PATH,
    STATS_PATH,
    DEFAULT_EPISODES_PATH,
    load_info,
    load_tasks,
    unflatten_dict,
    write_info,
    write_stats,
    write_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete specific episodes from an existing LeRobot dataset by rewriting only "
            "data/meta files. Existing video files are reused; only metadata references are updated."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the LeRobot dataset root.")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        required=True,
        help="Episode indices to delete.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the safety prompt and write changes in place.",
    )
    return parser.parse_args()


def parquet_files(root: Path) -> list[Path]:
    return sorted(root.glob("chunk-*/file-*.parquet"))


def video_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/chunk-*/file-*.mp4"))


def backup_file(path: Path) -> Path:
    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    shutil.copy2(path, backup)
    return backup


def to_python_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [to_python_value(v) for v in value]
    if isinstance(value, tuple):
        return [to_python_value(v) for v in value]
    return value


def dataframe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    df_to_write = df.copy()
    for column in df_to_write.columns:
        if df_to_write[column].dtype == object:
            df_to_write[column] = df_to_write[column].map(to_python_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df_to_write.to_parquet(tmp_path, index=False)
        shutil.move(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_all_episode_rows(dataset_root: Path) -> pd.DataFrame:
    files = parquet_files(dataset_root / "meta" / "episodes")
    if not files:
        raise FileNotFoundError(f"No episode parquet files found in {dataset_root / 'meta' / 'episodes'}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def unwrap_nested_singletons(value: Any) -> Any:
    while True:
        if isinstance(value, np.ndarray):
            if value.dtype != object:
                return value
            if value.size == 1:
                value = value.reshape(-1)[0]
                continue
            return [unwrap_nested_singletons(v) for v in value.tolist()]
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                value = value[0]
                continue
            return [unwrap_nested_singletons(v) for v in value]
        return value


def normalize_stat_array(value: Any, feature_name: str, stat_key: str) -> np.ndarray:
    normalized_value = unwrap_nested_singletons(value)
    array = np.array(normalized_value)

    if "image" in feature_name and stat_key != "count":
        array = np.squeeze(array)
        if array.shape == (3,):
            array = array[:, None, None]
        elif array.shape == (3, 1):
            array = array[:, :, None]
        elif array.shape != (3, 1, 1):
            raise ValueError(
                f"Unsupported image stat shape for feature '{feature_name}', stat '{stat_key}': {array.shape}"
            )

    if array.dtype == object:
        target_dtype = np.int64 if stat_key == "count" else np.float64
        array = np.array(array.tolist() if isinstance(array, np.ndarray) else array, dtype=target_dtype)
    elif stat_key == "count":
        array = array.astype(np.int64, copy=False)
    else:
        array = array.astype(np.float64, copy=False)

    return array


def build_episode_stats(row: pd.Series) -> dict[str, dict[str, np.ndarray]]:
    flat_stats = {}
    for key, value in row.items():
        if key.startswith("stats/"):
            stat_path = key[len("stats/") :]
            feature_name, stat_key = stat_path.rsplit("/", 1)
            flat_stats[stat_path] = normalize_stat_array(value, feature_name, stat_key)
    return unflatten_dict(flat_stats)


def rewrite_data_files(
    dataset_root: Path,
    keep_episode_map: dict[int, int],
    old_task_to_new_task: dict[int, int],
) -> tuple[dict[int, int], int]:
    data_root = dataset_root / "data"
    data_paths = parquet_files(data_root)
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files found in {data_root}")

    next_global_index = 0
    episode_lengths: dict[int, int] = {}

    for path in data_paths:
        df = pd.read_parquet(path)
        if "episode_index" not in df.columns:
            raise ValueError(f"'episode_index' column missing in {path}")

        keep_mask = df["episode_index"].isin(keep_episode_map)
        filtered = df.loc[keep_mask].copy()

        if filtered.empty:
            path.unlink()
            continue

        filtered["episode_index"] = filtered["episode_index"].map(keep_episode_map).astype(np.int64)

        if "task_index" in filtered.columns:
            filtered["task_index"] = filtered["task_index"].map(old_task_to_new_task).astype(np.int64)

        if "index" in filtered.columns:
            filtered["index"] = np.arange(next_global_index, next_global_index + len(filtered), dtype=np.int64)
        next_global_index += len(filtered)

        counts = filtered.groupby("episode_index").size().to_dict()
        for episode_index, count in counts.items():
            episode_lengths[int(episode_index)] = episode_lengths.get(int(episode_index), 0) + int(count)

        dataframe_to_parquet(filtered, path)

    return episode_lengths, next_global_index


def rewrite_episode_metadata(
    dataset_root: Path,
    episodes_df: pd.DataFrame,
    keep_episode_map: dict[int, int],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], list[str]]:
    kept = episodes_df[episodes_df["episode_index"].isin(keep_episode_map)].copy()
    if kept.empty:
        raise ValueError("Deleting all episodes is not supported.")

    kept["episode_index"] = kept["episode_index"].map(keep_episode_map).astype(np.int64)
    kept = kept.sort_values("episode_index").reset_index(drop=True)

    cumulative = 0
    dataset_from = []
    dataset_to = []
    stats_list = []
    task_names: list[str] = []

    for _, row in kept.iterrows():
        length = int(row["length"])
        dataset_from.append(cumulative)
        cumulative += length
        dataset_to.append(cumulative)
        stats_list.append(build_episode_stats(row))
        for task in row["tasks"]:
            if task not in task_names:
                task_names.append(task)

    kept["dataset_from_index"] = np.array(dataset_from, dtype=np.int64)
    kept["dataset_to_index"] = np.array(dataset_to, dtype=np.int64)

    aggregated_stats = aggregate_stats(stats_list)

    return kept, aggregated_stats, task_names


def rewrite_tasks(dataset_root: Path, kept_task_names: list[str]) -> dict[int, int]:
    tasks_df = load_tasks(dataset_root)
    kept_tasks_df = tasks_df.loc[[task for task in tasks_df.index if task in kept_task_names]].copy()
    kept_tasks_df["task_index"] = np.arange(len(kept_tasks_df), dtype=np.int64)

    old_task_to_new_task = {
        int(old_row.task_index): int(new_index)
        for new_index, (_, old_row) in enumerate(tasks_df.loc[kept_tasks_df.index].iterrows())
    }

    write_tasks(kept_tasks_df, dataset_root)
    return old_task_to_new_task


def rewrite_info(dataset_root: Path, total_episodes: int, total_frames: int, total_tasks: int) -> None:
    info = load_info(dataset_root)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = total_tasks
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_info(info, dataset_root)


def prune_empty_dirs(dataset_root: Path) -> None:
    for subdir in [dataset_root / "data", dataset_root / "meta" / "episodes"]:
        for chunk_dir in sorted(subdir.glob("chunk-*")):
            if chunk_dir.is_dir() and not any(chunk_dir.iterdir()):
                chunk_dir.rmdir()


def confirm_write(force: bool, dataset_root: Path, delete_set: list[int]) -> None:
    if force:
        return
    print(f"Target dataset: {dataset_root}")
    print(f"Episodes to delete: {delete_set}")
    answer = input("Write changes in place? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted.")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    delete_set = sorted(set(args.episodes))

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    info = load_info(dataset_root)
    total_episodes = int(info["total_episodes"])
    invalid = [ep for ep in delete_set if ep < 0 or ep >= total_episodes]
    if invalid:
        raise IndexError(f"Episode indices out of range: {invalid}. Available range: 0..{total_episodes - 1}")

    if len(delete_set) == total_episodes:
        raise ValueError("Deleting all episodes is not supported.")

    confirm_write(args.force, dataset_root, delete_set)

    backup_file(dataset_root / INFO_PATH)
    backup_file(dataset_root / STATS_PATH)
    backup_file(dataset_root / DEFAULT_TASKS_PATH)

    episodes_backup_root = dataset_root / "meta" / "episodes_backup_before_delete"
    if episodes_backup_root.exists():
        shutil.rmtree(episodes_backup_root)
    shutil.copytree(dataset_root / "meta" / "episodes", episodes_backup_root)

    keep_episode_map = {}
    next_ep = 0
    for old_ep in range(total_episodes):
        if old_ep in delete_set:
            continue
        keep_episode_map[old_ep] = next_ep
        next_ep += 1

    episodes_df = load_all_episode_rows(dataset_root)
    kept_episodes_df, aggregated_stats, kept_task_names = rewrite_episode_metadata(
        dataset_root,
        episodes_df,
        keep_episode_map,
    )

    old_task_to_new_task = rewrite_tasks(dataset_root, kept_task_names)
    episode_lengths, total_frames = rewrite_data_files(dataset_root, keep_episode_map, old_task_to_new_task)

    if sum(episode_lengths.values()) != total_frames:
        raise RuntimeError("Data parquet rewrite produced inconsistent frame counts.")
    if len(episode_lengths) != len(keep_episode_map):
        raise RuntimeError("Some kept episodes disappeared while rewriting data parquet files.")

    rewrite_info(
        dataset_root,
        total_episodes=len(keep_episode_map),
        total_frames=total_frames,
        total_tasks=len(kept_task_names),
    )
    write_stats(aggregated_stats, dataset_root)

    episodes_dir = dataset_root / "meta" / "episodes"
    shutil.rmtree(episodes_dir)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_metadata_path = dataset_root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    dataframe_to_parquet(kept_episodes_df, episode_metadata_path)

    prune_empty_dirs(dataset_root)

    print(f"Deleted episodes: {delete_set}")
    print(f"Remaining episodes: {len(keep_episode_map)}")
    print(f"Remaining frames: {total_frames}")
    print("Videos were not re-encoded; existing mp4 files are reused through updated metadata.")


if __name__ == "__main__":
    main()
