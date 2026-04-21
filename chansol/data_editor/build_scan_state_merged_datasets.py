from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.utils import load_info, load_stats, write_info, write_stats, write_tasks


SOURCE_ROOT = Path("/nas/Dataset/VLA/0419_2obj")
NAVY_DATASET_ROOT = SOURCE_ROOT / "20260419_navy_scan"
ORANGE_DATASET_ROOT = SOURCE_ROOT / "20260419_orange_scan"

OVERWRITE_OUTPUT = True

NAVY_TASK_DESCRIPTION = "scan barcode and place navy object in basket"
ORANGE_TASK_DESCRIPTION = "scan barcode and place orange object in basket"
MERGED_TASK_DESCRIPTION = "scan barcode and place object in basket"

SCAN_STATE_NAMES = ["scan_code_orange", "scan_code_navy"]
REMOVED_FEATURE_KEYS = ["observation.scan_info", "observation.environment_state"]


@dataclass(frozen=True)
class OutputConfig:
    name: str
    output_root: Path
    output_repo_id: str
    split_task_description: bool


OUTPUT_CONFIGS = [
    OutputConfig(
        name="split_description",
        output_root=SOURCE_ROOT / "20260419_2obj_state_merged_split_desc",
        output_repo_id="user1/0419_2obj_state_merged_split_desc",
        split_task_description=True,
    ),
    OutputConfig(
        name="merged_description",
        output_root=SOURCE_ROOT / "20260419_2obj_state_merged_single_desc",
        output_repo_id="user1/0419_2obj_state_merged_single_desc",
        split_task_description=False,
    ),
]


SCAN_CODE_TO_STATE = {
    "": np.array([0.0, 0.0], dtype=np.float32),
    "orange": np.array([1.0, 0.0], dtype=np.float32),
    "navy": np.array([0.0, 1.0], dtype=np.float32),
}


def normalize_scan_info(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def scan_info_to_state_suffix(scan_info: object) -> np.ndarray:
    normalized = normalize_scan_info(scan_info)
    if normalized and set(normalized) == {"1"}:
        normalized = "orange"
    elif normalized and set(normalized) == {"2"}:
        normalized = "navy"

    if normalized not in SCAN_CODE_TO_STATE:
        raise ValueError(
            f"Unsupported observation.scan_info value: {scan_info!r}. "
            f"Expected one of {sorted(SCAN_CODE_TO_STATE)}."
        )
    return SCAN_CODE_TO_STATE[normalized]


def augment_state(state: object, scan_info: object) -> np.ndarray:
    state_array = np.asarray(state, dtype=np.float32)
    return np.concatenate([state_array, scan_info_to_state_suffix(scan_info)], axis=0).astype(np.float32)


def get_source_episode_counts() -> tuple[int, int]:
    if not NAVY_DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing source dataset: {NAVY_DATASET_ROOT}")
    if not ORANGE_DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing source dataset: {ORANGE_DATASET_ROOT}")

    navy_info = load_info(NAVY_DATASET_ROOT)
    orange_info = load_info(ORANGE_DATASET_ROOT)
    return int(navy_info["total_episodes"]), int(orange_info["total_episodes"])


def build_episode_task_map(
    navy_total_episodes: int,
    orange_total_episodes: int,
    *,
    split_task_description: bool,
) -> dict[int, str]:
    episode_task_map: dict[int, str] = {}

    for episode_idx in range(navy_total_episodes):
        episode_task_map[episode_idx] = (
            NAVY_TASK_DESCRIPTION if split_task_description else MERGED_TASK_DESCRIPTION
        )

    for episode_idx in range(navy_total_episodes, navy_total_episodes + orange_total_episodes):
        episode_task_map[episode_idx] = (
            ORANGE_TASK_DESCRIPTION if split_task_description else MERGED_TASK_DESCRIPTION
        )

    return episode_task_map


def build_task_table(*, split_task_description: bool) -> pd.DataFrame:
    if split_task_description:
        task_names = [NAVY_TASK_DESCRIPTION, ORANGE_TASK_DESCRIPTION]
    else:
        task_names = [MERGED_TASK_DESCRIPTION]
    return pd.DataFrame({"task_index": range(len(task_names))}, index=task_names)


def ensure_clean_output_root(output_root: Path) -> None:
    if output_root.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}\n"
                "Set OVERWRITE_OUTPUT=True or change output path."
            )
        shutil.rmtree(output_root)


def aggregate_source_datasets(output_root: Path, output_repo_id: str) -> None:
    aggregate_datasets(
        repo_ids=[NAVY_DATASET_ROOT.name, ORANGE_DATASET_ROOT.name],
        aggr_repo_id=output_repo_id,
        roots=[NAVY_DATASET_ROOT, ORANGE_DATASET_ROOT],
        aggr_root=output_root,
    )


def rewrite_data_and_collect_episode_stats(
    output_root: Path,
    episode_task_map: dict[int, str],
    task_to_index: dict[str, int],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    data_paths = sorted((output_root / "data").glob("*/*.parquet"))
    episode_states: dict[int, list[np.ndarray]] = {}
    episode_lengths: dict[int, int] = {}

    for data_path in tqdm(data_paths, desc=f"Rewrite data parquet ({output_root.name})"):
        df = pd.read_parquet(data_path)

        augmented_states: list[np.ndarray] = []
        new_task_indices: list[int] = []

        for episode_idx, state, scan_info in zip(
            df["episode_index"].to_numpy(),
            df["observation.state"].tolist(),
            df["observation.scan_info"].tolist(),
            strict=True,
        ):
            episode_idx = int(episode_idx)
            task_name = episode_task_map[episode_idx]
            augmented_state = augment_state(state, scan_info)

            augmented_states.append(augmented_state)
            new_task_indices.append(task_to_index[task_name])

            episode_states.setdefault(episode_idx, []).append(augmented_state)
            episode_lengths[episode_idx] = episode_lengths.get(episode_idx, 0) + 1

        df["observation.state"] = augmented_states
        df["task_index"] = np.asarray(new_task_indices, dtype=np.int64)
        df = df.drop(columns=REMOVED_FEATURE_KEYS, errors="ignore")
        df.to_parquet(data_path, index=False)

    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for episode_idx, states in episode_states.items():
        task_idx = task_to_index[episode_task_map[episode_idx]]
        state_array = np.stack(states, axis=0).astype(np.float32)
        task_array = np.full((episode_lengths[episode_idx],), task_idx, dtype=np.int64)

        episode_stats[episode_idx] = {
            "observation.state": get_feature_stats(state_array, axis=0, keepdims=False),
            "task_index": get_feature_stats(task_array, axis=0, keepdims=True),
        }

    return episode_stats


def rewrite_episode_metadata(
    output_root: Path,
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]],
    episode_task_map: dict[int, str],
) -> None:
    episode_paths = sorted((output_root / "meta/episodes").glob("*/*.parquet"))

    for episode_path in tqdm(episode_paths, desc=f"Rewrite episodes meta ({output_root.name})"):
        df = pd.read_parquet(episode_path)

        env_state_stat_cols = [
            col for col in df.columns if col.startswith("stats/observation.environment_state/")
        ]
        if env_state_stat_cols:
            df = df.drop(columns=env_state_stat_cols)

        for row_idx, episode_idx in zip(df.index.tolist(), df["episode_index"].tolist(), strict=True):
            episode_idx = int(episode_idx)
            task_name = episode_task_map[episode_idx]
            df.at[row_idx, "tasks"] = [task_name]

            for stat_key, stat_value in episode_stats[episode_idx]["observation.state"].items():
                df.at[row_idx, f"stats/observation.state/{stat_key}"] = stat_value.tolist()

            for stat_key, stat_value in episode_stats[episode_idx]["task_index"].items():
                df.at[row_idx, f"stats/task_index/{stat_key}"] = stat_value.tolist()

        df.to_parquet(episode_path, index=False)


def rewrite_top_level_metadata(
    output_root: Path,
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]],
    task_table: pd.DataFrame,
) -> None:
    info = load_info(output_root)
    stats = load_stats(output_root)

    state_feature = info["features"]["observation.state"]
    state_feature["shape"] = (len(state_feature["names"]) + len(SCAN_STATE_NAMES),)
    state_feature["names"] = list(state_feature["names"]) + SCAN_STATE_NAMES

    for feature_key in REMOVED_FEATURE_KEYS:
        info["features"].pop(feature_key, None)

    info["total_tasks"] = len(task_table)

    updated_stats = aggregate_stats(list(episode_stats.values()))
    stats["observation.state"] = updated_stats["observation.state"]
    stats["task_index"] = updated_stats["task_index"]
    for feature_key in REMOVED_FEATURE_KEYS:
        stats.pop(feature_key, None)

    write_info(info, output_root)
    write_tasks(task_table, output_root)
    write_stats(stats, output_root)


def build_output_dataset(
    config: OutputConfig,
    navy_total_episodes: int,
    orange_total_episodes: int,
) -> None:
    logging.info("Build dataset variant: %s", config.name)

    ensure_clean_output_root(config.output_root)
    aggregate_source_datasets(config.output_root, config.output_repo_id)

    task_table = build_task_table(split_task_description=config.split_task_description)
    task_to_index = {task_name: int(row.task_index) for task_name, row in task_table.iterrows()}
    episode_task_map = build_episode_task_map(
        navy_total_episodes,
        orange_total_episodes,
        split_task_description=config.split_task_description,
    )

    episode_stats = rewrite_data_and_collect_episode_stats(
        config.output_root,
        episode_task_map,
        task_to_index,
    )
    rewrite_episode_metadata(config.output_root, episode_stats, episode_task_map)
    rewrite_top_level_metadata(config.output_root, episode_stats, task_table)

    logging.info("Finished dataset variant at %s", config.output_root)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    navy_total_episodes, orange_total_episodes = get_source_episode_counts()

    for config in OUTPUT_CONFIGS:
        build_output_dataset(config, navy_total_episodes, orange_total_episodes)


if __name__ == "__main__":
    main()
