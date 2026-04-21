from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.utils import load_info, load_stats, write_info, write_stats, write_tasks


SOURCE_ROOT = Path("/nas/Dataset/VLA/0419_2obj")
RAW_DATASET_ROOT = SOURCE_ROOT / "20260420_2obj_scan"
OUTPUT_ROOT = SOURCE_ROOT / "20260420_2obj_scan_converted"
OUTPUT_REPO_ID = "user1/20260420_2obj_scan_converted"

OVERWRITE_OUTPUT = True

MERGED_TASK_DESCRIPTION = "scan barcode and place object in basket"
SCAN_STATE_NAMES = ["scan_code_orange", "scan_code_navy"]
REMOVED_FEATURE_KEYS = ["observation.scan_info", "observation.environment_state"]

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


def ensure_clean_output_root() -> None:
    if OUTPUT_ROOT.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(f"Output dataset already exists: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)


def build_single_task_table() -> pd.DataFrame:
    return pd.DataFrame({"task_index": [0]}, index=[MERGED_TASK_DESCRIPTION])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not RAW_DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing source dataset: {RAW_DATASET_ROOT}")

    ensure_clean_output_root()

    aggregate_datasets(
        repo_ids=[RAW_DATASET_ROOT.name],
        aggr_repo_id=OUTPUT_REPO_ID,
        roots=[RAW_DATASET_ROOT],
        aggr_root=OUTPUT_ROOT,
    )

    task_table = build_single_task_table()
    task_to_index = {MERGED_TASK_DESCRIPTION: 0}

    data_paths = sorted((OUTPUT_ROOT / "data").glob("*/*.parquet"))
    episode_states: dict[int, list[np.ndarray]] = {}
    episode_lengths: dict[int, int] = {}

    for data_path in tqdm(data_paths, desc="Rewrite converted data"):
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
            augmented_state = augment_state(state, scan_info)

            augmented_states.append(augmented_state)
            new_task_indices.append(task_to_index[MERGED_TASK_DESCRIPTION])
            episode_states.setdefault(episode_idx, []).append(augmented_state)
            episode_lengths[episode_idx] = episode_lengths.get(episode_idx, 0) + 1

        df["observation.state"] = augmented_states
        df["task_index"] = np.asarray(new_task_indices, dtype=np.int64)
        df = df.drop(columns=REMOVED_FEATURE_KEYS, errors="ignore")
        df.to_parquet(data_path, index=False)

    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for episode_idx, states in episode_states.items():
        state_array = np.stack(states, axis=0).astype(np.float32)
        task_array = np.zeros((episode_lengths[episode_idx],), dtype=np.int64)
        episode_stats[episode_idx] = {
            "observation.state": get_feature_stats(state_array, axis=0, keepdims=False),
            "task_index": get_feature_stats(task_array, axis=0, keepdims=True),
        }

    episode_paths = sorted((OUTPUT_ROOT / "meta/episodes").glob("*/*.parquet"))
    for episode_path in tqdm(episode_paths, desc="Rewrite converted episodes meta"):
        df = pd.read_parquet(episode_path)
        env_state_stat_cols = [
            col for col in df.columns if col.startswith("stats/observation.environment_state/")
        ]
        if env_state_stat_cols:
            df = df.drop(columns=env_state_stat_cols)

        for row_idx, episode_idx in zip(df.index.tolist(), df["episode_index"].tolist(), strict=True):
            episode_idx = int(episode_idx)
            df.at[row_idx, "tasks"] = [MERGED_TASK_DESCRIPTION]

            for stat_key, stat_value in episode_stats[episode_idx]["observation.state"].items():
                df.at[row_idx, f"stats/observation.state/{stat_key}"] = stat_value.tolist()
            for stat_key, stat_value in episode_stats[episode_idx]["task_index"].items():
                df.at[row_idx, f"stats/task_index/{stat_key}"] = stat_value.tolist()

        df.to_parquet(episode_path, index=False)

    info = load_info(OUTPUT_ROOT)
    stats = load_stats(OUTPUT_ROOT)
    state_feature = info["features"]["observation.state"]
    state_feature["shape"] = (len(state_feature["names"]) + len(SCAN_STATE_NAMES),)
    state_feature["names"] = list(state_feature["names"]) + SCAN_STATE_NAMES
    for feature_key in REMOVED_FEATURE_KEYS:
        info["features"].pop(feature_key, None)
    info["total_tasks"] = 1

    updated_stats = aggregate_stats(list(episode_stats.values()))
    stats["observation.state"] = updated_stats["observation.state"]
    stats["task_index"] = updated_stats["task_index"]
    for feature_key in REMOVED_FEATURE_KEYS:
        stats.pop(feature_key, None)

    write_info(info, OUTPUT_ROOT)
    write_tasks(task_table, OUTPUT_ROOT)
    write_stats(stats, OUTPUT_ROOT)

    logging.info("Finished converted dataset at %s", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
