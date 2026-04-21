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

BASE_SINGLE_DESC_ROOT = SOURCE_ROOT / "20260419_2obj_state_merged_single_desc"
RAW_20260420_ROOT = SOURCE_ROOT / "20260420_2obj_scan"

TEMP_CONVERTED_ROOT = SOURCE_ROOT / "_tmp_20260420_2obj_scan_single_desc_converted"
TEMP_CONVERTED_REPO_ID = "user1/_tmp_20260420_2obj_scan_single_desc_converted"

FINAL_OUTPUT_ROOT = SOURCE_ROOT / "20260419_20260420_2obj_state_merged_single_desc"
FINAL_OUTPUT_REPO_ID = "user1/20260419_20260420_2obj_state_merged_single_desc"

OVERWRITE_OUTPUT = True
REMOVE_TEMP_AFTER_MERGE = True

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


def ensure_removed(path: Path) -> None:
    if path.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(f"Path already exists: {path}")
        shutil.rmtree(path)


def build_single_task_table() -> pd.DataFrame:
    return pd.DataFrame({"task_index": [0]}, index=[MERGED_TASK_DESCRIPTION])


def convert_raw_dataset_to_single_desc() -> None:
    if not RAW_20260420_ROOT.exists():
        raise FileNotFoundError(f"Missing source dataset: {RAW_20260420_ROOT}")

    ensure_removed(TEMP_CONVERTED_ROOT)

    aggregate_datasets(
        repo_ids=[RAW_20260420_ROOT.name],
        aggr_repo_id=TEMP_CONVERTED_REPO_ID,
        roots=[RAW_20260420_ROOT],
        aggr_root=TEMP_CONVERTED_ROOT,
    )

    task_table = build_single_task_table()
    task_to_index = {MERGED_TASK_DESCRIPTION: 0}

    data_paths = sorted((TEMP_CONVERTED_ROOT / "data").glob("*/*.parquet"))
    episode_states: dict[int, list[np.ndarray]] = {}
    episode_lengths: dict[int, int] = {}

    for data_path in tqdm(data_paths, desc="Rewrite raw 20260420 data"):
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

    episode_paths = sorted((TEMP_CONVERTED_ROOT / "meta/episodes").glob("*/*.parquet"))
    for episode_path in tqdm(episode_paths, desc="Rewrite raw 20260420 episodes meta"):
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

    info = load_info(TEMP_CONVERTED_ROOT)
    stats = load_stats(TEMP_CONVERTED_ROOT)
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

    write_info(info, TEMP_CONVERTED_ROOT)
    write_tasks(task_table, TEMP_CONVERTED_ROOT)
    write_stats(stats, TEMP_CONVERTED_ROOT)


def validate_matching_features() -> None:
    base_info = load_info(BASE_SINGLE_DESC_ROOT)
    converted_info = load_info(TEMP_CONVERTED_ROOT)
    if base_info["features"] != converted_info["features"]:
        raise ValueError("Converted 20260420 dataset features do not match base single-desc dataset features.")


def build_final_merged_dataset() -> None:
    if not BASE_SINGLE_DESC_ROOT.exists():
        raise FileNotFoundError(f"Missing base dataset: {BASE_SINGLE_DESC_ROOT}")

    ensure_removed(FINAL_OUTPUT_ROOT)
    aggregate_datasets(
        repo_ids=[BASE_SINGLE_DESC_ROOT.name, TEMP_CONVERTED_ROOT.name],
        aggr_repo_id=FINAL_OUTPUT_REPO_ID,
        roots=[BASE_SINGLE_DESC_ROOT, TEMP_CONVERTED_ROOT],
        aggr_root=FINAL_OUTPUT_ROOT,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    convert_raw_dataset_to_single_desc()
    validate_matching_features()
    build_final_merged_dataset()

    if REMOVE_TEMP_AFTER_MERGE and TEMP_CONVERTED_ROOT.exists():
        shutil.rmtree(TEMP_CONVERTED_ROOT)

    logging.info("Finished merged dataset at %s", FINAL_OUTPUT_ROOT)


if __name__ == "__main__":
    main()
