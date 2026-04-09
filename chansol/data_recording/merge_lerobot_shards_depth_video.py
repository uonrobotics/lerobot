#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple depth-video LeRobot shard datasets into one final dataset. "
            "This validates that all shards share the same depth video encoding metadata "
            "before merging data/meta/videos."
        )
    )
    parser.add_argument(
        "--shard-roots",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to shard dataset roots created with convert_to_lerobot_shard_depth_video.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Path to the merged final dataset root.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default="user1/merged_depth_shards",
        help="Repo id to store inside merged metadata.",
    )
    parser.add_argument(
        "--data-files-size-in-mb",
        type=float,
        default=None,
        help="Optional max parquet file size in MB for merged output.",
    )
    parser.add_argument(
        "--video-files-size-in-mb",
        type=float,
        default=None,
        help="Optional max video file size in MB for merged output.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional max file count per chunk in merged output.",
    )
    return parser.parse_args()


def load_depth_metadata(shard_root: Path) -> dict:
    meta_path = shard_root / "depth_video_encoding.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Depth encoding metadata not found: {meta_path}\n"
            "This merge wrapper expects shards created by convert_to_lerobot_shard_depth_video.py."
        )
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_depth_metadata(shard_roots: list[Path]) -> dict:
    reference_root = shard_roots[0]
    reference = load_depth_metadata(reference_root)

    for root in shard_roots[1:]:
        current = load_depth_metadata(root)
        if current != reference:
            raise ValueError(
                "Depth encoding metadata differs across shards.\n"
                f"Reference shard: {reference_root}\n"
                f"Mismatched shard: {root}\n"
                "For depth-video shards, all shards must be created with the same "
                "depth_min/depth_max and encoded shapes. Re-create the shards with fixed "
                "--depth-min and --depth-max values shared across all PCs, then merge again."
            )

    return reference


def write_depth_metadata(output_root: Path, metadata: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    meta_path = output_root / "depth_video_encoding.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()

    shard_roots = [root.resolve() for root in args.shard_roots]
    for root in shard_roots:
        if not root.exists():
            raise FileNotFoundError(f"Shard root does not exist: {root}")

    depth_metadata = validate_depth_metadata(shard_roots)
    repo_ids = [f"user1/depth_shard_{idx:03d}" for idx, _ in enumerate(shard_roots)]

    print("[Info] Merging depth-video LeRobot shards:")
    for idx, root in enumerate(shard_roots):
        print(f"  - shard {idx}: {root}")
    print(f"[Info] Output: {args.output_root}")

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=args.output_repo_id,
        roots=shard_roots,
        aggr_root=args.output_root,
        data_files_size_in_mb=args.data_files_size_in_mb,
        video_files_size_in_mb=args.video_files_size_in_mb,
        chunk_size=args.chunk_size,
    )

    write_depth_metadata(args.output_root, depth_metadata)
    print(f"[Info] Merge complete: {args.output_root}")


if __name__ == "__main__":
    main()
