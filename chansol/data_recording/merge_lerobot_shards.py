#!/usr/bin/env python3

import argparse
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple independently-created LeRobot shard datasets into one final dataset "
            "by reindexing data/meta/videos and rewriting aggregate stats."
        )
    )
    parser.add_argument(
        "--shard-roots",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to shard dataset roots created on different PCs.",
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
        default="user1/merged_shards",
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


def main() -> None:
    args = parse_args()

    shard_roots = [root.resolve() for root in args.shard_roots]
    for root in shard_roots:
        if not root.exists():
            raise FileNotFoundError(f"Shard root does not exist: {root}")

    # if args.output_root.exists() and any(args.output_root.iterdir()):
    #     raise FileExistsError(
    #         f"Output root must be empty or not exist yet: {args.output_root}"
    #     )

    repo_ids = [f"user1/shard_{idx:03d}" for idx, _ in enumerate(shard_roots)]

    print("[Info] Merging LeRobot shards:")
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

    print(f"[Info] Merge complete: {args.output_root}")


if __name__ == "__main__":
    main()
