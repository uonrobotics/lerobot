#!/usr/bin/env python3

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# User Settings
# ============================================================

# Directory that contains files like:
# step_0169500_episode_0001_frame_0055.png
INPUT_DIR = Path("/nas/AI_Checkpoints/VLA/act/Isaacsim_OMY_apple_picking_auto/shift4_filtered_gradcam/gradcam_snapshots")
OUTPUT_DIR = INPUT_DIR / "gifs_by_episode_frame"
GIF_FPS = 10


FILE_PATTERN = re.compile(
    r"^step_(?P<step>\d+)_episode_(?P<episode>\d+)_frame_(?P<frame>\d+)\.png$"
)


def collect_groups(input_dir: Path) -> dict[tuple[int, int], list[tuple[int, Path]]]:
    groups: dict[tuple[int, int], list[tuple[int, Path]]] = defaultdict(list)

    for image_path in input_dir.glob("*.png"):
        match = FILE_PATTERN.match(image_path.name)
        if match is None:
            continue

        step = int(match.group("step"))
        episode = int(match.group("episode"))
        frame = int(match.group("frame"))
        groups[(episode, frame)].append((step, image_path))

    for key in groups:
        groups[key].sort(key=lambda item: item[0])

    return groups


def add_title_banner(image: Image.Image, episode: int, frame: int, step: int) -> Image.Image:
    banner_height = 110
    labeled_image = Image.new("RGB", (image.width, image.height + banner_height), color=(255, 255, 255))
    labeled_image.paste(image, (0, banner_height))

    draw = ImageDraw.Draw(labeled_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    title = f"Episode {episode:04d} | Frame {frame:04d} | Step {step:07d}"
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = max(0, (labeled_image.width - text_width) // 2)
    text_y = max(0, (banner_height - text_height) // 2)
    draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)
    return labeled_image


def build_gif(
    step_and_paths: list[tuple[int, Path]],
    output_path: Path,
    duration_ms: int,
    episode: int,
    frame: int,
) -> None:
    frames = []
    for step, image_path in step_and_paths:
        image = Image.open(image_path).convert("RGB")
        frames.append(add_title_banner(image=image, episode=episode, frame=frame, step=step))
        image.close()

    if not frames:
        return

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

    for frame in frames:
        frame.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = collect_groups(INPUT_DIR)
    if not groups:
        raise FileNotFoundError(f"No matching png files found in: {INPUT_DIR}")

    duration_ms = int(1000 / max(1, GIF_FPS))
    created_count = 0

    for (episode, frame), step_and_paths in sorted(groups.items()):
        output_path = OUTPUT_DIR / f"episode_{episode:04d}_frame_{frame:04d}.gif"
        build_gif(
            step_and_paths=step_and_paths,
            output_path=output_path,
            duration_ms=duration_ms,
            episode=episode,
            frame=frame,
        )
        created_count += 1
        print(f"Saved: {output_path} ({len(step_and_paths)} frames)")

    print(f"Done. Created {created_count} GIF files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
