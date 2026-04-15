#!/usr/bin/env python

import argparse
import sys
from collections import deque
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
from action_lipo import ActionLiPo
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins


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


class GrootInferenceServer:
    def __init__(
        self,
        pretrained_path: str,
        dataset_root: str,
        dataset_id: str,
        device: str,
        host: str,
        port: int,
        socket_utils_root: str,
        tokenizer_assets_repo: str | None = None,
        use_lipo: bool = True,
        lipo_solver: str = "osqp",
        lipo_blending_horizon: int = 10,
        lipo_time_delay: int = 0,
        lipo_dt: float = 0.0333,
        lipo_epsilon_blending: float = 0.02,
        lipo_epsilon_path: float = 0.003,
        recorde: bool = False,
        record_dir: str | None = None,
        record_max_side: int = 480,
        record_fps: int = 10,
    ):
        register_third_party_plugins()
        _patch_groot_resize_wrapper()
        _patch_groot_processor_compat_wrapper()

        self.device = torch.device(device)
        self.dataset_metadata = LeRobotDatasetMetadata(
            repo_id=dataset_id,
            root=dataset_root,
        )
        self.model = GrootPolicy.from_pretrained(pretrained_name_or_path=pretrained_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.config.device = str(self.device)
        if tokenizer_assets_repo:
            self.model.config.tokenizer_assets_repo = tokenizer_assets_repo

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            dataset_stats=self.dataset_metadata.stats,
        )
        self.chunk_size = int(self.model.config.n_action_steps)
        self.action_dim = int(self.model.config.output_features["action"].shape[0])
        self.action_queue: deque[np.ndarray] = deque()
        self.prev_action_chunk: np.ndarray | None = None
        self.recorde = recorde
        self.record_dir = Path(record_dir).expanduser().resolve() if record_dir else REPO_ROOT / "outputs" / "groot_records"
        self.record_max_side = max(64, int(record_max_side))
        self.record_fps = max(1, int(record_fps))
        self.record_episode_idx = 0
        self.record_frame_idx = 0
        self.record_episode_dir: Path | None = None
        self._image_size_logged = False
        self._last_logged_task: str | None = None
        if self.recorde:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._start_new_record_episode()
        self.lipo = None
        if use_lipo:
            self.lipo = ActionLiPo(
                solver=lipo_solver,
                chunk_size=self.chunk_size,
                blending_horizon=min(lipo_blending_horizon, self.chunk_size // 2),
                action_dim=self.action_dim,
                len_time_delay=lipo_time_delay,
                dt=lipo_dt,
                epsilon_blending=lipo_epsilon_blending,
                epsilon_path=lipo_epsilon_path,
            )

        socket_root = Path(socket_utils_root).expanduser().resolve()
        if str(socket_root) not in sys.path:
            sys.path.append(str(socket_root))

        from socket_utils.vla_socket import vla_server
        print(f"[SERVER IMPORT] {vla_server.__file__}")

        cfg = vla_server.VLAServerConfig(host=host, port=port, decode_jpeg=True)
        self.server = vla_server.VLARpcServer(cfg, infer_fn=self.infer_fn)

    def reset(self) -> None:
        self._finalize_record_episode()
        self.model.reset()
        self.action_queue.clear()
        self.prev_action_chunk = None
        if self.lipo is not None:
            self.lipo.reset_log()
        if self.recorde:
            self._start_new_record_episode()

    def _build_raw_observation(self, images: dict[str, Any], obss: dict[str, Any]) -> dict[str, Any]:
        joint_state = obss["joint_state"]
        return {
            "cam_top": images["full"],
            "cam_wrist": images["wrist"],
            "joint1": joint_state[0],
            "joint2": joint_state[1],
            "joint3": joint_state[2],
            "joint4": joint_state[3],
            "joint5": joint_state[4],
            "joint6": joint_state[5],
            "rh_r1_joint": joint_state[6],
        }

    def _smooth_action_chunk(self, raw_chunk: np.ndarray) -> np.ndarray:
        raw_chunk = np.asarray(raw_chunk, dtype=np.float64)
        if self.lipo is None:
            solved_chunk = raw_chunk
        else:
            blend = 0 if self.prev_action_chunk is None else min(self.lipo.B, len(self.prev_action_chunk), len(raw_chunk))
            solved, error = self.lipo.solve(
                raw_chunk,
                self.prev_action_chunk,
                len_past_actions=blend,
            )
            if solved is None:
                print(f"[LIPO] solve failed, using raw chunk: {error}")
                solved_chunk = raw_chunk
            else:
                solved_chunk = np.asarray(solved, dtype=np.float64)

        self.prev_action_chunk = solved_chunk.copy()
        return solved_chunk

    def _ensure_action_chunk_2d(self, action_chunk: np.ndarray) -> np.ndarray:
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            if action_chunk.shape[0] != self.action_dim:
                raise ValueError(
                    f"Expected 1D action with dim {self.action_dim}, got shape {action_chunk.shape}"
                )
            return action_chunk[None, :]
        if action_chunk.ndim == 2:
            if action_chunk.shape[1] != self.action_dim:
                raise ValueError(
                    f"Expected action chunk shape (_, {self.action_dim}), got {action_chunk.shape}"
                )
            return action_chunk
        raise ValueError(f"Expected action chunk with 1 or 2 dims, got shape {action_chunk.shape}")

    def _start_new_record_episode(self) -> None:
        if not self.recorde:
            return

        self.record_episode_dir = self.record_dir / f"episode_{self.record_episode_idx:04d}"
        self.record_episode_dir.mkdir(parents=True, exist_ok=True)
        self.record_frame_idx = 0

    def _finalize_record_episode(self) -> None:
        if not self.recorde or self.record_episode_dir is None:
            return

        frame_paths = sorted(self.record_episode_dir.glob("frame_*.png"))
        if not frame_paths:
            return

        gif_frames: list[Image.Image] = []
        adaptive_palette = getattr(getattr(Image, "Palette", None), "ADAPTIVE", Image.ADAPTIVE)
        for frame_path in frame_paths:
            with Image.open(frame_path) as frame:
                gif_frames.append(
                    frame.convert(
                        "P",
                        palette=adaptive_palette,
                        colors=256,
                        dither=Image.FLOYDSTEINBERG,
                    )
                )

        gif_path = self.record_episode_dir / "inference.gif"
        duration_ms = max(1, int(round(1000 / self.record_fps)))
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
        print(
            f"[Record] Saved {len(frame_paths)} frames to {gif_path} "
            f"(episode {self.record_episode_idx:04d})"
        )
        self.record_episode_idx += 1
        self.record_frame_idx = 0
        self.record_episode_dir = None

    def _to_uint8_image(self, image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        image = np.asarray(image)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported image shape for recording: {image.shape}")

        if np.issubdtype(image.dtype, np.floating):
            max_value = float(np.nanmax(image)) if image.size else 0.0
            scale = 255.0 if max_value <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.shape[2] == 4:
            image = image[:, :, :3]
        return image

    def _resize_for_record(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_side = max(width, height)
        if max_side <= self.record_max_side:
            return image

        scale = self.record_max_side / max_side
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _annotate_panel(self, image: Image.Image, label: str) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle((0, 0, max(80, 10 + 8 * len(label)), 24), fill=(0, 0, 0))
        draw.text((8, 6), label, fill=(255, 255, 255))
        return annotated

    def _compose_record_frame(self, images: dict[str, Any], task: Any) -> Image.Image:
        top_np = self._to_uint8_image(images["full"])
        wrist_np = self._to_uint8_image(images["wrist"])

        if not self._image_size_logged:
            print(
                f"[Record] Input image sizes - top: {top_np.shape[1]}x{top_np.shape[0]}, "
                f"wrist: {wrist_np.shape[1]}x{wrist_np.shape[0]}"
            )
            self._image_size_logged = True

        top_img = self._resize_for_record(Image.fromarray(top_np))
        wrist_img = self._resize_for_record(Image.fromarray(wrist_np))
        top_img = self._annotate_panel(top_img, "top")
        wrist_img = self._annotate_panel(wrist_img, "wrist")

        gap = 8
        canvas_width = top_img.width + wrist_img.width + gap
        canvas_height = max(top_img.height, wrist_img.height) + 30
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 18, 18))
        canvas.paste(top_img, (0, 30))
        canvas.paste(wrist_img, (top_img.width + gap, 30))

        task_text = str(task)[:120]
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"frame {self.record_frame_idx:05d} | task: {task_text}", fill=(255, 255, 255))
        return canvas

    def _record_frame(self, images: dict[str, Any], task: Any) -> None:
        if not self.recorde or self.record_episode_dir is None:
            return

        frame = self._compose_record_frame(images, task)
        frame_path = self.record_episode_dir / f"frame_{self.record_frame_idx:05d}.png"
        frame.save(frame_path, format="PNG", compress_level=0, optimize=False)
        self.record_frame_idx += 1

    @torch.inference_mode()
    def infer_fn(self, images, obss, task, action_type=None):
        if isinstance(action_type, str) and "reset" in action_type.lower():
            self.reset()

        task_str = str(task)
        if self._last_logged_task != task_str:
            print(f"[SERVER TASK] {task_str}")
            self._last_logged_task = task_str

        if self.recorde:
            self._record_frame(images, task)

        raw_observation = self._build_raw_observation(images, obss)
        if len(self.action_queue) == 0:
            obs_frame = build_inference_frame(
                observation=raw_observation,
                task=task,
                ds_features=self.dataset_metadata.features,
                device=self.device,
            )

            batch = self.preprocess(obs_frame)
            action_chunk = self.model.predict_action_chunk(batch)
            action_chunk = self.postprocess(action_chunk)
            action_chunk = action_chunk.squeeze(0).to("cpu").numpy()
            action_chunk = self._ensure_action_chunk_2d(action_chunk)
            solved_chunk = self._smooth_action_chunk(action_chunk)
            solved_chunk = self._ensure_action_chunk_2d(solved_chunk)

            for action in solved_chunk:
                self.action_queue.append(np.asarray(action, dtype=np.float32))

        action = torch.from_numpy(self.action_queue.popleft()).unsqueeze(0)
        # action = self.postprocess(action)
        action_dict = make_robot_action(action, self.dataset_metadata.features)
        return np.asarray(list(action_dict.values()), dtype=np.float32)

    def start(self) -> None:
        self.server.start_forever()

    def close(self) -> None:
        self._finalize_record_episode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T inference server")
    parser.add_argument("--pretrained-path", required=True, help="Path to the pretrained GR00T checkpoint")
    parser.add_argument("--dataset-root", required=True, help="Path to the LeRobot dataset root")
    parser.add_argument("--dataset-id", required=True, help="Dataset repo id, e.g. local/my_dataset")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--host", default="0.0.0.0", help="RPC server host")
    parser.add_argument("--port", type=int, default=1823, help="RPC server port")
    parser.add_argument(
        "--socket-utils-root",
        default="/home/cubox/workspace/isaac_chansol",
        help="Path that contains socket_utils.vla_socket",
    )
    parser.add_argument(
        "--tokenizer-assets-repo",
        default=None,
        help="Optional HF repo id or local path for Eagle processor/tokenizer assets",
    )
    parser.add_argument(
        "--use-lipo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable LIPO smoothing",
    )
    parser.add_argument("--lipo-solver", default="osqp", choices=["osqp", "cvxpy"], help="LIPO solver backend")
    parser.add_argument("--lipo-blending-horizon", type=int, default=10, help="Number of steps to blend")
    parser.add_argument("--lipo-time-delay", type=int, default=0, help="LIPO time delay")
    parser.add_argument("--lipo-dt", type=float, default=0.0333, help="Control timestep in seconds")
    parser.add_argument(
        "--lipo-epsilon-blending",
        type=float,
        default=0.02,
        help="Allowed deviation during blending horizon",
    )
    parser.add_argument(
        "--lipo-epsilon-path",
        type=float,
        default=0.003,
        help="Allowed deviation after blending horizon",
    )
    parser.add_argument(
        "--recorde",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y", "on"},
        default=False,
        help="Save per-frame images and an inference GIF for each episode",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="Directory to save recorded frames and GIFs",
    )
    parser.add_argument(
        "--record-max-side",
        type=int,
        default=480,
        help="Maximum side length used when downscaling images for recording",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=10,
        help="FPS used for the saved GIF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = GrootInferenceServer(
        pretrained_path=args.pretrained_path,
        dataset_root=args.dataset_root,
        dataset_id=args.dataset_id,
        device=args.device,
        host=args.host,
        port=args.port,
        socket_utils_root=args.socket_utils_root,
        tokenizer_assets_repo=args.tokenizer_assets_repo,
        use_lipo=args.use_lipo,
        lipo_solver=args.lipo_solver,
        lipo_blending_horizon=args.lipo_blending_horizon,
        lipo_time_delay=args.lipo_time_delay,
        lipo_dt=args.lipo_dt,
        lipo_epsilon_blending=args.lipo_epsilon_blending,
        lipo_epsilon_path=args.lipo_epsilon_path,
        recorde=args.recorde,
        record_dir=args.record_dir,
        record_max_side=args.record_max_side,
        record_fps=args.record_fps,
    )
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[Info] KeyboardInterrupt received. Finalizing recording before exit...")
    finally:
        server.close()


if __name__ == "__main__":
    main()
