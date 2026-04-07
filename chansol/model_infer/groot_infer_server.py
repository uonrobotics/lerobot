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

        cfg = vla_server.VLAServerConfig(host=host, port=port, decode_jpeg=True)
        self.server = vla_server.VLARpcServer(cfg, infer_fn=self.infer_fn)

    def reset(self) -> None:
        self.model.reset()
        self.action_queue.clear()
        self.prev_action_chunk = None
        if self.lipo is not None:
            self.lipo.reset_log()

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

    @torch.inference_mode()
    def infer_fn(self, images, obss, action_type=None):
        if isinstance(action_type, str) and "reset" in action_type.lower():
            self.reset()

        raw_observation = self._build_raw_observation(images, obss)
        if len(self.action_queue) == 0:
            obs_frame = build_inference_frame(
                observation=raw_observation,
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
        default="/home/uon/ochansol/isaac_code/isaac_chansol",
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
    )
    server.start()


if __name__ == "__main__":
    main()
