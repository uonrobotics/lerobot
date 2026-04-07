#!/usr/bin/env python

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from action_lipo import ActionLiPo

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame


DEFAULT_PRETRAINED_PATH = (
    "/nas/AI_Checkpoints/VLA/act/Isaacsim/OMY_apple_picking/"
    "auto_fixed_place_ALL/noaug_noscheduler_lr1e-05_adamw_16bs/150000steps"
)
DEFAULT_DATASET_ROOT = "/nas/Dataset/VLA/UON/Isaacsim/OMY_apple_picking/auto_fixed_place_ALL"
DEFAULT_DATASET_ID = "user1/repo1"
DEFAULT_SOCKET_UTILS_ROOT = "/home/uon/ochansol/isaac_code/isaac_chansol"


class ActInferenceServer:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device(args.device)
        self.dataset_metadata = LeRobotDatasetMetadata(
            repo_id=args.dataset_id,
            root=args.dataset_root,
        )

        self.model = ACTPolicy.from_pretrained(pretrained_name_or_path=args.pretrained_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.reset()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            dataset_stats=self.dataset_metadata.stats,
        )

        self.use_lipo = args.use_lipo
        self.action_queue: deque[np.ndarray] = deque()
        self.prev_action_chunk: np.ndarray | None = None
        self.lipo = None
        if self.use_lipo:
            self.lipo = ActionLiPo(
                solver=args.lipo_solver,
                chunk_size=self.model.config.chunk_size,
                blending_horizon=min(args.lipo_blending_horizon, self.model.config.chunk_size),
                action_dim=self.model.config.action_feature.shape[0],
                len_time_delay=args.lipo_time_delay,
                dt=args.lipo_dt,
                epsilon_blending=args.lipo_epsilon_blending,
                epsilon_path=args.lipo_epsilon_path,
            )

        socket_root = Path(args.socket_utils_root).expanduser().resolve()
        if str(socket_root) not in sys.path:
            sys.path.append(str(socket_root))

        from socket_utils.vla_socket import vla_server

        cfg = vla_server.VLAServerConfig(host=args.host, port=args.port, decode_jpeg=True)
        self.server = vla_server.VLARpcServer(cfg, infer_fn=self.infer_fn)

    def reset(self) -> None:
        self.model.reset()
        if self.lipo is not None:
            self.lipo.reset_log()
        self.prev_action_chunk = None
        self.action_queue.clear()

    def _predict_action_chunk(self, obs_frame) -> np.ndarray:
        obs = self.preprocess(obs_frame)
        raw_chunk = self.model.predict_action_chunk(obs)
        raw_chunk = self.postprocess(raw_chunk)
        return raw_chunk.squeeze(0).to("cpu").numpy()

    def _ensure_action_chunk_2d(self, action_chunk: np.ndarray) -> np.ndarray:
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        action_dim = self.model.config.action_feature.shape[0]
        if action_chunk.ndim == 1:
            if action_chunk.shape[0] != action_dim:
                raise ValueError(f"Expected 1D action with dim {action_dim}, got shape {action_chunk.shape}")
            return action_chunk[None, :]
        if action_chunk.ndim == 2:
            if action_chunk.shape[1] != action_dim:
                raise ValueError(f"Expected action chunk shape (_, {action_dim}), got {action_chunk.shape}")
            return action_chunk
        raise ValueError(f"Expected action chunk with 1 or 2 dims, got shape {action_chunk.shape}")

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

    @torch.inference_mode()
    def infer_fn(self, images, obss, action_type=None):
        if isinstance(action_type, str) and "reset" in action_type.lower():
            self.reset()

        obs = {
            "cam_top": images["full"],
            "cam_wrist": images["wrist"],
            "joint1": obss["joint_state"][0],
            "joint2": obss["joint_state"][1],
            "joint3": obss["joint_state"][2],
            "joint4": obss["joint_state"][3],
            "joint5": obss["joint_state"][4],
            "joint6": obss["joint_state"][5],
            "rh_r1_joint": obss["joint_state"][6],
        }

        if len(self.action_queue) == 0:
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=self.dataset_metadata.features,
                device=self.device,
            )
            raw_chunk = self._predict_action_chunk(obs_frame)
            raw_chunk = self._ensure_action_chunk_2d(raw_chunk)
            solved_chunk = self._smooth_action_chunk(raw_chunk)
            solved_chunk = self._ensure_action_chunk_2d(solved_chunk)
            for action in solved_chunk[: self.model.config.n_action_steps]:
                self.action_queue.append(np.asarray(action, dtype=np.float32))

        return self.action_queue.popleft().copy()

    def start(self) -> None:
        self.server.start_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT inference server")
    parser.add_argument("--pretrained-path", default=DEFAULT_PRETRAINED_PATH, help="Path to the ACT checkpoint")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Path to the LeRobot dataset root")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Dataset repo id")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--host", default="0.0.0.0", help="RPC server host")
    parser.add_argument("--port", type=int, default=1823, help="RPC server port")
    parser.add_argument(
        "--socket-utils-root",
        default=DEFAULT_SOCKET_UTILS_ROOT,
        help="Path that contains socket_utils.vla_socket",
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
    server = ActInferenceServer(args)
    server.start()


if __name__ == "__main__":
    main()
