import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act_vit.modeling_act_vit import ACTViTPolicy

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action


import sys
sys.path.append("/home/cubox/workspace/isaac_chansol")
from socket_utils.vla_socket import vla_server
from typing import Any, Dict, Callable, Optional, Set, Tuple
from typing_extensions import override
import numpy as np

# rollout 
import os
import cv2
import math
import queue
import threading
import imageio.v2 as imageio
import torch.nn.functional as F
from pathlib import Path

pre_trained_path = "/nas/AI_Checkpoints/VLA/act_vit/Isaacsim/0331_act_vit_aug_fullft_trans/checkpoints/180000/pretrained_model"
dataset_root_path = "/nas/Dataset/VLA/UON/Isaacsim/OMY_apple_picking/auto_fixed_place_aug"
 

dataset_id = "user1/repo1"


model = ACTViTPolicy.from_pretrained(pretrained_name_or_path=pre_trained_path)
device = torch.device("cuda") 

# This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
dataset_metadata = LeRobotDatasetMetadata(
    repo_id=dataset_id,
    root=dataset_root_path,
)
preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

class ViTRolloutDebugger:
    def __init__(
        self,
        save_dir="./vit_rollout_debug_freeze",
        sample_every=1,
        max_frames: Optional[int] = 120,
        alpha=0.45,
        gif_every=0,
        max_pending_writes=32,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.frames_dir = self.save_dir / "tmp_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.sample_every = sample_every
        self.max_frames = max_frames
        self.alpha = alpha
        self.gif_every = gif_every

        self.frame_idx = 0
        self.saved_paths = []
        self.write_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=max_pending_writes)
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="vit-rollout-writer",
            daemon=True,
        )
        self.writer_thread.start()

    def _writer_loop(self):
        while True:
            job_type, payload = self.write_queue.get()
            try:
                if job_type == "frame":
                    out_path, overlay = payload
                    imageio.imwrite(out_path, overlay)
                    self.saved_paths.append(str(out_path))
                    print(f"[Rollout] saved: {out_path}")
                elif job_type == "gif":
                    self._make_gif_sync(payload)
                elif job_type == "stop":
                    if payload is not None:
                        self._make_gif_sync(payload)
                    return
            except Exception as e:
                print("[Rollout] writer failed:", repr(e))
            finally:
                self.write_queue.task_done()

    def _enqueue_job(self, job_type: str, payload: Any) -> bool:
        try:
            self.write_queue.put_nowait((job_type, payload))
            return True
        except queue.Full:
            print(f"[Rollout] skipped {job_type}: writer queue is full")
            return False

    def _find_topcam_tensor(self, proc_obs: Dict[str, Any]) -> torch.Tensor:
        # preprocess 결과에서 top cam tensor 찾기
        # 보통 key 예: observation.images.cam_top
        candidates = []
        for k, v in proc_obs.items():
            if torch.is_tensor(v) and v.ndim == 4:
                if ("cam_top" in k) or (k.endswith("images.top")) or (k.endswith("top")):
                    candidates.append((k, v))

        if len(candidates) == 0:
            raise KeyError(
                f"top cam tensor not found. proc_obs keys = {list(proc_obs.keys())}"
            )

        print("[Rollout] using topcam key:", candidates[0][0])
        return candidates[0][1]

    @torch.no_grad()
    def compute_rollout(self, model, topcam_tensor: torch.Tensor) -> np.ndarray:
        model.eval()

        vit_img = F.interpolate(
            topcam_tensor,
            size=(384, 384),
            mode="bilinear",
            align_corners=False,
        )

        backbone = model.model.top_cam_backbone

        x = backbone._process_input(vit_img)   # [B, 576, D]
        n = x.shape[0]

        cls_tok = backbone.class_token.expand(n, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)     # [B, 577, D]

        pos_embed = backbone.encoder.pos_embedding
        if pos_embed.shape[1] != x.shape[1]:
            raise ValueError(
                f"pos_embedding 길이 불일치: pos={pos_embed.shape}, x={x.shape}"
            )

        x = x + pos_embed
        x = backbone.encoder.dropout(x)

        attn_weights_all = []

        for layer_idx, layer in enumerate(backbone.encoder.layers):
            y = layer.ln_1(x)

            attn_out, attn_weights = layer.self_attention(
                y, y, y,
                need_weights=True,
                average_attn_weights=False
            )
            # attn_weights: [B, heads, T, T]
            attn_weights_all.append(attn_weights)

            x = x + layer.dropout(attn_out)

            y2 = layer.ln_2(x)
            y2 = layer.mlp(y2)
            x = x + y2

        x = backbone.encoder.ln(x)

        b, _, t, _ = attn_weights_all[0].shape
        device_ = attn_weights_all[0].device
        eye = torch.eye(t, device=device_).unsqueeze(0)   # [1, T, T]

        rollout = eye.repeat(b, 1, 1)

        for attn in attn_weights_all:
            attn_mean = attn.mean(dim=1)   # [B, T, T]
            attn_mean = attn_mean + eye
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
            rollout = attn_mean @ rollout

        cls_to_patch = rollout[:, 0, 1:]   # [B, 576]

        num_patches = cls_to_patch.shape[-1]
        side = int(math.sqrt(num_patches))
        assert side * side == num_patches, f"unexpected number of patches: {num_patches}"

        mask = cls_to_patch[0].reshape(side, side).detach().float().cpu().numpy()
        mask = mask - mask.min()
        if mask.max() > 1e-8:
            mask = mask / mask.max()

        return mask

    def overlay_on_image(self, raw_img: np.ndarray, mask_24: np.ndarray) -> np.ndarray:
        """
        raw_img: HWC uint8
        mask_24: [24,24]
        return: overlay image HWC uint8
        """
        img = raw_img.copy()

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # 원본이 BGR이면 RGB로 바꾸고 싶으면 여기서 처리
        # 필요하면 아래 주석 해제
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        heat = cv2.resize(mask_24, (w, h), interpolation=cv2.INTER_CUBIC)
        heat = np.clip(heat, 0.0, 1.0)
        heat_u8 = (heat * 255).astype(np.uint8)

        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        if img.shape[2] == 3:
            base = img
        else:
            raise ValueError(f"unexpected image shape: {img.shape}")

        overlay = cv2.addWeighted(base, 1.0 - self.alpha, heat_color, self.alpha, 0)

        # 좌측 상단에 작은 mask도 표시
        small = cv2.resize(heat_u8, (160, 160), interpolation=cv2.INTER_NEAREST)
        small = cv2.applyColorMap(small, cv2.COLORMAP_JET)
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        overlay[10:170, 10:170] = small

        return overlay

    def maybe_save(self, model, proc_obs: Dict[str, Any], raw_top_img: np.ndarray):
        if self.frame_idx % self.sample_every != 0:
            self.frame_idx += 1
            return

        expected_saved = self.frame_idx // max(self.sample_every, 1)
        if self.max_frames is not None and self.max_frames > 0 and expected_saved >= self.max_frames:
            self.frame_idx += 1
            return

        topcam_tensor = self._find_topcam_tensor(proc_obs)
        mask = self.compute_rollout(model, topcam_tensor)
        overlay = self.overlay_on_image(raw_top_img, mask)

        out_path = self.frames_dir / f"frame_{expected_saved:04d}.png"
        self._enqueue_job("frame", (out_path, overlay))
        self.frame_idx += 1

        # 필요할 때만 백그라운드에서 gif 생성
        if self.gif_every > 0 and (expected_saved + 1) % self.gif_every == 0:
            self._enqueue_job("gif", 5)

    def _make_gif_sync(self, fps=5):
        if len(self.saved_paths) == 0:
            return

        images = [imageio.imread(p) for p in self.saved_paths]
        gif_path = self.save_dir / "rollout.gif"
        imageio.mimsave(gif_path, images, fps=fps)
        print(f"[Rollout] gif saved: {gif_path}")

    def make_gif(self, fps=5):
        self.write_queue.join()
        self._make_gif_sync(fps=fps)

    def close(self, fps=5):
        self.write_queue.put(("stop", fps))
        self.writer_thread.join()
    
rollout_debugger = ViTRolloutDebugger(
    save_dir="./0401_vit_act_aug_fullft_add_trans_180k",
    sample_every=3,     # 매 프레임 저장
    max_frames=None,    # None 또는 0이면 종료할 때까지 계속 저장
    alpha=0.45,
    gif_every=0,        # 0이면 종료 시점에만 gif 생성
)

def infer_fn(images, obss, action_type) -> Dict[str, Any]:

    obs = {
        'cam_top': images['full'],
        'cam_wrist': images['wrist'],
        "joint1":obss["joint_state"][0],
        "joint2":obss["joint_state"][1],
        "joint3":obss["joint_state"][2],
        "joint4":obss["joint_state"][3],
        "joint5":obss["joint_state"][4],
        "joint6":obss["joint_state"][5],
        "rh_r1_joint":obss["joint_state"][6],

    }
    print(obs["joint1"], obs["joint2"], obs["joint3"], 
          obs["joint4"], obs["joint5"], obs["joint6"], obs["rh_r1_joint"])
    obs_frame = build_inference_frame(
        observation=obs, 
        ds_features=dataset_metadata.features, 
        device=device
    )

    proc_obs = preprocess(obs_frame)

    # ----------------------------
    # rollout 저장
    # ----------------------------
    try:
        raw_top_img = images["full"]
        rollout_debugger.maybe_save(model, proc_obs, raw_top_img)
    except Exception as e:
        print("[Rollout] failed:", repr(e))

    action = model.select_action(proc_obs)
    action = postprocess(action)

    action = make_robot_action(action, dataset_metadata.features)
    action = np.array([i for i in action.values()])

    return action



cfg = vla_server.VLAServerConfig(host="0.0.0.0", port=1823, decode_jpeg=True)
server = vla_server.VLARpcServer(cfg, infer_fn=infer_fn)


try:
    server.start_forever()
finally:
    rollout_debugger.close(fps=5)
