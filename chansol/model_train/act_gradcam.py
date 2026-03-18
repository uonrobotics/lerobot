#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@dataclass
class GradCamResult:
    camera_key: str
    camera_index: int
    heatmap: np.ndarray
    overlay: np.ndarray
    score: float


@dataclass
class JointSaliencyResult:
    saliency: np.ndarray
    normalized_saliency: np.ndarray
    score: float


def _to_uint8_hwc(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().float().numpy()

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")

    if image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = np.rint(image * 255.0).astype(np.uint8)

    return image


def _normalize_heatmap(cam: torch.Tensor, output_hw: tuple[int, int]) -> np.ndarray:
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=output_hw, mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    cam -= cam.min()
    cam /= cam.max().clamp(min=1e-8)
    return cam.detach().cpu().numpy()


def _make_overlay(image_hwc: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    # Use a standard continuous heatmap: blue -> green -> yellow -> red.
    heat_rgb = cm.get_cmap("jet")(heatmap)[..., :3]
    heat_rgb = np.rint(heat_rgb * 255.0).astype(np.uint8)
    overlay = (1.0 - alpha) * image_hwc.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)


def build_single_sample_batch(sample: dict[str, Any], preprocessor) -> dict[str, torch.Tensor]:
    return preprocessor(sample)


def _prepare_act_batch(policy: ACTPolicy, processed_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch = dict(processed_batch)
    if policy.config.image_features:
        batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
    return batch


class _BackboneFeatureHook:
    def __init__(self, backbone_module):
        self.activations: list[torch.Tensor] = []
        self.gradients: list[torch.Tensor] = []
        self.handle = backbone_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        feature_map = output["feature_map"]
        self.activations.append(feature_map)

        def _save_grad(grad: torch.Tensor):
            self.gradients.append(grad)

        feature_map.register_hook(_save_grad)

    def close(self):
        self.handle.remove()


def _resolve_camera_key(policy: ACTPolicy, camera_index: int) -> str:
    image_features = list(policy.config.image_features)
    if camera_index < 0 or camera_index >= len(image_features):
        raise IndexError(f"camera_index={camera_index} out of range for image features {image_features}")
    return image_features[camera_index]


# def compute_action_aware_gradcam(
#     policy: ACTPolicy,
#     processed_batch: dict[str, torch.Tensor],
#     raw_sample: dict[str, Any],
#     camera_index: int = 0,
#     action_timestep: int | None = 0,
#     action_dim: int | None = None,
# ) -> GradCamResult:
#     policy.eval()
#     model = policy.model
#     camera_key = _resolve_camera_key(policy, camera_index)
#     batch_for_model = _prepare_act_batch(policy, processed_batch)

#     hook = _BackboneFeatureHook(model.backbone)
#     try:
#         model.zero_grad(set_to_none=True)
#         actions, _ = model(batch_for_model)

#         if action_timestep is None:
#             if action_dim is None:
#                 target = actions[0].abs().sum()
#             else:
#                 if action_dim < 0 or action_dim >= actions.shape[2]:
#                     raise IndexError(
#                         f"action_dim={action_dim} out of range for action dim {actions.shape[2]}"
#                     )
#                 target = actions[0, :, action_dim].abs().sum()
#         else:
#             if action_timestep < 0 or action_timestep >= actions.shape[1]:
#                 raise IndexError(
#                     f"action_timestep={action_timestep} out of range for predicted chunk size {actions.shape[1]}"
#                 )

#             if action_dim is None:
#                 target = actions[0, action_timestep].abs().sum()
#             else:
#                 if action_dim < 0 or action_dim >= actions.shape[2]:
#                     raise IndexError(
#                         f"action_dim={action_dim} out of range for action dim {actions.shape[2]}"
#                     )
#                 target = actions[0, action_timestep, action_dim]

#         target.backward()

#         if len(hook.activations) != len(batch_for_model[OBS_IMAGES]):
#             raise RuntimeError("Backbone hook capture count does not match number of camera inputs.")

#         activation = hook.activations[camera_index]
#         gradient = hook.gradients[camera_index]
#         weights = gradient.mean(dim=(2, 3), keepdim=True)
#         cam = (weights * activation).sum(dim=1, keepdim=True)

#         image_hwc = _to_uint8_hwc(raw_sample[camera_key])
#         heatmap = _normalize_heatmap(cam, output_hw=image_hwc.shape[:2])
#         overlay = _make_overlay(image_hwc, heatmap)

#         return GradCamResult(
#             camera_key=camera_key,
#             camera_index=camera_index,
#             heatmap=heatmap,
#             overlay=overlay,
#             score=float(target.detach().cpu().item()),
#         )
#     finally:
#         hook.close()
#         model.zero_grad(set_to_none=True)


# def compute_joint_observation_saliency(
#     policy: ACTPolicy,
#     processed_batch: dict[str, torch.Tensor],
#     action_timestep: int | None = 0,
#     action_dim: int | None = None,
# ) -> JointSaliencyResult:
#     policy.eval()
#     model = policy.model
#     batch_for_model = _prepare_act_batch(policy, processed_batch)

#     if OBS_STATE not in batch_for_model:
#         raise ValueError("Processed batch does not contain observation.state.")

#     obs_state = batch_for_model[OBS_STATE].detach().clone().requires_grad_(True)
#     batch_for_model[OBS_STATE] = obs_state

#     model.zero_grad(set_to_none=True)
#     actions, _ = model(batch_for_model)

#     if action_timestep is None:
#         if action_dim is None:
#             target = actions[0].abs().sum()
#         else:
#             if action_dim < 0 or action_dim >= actions.shape[2]:
#                 raise IndexError(f"action_dim={action_dim} out of range for action dim {actions.shape[2]}")
#             target = actions[0, :, action_dim].abs().sum()
#     else:
#         if action_timestep < 0 or action_timestep >= actions.shape[1]:
#             raise IndexError(
#                 f"action_timestep={action_timestep} out of range for predicted chunk size {actions.shape[1]}"
#             )
#         if action_dim is None:
#             target = actions[0, action_timestep].abs().sum()
#         else:
#             if action_dim < 0 or action_dim >= actions.shape[2]:
#                 raise IndexError(f"action_dim={action_dim} out of range for action dim {actions.shape[2]}")
#             target = actions[0, action_timestep, action_dim]

#     target.backward()

#     if obs_state.grad is None:
#         raise RuntimeError("No gradient captured for observation.state.")

#     saliency = (obs_state.grad[0] * obs_state[0]).abs().detach().cpu().numpy()
#     normalized = saliency.copy()
#     normalized -= normalized.min()
#     normalized /= max(normalized.max(), 1e-8)

#     model.zero_grad(set_to_none=True)
#     return JointSaliencyResult(
#         saliency=saliency,
#         normalized_saliency=normalized,
#         score=float(target.detach().cpu().item()),
#     )


# def compute_cnn_only_gradcam(
#     policy: ACTPolicy,
#     processed_batch: dict[str, torch.Tensor],
#     raw_sample: dict[str, Any],
#     camera_index: int = 0,
# ) -> GradCamResult:
#     policy.eval()
#     model = policy.model
#     camera_key = _resolve_camera_key(policy, camera_index)
#     batch_for_model = _prepare_act_batch(policy, processed_batch)

#     if OBS_IMAGES not in batch_for_model:
#         raise ValueError("Processed batch does not contain image inputs.")

#     image_tensor = batch_for_model[OBS_IMAGES][camera_index]
#     image_tensor = image_tensor.requires_grad_(True)

#     model.zero_grad(set_to_none=True)
#     backbone_features = model.backbone(image_tensor)["feature_map"]
#     backbone_features.retain_grad()
#     projected_features = model.encoder_img_feat_input_proj(backbone_features)
#     target = projected_features.abs().mean()
#     target.backward()

#     gradient = backbone_features.grad
#     if gradient is None:
#         raise RuntimeError("No gradient captured for CNN feature map.")

#     weights = gradient.mean(dim=(2, 3), keepdim=True)
#     cam = (weights * backbone_features).sum(dim=1, keepdim=True)

#     image_hwc = _to_uint8_hwc(raw_sample[camera_key])
#     heatmap = _normalize_heatmap(cam, output_hw=image_hwc.shape[:2])
#     overlay = _make_overlay(image_hwc, heatmap)

#     model.zero_grad(set_to_none=True)
    
#     return GradCamResult(
#         camera_key=camera_key,
#         camera_index=camera_index,
#         heatmap=heatmap,
#         overlay=overlay,
#         score=float(target.detach().cpu().item()),
#     )

def compute_cnn_only_gradcam(
    policy: ACTPolicy,
    processed_batch: dict[str, torch.Tensor],
    raw_sample: dict[str, Any],
    camera_index: int = 0,
) -> GradCamResult:
    policy.eval()
    model = policy.model
    camera_key = _resolve_camera_key(policy, camera_index)
    batch_for_model = _prepare_act_batch(policy, processed_batch)

    if OBS_IMAGES not in batch_for_model:
        raise ValueError("Processed batch does not contain image inputs.")

    image_tensor = batch_for_model[OBS_IMAGES][camera_index]

    # 혹시 이전 호출에서 남은 param.grad가 있다면 제거
    model.zero_grad(set_to_none=True)

    # Grad-CAM에 필요한 부분만 grad 추적
    backbone_features = model.backbone(image_tensor)["feature_map"]
    projected_features = model.encoder_img_feat_input_proj(backbone_features)
    target = projected_features.abs().mean()

    # backward() 대신 필요한 텐서에 대해서만 grad 계산
    gradient = torch.autograd.grad(
        outputs=target,
        inputs=backbone_features,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * backbone_features).sum(dim=1, keepdim=True)

    image_hwc = _to_uint8_hwc(raw_sample[camera_key])

    # 후처리는 grad 필요 없음
    with torch.no_grad():
        heatmap = _normalize_heatmap(cam.detach(), output_hw=image_hwc.shape[:2])
        overlay = _make_overlay(image_hwc, heatmap)

    if torch.is_tensor(heatmap):
        heatmap = heatmap.detach().cpu().numpy()
    if torch.is_tensor(overlay):
        overlay = overlay.detach().cpu().numpy()

    score = float(target.detach().cpu().item())

    # 안전하게 참조 제거
    del projected_features, target, gradient, weights, cam, backbone_features

    return GradCamResult(
        camera_key=camera_key,
        camera_index=camera_index,
        heatmap=heatmap,
        overlay=overlay,
        score=score,
    )

def compute_action_aware_gradcam(
    policy: ACTPolicy,
    processed_batch: dict[str, torch.Tensor],
    raw_sample: dict[str, Any],
    camera_index: int = 0,
    action_timestep: int | None = 0,
    action_dim: int | None = None,
) -> GradCamResult:
    policy.eval()
    model = policy.model
    camera_key = _resolve_camera_key(policy, camera_index)
    batch_for_model = _prepare_act_batch(policy, processed_batch)

    # activation만 저장하는 forward hook
    activations: list[torch.Tensor] = []

    def _forward_hook(_module, _inputs, output):
        if isinstance(output, dict):
            feat = output["feature_map"]
        else:
            feat = output
        activations.append(feat)

    hook_handle = model.backbone.register_forward_hook(_forward_hook)

    try:
        model.zero_grad(set_to_none=True)
        actions, _ = model(batch_for_model)

        if action_timestep is None:
            if action_dim is None:
                target = actions[0].abs().sum()
            else:
                if action_dim < 0 or action_dim >= actions.shape[2]:
                    raise IndexError(
                        f"action_dim={action_dim} out of range for action dim {actions.shape[2]}"
                    )
                target = actions[0, :, action_dim].abs().sum()
        else:
            if action_timestep < 0 or action_timestep >= actions.shape[1]:
                raise IndexError(
                    f"action_timestep={action_timestep} out of range for predicted chunk size {actions.shape[1]}"
                )

            if action_dim is None:
                target = actions[0, action_timestep].abs().sum()
            else:
                if action_dim < 0 or action_dim >= actions.shape[2]:
                    raise IndexError(
                        f"action_dim={action_dim} out of range for action dim {actions.shape[2]}"
                    )
                target = actions[0, action_timestep, action_dim]

        if len(activations) != len(batch_for_model[OBS_IMAGES]):
            raise RuntimeError(
                "Backbone hook capture count does not match number of camera inputs."
            )

        activation = activations[camera_index]

        gradient = torch.autograd.grad(
            outputs=target,
            inputs=activation,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        if gradient is None:
            raise RuntimeError("No gradient captured for selected backbone activation.")

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)

        image_hwc = _to_uint8_hwc(raw_sample[camera_key])

        with torch.no_grad():
            heatmap = _normalize_heatmap(cam.detach(), output_hw=image_hwc.shape[:2])
            overlay = _make_overlay(image_hwc, heatmap)

        if torch.is_tensor(heatmap):
            heatmap = heatmap.detach().cpu().numpy()
        if torch.is_tensor(overlay):
            overlay = overlay.detach().cpu().numpy()

        score = float(target.detach().cpu().item())

        del actions, target, activation, gradient, weights, cam

        return GradCamResult(
            camera_key=camera_key,
            camera_index=camera_index,
            heatmap=heatmap,
            overlay=overlay,
            score=score,
        )
    finally:
        hook_handle.remove()
        model.zero_grad(set_to_none=True)

def compute_joint_observation_saliency(
    policy: ACTPolicy,
    processed_batch: dict[str, torch.Tensor],
    action_timestep: int | None = 0,
    action_dim: int | None = None,
) -> JointSaliencyResult:
    policy.eval()
    model = policy.model
    batch_for_model = _prepare_act_batch(policy, processed_batch)

    if OBS_STATE not in batch_for_model:
        raise ValueError("Processed batch does not contain observation.state.")

    obs_state = batch_for_model[OBS_STATE].detach().clone().requires_grad_(True)
    batch_for_model[OBS_STATE] = obs_state

    model.zero_grad(set_to_none=True)
    actions, _ = model(batch_for_model)

    if action_timestep is None:
        if action_dim is None:
            target = actions[0].abs().sum()
        else:
            if action_dim < 0 or action_dim >= actions.shape[2]:
                raise IndexError(
                    f"action_dim={action_dim} out of range for action dim {actions.shape[2]}"
                )
            target = actions[0, :, action_dim].abs().sum()
    else:
        if action_timestep < 0 or action_timestep >= actions.shape[1]:
            raise IndexError(
                f"action_timestep={action_timestep} out of range for predicted chunk size {actions.shape[1]}"
            )
        if action_dim is None:
            target = actions[0, action_timestep].abs().sum()
        else:
            if action_dim < 0 or action_dim >= actions.shape[2]:
                raise IndexError(
                    f"action_dim={action_dim} out of range for action dim {actions.shape[2]}"
                )
            target = actions[0, action_timestep, action_dim]

    obs_grad = torch.autograd.grad(
        outputs=target,
        inputs=obs_state,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]

    if obs_grad is None:
        raise RuntimeError("No gradient captured for observation.state.")

    saliency = (obs_grad[0] * obs_state[0]).abs().detach().cpu().numpy()
    normalized = saliency.copy()
    normalized -= normalized.min()
    normalized /= max(normalized.max(), 1e-8)

    score = float(target.detach().cpu().item())

    del actions, target, obs_grad, obs_state

    model.zero_grad(set_to_none=True)

    return JointSaliencyResult(
        saliency=saliency,
        normalized_saliency=normalized,
        score=score,
    )
