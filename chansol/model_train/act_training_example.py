"""This script demonstrates how to train ACT Policy on a real-world dataset."""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from tqdm import tqdm
import wandb
wandb.init(
    project="Isaacsim_OMY_apple_picking_auto",   # 프로젝트 이름
    name="shift4_filtered",                   # 실험 이름 (선택)
    resume="allow",                   # 이전 실험이 있으면 이어서 진행
    config={
        "lr": 1e-5,
        "batch_size": 16,
        "model": "act",
        "dataset_path": "/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking_auto_shift4_filtered",
        "training_steps": int(100e4),
    }
)
def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def main():
    output_directory = Path(f"/nas/AI_Checkpoints/VLA/act/{wandb.run.project}/{wandb.run.name}")

    dataset_id = "user1/repo1"
    dataset_root_path = "/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking_auto_shift4_filtered" #"/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking_auto" --- IGNORE ---
    pre_checkpoint_path = ""#"/home/uon/ochansol/lerobot/chansol/model_train/weights/isaac_omy_put_food_act"
    optim_name = "adamw" ## "adamw" or "sgd"
    scheduler_name = "cosine_decay_with_warmup"
    batch_size = wandb.config.batch_size
    training_steps = wandb.config.training_steps
    log_freq = 20
    save_step = 10000

    # Select your device
    device = torch.device("cuda")  # or "cuda" or "cpu"


    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id=dataset_id,
        root=dataset_root_path,
    )

    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        optimizer_lr=wandb.config.lr,
    )
    if pre_checkpoint_path != "":
        policy = ACTPolicy(cfg).from_pretrained(pretrained_name_or_path=pre_checkpoint_path)
    else:
        policy = ACTPolicy(cfg)

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # To perform action chunking, ACT expects a given number of actions as targets
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }

    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    # Instantiate the dataset
    dataset = LeRobotDataset(
        dataset_id, 
        root=dataset_root_path,
        delta_timestamps=delta_timestamps)

    # Create the optimizer and dataloader for offline training
    if optim_name =="adamw":
        optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    elif optim_name =="sgd":
        from lerobot.optim.optimizers import SGDConfig
        optimizer = SGDConfig(lr=cfg.optimizer_lr, 
                              momentum=0.9, 
                              weight_decay=cfg.optimizer_weight_decay).build(policy.parameters())

    scheduler = None
    if scheduler_name == "cosine_decay_with_warmup":
        scheduler_cfg = CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=min(1000, max(1, training_steps // 100)),
            num_decay_steps=training_steps,
            peak_lr=cfg.optimizer_lr,
            decay_lr=cfg.optimizer_lr * 0.1,
        )
        scheduler = scheduler_cfg.build(optimizer, training_steps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    ###### debuging #############
    # data = iter(dataloader)
    # batch = next(data)
    # batch_action = batch['action']
    # pre_action  = preprocessor(batch)["action"]
    # import matplotlib.pyplot as plt
    # plt.plot(batch_action[0].cpu().numpy(), label='orig', c ='orange')
    # plt.plot(pre_action[0].cpu().numpy(), label='preproc', c ='blue')
    # plt.legend()
    # plt.show()

    

    # Run training loop
    step = 0
    done = False
    while not done:
        pbar = tqdm(dataloader, desc="Training", unit="batch")
        for batch in pbar:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                step=step,
                lr=current_lr,
            )

            if step % log_freq == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": step,
                    "train/lr": current_lr,
                })
            if step % save_step == 0 and step != 0:
                output_path = Path(f"{output_directory}_{optim_name}_{step:06d}steps_{batch_size}bs")
                output_path.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(output_path)
                preprocessor.save_pretrained(output_path)
                postprocessor.save_pretrained(output_path)
            step += 1
            if step >= training_steps:
                done = True
                break


    wandb.finish()

    # Save all assets to the Hub
    # policy.push_to_hub(dataset_id)
    # preprocessor.push_to_hub(dataset_id)
    # postprocessor.push_to_hub(dataset_id)


if __name__ == "__main__":
    main()
