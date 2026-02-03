"""This script demonstrates how to train ACT Policy on a real-world dataset."""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from tqdm import tqdm


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def main():
    output_directory = Path("/home/uon/ochansol/lerobot/chansol/model_train/weights/isaac_omy_put_food_act")

    dataset_id = "user1/repo1"
    dataset_root_path = "/nas/Dataset/VLA/UON/Isaacsim_OMY"
    pre_checkpoint_path = ""#"/home/uon/ochansol/lerobot/chansol/model_train/weights/isaac_omy_put_food_act"
    optim_name = "adamw" ## "adamw" or "sgd"
    batch_size = 16
    training_steps = int(1e5)
    log_freq = 1
    save_step = 100

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

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
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
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

    # Create the optimizer and dataloader for offline training
    if optim_name =="adamw":
        optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    elif optim_name =="sgd":
        from lerobot.optim.optimizers import SGDConfig
        optimizer = SGDConfig(lr=cfg.optimizer_lr, 
                              momentum=0.9, 
                              weight_decay=cfg.optimizer_weight_decay).build(policy.parameters())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    import pdb; pdb.set_trace()

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
            optimizer.zero_grad()
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                step=step,
                lr = optimizer.param_groups[0]["lr"],
            )

            if step % log_freq == 0:
                pass
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




    # Save all assets to the Hub
    # policy.push_to_hub(dataset_id)
    # preprocessor.push_to_hub(dataset_id)
    # postprocessor.push_to_hub(dataset_id)


if __name__ == "__main__":
    main()
