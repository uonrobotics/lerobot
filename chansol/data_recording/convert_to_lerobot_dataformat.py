import argparse
import time
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rerun_visualizer import init_rerun, log_rerun_visualization
import Robotis_OMY_isaac_configs as cfg
from get_data import DataAggregator
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert raw data to LeRobot dataset format")
parser.add_argument("--data_root", type=str, help="Root directory of the raw data")
parser.add_argument("--save_path", type=str, default="",   help="Path to save the converted dataset")
parser.add_argument("--shift", type=int, default=4, help="Time step shift for leader action")
parser.add_argument("--start_num", type=int, default=0, help="Starting episode number")
parser.add_argument("--end_num", type=int, default=-1, help="Ending episode number")
args = parser.parse_args()
# ==============================
# 전역 변수
# ==============================


if args.save_path == "":
    save_path = f"{args.data_root}"
else :
    save_path = f"{args.save_path}"
# DEFAULT_SAVE_ROOT_PATH = Path("/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking_auto_fixed_box")

data = DataAggregator(args.data_root)

shift = args.shift
start_num = args.start_num
end_num = args.end_num if args.end_num >= 0 else len(data.episode_exist_list)
def main():
    # ------------------------------
    # Rerun 초기화
    # ------------------------------
    # init_rerun('Lerobot Data Collection')


    # ------------------------------
    # 데이터셋 생성
    # ------------------------------
    # if (cfg.DEFAULT_SAVE_ROOT_PATH / cfg.HF_REPO_ID).exists():
    #     import shutil
    #     shutil.rmtree(cfg.DEFAULT_SAVE_ROOT_PATH / cfg.HF_REPO_ID)
    #     print(f'[Info ] 이미 있는 데이터셋 제거함: {cfg.DEFAULT_SAVE_ROOT_PATH / cfg.HF_REPO_ID}')

    dataset = LeRobotDataset.create(
        repo_id=cfg.HF_REPO_ID,
        fps=cfg.FPS,
        features=cfg.FEATURES,
        root = save_path,
        robot_type='omy_f3m',
        use_videos=True,
        image_writer_processes=8,
        image_writer_threads=16,
    )
    print(f'[Info ] 새 데이터셋 생성됨: {save_path}')

    for ep_num in data.episode_exist_list[start_num:end_num]:
        ep_num = int(ep_num)
        # print(f'[Info ] 에피소드 {ep_num} 녹화 시작...')
        
        # ep_num = 16
        data.setup(episode_num=ep_num)
        for i in tqdm(range(data.total_data_num), desc=f'Episode {ep_num} Recording'):


            # 관절 변환
            follower_numpy, time_stamp = data.get_follower_action(dtype=np.float32)
            leader_numpy = data.get_leader_action(shift=shift, dtype=np.float32)
            data.step_idx += 1



            frame_data = {
                'observation.images.cam_top': data.get_image_top(),
                'observation.images.cam_wrist': data.get_image_wrist(),
                'observation.depth.cam_top': data.get_depth_top(),
                'observation.depth.cam_wrist': data.get_depth_wrist(),
                'observation.state': follower_numpy,
                'action': leader_numpy,
                'task': cfg.TASK_DESCRIPTION,
                'timestamp': time_stamp,
            }
            dataset.add_frame(frame_data)


            # 3. Rerun 시각화
            # log_rerun_visualization(
            #     images={'cam_top': img_top, 'cam_wrist': img_wrist},
            #     follower_joints=follower_numpy,
            #     leader_joints=leader_numpy,
            # )


        print(f'[Info ] 에피소드 저장중...')
        dataset.save_episode()
        print(f'[Info ] 에피소드 저장 완료')



    dataset.finalize()
    print(f'[Info ] 데이터셋 Finalize 완료')
    print(f'[Info ] 종료')



if __name__ == "__main__":
    main()


