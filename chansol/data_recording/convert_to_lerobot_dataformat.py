import time
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rerun_visualizer import init_rerun, log_rerun_visualization
import Robotis_OMY_isaac_configs as cfg
from get_data import DataAggregator
from tqdm import tqdm


# ==============================
# 전역 변수
# ==============================

TASK_DESCRIPTION = "put the apple in the box"     # Task Instruction



data = DataAggregator(cfg.DEFAULT_SAVE_ROOT_PATH)
save_path = f"{cfg.DEFAULT_SAVE_ROOT_PATH}_shift4"#cfg.DEFAULT_SAVE_ROOT_PATH

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

    for ep_num in range(data.episode_exist_num):
        # print(f'[Info ] 에피소드 {ep_num} 녹화 시작...')
        
        # ep_num = 16
        data.setup(episode_num=ep_num)
        for i in tqdm(range(data.total_data_num), desc=f'Episode {ep_num} Recording'):


            img_top = data.get_image_top()
            img_wrist = data.get_image_wrist()

            # 관절 변환
            follower_numpy, time_stamp = data.get_follower_action(dtype=np.float32)
            leader_numpy = data.get_leader_action(shift=4, dtype=np.float32)
            data.step_idx += 1



            frame_data = {
                'observation.images.cam_top': img_top,
                'observation.images.cam_wrist': img_wrist,
                'observation.state': follower_numpy,
                'action': leader_numpy,
                'task': TASK_DESCRIPTION,
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


