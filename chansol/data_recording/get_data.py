import os
import numpy as np
from PIL import Image
import json

class DataAggregator():
    def __init__(self, data_root_path):
        self.data_root_path = data_root_path
        self.action_dir_path = "action"
        self.rgb_dir_path = "rgb"
        self.depth_dir_path = "depth"

        self.wrist_dir_path = "cam_wrist"
        self.top_dir_path = "cam_top"
        
        self.step_idx = 0
        self.episode_exist_list= sorted([i for i in os.listdir(os.path.join(self.data_root_path, self.rgb_dir_path)) if os.path.isdir(os.path.join(self.data_root_path, self.rgb_dir_path, i))])


    def setup(self, episode_num):
        self.step_idx = 0
        wrist_img_path = os.path.join(self.data_root_path, self.rgb_dir_path, f"{episode_num:04d}", self.wrist_dir_path)
        top_img_path   = os.path.join(self.data_root_path, self.rgb_dir_path, f"{episode_num:04d}", self.top_dir_path)
        wrist_depth_path = os.path.join(self.data_root_path, self.depth_dir_path, f"{episode_num:04d}", self.wrist_dir_path)
        top_depth_path   = os.path.join(self.data_root_path, self.depth_dir_path, f"{episode_num:04d}", self.top_dir_path)
        action_path    = os.path.join(self.data_root_path, self.action_dir_path)

        if not os.path.exists(wrist_img_path):
            raise FileNotFoundError(f"Wrist image path not found: {wrist_img_path}")
        if not os.path.exists(top_img_path):
            raise FileNotFoundError(f"Top image path not found: {top_img_path}")
        if not os.path.exists(wrist_depth_path):
            raise FileNotFoundError(f"Wrist depth path not found: {wrist_depth_path}")
        if not os.path.exists(top_depth_path):
            raise FileNotFoundError(f"Top depth path not found: {top_depth_path}")
        if not os.path.exists(action_path):
            raise FileNotFoundError(f"Action path not found: {action_path}")

        self.wrist_img_list = sorted([os.path.join(wrist_img_path, img)       for img in os.listdir(wrist_img_path) if img.endswith('.png')])
        self.top_img_list   = sorted([os.path.join(top_img_path, img)         for img in os.listdir(top_img_path) if img.endswith('.png')])
        self.wrist_depth_list = sorted([os.path.join(wrist_depth_path, img)   for img in os.listdir(wrist_depth_path) if img.endswith('.npy')])
        self.top_depth_list   = sorted([os.path.join(top_depth_path, img)     for img in os.listdir(top_depth_path) if img.endswith('.npy')])
        self.action_list    = os.path.join(action_path, f"{episode_num:04d}.json")

        self.action = json.load(open(self.action_list, 'r'))
        self.total_data_num = len(self.wrist_img_list)
        


    def get_image_wrist(self):
        img = Image.open(self.wrist_img_list[self.step_idx])
        img = np.array(img)[..., :3]
        return img
    
    def get_image_top(self):
        img = Image.open(self.top_img_list[self.step_idx])
        img = np.array(img)[..., :3]
        return img
    def get_depth_wrist(self):
        img = np.load(self.wrist_depth_list[self.step_idx])
        return img
    def get_depth_top(self):
        img = np.load(self.top_depth_list[self.step_idx])
        return img
    
    def cal_action_idx(self):
        return int(self.top_img_list[self.step_idx].split("/")[-1].rstrip(".png"))
    
    def get_follower_action(self, dtype=np.float32):
        action_info = self.action[self.cal_action_idx()]
        joint_names = action_info['robot']['joint_names']
        joint_positions = action_info['robot']['joint_positions']
        joint_velocities = action_info['robot']['joint_velocities']
        time = action_info['time']
        index = action_info['index']
        # print(f'[Debug] Follower action index: {index}, time: {time:.4f}s')
        actions = [joint_positions[i] for i,name in enumerate(joint_names) if 'joint' in name]
        return np.array(actions, dtype=dtype) ,time
    
    
    def get_leader_action(self,shift=5, dtype=np.float32):
        idx = self.cal_action_idx() + shift
        if idx >= len(self.action):
            idx = -1

        action_info = self.action[idx]
        joint_names = action_info['robot']['joint_names']
        joint_positions = action_info['robot']['joint_positions']
        joint_velocities = action_info['robot']['joint_velocities']
        time = action_info['time']
        index = action_info['index']
        # print(f'[Debug] Leader action index: {index}, time: {time:.4f}s')

        actions = [joint_positions[i] for i,name in enumerate(joint_names) if 'joint' in name]
        return np.array(actions, dtype=dtype)