import numpy as np
import cv2
from typing import List

# ros2
from sensor_msgs.msg import JointState, CompressedImage
from trajectory_msgs.msg import JointTrajectory

def _convert_compressedImage_to_cvmat(msg: CompressedImage) -> np.ndarray:
    """CompressedImage 메시지를 OpenCV Mat 형식으로 변환"""
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if cv_image is not None and cv_image.dtype == np.uint16:
        cv_image = cv2.normalize(
            cv_image,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U)
    return cv_image

def convert_compressedImage_to_numpy(msg: CompressedImage) -> np.ndarray:
    """CompressedImage 메시지를 NumPy 배열로 변환"""
    cv_mat = _convert_compressedImage_to_cvmat(msg)
    if cv_mat is None:
        return None
    return cv2.cvtColor(cv_mat, cv2.COLOR_BGR2RGB)

def convert_jointState_to_numpy_list(msg: JointState, observation_order: List[str], action_order: List[str], kinematics_solver=None) -> np.ndarray:
    """JointState 메시지를 NumPy 배열로 변환"""
    joint_pos_map = dict(zip(msg.name, msg.position))

    print("===jointstate")
    print(joint_pos_map)

    

    ordered_positions = [np.float32(joint_pos_map.get(name, 0.0)) for name in observation_order]

    print("===ordered")
    print(ordered_positions)
    return np.array(ordered_positions, dtype=np.float32)

def convert_jointTrajectory_to_numpy_list(msg: JointTrajectory, observation_order: List[str], action_order: List[str], kinematics_solver=None) -> np.ndarray:
    """JointTrajectory 메시지를 NumPy 배열로 변환"""
    target_point = msg.points[-1]
    joint_pos_map = dict(zip(msg.joint_names, target_point.positions))
    print("===jointTraj")
    print(joint_pos_map)
    ordered_positions = [np.float32(joint_pos_map.get(name, 0.0)) for name in observation_order]
    print("===ordered")
    print(ordered_positions)
    return np.array(ordered_positions, dtype=np.float32)



class DataConverter():
    def __init__(self, args, observation_order:List[str], action_order:List[str], kinematics_solver=None):
        self.action_type = args.action_type

        self.joint_names = args.joint_names
        self.observation_order = observation_order
        self.action_order = action_order

        self.kinematics_solver = kinematics_solver
        print("initialization")

    def convert(self, msg_img1, msg_img2, msg_joint_states, msg_joint_trajectory):
        img_1 = self.convert_compressedImage_to_numpy(msg_img1)
        img_2 = self.convert_compressedImage_to_numpy(msg_img2)

        if self.action_type == "abs_joint":
            state_numpy = self.convert_jointState_to_numpy_list(msg_joint_states, self.observation_order)   # state
            action_numpy = self.convert_jointTrajectory_to_numpy_list(msg_joint_trajectory, self.action_order)  # action

        elif self.action_type == "delta_pose":
            follower_numpy = self.convert_jointState_to_numpy_list(msg_joint_states, self.joint_names)   # state
            leader_numpy = self.convert_jointTrajectory_to_numpy_list(msg_joint_trajectory, self.joint_names)  # action
            state_numpy, action_numpy = self.convert_jointSpace_numpy_to_pose(follower_numpy, leader_numpy) # state : abs_pose, action : delta_pose

        return img_1, img_2, state_numpy, action_numpy

    def _convert_compressedImage_to_cvmat(self,msg: CompressedImage) -> np.ndarray:
        """CompressedImage 메시지를 OpenCV Mat 형식으로 변환"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is not None and cv_image.dtype == np.uint16:
            cv_image = cv2.normalize(
                cv_image,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U)
        return cv_image

    def convert_compressedImage_to_numpy(self, msg: CompressedImage) -> np.ndarray:
        """CompressedImage 메시지를 NumPy 배열로 변환"""
        cv_mat = _convert_compressedImage_to_cvmat(msg)
        if cv_mat is None:
            return None
        return cv2.cvtColor(cv_mat, cv2.COLOR_BGR2RGB)

    def convert_jointState_to_numpy_list(self, msg: JointState, order:list[str]) -> np.ndarray:
        """JointState 메시지를 NumPy 배열로 변환"""
        joint_pos_map = dict(zip(msg.name, msg.position))
        ordered_positions = [np.float32(joint_pos_map.get(name, 0.0)) for name in order]
        return np.array(ordered_positions, dtype=np.float32)

    def convert_jointTrajectory_to_numpy_list(self, msg: JointTrajectory, order:list[str]) -> np.ndarray:
        """JointTrajectory 메시지를 NumPy 배열로 변환"""
        target_point = msg.points[-1]
        joint_pos_map = dict(zip(msg.joint_names, target_point.positions))
        ordered_positions = [np.float32(joint_pos_map.get(name, 0.0)) for name in order]
        return np.array(ordered_positions, dtype=np.float32)

    def convert_jointSpace_numpy_to_pose(self, follower_: np.ndarray, leader_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """sadsdafsadfasfdasdfasfdasdf"""
        follower = np.concatenate([follower_, np.repeat(follower_[-1], 3)]) # (10,)
        leader = np.concatenate([leader_, np.repeat(leader_[-1], 3)]) # (10,)

        follower_pose = self.kinematics_solver._forward(follower)
        leader_pose = self.kinematics_solver._forward(leader)

        delta_pose = leader_pose - follower_pose
        
        return follower_pose, delta_pose # state, action