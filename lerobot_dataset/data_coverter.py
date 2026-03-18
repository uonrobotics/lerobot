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

def convert_jointState_to_numpy_list(msg: JointState, joint_order: List[str]) -> np.ndarray:
    """JointState 메시지를 NumPy 배열로 변환"""
    joint_pos_map = dict(zip(msg.name, msg.position))
    ordered_positions = [np.float32(joint_pos_map.get(name, 0.0)) for name in joint_order]
    return np.array(ordered_positions, dtype=np.float32)

def convert_jointTrajectory_to_numpy_list(msg: JointTrajectory, joint_order: List[str]) -> np.ndarray:
    """JointTrajectory 메시지를 NumPy 배열로 변환"""
    target_point = msg.points[-1]
    joint_pos_map = dict(zip(msg.joint_names, target_point.positions))
    ordered_positions = [np.float32(joint_pos_map.get(name, 0.0)) for name in joint_order]
    return np.array(ordered_positions, dtype=np.float32)