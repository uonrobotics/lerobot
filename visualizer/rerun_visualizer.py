import rerun as rr
import numpy as np

# ros2
from sensor_msgs.msg import JointState, CompressedImage
from trajectory_msgs.msg import JointTrajectory


def init_rerun(title: str):
    rr.init(title, spawn=True)

def log_rerun_visualization(images, follower_joints, leader_joints):
    # 카메라 이미지 로깅
    for cam_name, img in images.items():
        if img is not None:
            rr.log(f"robot/camera/{cam_name}", rr.Image(img))

    # Follower (현재 상태)
    if follower_joints is not None:
        follower_str = ", ".join([f"{x:.2f}" for x in np.rad2deg(follower_joints)])
        rr.log("robot/text/follower", rr.TextLog(f"Fo(deg): [{follower_str}]"))

    # Leader (Action)
    if leader_joints is not None:
        leader_str = ", ".join([f"{x:.2f}" for x in np.rad2deg(leader_joints)])
        rr.log("robot/text/leader", rr.TextLog(f"Le(deg): [{leader_str}]"))
