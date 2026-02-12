import numpy as np
import pinocchio as pin



def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    """
    Pinocchio 기본 RPY convention: roll-pitch-yaw = (x, y, z) 순서의 회전
    - 반환: [roll, pitch, yaw] (rad)
    """
    return pin.rpy.matrixToRpy(R)

class KinematicsSolver():
    def __init__(self, path_urdf, base_frame:str, ee_frame:str):
        # Load the urdf model
        self.model = pin.buildModelFromUrdf(path_urdf)
        print("model name: " + self.model.name)
        
        # Create data required by the algorithms
        self.data = self.model.createData()
        
        # 2) 프레임 id 찾기 (joint가 아니라 "frame" 기준 추천)
        self.base_fid = self.model.getFrameId(base_frame)
        self.ee_fid   = self.model.getFrameId(ee_frame)

    def _get_random_configuration(self):
        # Sample a random configuration
        q = pin.randomConfiguration(self.model) * 0
        print("===")
        print(q)
        print(f"q: {q.T}")

        return q

    def _forward(self, joint_states=None, test=False):
        if test:
            joint_states = self._get_random_configuration()

        # 3) FK 실행 + 프레임 위치 갱신
        pin.forwardKinematics(self.model, self.data, joint_states)
        pin.updateFramePlacements(self.model, self.data)

        # 4) 월드(URDF root) 기준 base, ee pose
        oMb = self.data.oMf[self.base_fid]  # world->base
        oMe = self.data.oMf[self.ee_fid]    # world->ee

        # 5) base 기준 ee pose: bMe = (oMb)^(-1) * oMe
        bMe = oMb.inverse() * oMe
        # 6) xyz + rpy
        xyz = bMe.translation.copy()
        # rpy = rot_to_rpy(bMe.rotation)

        ############### modify!!!!!!!!!1 ############
        rpy = np.zeros(3).astype(np.float32) # tmp. need to modify!!!
        gripper = np.zeros(1).astype(np.float32) # tmp. need to modify!!!
        ############### modify!!!!!!!!!1 ############

        pose_7d = np.concatenate([xyz, rpy, gripper]).astype(np.float32)

        return pose_7d
        # return xyz, rpy, bMe  # bMe는 SE3 전체
