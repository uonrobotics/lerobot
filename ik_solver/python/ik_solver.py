import os
import sys
import numpy as np
from typing import Any, Optional, Union, List, Tuple, Dict


# ------------------------------------------------------------------
# Ik Solver 모듈 불러오기
# .so 파일이 있는 디렉토리를 sys.path.append로 추가
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(current_dir, '.'))
sys.path.append(lib_dir)
import ik_solver_py


# ------------------------------------------------------------------
# Ik Solver 상위 래퍼 클래스
# ------------------------------------------------------------------
class IkSolver(ik_solver_py.IkSolver):
    def __init__(self, urdf_path: str):
        super().__init__(urdf_path)
        self.is_initialized = False

    def init(self) -> bool:
        if super().init():
            super().set_tcp_max_speed(1.0)
            super().set_joint_velocity_limit_scale(0.8)

            self.is_initialized = True
            return True
        return False

    def movej(self, q: Union[List[float], np.ndarray]):
        """q: [q1, q2, q3, q4, q5, q6] (radian)"""
        super().movej(q)

    def movel(self, matrix: np.ndarray):
        """matrix: 4x4 numpy array (float64)"""
        if not self.is_initialized:
            return None

        if matrix.shape != (4, 4):
            raise ValueError("Input matrix must be a 4x4 numpy array")

        target_tf = ik_solver_py.Transform(matrix.astype(np.float64))

        super().movel(target_tf)

    # -------------------------------------------------------
    # Getter
    # -------------------------------------------------------
    def get_curr_joint_deg(self) -> np.ndarray:
        """현재 조인트 각도 반환 (deg)"""
        return super().get_curr_joint_deg().flatten()

    def get_curr_joint_rad(self) -> np.ndarray:
        """현재 조인트 각도 반환 (rad)"""
        return super().get_curr_joint_rad().flatten()

    def get_curr_joint_velocity_deg(self) -> np.ndarray:
        """현재 조인트 속도 반환 (deg/s)"""
        return super().get_curr_joint_velocity_deg().flatten()

    def get_curr_joint_velocity_rad(self) -> np.ndarray:
        """현재 조인트 속도 반환 (rad/s)"""
        return super().get_curr_joint_velocity_rad().flatten()

    def get_curr_tcp_speed(self) -> np.ndarray:
        """현재 TCP 속도 반환 (m/s)"""
        return super().get_curr_tcp_speed().flatten()

    def get_curr_jacobian(self) -> np.ndarray:
        """6x6 자코비안 행렬 반환"""
        return super().get_curr_jacobian()

    def get_curr_tcp_tf(self) -> np.ndarray:
        """현재 TCP 변환 행렬 반환 (4x4)"""
        return super().get_curr_tcp_tf().matrix()

    def get_curr_tcp_rpy(self) -> np.ndarray:
        """현재 TCP의 RPY 반환 (deg)"""
        rpy_rad = super().get_curr_tcp_tf().rpy()
        return np.degrees(rpy_rad).flatten()

    def get_curr_tf(self, index: int) -> np.ndarray:
        """특정 조인트/링크의 변환 행렬 반환 (4x4)"""
        return super().get_curr_tf(index).matrix()


    # -------------------------------------------------------
    # Setter
    # -------------------------------------------------------
    def set_joint(self, joint_values: Union[List[float], np.ndarray]) -> None:
        """각 Joint를 즉시 설정 [rad]"""
        safe_q = np.array(joint_values, dtype=np.float64).flatten().tolist()
        super().set_joint(safe_q)

    def set_end_effector_offset(self, matrix: np.ndarray) -> None:
        """End-Effector offset 설정 (meters and radians)"""
        if matrix.shape != (4, 4):
            raise ValueError("Input matrix must be a 4x4 numpy array")

        tf = ik_solver_py.Transform.make_tf_from_matrix(matrix)
        super().set_end_effector_offset(tf)

    def set_workspace_limits(self, x_range: Optional[Tuple[float, float]] = None, y_range: Optional[Tuple[float, float]] = None, z_range: Optional[Tuple[float, float]] = None) -> None:
        """작업 영역 제한 설정 (units: meters)x_range: (min, max)"""
        if x_range: super().set_workspace_limitX(x_range[0], x_range[1])
        if y_range: super().set_workspace_limitY(y_range[0], y_range[1])
        if z_range: super().set_workspace_limitZ(z_range[0], z_range[1])

    def set_tcp_max_speed(self, max_speed: float) -> None:
        """TCP 최대 속도 제한 (m/s)"""
        super().set_tcp_max_speed(max_speed)

    def set_joint_limit(self, index: int, min_angle: float, max_angle: float) -> None:
        """조인트 제한 설정 (radian)"""
        super().set_joint_limit(index, min_angle, max_angle)

    def set_joint_velocity_limit(self, index: int, max_speed: float) -> None:
        """조인트 속도 제한 설정 (radian/s)"""
        super().set_joint_velocity_limit(index, max_speed)
