import os
import sys
import numpy as np
import atexit
from typing import Any, Optional, Union, List, Tuple, Dict
from multiprocessing import shared_memory, resource_tracker

# ------------------------------------------------------------------
# Ik Solver 모듈 불러오기
# .so 파일이 있는 디렉토리를 sys.path.append로 추가
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import ik_solver_py
    IK_AVAILABLE = True
except ImportError as e:
    IK_AVAILABLE = False
    print(f'Python Ik Solver [Error] ik_solver 모듈을 찾을 수 없음: {e}')

# ------------------------------------------------------------------
# 공유 메모리 클래스
# ------------------------------------------------------------------
class SharedMemory:
    def __init__(self, name: str, fields_config: Dict[str, Tuple[int, Any]]):
        self.name = name
        self.fields = {}
        offset = 0
        for f_name, (count, dtype) in fields_config.items():
            item_size = 1 if dtype == str else np.dtype(dtype).itemsize
            byte_size = count * item_size
            self.fields[f_name] = {'offset': offset, 'count': count, 'dtype': dtype, 'byte_size': byte_size}
            offset += byte_size
        self.total_size = offset

        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.total_size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name)

        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except: pass
        atexit.register(self.close)

    def set(self, field_name: str, value: Any) -> None:
        f = self.fields[field_name]
        if f['dtype'] == str:
            encoded = str(value).encode('utf-8')[:f['byte_size']]
            self.shm.buf[f['offset'] : f['offset'] + len(encoded)] = encoded
        else:
            arr = np.ndarray((f['count'],), dtype=f['dtype'], buffer=self.shm.buf, offset=f['offset'])
            arr[:] = value

    def close(self) -> None:
        if hasattr(self, 'shm'): self.shm.close()

# ------------------------------------------------------------------
# Ik Solver 상위 래퍼 클래스
# ------------------------------------------------------------------
class ik_solver:
    def __init__(self, urdf_path: str):
        if not IK_AVAILABLE:
            raise RuntimeError("ik_solver_py module is not available.")
            
        self.solver = ik_solver_py.IkSolver(urdf_path)
        self.is_initialized = False

    def init(self) -> bool:
        if self.solver.init():
            # 기본 제한 설정 (example.py에 있던 기본값들)
            self.solver.set_tcp_speed_limit(1.0)

            # 조인트 범위 제한 (커스텀 할경우 주석을 해제 하고 사용하세요)
            # self.solver.set_joint_limit(0, np.radians(-360), np.radians(360))
            # self.solver.set_joint_limit(1, np.radians(-360), np.radians(360))
            # self.solver.set_joint_limit(2, np.radians(-160), np.radians(160))
            # self.solver.set_joint_limit(3, np.radians(-360), np.radians(360))
            # self.solver.set_joint_limit(4, np.radians(-360), np.radians(360))
            # self.solver.set_joint_limit(5, np.radians(-360), np.radians(360))


            # 조인트 속도 제한 (rad/s) (커스텀 할경우 주석을 해제 하고 사용하세요)
            # self.solver.set_joint_vlimit(0, np.radians(120))
            # self.solver.set_joint_vlimit(1, np.radians(120))
            # self.solver.set_joint_vlimit(2, np.radians(180))
            # self.solver.set_joint_vlimit(3, np.radians(255))
            # self.solver.set_joint_vlimit(4, np.radians(255))
            # self.solver.set_joint_vlimit(5, np.radians(255))
            
            # 안전 계수
            self.solver.set_safety_scale(1.2)
            
            self.is_initialized = True
            return True
        return False

    def movej(self, q: Union[List[float], np.ndarray]):
        """
        q: [q1, q2, q3, q4, q5, q6] (radian)
        """
        self.solver.movej(q)

    def movel(self, pose: Union[List[float], np.ndarray]):
        """
        pose: [x, y, z, r, p, y] (meters and rad)
        """
        if not self.is_initialized:
            return None
            
        # Transform 생성
        target_tf = ik_solver_py.Transform.make_tf(
            pose[0], pose[1], pose[2],  # position x, y, z [m]
            pose[3], pose[4], pose[5] # rotation roll, pitch, yaw [rad]
        )
        
        # 이동 (내부적으로 IK 계산 및 상태 업데이트)
        self.solver.movel(target_tf)


    def get_current_joints(self) -> np.ndarray:
        """현재 조인트 각도 반환 (deg)"""
        return self.solver.get_curr_joint_deg().flatten()

    def get_current_joints_rad(self) -> np.ndarray:
        """현재 조인트 각도 반환 (rad)"""
        return self.solver.get_curr_joint_rad().flatten()

    def get_current_jvel(self) -> np.ndarray:
        """현재 조인트 속도 반환 (deg/s)"""
        return self.solver.get_curr_jvel_deg().flatten()

    def get_current_jvel_rad(self) -> np.ndarray:
        """현재 조인트 속도 반환 (rad/s)"""
        return self.solver.get_curr_jvel_rad().flatten()

    def get_current_tcp_speed(self) -> np.ndarray:
        """현재 TCP 속도 반환 (m/s)"""
        return self.solver.get_curr_tcp_speed().flatten()

    def get_jacobian(self) -> np.ndarray:
        """6x6 자코비안 행렬 반환"""
        return self.solver.get_jacobian()

    def get_tcp_tf(self) -> np.ndarray:
        """현재 TCP 변환 행렬 반환 (4x4)"""
        return self.solver.get_tcp_tf().matrix()

    def get_tcp_rpy(self) -> np.ndarray:
        """현재 TCP의 RPY 반환 (deg)"""
        rpy_rad = self.solver.get_tcp_tf().rpy()
        return np.degrees(rpy_rad).flatten()

    def get_tf(self, index: int) -> np.ndarray:
        """특정 조인트/링크의 변환 행렬 반환 (4x4)"""
        return self.solver.get_tf(index).matrix()

    # ------------------------------------------------------------------
    # End-Effector 설정
    # ------------------------------------------------------------------
    def set_end_effector(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        """
        End-Effector offset 설정 (meters and radians)
        """
        tf = ik_solver_py.Transform.make_tf(
            x, y, z,
            roll, pitch, yaw
        )
        self.solver.set_end_effector_tf(tf)

    def remove_end_effector(self) -> None:
        """End-Effector offset 제거"""
        self.solver.remove_end_effector()

    def has_end_effector(self) -> bool:
        """EE 설정 여부 확인"""
        return self.solver.has_end_effector()

    # ------------------------------------------------------------------
    # 모드 및 제한 설정
    # ------------------------------------------------------------------
    def set_strict_mode(self) -> None:
        """경로 정확도 우선 모드"""
        self.solver.set_strict_mode()

    def set_relax_mode(self) -> None:
        """조인트 속도 우선 모드 (싱귤래리티 대응)"""
        self.solver.set_relax_mode()

    def set_workspace_limits(self, x_range: Optional[Tuple[float, float]] = None, y_range: Optional[Tuple[float, float]] = None, z_range: Optional[Tuple[float, float]] = None) -> None:
        """
        작업 영역 제한 설정 (units: meters)
        x_range: (min, max)
        """
        if x_range: self.solver.set_workspace_limitX(x_range[0], x_range[1])
        if y_range: self.solver.set_workspace_limitY(y_range[0], y_range[1])
        if z_range: self.solver.set_workspace_limitZ(z_range[0], z_range[1])

    def set_tcp_max_speed(self, max_speed: float) -> None:
        """TCP 최대 속도 제한 (m/s)"""
        self.solver.set_tcp_max_speed(max_speed)

    def set_joint_limit(self, index: int, min_angle: float, max_angle: float) -> None:
        """조인트 제한 설정 (radian)"""
        self.solver.set_joint_limit(index, min_angle, max_angle)

    def set_joint_vlimit(self, index: int, max_speed: float) -> None:
        """조인트 속도 제한 설정 (radian/s)"""
        self.solver.set_joint_vlimit(index, max_speed)
