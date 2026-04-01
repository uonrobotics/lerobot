import sys
import os
import time
import numpy as np
np.set_printoptions(precision=6, suppress=True)
import copy
import math
import yaml
import argparse

# ROS 2 관련 임포트
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_msgs.msg import Float32


# ==================================================================
# Ik Solver
# ==================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(current_dir, '../ik_solver/python'))
sys.path.append(lib_dir)
from ik_solver import IkSolver



# ==================================================================
# 두산 로봇 - M10103
# ==================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(current_dir, '../doosan_robot_controller/python'))
sys.path.append(lib_dir)
from doosan_robot_controller import DoosanRobotController


# ==================================================================
# 다이나믹셀 - 3지 그리퍼
# ==================================================================
from dynamixel_sdk import *
class GripperController:
    # --- Control Table Addresses (X-Series) ---
    ADDR_OPERATING_MODE     = 11
    ADDR_MAX_POS_LIMIT      = 48
    ADDR_MIN_POS_LIMIT      = 52
    ADDR_TORQUE_ENABLE      = 64
    ADDR_GOAL_POSITION      = 116
    ADDR_PRESENT_POSITION   = 132
    ADDR_CURRENT_LIMIT      = 38

    # Operating Modes
    CURRENT_BASED_POSITION_MODE = 5

    # --- Parameters ---
    PROTOCOL_VERSION = 2.0
    BAUDRATE         = 115200
    DEVICENAME       = '/dev/ttyUSB1' #
    DXL_ID           = 0

    STROKE_MIN       = 200
    STROKE_MAX       = 2800 + 200
    GRASPING_FORCE   = 160 # Current Limit

    def __init__(self):
        # 포트 및 패킷 핸들러 초기화
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        self.last_error = 0

    def __del__(self):
        # 종료 시 토크 해제 및 포트 닫기
        self._write1Byte(self.ADDR_TORQUE_ENABLE, 0)
        self.portHandler.closePort()

    def init(self):
        # 포트 열기
        if not self.portHandler.openPort():
            print(f"[Error] [Gripper] 포트 연결 실패: {self.DEVICENAME}")
            return False

        # 보드레이트 설정
        if not self.portHandler.setBaudRate(self.BAUDRATE):
            print(f"[Error] [Gripper] 보드레이트 설정 실패: {self.BAUDRATE}")
            return False

        # 모터 ID 확인 (Ping)
        model_number, result, error = self.packetHandler.ping(self.portHandler, self.DXL_ID)
        if result != COMM_SUCCESS:
            print(f"[Error] [Gripper] Ping 실패: ID {self.DXL_ID}, Result {result}")
            return False

        print("[Info] [Gripper] 다이나믹셀 초기화 성공")
        return self._configure_motor()

    def _configure_motor(self):
        # 설정 변경을 위해 토크 끄기
        self._write1Byte(self.ADDR_TORQUE_ENABLE, 0)

        # 운전 모드 설정 (Current-based Position Control)
        self._write1Byte(self.ADDR_OPERATING_MODE, self.CURRENT_BASED_POSITION_MODE)

        # 전류 제한(파워) 및 위치 제한 설정
        self._write2Byte(self.ADDR_CURRENT_LIMIT, self.GRASPING_FORCE)
        self._write4Byte(self.ADDR_MAX_POS_LIMIT, self.STROKE_MAX)
        self._write4Byte(self.ADDR_MIN_POS_LIMIT, self.STROKE_MIN)

        # 토크 켜기
        result = self._write1Byte(self.ADDR_TORQUE_ENABLE, 1)
        if result != COMM_SUCCESS:
            print("[Error] [Gripper] 다이나믹셀 토크 활성화 실패")
            return False

        print(f"[Info ] [Gripper] 설정 완료: 모드 5, Force: {self.GRASPING_FORCE}")
        return True

    def set_position(self, pos:int):
        # Safety Clamping
        pos = max(self.STROKE_MIN, min(pos, self.STROKE_MAX))

        result = self._write4Byte(self.ADDR_GOAL_POSITION, pos)
        if result != COMM_SUCCESS:
            print(f"[Warn ] [Gripper] 위치 제어 실패: {pos}")

    def get_position(self):
        present_pos, result, error = self.packetHandler.read4ByteTxRx(
            self.portHandler, self.DXL_ID, self.ADDR_PRESENT_POSITION)

        if result != COMM_SUCCESS or error != 0:
            return -1
        return present_pos

    def open(self):
        self.set_position(self.STROKE_MAX)

    def close(self):
        self.set_position(self.STROKE_MIN)

    def mapping(self, x, in_min=-0.3, in_max=0.9, out_min=None, out_max=None):
        """그리퍼 제어값 매핑"""
        if out_min is None:
            out_min = self.STROKE_MIN
        if out_max is None:
            out_max = self.STROKE_MAX

        # 매핑 로직 수행
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    # --- Write Wrappers ---
    def _write1Byte(self, address, value):
        res, err = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, address, value)
        self._handle_error(res, err)
        return res

    def _write2Byte(self, address, value):
        res, err = self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID, address, value)
        self._handle_error(res, err)
        return res

    def _write4Byte(self, address, value):
        res, err = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID, address, value)
        self._handle_error(res, err)
        return res

    def _handle_error(self, result, error):
        if result != COMM_SUCCESS:
            print(f"[Error] [Gripper] SDK Error: {self.packetHandler.getTxRxResult(result)}")
        elif error != 0:
            print(f"[Error] [Gripper] Dxl Error: {self.packetHandler.getRxPacketError(error)}")

    def scan(self):
        BAUDRATE_LIST = [9600, 57600, 115200, 1000000, 2000000, 3000000, 4000000]
        print("--- 다이나믹셀 전체 스캔 시작 ---")
        for baud in BAUDRATE_LIST:
            print(f"\n[Baudrate: {baud} bps] 스캔 중...")
            self.portHandler.setBaudRate(baud)
            found = False
            for dxl_id in range(10): # ID 0~9 스캔
                model_number, result, error = self.packetHandler.ping(self.portHandler, dxl_id)
                if result == COMM_SUCCESS:
                    print(f"  >> [발견] ID: {dxl_id:3d} | 모델: {model_number} | Baud: {baud}")
                    found = True
            if not found:
                print("   응답 없음.")
        print("\n--- 스캔 종료 ---")






# ==================================================================
# 멀티 프로세서
# ==================================================================
from multiprocessing import Process, Manager

manager = Manager()
shared = manager.dict(
    {
        'input_joint_rad': np.zeros(6),
        'output_joint_rad': np.zeros(6),
    })

# 서로 다른 작업 함수들
def worker1(shared):
    while True:
        time.sleep(1)




import numpy as np
from multiprocessing import shared_memory, resource_tracker
import atexit
from typing import Dict, Tuple, Any

class SharedMemoryManager:
    def __init__(self, name: str, fields_config: Dict[str, Tuple[int, Any]], create: bool = True):
        self.name = name
        self.fields = {}
        offset = 0
        for f_name, (count, dtype) in fields_config.items():
            item_size = 1 if dtype == str else np.dtype(dtype).itemsize
            byte_size = count * item_size
            self.fields[f_name] = {
                'offset': offset,
                'count': count,
                'dtype': dtype,
                'byte_size': byte_size
            }
            offset += byte_size

        self.total_size = offset

        try:
            # 일단 생성
            self.shm = shared_memory.SharedMemory(name=name, create=create, size=self.total_size)
        except FileExistsError:
            # 이미 있으면 메모리 링크
            self.shm = shared_memory.SharedMemory(name=name)

        # 리소스 트래커에서 제외 (프로세스 종료 시에도 메모리 유지 목적)
        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except:
            pass

        atexit.register(self.close)

    def set(self, field_name: str, value: Any) -> None:
        f = self.fields[field_name]
        if f['dtype'] == str:
            # 문자열을 바이트로 변환 후 지정된 크기만큼만 복사
            encoded = str(value).encode('utf-8')[:f['byte_size']]
            # 버퍼 초기화 (이전 데이터 잔상 제거)
            self.shm.buf[f['offset'] : f['offset'] + f['byte_size']] = b'\x00' * f['byte_size']
            self.shm.buf[f['offset'] : f['offset'] + len(encoded)] = encoded
        else:
            # numpy view를 생성하여 값 복사
            arr = np.ndarray((f['count'],), dtype=f['dtype'], buffer=self.shm.buf, offset=f['offset'])
            arr[:] = value

    def get(self, field_name: str) -> Any:
        f = self.fields[field_name]
        # buf에서 해당 영역만 먼저 bytearray로 추출하여 참조를 최소화
        target_buf = self.shm.buf[f['offset'] : f['offset'] + f['byte_size']]

        if f['dtype'] == str:
            return bytes(target_buf).decode('utf-8', errors='ignore').split('\x00')[0]
        else:
            # frombuffer는 내부적으로 copy를 수행하지 않으므로 .copy() 필수
            return np.frombuffer(target_buf, dtype=f['dtype']).copy()

    def close(self) -> None:
        if hasattr(self, 'shm'):
            self.shm.close()


fields_config = {
    'joint': (6, np.float32), # 24
    'status': (20, str)  # 20바이트 크기 문자열
}
shm = SharedMemoryManager(name='movej', fields_config=fields_config)



# ==================================================================
# DSR 노드: 로봇 제어 & 데이터 퍼블리셔
# ==================================================================
class DsrNode(Node):
    # 퍼블리셔용
    DSR_PUB_JOINT_TOPIC_NAME   = '/robot/dsr/jointStates' # 팔로워암의 joint 값
    DSR_PUB_TF_TOPIC_NAME      = '/robot/dsr/tcp_tf'
    DSR_PUB_GRIPPER_TOPIC_NAME = '/robot/dsr/gripper_val'
    URDF_FILE                  = '/home/uon/workspace/test_realsense3/ik_solver/contents/dsr-m1013.urdf'

    # 서브스크라이버용
    DSR_SUB_JOINT_TOPIC_NAME   = '/leader/joint_states'   # 리더암의 joint값
    DSR_SUB_INFER_JOINT_TOPIC_NAME = '/infer_joint_states' # 추론시 사용
    DSR_SUB_INFER_GRIPPER_TOPIC_NAME = '/infer_gripper'


    def __init__(self, config_path: str = ""):
        super().__init__('dsr_node')
        self._is_init = False
        self.running = False

        # 기본 설정값-
        self.robot = None
        self.robot_ip = '192.168.1.30'
        self.robot_idle_pose = [90.0, -25.0, 120.0, 9.0, 50.0, 0.0]
        self.infer_joint = self.robot_idle_pose
        self.fps = 30
        self.joint_topic = self.DSR_PUB_JOINT_TOPIC_NAME
        self.joint_order = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        # fk 계산 관련
        self.sub_topic_name = self.DSR_SUB_JOINT_TOPIC_NAME
        self.current_tf = np.identity(4)
        self.current_gripper_val = 0.0
        self.XYZ_SCALE = 2.2
        self.is_move_completed = False

        # 그리퍼 관련
        self.gripper = None

        # YAML 설정 파일 로드
        print(f'[Info ] [dsr node] Config 파일 로드: {config_path}')
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            print(f'[Error] [dsr node] Config 파일 로드 실패: {e}')
            raise e

        # 파라미터 파싱
        params = config_data.get('dsr_node', {}).get('ros__parameters', {})
        self.fps = params.get('fps', self.fps)
        self.joint_topic = params.get('joint_topic', self.joint_topic)
        self.robot_ip = params.get('robot_ip', self.robot_ip)
        self.robot_idle_pose = params.get('robot_idle_pose', self.robot_idle_pose)
        self.joint_order = params.get('joint_order', self.joint_order)

        # 제어 관련 변수 초기화
        self.start_time = None
        self.solver = None
        self.solver2 = None

    def joint_T(self, theta, offset_T):
        """조인트 회전 행렬과 오프셋 행렬을 곱함"""
        c = math.cos(theta)
        s = math.sin(theta)

        R_joint = np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

        return np.dot(offset_T, R_joint)

    def calculate_fk(self, joints):
        # Fixed Offsets
        T_offset_01 = np.array([[-1,  0, 0, 0],
                                [ 0, -1, 0, 0],
                                [ 0,  0, 1, 0.1285],
                                [ 0,  0, 0, 1]])

        T_offset_12 = np.array([[1, 0,  0, 0],
                                [0, 0, -1, 0],
                                [0, 1,  0, 0],
                                [0, 0,  0, 1]])

        T_offset_23 = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0.2658],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        T_offset_34 = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0.2169],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        T_offset_45 = np.array([[1, 0,  0, 0],
                                [0, 0, -1, 0],
                                [0, 1,  0, 0.0589],
                                [0, 0,  0, 1]])

        T_offset_56 = np.array([[1,  0, 0, -0.003],
                                [0,  0, 1, 0],
                                [0, -1, 0, 0.05475],
                                [0,  0, 0, 1]])

        T_offset_6e = np.array([[1,  0,  0, 0],
                                [0, -1,  0, 0],
                                [0,  0, -1, -0.16625],
                                [0,  0,  0, 1]])

        # Joint Transform Matrix Calculation
        T_01 = self.joint_T(joints[0] + 1.570796327, T_offset_01)
        T_12 = self.joint_T(joints[1], T_offset_12)
        T_23 = self.joint_T(joints[2], T_offset_23)
        T_34 = self.joint_T(joints[3], T_offset_34)
        T_45 = self.joint_T(-joints[4], T_offset_45)
        T_56 = self.joint_T(joints[5], T_offset_56)

        # T_0e = T_01 @ ... @ T_6e
        T_0e = T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56 @ T_offset_6e

        return T_0e

    def init(self):
        """로봇 연결 및 RT 모드 활성화"""
        self.start_time = time.time()

        # 현제 joint 값을 퍼블리시 (데이터 수집용)
        self.joint_pub = self.create_publisher(
            JointState,
            self.joint_topic,
            1
        )

        # 현제 tcp tf를 퍼블리시 (데이터 수집용)
        self.tf_pub = self.create_publisher(
            Float64MultiArray,
            self.DSR_PUB_TF_TOPIC_NAME,  # 원하는 토픽명
            1
        )

        # 현제 그리퍼 값을 퍼블리시 (데이터 수집용)
        self.gripper_pub = self.create_publisher(
            Float32,
            self.DSR_PUB_GRIPPER_TOPIC_NAME,
            1
        )

        # 추론 결과 그리퍼 값을 서브스크라이브 (인퍼런스용)
        self.infer_gripper_sub = self.create_subscription(
            Float32,
            self.DSR_SUB_INFER_GRIPPER_TOPIC_NAME,
            self.infer_gripper_msg_callback,
            10
        )

        # 추론 결과 joint 값을 서브스크라이브 (인퍼런스용)
        self.infer_joint_sub = self.create_subscription(
            JointState,
            self.DSR_SUB_INFER_JOINT_TOPIC_NAME,
            self.infer_joint_msg_callback,
            10
        )

        # 두산 로봇 연결
        print(f'[Info ] [dsr node] 로봇 연결중: {self.robot_ip}...')
        self.robot = DoosanRobotController(self.robot_ip)
        self.robot.set_kp_gain(3.0)
        self.robot.set_kd_gain(0.00001)
        self.robot.set_target_time(0.04)

        self.robot.connect()
        time.sleep(0.1)
        self.robot.stop()
        time.sleep(0.1)

        # 두산 로봇 서보온
        print(f'[Info ] [dsr node] 서보 on')
        self.robot.servo_on()
        time.sleep(1.5)

        # idle 포즈로 이동
        print(f'[Info ] [dsr node] Idle 포즈 이동')
        self.robot.movej(self.robot_idle_pose, 2.5)

        # 그리퍼 연결 및 열기
        print(f'[Info ] [dsr node] 그리퍼 연결중...')
        self.gripper = GripperController()
        if not self.gripper.init():
            self.gripper.scan()
        self.gripper.open()


        # ik solver 초기화
        if not self._iksolver_init():
            return False
        # idle 포즈로 이동
        self.solver.set_joint(np.radians(self.robot_idle_pose))
        self.solver2.set_joint(np.radians(self.robot_idle_pose))

        # rt 제어 시작
        print(f'[Info ] [dsr node] RT 제어 모드 시작')
        cur_joint = self.solver.get_curr_joint_deg() # tf 추론 버전
        self.robot.start_rt(cur_joint)
        time.sleep(1)

        self.is_move_completed = True
        # --- ---

        self._is_init = True
        self.running = True

        return True

    def infer_gripper_msg_callback(self, msg: Float32):
        self.current_gripper_val = msg.data
        # print(f'[Info ] [dsr node] infer gripper val= {self.current_gripper_val}')
        # print(f'[Info ] [dsr node] infer gripper val= {self.c}')

    def infer_joint_msg_callback(self, msg: JointState):
        self.infer_joint = msg
        # print(f'[Info ] [dsr infer] joint= \n{msg}')

    def publish_joint(self):
        """로봇의 현재 조인트 상태를 ROS 토픽으로 발행 (루프에서 직접 호출)"""
        # current_joints = self.robot.get_current_joint()
        current_joints = self.solver.get_curr_joint_deg();
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_order
        msg.position = [math.radians(j) for j in current_joints]
        self.joint_pub.publish(msg)

    def publish_tf(self):
        """현재 로봇의 TCP TF 행렬(4x4)을 발행"""
        # IK Solver로부터 현재 TCP의 4x4 TF 행렬 획득
        tf_matrix = self.solver.get_curr_tcp_tf()

        msg = Float64MultiArray()
        # 1차원 리스트로 변환 (4x4 -> 16개 요소)
        msg.data = tf_matrix.flatten().tolist()

        # 레이아웃 설정
        msg.layout.dim.append(MultiArrayDimension(label="rows", size=4, stride=16))
        msg.layout.dim.append(MultiArrayDimension(label="cols", size=4, stride=4))

        self.tf_pub.publish(msg)

    def publish_gripper(self):
        """현재 로봇의 그리퍼 값을 발행"""
        # gripper_val = float(self.gripper.mapping(self.current_gripper_val))  # 200 ~ 3000
        gripper_val = float(np.clip(self.current_gripper_val, -0.4, 0.7)) # -0.4 ~ 0.7

        msg = Float32()
        msg.data = gripper_val
        self.gripper_pub.publish(msg)




    def stop(self):
        """시스템 안전 종료"""
        print(f'[Info ] [dsr node] 노드 종료')
        self.running = False
        self.robot.stop()
        self.robot.disconnect()
        self.destroy_node()


    def get_ordered_joint_deg(self, msg):
        """
        msg가 JointState 객체면 이름에 맞춰 정렬 후 Degree 변환,
        단순 리스트(이미 Degree)면 그대로 반환합니다.
        """
        if msg is None:
            return None

        # 1. msg가 ROS 2 JointState 메시지 객체인 경우
        if hasattr(msg, 'name') and hasattr(msg, 'position'):
            joint_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            try:
                # Radian -> Degree 변환
                joint_deg = [math.degrees(joint_map[name]) for name in self.joint_order]
                return joint_deg
            except KeyError as e:
                print(f"[Error] missing joint in msg: {e}")
                return None

        # 2. msg가 단순 리스트인 경우 (초기값 등)
        elif isinstance(msg, list):
            # 초기값 robot_idle_pose 등은 이미 Degree 단위라고 가정
            return msg

        return None

    def move(self):
        if not self.is_move_completed or self.infer_joint is None:
            return

        # 주기 계산
        dt = time.time() - self.start_time
        self.start_time = time.time()

        # 그리퍼 제어
        gripper_pos = int(self.gripper.mapping(self.current_gripper_val))
        # print(f'[Info ] [dsr node] gripper_pos = {gripper_pos}')
        self.gripper.set_position(gripper_pos)

        # 변환 함수 호출
        joint_deg = self.get_ordered_joint_deg(self.infer_joint)


        self.solver.movej(np.radians(joint_deg))


        if joint_deg is not None:
            # 5. RT 제어 명령 수행 (주석 해제)
            pass
            # print(f'[RT] Target: {joint_deg}')
            tf = self.solver.get_curr_tcp_tf()

            self.solver2.movel(tf)
            joint_deg2 = self.solver2.get_curr_joint_deg()

            np_joint_deg = np.array(joint_deg2)

            shm.set('joint', np_joint_deg)
            shm.set('status', 'run')


            self.robot.movej_rt(joint_deg2, dt)
            print(f'dt= {dt}')
        else:
            print("[Warn] 유효하지 않은 조인트 데이터")



    def _iksolver_init(self):
        """Ik Solver 초기화"""
        self.solver = IkSolver(self.URDF_FILE)                     # 솔버 생성
        if not self.solver.init():
            return False


        self.solver.set_tcp_max_speed(2.0)                          # tcp speed
        self.solver.set_joint_limit(2, 1.0, 160)                    # joint3
        self.solver.set_workspace_limits([-1, 1], [-1, 1], [0.255, 1])# workspace

        self.solver2 = IkSolver(self.URDF_FILE)
        if not self.solver2.init():
            return False

        self.solver2.set_tcp_max_speed(2.0)                          # tcp speed
        self.solver2.set_joint_limit(2, 1.0, 160)                    # joint3
        self.solver2.set_workspace_limits([-1, 1], [-1, 1], [0.255, 1])# workspace

        return True

    def _iksolver_move_ready(self):
        """리더암으로 계산된 TCP 위치로 이동 (루프 내부에서 spin_once 수행)"""
        print(f'[Info ] [dsr node] 두산 로봇 리더암 자세로 이동')
        for i in range(1000):
            # 루프 내부에서 콜백이 실행될 수 있도록 spin_once 호출
            rclpy.spin_once(self, timeout_sec=0)

            tf = self.current_tf
            self.solver.movel(tf)
            res = self.solver.get_curr_joint_deg()

            if i % 500 == 0:
                print(f'[Info ] [dsr node] 리더암 자세 계산중... {i}: Position = {tf[0:3, 3]}')

            time.sleep(0.001)

        self.robot.movej(res, 2.5)
        self.is_move_completed = True


def main():
    # ------------------------------------------------------------------
    # 파라미터 파싱
    # python dsr_node.py --config config/dsr.config.yaml
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    default_config = os.path.join(current_dir, '../config/dsr.config.yaml')
    parser.add_argument('--config', type=str, default=default_config)
    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------
    # ros2 초기화
    # ------------------------------------------------------------------
    if not rclpy.ok():
        rclpy.init()

    # ------------------------------------------------------------------
    # 멀티 프로세서 생성
    # ------------------------------------------------------------------
    workers = {}
    workers["movel_loop"] = Process(target=worker1, args=(shared,))

    for name, p in workers.items():
        print(f"[{name}] 프로세서 시작")
        p.start()


    # ------------------------------------------------------------------
    # 두산 로봇 노드 생성
    # ------------------------------------------------------------------
    node = DsrNode(config_path=args.config)

    if not node.init():
        print(f'[Error] [dsr node] 두산 로봇 노드 생성 실패')
        node.stop()
        rclpy.shutdown()
        return

    # ------------------------------------------------------------------
    #  메인 루프
    # ------------------------------------------------------------------
    start_time = time.time()
    try:
        while rclpy.ok():
            # dt = time.time() - start_time
            # start_time = time.time()
            # print(f'dt = {dt:.6f}')
            node.move()
            node.publish_joint()                 # 현재 조인트 퍼블리시
            node.publish_tf()                    # tcp 퍼블리시
            node.publish_gripper()               # 그리퍼 값 퍼블리시
            rclpy.spin_once(node, timeout_sec=0) # 이벤트 처리 (서브스크라이버 콜백 실행 포함)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 종료")

    except Exception as e:
        print(f"실행 중 오류 발생: {e}")

    finally:
        for name, p in workers.items():
            p.terminate()
            p.join()
            print(f"[{name}] 프로세서 종료")

        node.stop()
        rclpy.shutdown()

if __name__ == "__main__":
    main()