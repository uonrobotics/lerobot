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
    DSR_SUB_INFER_TF_TOPIC_NAME = '/infer_tf_matrix' # 추론시 사용
    DSR_SUB_INFER_GRIPPER_TOPIC_NAME = '/infer_gripper'


    def __init__(self, config_path: str = ""):
        super().__init__('dsr_node')
        self._is_init = False
        self.running = False

        # 기본 설정값
        self.robot = None
        self.robot_ip = '192.168.1.30'
        self.robot_idle_pose = [90.0, -25.0, 120.0, 9.0, 50.0, 0.0]
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

        # 퍼플리셔 생성: 두산 로봇 조인트 값
        self.joint_pub = self.create_publisher(
            JointState,
            self.joint_topic,
            1
        )

        # 서브스크라이버 생성: 리더암의 조인트 값을 받음
        self.leader_joint_sub = self.create_subscription(
            JointState,
            self.sub_topic_name,
            self.joint_msg_callback,
            10
        )

        # tcp tf 퍼블리셔
        self.tf_pub = self.create_publisher(
            Float64MultiArray,
            self.DSR_PUB_TF_TOPIC_NAME,  # 원하는 토픽명
            1
        )

        # 그리퍼 값 퍼플리셔
        self.gripper_pub = self.create_publisher(
            Float32,
            self.DSR_PUB_GRIPPER_TOPIC_NAME,
            1
        )

        # 두산 로봇 연결
        print(f'[Info ] [dsr node] 로봇 연결중: {self.robot_ip}...')
        self.robot = DoosanRobotController(self.robot_ip)
        self.robot.connect()
        self.robot.stop()
        time.sleep(1)

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
        self.solver.set_joint(np.radians(self.robot_idle_pose))

        # 토픽 대기
        print(f'[Info ] [dsr node] 리더암 데이터 기다리는 중...')
        start_wait = time.time()
        while rclpy.ok() and np.array_equal(self.current_tf, np.identity(4)):
            rclpy.spin_once(self, timeout_sec=0.0)
            if time.time() - start_wait > 5.0: # 5초 타임아웃
                print(f'[Error] [dsr node] 리더암 토픽 수신 타임아웃!')
                return False

        # 리더암 위치로 이동
        print(f'[Info ] [dsr node] 리더암 위치로 이동중...')
        self._iksolver_move_ready()

        # rt 제어 시작
        print(f'[Info ] [dsr node] RT 제어 모드 시작')
        cur_joint = self.solver.get_curr_joint_deg()
        self.robot.start_rt(cur_joint)
        time.sleep(1)
        # --- ---

        self._is_init = True
        self.running = True

        return True


    def joint_msg_callback(self, msg: JointState):
        # 데이터 개수 확인 (6축 + 그리퍼 1개 = 최소 7개)
        if len(msg.position) < 7:
            print(f'[Error] [dsr node] JointStates 개수가 7개가 아님: {len(msg.position)}')
            return

        # JointState에서 추출 (상위 6개는 암, 7번째는 그리퍼)
        arm_joints = list(msg.position[:6])

        # FK 계산
        T_result = self.calculate_fk(arm_joints)

        # 스케일링 적용 (Translation 부분에만 적용)
        T_result[0, 3] *= self.XYZ_SCALE
        T_result[1, 3] *= self.XYZ_SCALE
        T_result[2, 3] *= self.XYZ_SCALE

        self.current_tf = T_result
        self.current_gripper_val = msg.position[6]




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
        gripper_val = float(self.current_gripper_val)# -0.4 ~ 0.7
        # print(f'[Info ] [dsr node] gripper_val = {gripper_val}')

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

    def move(self):
        if not self.is_move_completed:
            return


        dt = time.time() - self.start_time
        self.start_time = time.time()

        # rt 제어
        self.solver.movel(self.current_tf)
        joint = self.solver.get_curr_joint_deg() # deg
        curr_tf = self.solver.get_curr_tcp_tf()
        gripper_val = int(self.gripper.mapping(self.current_gripper_val))

        # print(f"dt = {dt}")
        # print(f"tf = {curr_tf}")

        # 그리퍼와 로봇 제어
        self.gripper.set_position(gripper_val)
        self.robot.movej_rt(joint, dt)


    def _iksolver_init(self):
        """Ik Solver 초기화"""
        self.solver = IkSolver(self.URDF_FILE)                     # 솔버 생성
        if not self.solver.init():
            return False

        self.solver.set_tcp_max_speed(2.0)                          # tcp speed
        self.solver.set_joint_limit(2, 1.0, 160)                    # joint3
        self.solver.set_workspace_limits([-1, 1], [0.16, 0.840], [0.255, 1])# workspace

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
        node.stop()
        rclpy.shutdown()

if __name__ == "__main__":
    main()