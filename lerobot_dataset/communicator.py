import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, Int32, Float32
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data

class Communicator(Node):
    def __init__(self, config_path='config/comm.config.yaml', shared_data=None, send_queue=None):
        super().__init__('comm_node')
        self.sub_cb_group = ReentrantCallbackGroup()
        self.config = self._load_config(config_path)
        self.latest_msgs = shared_data
        self.send_queue = send_queue

        self.subscribers_list = {}
        self.publishers_list = {}

        self.type_map = {
            "CompressedImage": CompressedImage,
            "JointState": JointState,
            "JointTrajectory": JointTrajectory,
            "Bool": Bool,
            "Float64MultiArray": Float64MultiArray,
            "Int32": Int32,
            "Float32": Float32
        }

    def _load_config(self, path):
        """설정 파일 로드 및 경로 확인 로거"""
        if not os.path.exists(path):
            self.get_logger().error(f'Config 파일을 찾을 수 없음: {os.path.abspath(path)}')
            return {}

        with open(path, 'r') as f:
            conf = yaml.safe_load(f)
            self.get_logger().info(f"설정 파일 로드 성공: {path}")
            return conf

    def init(self):
        """커뮤니케이터 초기화 (로거 포함)"""
        if not self.config:
            return False

        # --- 서브스크라이버 자동 생성 ---
        if 'subscribers' in self.config:
            self._init_subscribers()

        # --- 퍼블리셔 자동 생성 ---
        if 'publishers' in self.config:
            self._init_publishers()
            # 메인 프로세스에서 보낸 데이터를 퍼블리시하기 위한 타이머
            if self.send_queue is not None:
                self.create_timer(0.01, self._check_queue_callback)
                self.get_logger().info("메시지 전송 큐 타이머 시작 (100Hz)")

        self.get_logger().info("=== Communicator 모든 초기화 완료 ===")
        return True

    def _init_subscribers(self):
        """서브스크라이버 생성 및 정보 출력"""
        sub_configs = self.config.get('subscribers', {})
        for key, info in sub_configs.items():
            topic = info['topic']
            msg_type_str = info['msg_type']
            msg_class = self.type_map.get(msg_type_str)

            if msg_class:
                self.subscribers_list[key] = self.create_subscription(
                    msg_class,
                    topic,
                    self._make_callback(key),
                    qos_profile_sensor_data,
                    callback_group=self.sub_cb_group
                )

                self.get_logger().info(f"서브스크라이버 생성: {key} -> {topic} ({msg_type_str})")
            else:
                self.get_logger().error(f"지원되지 않는 메시지 타입: {msg_type_str} (key: {key})")

    def _init_publishers(self):
        """퍼블리셔 생성 및 정보 출력"""
        pub_configs = self.config.get('publishers', {})
        for key, info in pub_configs.items():
            if not isinstance(info, dict): continue

            topic = info.get('topic')
            msg_type_str = info.get('msg_type')
            msg_class = self.type_map.get(msg_type_str)

            if msg_class:
                self.publishers_list[key] = self.create_publisher(
                    msg_class,
                    topic,
                    10
                )

                self.get_logger().info(f"퍼블리셔 생성: {key} -> {topic} ({msg_type_str})")
            else:
                self.get_logger().error(f"지원되지 않는 메시지 타입: {msg_type_str} (key: {key})")

    def _make_callback(self, key):
        def callback(msg):
            if self.latest_msgs is not None:
                self.latest_msgs[key] = msg
        return callback

    def _check_queue_callback(self):
        """큐 모니터링 및 실제 퍼블리시 수행"""
        while self.send_queue and not self.send_queue.empty():
            try:
                key, msg = self.send_queue.get_nowait()
                if key in self.publishers_list:
                    self.publishers_list[key].publish(msg)
                    # self.get_logger().debug(f"데이터 퍼블리시 완료: {key}")
            except Exception:
                break

    def publish(self, key, msg):
        if key in self.publishers_list:
            self.publishers_list[key].publish(msg)