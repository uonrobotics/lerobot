import threading
import time
import numpy as np

# ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Communicator
class Communicator(Node):
    def __init__(self):
        super().__init__('Communicator_node')
        self.topic_msgs = {
            'cam_wrist': None,
            'cam_top': None,
            'leader': None,
            'follower': None
        }

        # 리더암 토픽 퍼블리셔 (팔로우암이 이 토픽을 읽어서 움직임)
        self.joint_pub = self.create_publisher(
            JointTrajectory,
            '/right_robot/leader/joint_trajectory',
            100
        )

        # 손목 카메라 토픽
        self.sub_1 = self.create_subscription(
            CompressedImage,
            '/right/camera/cam_wrist/color/image_rect_raw/compressed',
            self.callback_topic_1,
            10
        )

        # 키넥트 토픽
        self.sub_2 = self.create_subscription(
            CompressedImage,
            '/right/camera/cam_top/color/image_rect_raw/compressed',
            self.callback_topic_2,
            10
        )

        # 리더암
        self.sub3 = self.create_subscription(
            JointTrajectory,
            '/right_robot/leader/joint_trajectory',
            self.callback_topic_3,
            10
        )

        # 팔로우암
        self.sub4 = self.create_subscription(
            JointState,
            '/right/joint_states',
            self.callback_topic_4,
            10
        )


    def callback_topic_1(self, msg: CompressedImage):
        self.topic_msgs['cam_wrist'] = msg
        # print(f'Received from Topic 1: "{msg.data}"')

    def callback_topic_2(self, msg: CompressedImage):
        self.topic_msgs['cam_top'] = msg
        # print(f'Received from Topic 2: "{msg.data}"')

    def callback_topic_3(self, msg: JointTrajectory):
        self.topic_msgs['leader'] = msg
        # print(f'Received from Topic 3: "{msg.data}"')

    def callback_topic_4(self, msg: JointState):
        self.topic_msgs['follower'] = msg
        # print(f'Received from Topic 4: "{msg.data}"')

    def action_publish(self, action: np.ndarray):
        # ACT 모델은 길게 예측을 하는데 앞에 것만 사용함
        action = action[0:7]

        # 트레젝토리 설정
        joint_msg = JointTrajectory()
        joint_msg.joint_names = [
            'right_joint1',
            'right_joint2',
            'right_joint3',
            'right_joint4',
            'right_joint5',
            'right_joint6',
            'right_rh_r1_joint'
        ]
        joint_msg.points = [JointTrajectoryPoint(positions=action)]

        # 퍼블리시
        self.joint_pub.publish(joint_msg)

    def get_latest_msgs(self):
        return self.topic_msgs

    def start(self):
        """노드 비동기 실행(non blocking)"""
        def _node_run():
            if not rclpy.ok():
                print(f'[Error] rclpy 초기화 필요')
                return
            rclpy.spin(self)

        thread = threading.Thread(target=_node_run, daemon=True)
        thread.start()
        print(f'[Info ] 토픽 서브스크라이버 시작')






def main(args=None):
    rclpy.init(args=args)
    node = Communicator()
    node.start()


    try:
        while True:
            data = node.get_latest_msgs()
            print(f'data: {data['leader']}')
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()