import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import rerun as rr

class KinectRerunViewer(Node):
    def __init__(self):
        super().__init__('kinect_rerun_viewer_node')

        # 1. Rerun 초기화 (Viewer 실행)
        # spawn=True는 스크립트 실행 시 Rerun Viewer 창을 자동으로 띄웁니다.
        rr.init("azure_kinect_monitoring", spawn=True)

        # 2. 토픽 구독 설정 (이전 답변에서 생성한 노드의 토픽명과 매칭)
        # 컬러 이미지 구독
        self.color_sub = self.create_subscription(
            CompressedImage,
            '/kinect/camera/cam_top/color/image_rect_raw/compressed',
            self.color_callback,
            10)

        # 뎁스 이미지 구독
        self.depth_sub = self.create_subscription(
            CompressedImage,
            '/kinect/camera/cam_top/depth/image_rect_raw/compressed',
            self.depth_callback,
            10)

        self.get_logger().info("Azure Kinect Rerun Viewer Node started.")

    def color_callback(self, msg):
        """컬러 압축 이미지를 수신하여 Rerun에 로그"""
        try:
            # JPEG/PNG 압축 데이터 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # BGR(OpenCV) -> RGB(Rerun) 변환
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Rerun 타임라인 설정 (ROS2 헤더 스탬프 사용)
                time_sns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
                rr.set_time_nanos("log_time", int(time_sns))

                # Rerun 로그 전송
                rr.log("kinect/color", rr.Image(img_rgb))
        except Exception as e:
            self.get_logger().error(f"Color callback error: {e}")

    def depth_callback(self, msg):
        """뎁스 압축 이미지를 수신하여 Rerun에 로그"""
        try:
            # 뎁스는 보통 PNG 압축을 사용하므로 IMREAD_UNCHANGED로 로드
            np_arr = np.frombuffer(msg.data, np.uint8)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if depth_img is not None:
                # 타임라인 설정
                time_sns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
                rr.set_time_nanos("log_time", int(time_sns))

                # Azure Kinect의 실제 거리값(mm)을 미터 단위 시각화로 매핑
                # meter=1000.0 은 입력 데이터가 mm 단위임을 Rerun에 알림
                rr.log("kinect/depth", rr.DepthImage(depth_img, meter=1000.0))

        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = KinectRerunViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Viewer stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()