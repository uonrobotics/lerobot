import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import rerun as rr  # Rerun 라이브러리 추가

class D405RerunViewer(Node):
    def __init__(self):
        super().__init__('d405_rerun_viewer_node')

        # 1. Rerun 초기화 (Viewer 실행)
        rr.init("d405_monitoring", spawn=True)

        # 컬러 및 뎁스 압축 토픽 구독
        self.color_sub = self.create_subscription(
            CompressedImage,
            '/camera/cam_wrist/realsensc/color/compressed',
            self.color_callback,
            10)

        self.depth_sub = self.create_subscription(
            CompressedImage,
            'camera/depth/image_raw/compressed',
            self.depth_callback,
            10)

        self.get_logger().info("D405 Rerun Viewer Node started.")

    def color_callback(self, msg):
        try:
            # 1. 압축 데이터 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # 2. BGR -> RGB 변환 (Rerun은 RGB 사용)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 3. Rerun 로그 전송
                rr.log("camera/color", rr.Image(img_rgb))
        except Exception as e:
            self.get_logger().error(f"Color callback error: {e}")

    def depth_callback(self, msg):
        try:
            # 1. 압축 데이터 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if depth_img is not None:
                # 2. Rerun에 Depth 이미지 전송 (자동으로 시각화 가능)
                rr.log("camera/depth", rr.DepthImage(depth_img, meter=1.0))
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = D405RerunViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()