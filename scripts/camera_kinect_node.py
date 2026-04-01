import numpy as np
import cv2
import yaml
import os
import sys
import argparse
import time

# ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# azure kinect (pyk4a)
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, ImageFormat, FPS

class AzureKinectNode(Node):
    def __init__(self, config_path: str):
        super().__init__('kinect_node')

        self._is_init = False

        # yaml 파일 불러오기
        self.get_logger().info(f"Config 파일 로드: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Config 파일 로드 실패: {e}")
            raise e

        # 파라미터 파싱
        params = config_data.get('kinect_node', {}).get('ros__parameters', {})

        # Kinect 설정
        self.device_id = params.get('device_id', 0)
        self.color_res_val = params.get('color_resolution', 720)
        self.depth_mode_str = params.get('depth_mode', 'NFOV_UNBINNED')
        self.fps_val = params.get('fps', 30)

        roi_params = params.get('roi', {})
        self.roi_enabled = roi_params.get('enabled', False)
        self.roi_x = roi_params.get('x', 0)
        self.roi_y = roi_params.get('y', 0)
        self.roi_w = roi_params.get('width', 640)
        self.roi_h = roi_params.get('height', 480)

        self.enable_depth = params.get('enable_depth', True)
        self.color_topic = params.get('color_topic', '/kinect/camera/cam_top/color/image_rect_raw/compressed')
        self.depth_topic = params.get('depth_topic', '/kinect/camera/cam_top/depth/image_rect_raw/compressed')

    def _get_color_res(self, res):
        res_map = {
            720: ColorResolution.RES_720P,
            1080: ColorResolution.RES_1080P,
            1440: ColorResolution.RES_1440P,
            2160: ColorResolution.RES_2160P,
        }
        return res_map.get(res, ColorResolution.RES_720P)

    def _get_depth_mode(self, mode_str):
        mode_map = {
            'NFOV_UNBINNED': DepthMode.NFOV_UNBINNED,
            'NFOV_2X2BINNED': DepthMode.NFOV_2X2BINNED,
            'WFOV_UNBINNED': DepthMode.WFOV_UNBINNED,
            'WFOV_2X2BINNED': DepthMode.WFOV_2X2BINNED,
            'OFF': DepthMode.OFF
        }
        return mode_map.get(mode_str, DepthMode.NFOV_UNBINNED)

    def _get_fps(self, fps_val):
        """Kinect SDK 에러 방지를 위한 FPS Enum 매핑"""
        fps_map = {
            5: FPS.FPS_5,
            15: FPS.FPS_15,
            30: FPS.FPS_30
        }
        return fps_map.get(fps_val, FPS.FPS_30)

    def init(self) -> bool:
        """Azure Kinect 초기화"""
        try:
            self.get_logger().info("Azure Kinect 초기화 시도 중...")

            # 하드웨어 설정
            k4a_config = Config(
                color_resolution=self._get_color_res(self.color_res_val),
                depth_mode=DepthMode.OFF,
                camera_fps=self._get_fps(self.fps_val),
                color_format=ImageFormat.COLOR_BGRA32,
                synchronized_images_only=False,
            )

            # PyK4A 인스턴스 생성 및 시작
            self.k4a = PyK4A(device_id=self.device_id, config=k4a_config)
            self.k4a.start()

            # 퍼블리셔 설정
            self.color_pub = self.create_publisher(CompressedImage, self.color_topic, 10)
            if self.enable_depth:
                self.depth_pub = self.create_publisher(CompressedImage, self.depth_topic, 10)

            self.print_summary()
            self.get_logger().info("Azure Kinect 초기화 성공")

            self._is_init = True
            return True

        except Exception as e:
            self.get_logger().error(f"Azure Kinect 초기화 실패: {str(e)}")
            self._is_init = False
            return False

    def run_once(self):
        """메인 루프에서 호출될 단일 프레임 처리 함수"""
        if not self._is_init:
            return

        try:
            # 캡처 데이터 가져오기
            capture = self.k4a.get_capture()

            # Color 이미지 처리
            if capture.color is not None:
                # 1. BGRA -> BGR
                full_color_img = capture.color[:, :, :3]

                # 2. ROI 잘라내기 실행
                color_img = self.crop_roi(full_color_img)

                # 3. 메시지 생성 및 발행
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.format = "jpeg"
                msg.data = cv2.imencode('.jpg', color_img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                self.color_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"프레임 처리 중 오류 발생: {e}")

    def print_summary(self):
        border = "=" * 50
        summary = (
            f"\n{border}\n"
            f"  KINECT SINGLE THREAD PUBLISHER STARTING\n"
            f"{border}\n"
            f"  Device ID     : {self.device_id}\n"
            f"  Resolution (H): {self.color_res_val}p @ {self.fps_val}fps\n"
            f"  Depth Mode    : {self.depth_mode_str}\n"
            f"  Color Topic   : {self.color_topic}\n"
            f"{border}"
        )
        self.get_logger().info(summary)

    def stop(self):
        self.get_logger().info("Kinect Node 종료!")
        try:
            self.k4a.stop()
        except:
            pass
        self.destroy_node()

    def crop_roi(self, frame):
        """이미지에서 설정된 ROI 영역을 잘라내는 함수"""
        if not self.roi_enabled:
            return frame

        # 이미지 크기 확인 (Height, Width)
        h, w = frame.shape[:2]

        # 좌표 계산 (이미지 범위를 벗어나지 않도록 방어 코드 추가)
        x1 = max(0, self.roi_x)
        y1 = max(0, self.roi_y)
        x2 = min(w, x1 + self.roi_w)
        y2 = min(h, y1 + self.roi_h)

        # 슬라이싱으로 이미지 절단 [y영역, x영역]
        cropped_img = frame[y1:y2, x1:x2]
        return cropped_img


def main():
    # ------------------------------------------------------------------
    # 파라미터 파싱
    # python dsr_node.py --config config/d405.config.yaml
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Azure Kinect 노드 실행 스크립트')
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config/kinect.config.yaml')
    parser.add_argument('--config', type=str, default=default_config, help='설정 파일(.yaml)의 경로')

    args, unknown = parser.parse_known_args()
    config_path = args.config


    # ------------------------------------------------------------------
    # ros2 초기화
    # ------------------------------------------------------------------
    if not rclpy.ok():
        rclpy.init()


    # ------------------------------------------------------------------
    # d405 노드 생성
    # ------------------------------------------------------------------
    node = AzureKinectNode(config_path)
    if not node.init():
        return


    # ------------------------------------------------------------------
    # 메인루프
    # ------------------------------------------------------------------
    try:
        while rclpy.ok():
            node.run_once()
            rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 종료")
    finally:
        node.stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()