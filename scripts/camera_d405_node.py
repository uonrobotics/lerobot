# RealSense D405 영상 데이터 퍼블리시 스크립트 (단일 스레드 버전)

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

# realsense
import pyrealsense2 as rs

class D405Node(Node):
    def __init__(self, config_path: str):
        super().__init__('d405_node')

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
        params = config_data.get('d405_node', {}).get('ros__parameters', {})


        self.serial_no = params.get('serial_no', '')
        self.width = params.get('width', 848)
        self.height = params.get('height', 480)
        self.fps = params.get('fps', 30)
        self.enable_depth = params.get('enable_depth', True)
        self.color_topic = params.get('color_topic', '/robot1/camera/cam_wrist/color/image_rect_raw/compressed')
        self.depth_topic = params.get('depth_topic', '/robot1/camera/cam_wrist/depth/image_rect_raw/compressed')
        self.auto_exp = params.get('auto_exposure', True)
        self.exp_val = params.get('exposure_value', 8000)
        self.auto_wb = params.get('auto_white_balance', True)
        self.wb_val = params.get('white_balance_value', 4500)

    def init(self) -> bool:
        """RealSense 초기화"""
        try:
            self.get_logger().info("RealSense D405 초기화 시도 중...")

            # 파이프라인 및 설정 초기화
            self.pipeline = rs.pipeline()
            config = rs.config()

            self.print_device_info()


            if self.serial_no:
                self.get_logger().info(f"특정 기기 연결 시도: {self.serial_no}")
                config.enable_device(str(self.serial_no))
            else:
                self.get_logger().info("시리얼 번호 미지정: 첫 번째 발견된 기기를 연결합니다.")

            # 스트림 설정
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            if self.enable_depth:
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            # 파이프라인 시작
            profile = self.pipeline.start(config)

            # 센서 옵션 및 퍼블리셔 설정
            self.set_sensor_options(profile)
            self.color_pub = self.create_publisher(CompressedImage, self.color_topic, 10)
            if self.enable_depth:
                self.depth_pub = self.create_publisher(CompressedImage, self.depth_topic, 10)

            self.print_summary()
            self.get_logger().info("RealSense D405 초기화 성공")

            self._is_init = True
            return True

        except Exception as e:
            self.get_logger().error(f"RealSense 초기화 실패: {str(e)}")
            self._is_init = False
            return False

    def run_once(self):
        """메인 루프에서 호출될 단일 프레임 처리 함수"""
        if not self._is_init:
            return

        try:
            # 하드웨어에서 프레임 대기 (타임아웃 설정)
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            if not frames:
                return

            color_frame = frames.get_color_frame()
            if not color_frame:
                return

            # 데이터 변환 및 메시지 생성
            color_img = np.asanyarray(color_frame.get_data())
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = cv2.imencode('.jpg', color_img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

            # 즉시 퍼블리시
            self.color_pub.publish(msg)

            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_img = np.asanyarray(depth_frame.get_data())
                    d_msg = CompressedImage()
                    d_msg.header.stamp = msg.header.stamp
                    d_msg.format = "png"
                    # Depth 데이터를 시각화 가능한 8bit로 변환 (필요에 따라 조정 가능)
                    depth_8bit = cv2.convertScaleAbs(depth_img, alpha=0.03)
                    d_msg.data = cv2.imencode('.png', depth_8bit)[1].tobytes()
                    self.depth_pub.publish(d_msg)

        except Exception as e:
            self.get_logger().error(f"프레임 처리 중 오류 발생: {e}")

    def print_summary(self):
        border = "=" * 50
        summary = (
            f"\n{border}\n"
            f"  D405 SINGLE THREAD PUBLISHER STARTING\n"
            f"{border}\n"
            f"  Resolution    : {self.width}x{self.height} @ {self.fps}fps\n"
            f"  Depth Stream  : {'ENABLED' if self.enable_depth else 'DISABLED'}\n"
            f"  Color Topic   : {self.color_topic}\n"
            f"--------------------------------------------------\n"
            f"  Auto Exposure : {self.auto_exp}\n"
            f"  Auto W.Balance: {self.auto_wb}\n"
            f"{border}"
        )
        self.get_logger().info(summary)

    def set_sensor_options(self, profile):
        sensor = profile.get_device().query_sensors()[0]
        sensor.set_option(rs.option.enable_auto_exposure, 1 if self.auto_exp else 0)
        if not self.auto_exp:
            sensor.set_option(rs.option.exposure, float(self.exp_val))
        sensor.set_option(rs.option.enable_auto_white_balance, 1 if self.auto_wb else 0)
        if not self.auto_wb:
            sensor.set_option(rs.option.white_balance, float(self.wb_val))

    def stop(self):
        self.get_logger().info("D405 Node 종료!")
        try:
            self.pipeline.stop()
        except:
            pass
        self.destroy_node()

    def print_device_info(self):
        ctx = rs.context()
        devices = ctx.query_devices()

        if not devices:
            print("연결된 RealSense 장치가 없습니다.")
            return

        for dev in devices:
            print(f"\n{'='*50}")
            print(f"  Device Name          : {dev.get_info(rs.camera_info.name)}")
            print(f"  Serial Number        : {dev.get_info(rs.camera_info.serial_number)}")
            print(f"  Firmware Version     : {dev.get_info(rs.camera_info.firmware_version)}")
            print(f"  USB Type             : {dev.get_info(rs.camera_info.usb_type_descriptor)}")
            print(f"{'='*50}")

            # 지원하는 스트림 프로파일(해상도, FPS) 확인
            print("\n[지원되는 스트림 모드 리스트]")
            sensors = dev.query_sensors()
            for sensor in sensors:
                print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")
                profiles = sensor.get_stream_profiles()

                # 중복 제거를 위해 set 사용 (선택 사항)
                modes = set()
                for p in profiles:
                    v_p = p.as_video_stream_profile()
                    mode_str = f"  {v_p.stream_name():<10} {v_p.width()}x{v_p.height()} @ {v_p.fps()}fps ({v_p.format()})"
                    modes.add(mode_str)

                for m in sorted(list(modes)):
                    print(m)
            print(f"{'='*50}\n")



def main():
    # ------------------------------------------------------------------
    # 파라미터 파싱
    # python dsr_node.py --config config/d405.config.yaml
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='D405 카메라 노드 실행 스크립트 (단일 스레드)')
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config/d405.config.yaml')
    parser.add_argument('--config', type=str, default=default_config, help='설정 파일(.yaml)의 경로를 입력하세요.')

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
    node = D405Node(config_path)

    if not node.init():
        node.get_logger().error("d405 노드 초기화 실패")
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

    except Exception as e:
        print(f"실행 중 오류 발생: {e}")

    finally:
        node.stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()