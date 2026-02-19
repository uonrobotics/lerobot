#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""UON AMR robot driver with Orbbec SDK Integration."""

import logging
import time
import threading
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict

import numpy as np
import serial
import cv2 

# Camera webstream
from flask import Flask, Response

# LeRobot Imports
from lerobot.cameras import CameraConfig
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .uon_amr_driver import UONAMRDriver

# --- Orbbec SDK Import ---
try:
    from pyorbbecsdk import (
        Pipeline, Config, OBSensorType, OBFormat, OBStreamType, 
        OBFrameAggregateOutputMode, AlignFilter, OBPropertyID
    )
    ORBBEC_AVAILABLE = True
except ImportError:
    ORBBEC_AVAILABLE = False
    print("❌ pyorbbecsdk not installed. Camera will fail.")

logger = logging.getLogger(__name__)

# --- Standardized Feature Names ---
ACTION_LINEAR_VEL = "linear.vel"
ACTION_ANGULAR_VEL = "angular.vel"

OBS_FRONT_RGB = "front_rgb"
OBS_FRONT_DEPTH = "front_depth"        # Visualization (Color Heatmap)
OBS_FRONT_DEPTH_RAW = "front_depth_raw" # Actual Data (16-bit encoded)

OBS_LINEAR_VEL = "linear.vel"
OBS_ANGULAR_VEL = "angular.vel"
OBS_BATTERY_LEVEL = "battery.level"

LIDAR_KEYS = [f"scan.lidar_{i}" for i in range(360)]

try:
    from rplidar import RPLidar
    LIDAR_AVAILABLE = True
except ImportError:
    LIDAR_AVAILABLE = False
    logger.warning("RPLidar library not found. LiDAR features will be disabled.")

# ==============================================================================
# 1. Threaded Orbbec Driver
# ==============================================================================
class ThreadedOrbbec:
    def __init__(self, width=640, height=480, min_mm=200, max_mm=3000):
        self.width = width
        self.height = height
        self.min_mm = min_mm
        self.max_mm = max_mm
        
        self.pipeline = None
        self.align_filter = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Flask setup
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Placeholders for data
        self.latest_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.latest_depth_vis = np.zeros((height, width, 3), dtype=np.uint8)
        self.latest_depth_raw = np.zeros((height, width), dtype=np.uint16)

    def start(self):
        if not ORBBEC_AVAILABLE:
            logger.error("❌ Orbbec SDK missing.")
            return

        try:
            self.pipeline = Pipeline()
            config = Config()
            
            # 1. Configure RGB
            try:
                profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile = profiles.get_video_stream_profile(self.width, self.height, OBFormat.RGB, 30)
                config.enable_stream(color_profile)
            except Exception as e:
                logger.error(f"❌ Failed to config RGB: {e}")
                return

            # 2. Configure Depth
            try:
                # Use native resolution for Depth (usually 640x400)
                profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                try:
                    depth_profile = profiles.get_video_stream_profile(640, 400, OBFormat.Y16, 30)
                except:
                    depth_profile = profiles.get_video_stream_profile(self.width, self.height, OBFormat.Y16, 30)
                config.enable_stream(depth_profile)
            except Exception as e:
                logger.error(f"❌ Failed to config Depth: {e}")
                return

            # 3. Enable Align (D2C)
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            
            # 4. Start
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            self.pipeline.enable_frame_sync()
            self.pipeline.start(config)
            
            device = self.pipeline.get_device()
            
            #### Apply Camera Settings ####
            try:
                # 1. Anti-Flicker (Fixed)
                device.set_int_property(OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, 2)
                
                # 2. Exposure (Fixed at flicker-free value)
                device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
                device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 80)
                
                # 3. FIX THE WHITENESS: Lower the Gain
                # Try a value between 10 and 30. Higher = brighter/noisier.
                device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 1) 
                
                logger.info("✅ [Orbbec] Flicker gone and brightness adjusted via Gain.")
                
            except Exception as e:
                logger.error(f"❌ Failed to apply Orbbec settings: {e}")
            #### Apply Camera Settings ####
                
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("✅ [Orbbec] SDK Pipeline Started")
            
            # Start Flask thread on port 5000
            self.flask_thread = threading.Thread(
                target=lambda: self.app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False),
                daemon=True
            )
            self.flask_thread.start()
            logger.info("✅ [Orbbec] POV Stream started at http://<robot_ip>:5000")
                
        except Exception as e:
            logger.error(f"❌ [Orbbec] Init failed: {e}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.pipeline:
            try:
                self.pipeline.stop()
            except: pass

    def _run(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(50)
                if frames is None: 
                    continue

                if self.align_filter:
                    frames = self.align_filter.process(frames)
                    if not frames: continue
                    frames = frames.as_frame_set()

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame and depth_frame:
                    # 1. Raw 데이터 추출 (최소한의 연산)
                    w, h = color_frame.get_width(), color_frame.get_height()
                    rgb = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape((h, w, 3))
                    
                    scale = float(depth_frame.get_depth_scale())
                    depth_u16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((depth_frame.get_height(), depth_frame.get_width()))

                    # 2. 크기 조정 (필요한 경우에만 수행, NEAREST 보간법으로 CPU 절약)
                    if h != self.height or w != self.width:
                        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                        depth_raw = cv2.resize(depth_u16, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    else:
                        depth_raw = depth_u16

                    # 3. 시각화용 데이터(Depth Vis)는 락 외부에서 필요할 때만 생성하거나 
                    # 부하가 크면 주기를 조절 (여기서는 최적화하여 유지)
                    clipped = np.clip(depth_raw.astype(np.int32), self.min_mm, self.max_mm).astype(np.uint16)
                    depth_8u = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

                    with self.lock:
                        self.latest_rgb = rgb
                        self.latest_depth_vis = depth_vis
                        self.latest_depth_raw = depth_raw

            except Exception as e:
                logger.error(f"❌ [Orbbec] Run loop error: {e}")
                time.sleep(0.1)

    def get_data(self):
        with self.lock:
            return (
                self.latest_rgb.copy(), 
                self.latest_depth_vis.copy(),
                self.latest_depth_raw.copy()
            )

    # --- Helpers ---
    def _process_color_frame(self, color_frame) -> np.ndarray:
        w = color_frame.get_width()
        h = color_frame.get_height()
        fmt = color_frame.get_format()
        buf = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

        if fmt == OBFormat.RGB:
            # SDK gives RGB, return as RGB (No conversion needed)
            return buf.reshape((h, w, 3))
        elif fmt == OBFormat.MJPG:
            # MJPG decodes to BGR, convert to RGB
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _process_depth_frame(self, depth_frame, min_mm, max_mm):
        w = depth_frame.get_width()
        h = depth_frame.get_height()
        
        scale = float(depth_frame.get_depth_scale())
        depth_u16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))

        if scale < 0.01:
            depth_mm = (depth_u16.astype(np.float32) * scale * 1000.0)
        else:
            depth_mm = (depth_u16.astype(np.float32) * scale)

        depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

        # Visualization
        # 1. Normalize
        invalid = (depth_mm == 0) | (depth_mm == 65535)
        clipped = np.clip(depth_mm.astype(np.int32), min_mm, max_mm).astype(np.uint16)
        depth_8u = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_8u[invalid] = 0
        
        # 2. Apply Colormap (Returns BGR)
        depth_vis_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        
        # 3. Convert BGR -> RGB (Fixes the "Vice Versa" Blue/Red issue)
        depth_vis_rgb = cv2.cvtColor(depth_vis_bgr, cv2.COLOR_BGR2RGB)

        return depth_mm, depth_vis_rgb
    
    # --- Flask Routes ---
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return "<h1>UON AMR - Remote POV</h1><img src='/video_feed' width='640'>"

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate_stream(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
            
    def _generate_stream(self):
        # 🟢 최적화 1: 10 FPS로 제한 (사람이 조종하기에 충분한 속도)
        stream_delay = 1.0 / 10.0 
        last_frame_time = time.time()

        while self.running:
            current_time = time.time()
            if current_time - last_frame_time < stream_delay:
                time.sleep(0.01) # CPU 휴식
                continue
            
            last_frame_time = current_time

            # 🟢 최적화 2: 락 안에서는 '복사'만 수행하여 LiDAR 루프 방해 차단
            with self.lock:
                frame_to_process = self.latest_rgb.copy()
            
            # 🟢 최적화 3: 해상도를 더 낮춤 (240x180) 및 가장 빠른 보간법(NEAREST) 사용
            small_frame = cv2.resize(frame_to_process, (240, 180), interpolation=cv2.INTER_NEAREST)
            frame_bgr = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)
            
            # 🟢 최적화 4: JPEG 품질을 15로 낮춤 (노이즈는 늘지만 CPU 사용량 급감)
            ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 15])
            
            if not ret: continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==============================================================================
# 2. Other Drivers
# ==============================================================================
class ThreadedLidar:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.lidar = None
        self.latest_scan = np.zeros(360, dtype=np.float32)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        try:
            self.lidar = RPLidar(self.port, baudrate=self.baudrate, timeout=3)
            # Essential for S2: Force a clean start state
            self.lidar.stop()
            self.lidar.start_motor()
            
            time.sleep(1.0) 
            
            self.lidar.clean_input()
            
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("✅ [Lidar] Thread started")
            
        except Exception as e: 
            logger.error(f"❌ [Lidar] Start failed: {e}")

    def stop(self):
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
            except: 
                pass

    def _run(self):
        """Your original logic, wrapped for resilience."""
        while self.running:
            try:
                # max_buf_meas를 늘려 CPU가 바쁠 때 버퍼링 허용
                # S2는 초당 최대 32,000포트를 쏘기 때문에 버퍼가 커야 합니다.
                for scan in self.lidar.iter_scans(max_buf_meas=4000):
                    if not self.running: 
                        break
                    
                    temp = np.zeros(360, dtype=np.float32)
                    for (_, angle, distance) in scan: 
                        # 각도 정밀도를 정수로 매핑 (데이터 품질 유지)
                        temp[min(359, int(angle))] = distance
                        
                    with self.lock: 
                        self.latest_scan = temp

            except Exception as e:
                if self.running:
                    logger.warning(f"⚠️ [Lidar] Sync lost: {e}. Rapid recovery...")
                    try:
                        self.lidar.clean_input() 
                    except: 
                        pass
                continue

    def get_scan(self):
        with self.lock:
            return self.latest_scan.copy()

class UONBatteryDriver:
    def __init__(self, port, baudrate=19200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.soc = 0.0
        self.voltage = 0.0
        self.ampere = 0.0
        self.has_printed_banner = False
        self.req_msg = bytearray([0xAF, 0xFA, 0x60, 0x05, 0x01, 0x60, 0x7F, 0x07, 0x00, 0xAF, 0xA0])
        self.req_msg[8] = sum(self.req_msg[2:8]) & 0xFF

    def start(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info(f"✅ [Battery] Connected to {self.port}")
            
        except Exception as e: 
            logger.warning(f"⚠️ [Battery] Connection failed: {e}")

    def stop(self):
        self.running = False
        
        if self.thread: 
            self.thread.join(timeout=1.0)
            
        if self.ser:
            self.ser.close()

    def _run(self):
        while self.running and self.ser:
            try:
                self.ser.reset_input_buffer()
                self.ser.write(self.req_msg)
                
                buffer = bytearray()
                chunk = self.ser.read(35)
                
                if chunk: 
                    buffer.extend(chunk)
                    
                if len(buffer) >= 29:
                    for i in range(len(buffer) - 28):
                        if (buffer[i] == 0xAF and buffer[i+1] == 0xFA and buffer[i+2] == 0x60 and buffer[i+3] == 0x17):
                            with self.lock:
                                self.voltage = ((buffer[i+6] << 8) | buffer[i+7]) * 0.01
                                raw_amps = (buffer[i+8] << 8) | buffer[i+9]
                                self.ampere = (raw_amps - 65536 if raw_amps > 32767 else raw_amps) * 0.01
                                self.soc = ((buffer[i+10] << 8) | buffer[i+11]) / 100.0
                            if not self.has_printed_banner: 
                                self._print_banner()
                                self.has_printed_banner = True
                            break
                        
                for _ in range(20): 
                    if not self.running:
                        break
                    
                    time.sleep(0.1)
                    
            except Exception: 
                time.sleep(1.0)

    def _print_banner(self):
        soc_val = int(self.soc * 100)
        color = '\033[92m' if soc_val > 60 else ('\033[93m' if soc_val > 20 else '\033[91m')
        print(f"\n\033[96m\033[1m" + "="*45 + f"\n| 🔋 UON AMR BATTERY STATUS                  |\n|---------------------------------------------|\033[0m")
        print(f"\033[96m|\033[0m  ⚡ Voltage: {self.voltage:5.2f} V                    \033[96m|\033[0m")
        print(f"\033[96m|\033[0m  🔌 Current: {self.ampere:5.2f} A                    \033[96m|\033[0m")
        print(f"\033[96m|\033[0m  {color}\033[1m🔋 SOC:     {soc_val:3d} %\033[0m                        \033[96m|\033[0m")
        print(f"\033[96m\033[1m" + "="*45 + f"\033[0m\n")

    def get_soc(self): 
        with self.lock: 
            return self.soc

# ==============================================================================
# 5. Robot Class
# ==============================================================================

@RobotConfig.register_subclass("uon_amr")
@dataclass
class UONAMRConfig(RobotConfig):
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    motor_port: str = "/dev/ttyUSB11"
    lidar_port: str = "/dev/ttyUSB12"
    lidar_baudrate: int = 460800
    battery_port: str = "/dev/ttyUSB15" 
    battery_baudrate: int = 19200
    camera_w: int = 640
    camera_h: int = 480

class UONAMR(Robot):
    config_class = UONAMRConfig
    name = "uon_amr"

    def __init__(self, config: UONAMRConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.cameras = config.cameras
        self.driver = UONAMRDriver(config.motor_port)
        self.lidar_driver = None
        self.battery_driver = None
        self.orbbec_driver = None
        self._latest_scan = np.zeros(360, dtype=np.float32)
        
    @property
    def is_connected(self) -> bool: 
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool: 
        return True

    def connect(self) -> None:
        if self._is_connected: 
            raise DeviceAlreadyConnectedError(f"{self.name} connected")
        
        logger.info(f"Connecting {self.name}...")
        
        try: 
            self.driver.connect()
            logger.info("✅ Motors connected")
        except Exception as e: 
            logger.error(f"❌ Motors failed: {e}")
            
        if LIDAR_AVAILABLE: 
            self._connect_lidar()
            
        self._connect_battery()
        self._connect_camera()
        self._is_connected = True

    def disconnect(self) -> None:
        if not self._is_connected: 
            raise DeviceNotConnectedError(f"{self.name} disconnected")
        
        try: 
            self.driver.send_velocity(0.0, 0.0)
            self.driver.disconnect()
        except Exception:
            pass
        
        if self.orbbec_driver: 
            self.orbbec_driver.stop()
        if self.lidar_driver: 
            self.lidar_driver.stop()
        if self.battery_driver: 
            self.battery_driver.stop()
            
        self._is_connected = False
        logger.info(f"{self.name} disconnected")

    def calibrate(self) -> None: 
        pass
    
    def configure(self) -> None:
        pass

    @cached_property
    def observation_features(self) -> Dict[str, Any]:
        features = {
            OBS_FRONT_RGB: (self.config.camera_h, self.config.camera_w, 3),
            OBS_FRONT_DEPTH: (self.config.camera_h, self.config.camera_w, 3), 
            OBS_FRONT_DEPTH_RAW: (self.config.camera_h, self.config.camera_w, 3), # 3-ch encoded
            OBS_LINEAR_VEL: float,
            OBS_ANGULAR_VEL: float,
            OBS_BATTERY_LEVEL: float,
        }
        
        for key in LIDAR_KEYS: 
            features[key] = float
            
        return features

    @cached_property
    def action_features(self) -> Dict[str, Any]:
        return {ACTION_LINEAR_VEL: float, ACTION_ANGULAR_VEL: float}

    def get_observation(self) -> Dict[str, Any]:
        if not self._is_connected: 
            raise DeviceNotConnectedError(f"{self.name} not connected")
        
        if self.lidar_driver: 
            self._latest_scan = self.lidar_driver.get_scan()

        rgb = np.zeros((self.config.camera_h, self.config.camera_w, 3), dtype=np.uint8)
        depth_vis = np.zeros((self.config.camera_h, self.config.camera_w, 3), dtype=np.uint8)
        depth_raw = np.zeros((self.config.camera_h, self.config.camera_w), dtype=np.uint16)
        
        if self.orbbec_driver:
            rgb, depth_vis, depth_raw = self.orbbec_driver.get_data()
        
        # Encode 16-bit depth to 3-channel uint8
        depth_encoded = np.zeros((depth_raw.shape[0], depth_raw.shape[1], 3), dtype=np.uint8)
        depth_encoded[:, :, 0] = depth_raw & 0xFF
        depth_encoded[:, :, 1] = (depth_raw >> 8) & 0xFF

        batt_level = self.battery_driver.get_soc() if self.battery_driver else 1.0
        
        curr_v, curr_w = self.driver.get_feedback()

        obs = {
            OBS_FRONT_RGB: rgb,
            OBS_FRONT_DEPTH: depth_vis,
            OBS_FRONT_DEPTH_RAW: depth_encoded,
            OBS_LINEAR_VEL: curr_v,
            OBS_ANGULAR_VEL: curr_w,
            OBS_BATTERY_LEVEL: batt_level,
        }
        obs.update(zip(LIDAR_KEYS, self._latest_scan))
        return obs

    def send_action(self, action: Dict[str, Any] | np.ndarray) -> Dict[str, Any]:
        if not self._is_connected: 
            raise DeviceNotConnectedError(f"{self.name} not connected")
        
        lin, ang = 0.0, 0.0
        
        if isinstance(action, dict):
            if ACTION_LINEAR_VEL in action: 
                lin, ang = float(action[ACTION_LINEAR_VEL]), float(action[ACTION_ANGULAR_VEL])
            elif "v" in action: 
                lin, ang = float(action["v"]), float(action["w"])
            else:
                raw = list(action.values())[0]
                if hasattr(raw, "cpu"): 
                    raw = raw.cpu().numpy()
                lin, ang = float(raw[0]), float(raw[1])
        else:
            raw = action
            if hasattr(raw, "cpu"): 
                raw = raw.cpu().numpy()
            if len(raw.shape) > 1: 
                raw = raw.flatten()
            lin, ang = float(raw[0]), float(raw[1])

        try:
            self.driver.send_velocity(lin, ang)

        except Exception: pass
        return {ACTION_LINEAR_VEL: lin, ACTION_ANGULAR_VEL: ang}

    def _connect_lidar(self):
        if not LIDAR_AVAILABLE: 
            return
        
        try:
            with serial.Serial(self.config.lidar_port, self.config.lidar_baudrate, timeout=0.1) as tmp:
                tmp.dtr = False
                tmp.rts = False
                time.sleep(0.1)
                tmp.dtr = True
                tmp.rts = True
                time.sleep(0.2)
                tmp.write(b'\xA5\x25')
                time.sleep(0.1)
                tmp.reset_input_buffer()
        except Exception: 
                pass
            
        self.lidar_driver = ThreadedLidar(self.config.lidar_port, self.config.lidar_baudrate)
        self.lidar_driver.start()

    def _connect_battery(self):
        self.battery_driver = UONBatteryDriver(self.config.battery_port, self.config.battery_baudrate)
        self.battery_driver.start()

    def _connect_camera(self):
        self.orbbec_driver = ThreadedOrbbec(
            width=self.config.camera_w, height=self.config.camera_h, min_mm=200, max_mm=3000
        )
        self.orbbec_driver.start()