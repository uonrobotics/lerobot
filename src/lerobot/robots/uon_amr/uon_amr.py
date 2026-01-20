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
import struct
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Optional, Tuple

import numpy as np
import serial
import cv2 

# LeRobot Imports
from lerobot.cameras import CameraConfig
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .uon_amr_driver import UONAMRDriver

# --- Orbbec SDK Import ---
try:
    from pyorbbecsdk import (
        Pipeline, Config, OBSensorType, OBFormat, OBStreamType, 
        OBFrameAggregateOutputMode, AlignFilter
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
            
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("✅ [Orbbec] SDK Pipeline Started")
            
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
                frames = self.pipeline.wait_for_frames(100)
                if frames is None: continue

                if self.align_filter:
                    frames = self.align_filter.process(frames)
                    if not frames: continue
                    frames = frames.as_frame_set()

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame and depth_frame:
                    rgb = self._process_color_frame(color_frame)
                    depth_raw, depth_vis = self._process_depth_frame(depth_frame, self.min_mm, self.max_mm)
                    
                    # Ensure resize
                    if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
                        rgb = cv2.resize(rgb, (self.width, self.height))
                    if depth_raw.shape[0] != self.height or depth_raw.shape[1] != self.width:
                        depth_raw = cv2.resize(depth_raw, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    if depth_vis.shape[0] != self.height or depth_vis.shape[1] != self.width:
                        depth_vis = cv2.resize(depth_vis, (self.width, self.height))

                    with self.lock:
                        self.latest_rgb = rgb
                        self.latest_depth_vis = depth_vis
                        self.latest_depth_raw = depth_raw

            except Exception as e:
                pass

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

# ==============================================================================
# 2. Other Drivers
# ==============================================================================
class ThreadedLidar:
    def __init__(self, port, baudrate):
        self.port = port; self.baudrate = baudrate
        self.lidar = None; self.latest_scan = np.zeros(360, dtype=np.float32)
        self.running = False; self.thread = None; self.lock = threading.Lock()

    def start(self):
        try:
            self.lidar = RPLidar(self.port, baudrate=self.baudrate, timeout=3)
            self.lidar.clean_input()
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("✅ [Lidar] Thread started")
        except Exception as e: logger.error(f"❌ [Lidar] Start failed: {e}")

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        if self.lidar:
            try: self.lidar.stop(); self.lidar.disconnect()
            except: pass

    def _run(self):
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=500):
                if not self.running: break
                temp = np.zeros(360, dtype=np.float32)
                for (_, angle, distance) in scan: temp[min(359, int(angle))] = distance
                with self.lock: self.latest_scan = temp
                time.sleep(0.001)
        except Exception: pass

    def get_scan(self):
        with self.lock: return self.latest_scan.copy()

class UONBatteryDriver:
    def __init__(self, port, baudrate=19200):
        self.port = port; self.baudrate = baudrate
        self.ser = None; self.running = False; self.thread = None; self.lock = threading.Lock()
        self.soc = 0.0; self.voltage = 0.0; self.ampere = 0.0; self.has_printed_banner = False
        self.req_msg = bytearray([0xAF, 0xFA, 0x60, 0x05, 0x01, 0x60, 0x7F, 0x07, 0x00, 0xAF, 0xA0])
        self.req_msg[8] = sum(self.req_msg[2:8]) & 0xFF

    def start(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
            self.running = True; self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info(f"✅ [Battery] Connected to {self.port}")
        except Exception as e: logger.warning(f"⚠️ [Battery] Connection failed: {e}")

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1.0)
        if self.ser: self.ser.close()

    def _run(self):
        while self.running and self.ser:
            try:
                self.ser.reset_input_buffer(); self.ser.write(self.req_msg)
                buffer = bytearray(); chunk = self.ser.read(35)
                if chunk: buffer.extend(chunk)
                if len(buffer) >= 29:
                    for i in range(len(buffer) - 28):
                        if (buffer[i] == 0xAF and buffer[i+1] == 0xFA and buffer[i+2] == 0x60 and buffer[i+3] == 0x17):
                            with self.lock:
                                self.voltage = ((buffer[i+6] << 8) | buffer[i+7]) * 0.01
                                raw_amps = (buffer[i+8] << 8) | buffer[i+9]
                                self.ampere = (raw_amps - 65536 if raw_amps > 32767 else raw_amps) * 0.01
                                self.soc = ((buffer[i+10] << 8) | buffer[i+11]) / 100.0
                            if not self.has_printed_banner: self._print_banner(); self.has_printed_banner = True
                            break
                for _ in range(20): 
                    if not self.running: break
                    time.sleep(0.1)
            except Exception: time.sleep(1.0)

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
        self.lidar_driver = None; self.battery_driver = None; self.orbbec_driver = None
        self._current_lin = 0.0; self._current_ang = 0.0
        self._latest_scan = np.zeros(360, dtype=np.float32)
        
    @property
    def is_connected(self) -> bool: return self._is_connected
    @property
    def is_calibrated(self) -> bool: return True

    def connect(self) -> None:
        if self._is_connected: raise DeviceAlreadyConnectedError(f"{self.name} connected")
        logger.info(f"Connecting {self.name}...")
        try: self.driver.connect(); logger.info("✅ Motors connected")
        except Exception as e: logger.error(f"❌ Motors failed: {e}")
        if LIDAR_AVAILABLE: self._connect_lidar()
        self._connect_battery()
        self._connect_camera()
        self._is_connected = True

    def disconnect(self) -> None:
        if not self._is_connected: raise DeviceNotConnectedError(f"{self.name} disconnected")
        try: self.driver.send_velocity(0.0, 0.0); self.driver.disconnect()
        except Exception: pass
        if self.orbbec_driver: self.orbbec_driver.stop()
        if self.lidar_driver: self.lidar_driver.stop()
        if self.battery_driver: self.battery_driver.stop()
        self._is_connected = False
        logger.info(f"{self.name} disconnected")

    def calibrate(self) -> None: pass
    def configure(self) -> None: pass

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
        for key in LIDAR_KEYS: features[key] = float
        return features

    @cached_property
    def action_features(self) -> Dict[str, Any]:
        return {ACTION_LINEAR_VEL: float, ACTION_ANGULAR_VEL: float}

    def get_observation(self) -> Dict[str, Any]:
        if not self._is_connected: raise DeviceNotConnectedError(f"{self.name} not connected")
        if self.lidar_driver: self._latest_scan = self.lidar_driver.get_scan()

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

        obs = {
            OBS_FRONT_RGB: rgb,
            OBS_FRONT_DEPTH: depth_vis,
            OBS_FRONT_DEPTH_RAW: depth_encoded,
            OBS_LINEAR_VEL: self._current_lin,
            OBS_ANGULAR_VEL: self._current_ang,
            OBS_BATTERY_LEVEL: batt_level,
        }
        obs.update(zip(LIDAR_KEYS, self._latest_scan))
        return obs

    def send_action(self, action: Dict[str, Any] | np.ndarray) -> Dict[str, Any]:
        if not self._is_connected: raise DeviceNotConnectedError(f"{self.name} not connected")
        lin, ang = 0.0, 0.0
        if isinstance(action, dict):
            if ACTION_LINEAR_VEL in action: lin, ang = float(action[ACTION_LINEAR_VEL]), float(action[ACTION_ANGULAR_VEL])
            elif "v" in action: lin, ang = float(action["v"]), float(action["w"])
            else:
                raw = list(action.values())[0]
                if hasattr(raw, "cpu"): raw = raw.cpu().numpy()
                lin, ang = float(raw[0]), float(raw[1])
        else:
            raw = action
            if hasattr(raw, "cpu"): raw = raw.cpu().numpy()
            if len(raw.shape) > 1: raw = raw.flatten()
            lin, ang = float(raw[0]), float(raw[1])

        try:
            self.driver.send_velocity(lin, ang)
            self._current_lin, self._current_ang = lin, ang
        except Exception: pass
        return {ACTION_LINEAR_VEL: lin, ACTION_ANGULAR_VEL: ang}

    def _connect_lidar(self):
        if not LIDAR_AVAILABLE: return
        try:
            with serial.Serial(self.config.lidar_port, self.config.lidar_baudrate, timeout=0.1) as tmp:
                tmp.dtr = False; tmp.rts = False; time.sleep(0.1)
                tmp.dtr = True; tmp.rts = True; time.sleep(0.2)
                tmp.write(b'\xA5\x25'); time.sleep(0.1)
                tmp.reset_input_buffer()
        except Exception: pass
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