import threading
import time
import serial  
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from lerobot.robots.uon_amr.uon_utils import calculate_crc
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

# ---------------------------------------------------------
# 1. Low-Level Serial Driver
# ---------------------------------------------------------
class ST10SerialDriver:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.thread = None
        # Initialize with 10 channels centered at 1500
        self.channels = [1500] * 10 
        self.dev_status = -1
        self.lock = threading.Lock()
        self.connected = False

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"[ST10] Connected to {self.port} ({self.baudrate})")
            self.connected = True
        except Exception as e:
            print(f"[ST10] Connection Failed: {e}")
            self.connected = False

    def disconnect(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.ser:
            self.ser.close()
        self.connected = False

    def _read_loop(self):
        buffer = bytearray()
        print("[RC-Driver] Starting read loop...")
        
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    chunk = self.ser.read(self.ser.in_waiting)
                    buffer.extend(chunk)
                    
                    # Process all complete packets in buffer
                    while len(buffer) >= 26:
                        # Check Header: FF FF 17
                        if buffer[0] == 0xFF and buffer[1] == 0xFF and buffer[2] == 0x17:
                            payload_for_crc = buffer[3:24] 
                            calc_crc = calculate_crc(payload_for_crc)
                            # Little Endian CRC check
                            recv_crc = buffer[24] | (buffer[25] << 8)
                            
                            if calc_crc == recv_crc:
                                with self.lock:
                                    self.dev_status = buffer[3]
                                    # Parse 10 channels (2 bytes each)
                                    for k in range(10):
                                        idx = 4 + (k * 2)
                                        val = int.from_bytes(buffer[idx:idx+2], byteorder='little', signed=False)
                                        self.channels[k] = val
                                # Remove processed packet
                                del buffer[:26]
                            else:
                                # Bad CRC, shift by 1 to find next header
                                del buffer[0]
                        else:
                            # Bad Header, shift by 1
                            del buffer[0]
                else:
                    time.sleep(0.002) 
            except Exception as e:
                # print(f"[RC Error] {e}")
                time.sleep(0.1)

    def get_channels(self):
        with self.lock:
            return list(self.channels)

# ---------------------------------------------------------
# 2. LeRobot Teleoperator Config
# ---------------------------------------------------------
@TeleoperatorConfig.register_subclass("uon_rc_teleoperator")
@dataclass
class UONRcTeleoperatorConfig(TeleoperatorConfig):
    name: str = "uon_rc_teleoperator"
    port: str = "/dev/ttyUSB14"
    baudrate: int = 115200
    max_lin_speed: float = 0.5
    max_ang_speed: float = 0.5
    axis_linear_channel: int = 1  
    axis_angular_channel: int = 3
    raw_min: int = 1000     
    raw_center: int = 1500  
    raw_max: int = 2000     
    raw_deadzone: int = 50  

    @classmethod
    def get_valid_devices(cls):
        return [UONRcTeleoperator]

# ---------------------------------------------------------
# 3. LeRobot Teleoperator Class
# ---------------------------------------------------------
class UONRcTeleoperator(Teleoperator):
    config_class = UONRcTeleoperatorConfig
    name = "uon_rc_teleoperator"

    def __init__(self, config: UONRcTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.driver = None

    def connect(self):
        self.driver = ST10SerialDriver(self.config.port, self.config.baudrate)
        self.driver.connect()

    def disconnect(self):
        if self.driver:
            self.driver.disconnect()

    @property
    def is_connected(self) -> bool:
        return self.driver.connected if self.driver else False

    @property
    def action_features(self) -> Dict:
        return {
            "linear.vel": {"dtype": "float32", "shape": (1,), "names": ["linear.vel"]},
            "angular.vel": {"dtype": "float32", "shape": (1,), "names": ["angular.vel"]}
        }

    def _normalize(self, value):
        centered = value - self.config.raw_center
        if abs(centered) < self.config.raw_deadzone:
            return 0.0
        range_span = self.config.raw_max - self.config.raw_center
        if range_span == 0: return 0.0
        normalized = centered / range_span
        return max(-1.0, min(1.0, normalized))

    def get_action(self, observation=None) -> Dict[str, Any]:
        safe_action = {"linear.vel": 0.0, "angular.vel": 0.0}

        if not self.is_connected:
            return safe_action

        channels = self.driver.get_channels()
        
        # Safety check for empty or zeroed channels
        if not channels or (channels[0] == 0 and channels[1] == 0):
            return safe_action
        
        max_idx = max(self.config.axis_linear_channel, self.config.axis_angular_channel)
        if len(channels) <= max_idx:
            return safe_action
        
        raw_lin = channels[self.config.axis_linear_channel]
        raw_ang = channels[self.config.axis_angular_channel]
        
        norm_lin = self._normalize(raw_lin)
        norm_ang = self._normalize(raw_ang)

        lin = float(norm_lin * self.config.max_lin_speed)
        ang = float(norm_ang * self.config.max_ang_speed)
        
        return {
            "linear.vel": lin, 
            "angular.vel": ang
        }

    @property
    def feedback_features(self): return None
    def send_feedback(self, feedback): pass
    def configure(self): pass
    def calibrate(self): pass
    @property
    def is_calibrated(self) -> bool: return True
    
    # --- FIXED: Use the parsed channels from the driver ---
    
    def _get_channel_10_value(self):
        """Helper to get Channel 10 value directly from the driver."""
        if not self.is_connected:
            return 1500
        
        channels = self.driver.get_channels()
        # Channel 10 is at index 9 (0-based list)
        if len(channels) >= 10:
            return channels[9]
        
        return 1500

    def is_next_episode_command(self):
        """Roller UP (> 1800) means Save & Next."""
        val = self._get_channel_10_value()
        
        # # Debug print to verify in console (Optional: remove later)
        # if val > 1600 or val < 1400:
        #     print(f"\r[DEBUG] Ch10 Value: {val}   ", end="", flush=True)
            
        return val > 1800

    def is_rerecord_command(self):
        """Roller DOWN (< 1200) means Discard & Retry."""
        val = self._get_channel_10_value()
        return val < 1200