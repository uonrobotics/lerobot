import serial
import time
import math
import threading
import struct

# ========================= CONSTANTS =========================

MAX_SPD             = 1500
DEFAULT_MOTOR_BPS   = 115200

MotorDriver_ID      = 2
MotorDriver_ID_R    = 3

READ_CMD            = 0x03
READ_SPD_H          = 0x20
READ_SPD_L          = 0x2C
READ_BULK_H         = 0x20
READ_BULK_L         = 0x2C

WRITE_CMD_S         = 0x06
WRITE_SPD_H         = 0x20
WRITE_SPD_L         = 0x3A

WR_WORD_H           = 0x20
WR_WORD_L           = 0x31
WR_WORD_QUICKSTOP   = 0x05
WR_WORD_CLEARFAULT  = 0x06
WR_WORD_STOP        = 0x07
WR_WORD_ENABLE      = 0x08

WR_MODE_H           = 0x20
WR_MODE_L           = 0x32
WR_MODE_VEL         = 0x03
WR_OFFLINE_TIME_L   = 0x00

WRITE_ACC_TIME_H    = 0x20
WRITE_ACC_TIME_L    = 0x37
WRITE_DACC_TIME_H   = 0x20
WRITE_DACC_TIME_L   = 0x38

DEFAULT_ACC_TIME    = 1
DEFAULT_DACC_TIME   = 1

# ========================= CRC LOGIC =========================

wCRCTable = [
    0x0000, 0xC0C1, 0xC181, 0x0140, 0xC301, 0x03C0, 0x0280, 0xC241,
    0xC601, 0x06C0, 0x0780, 0xC741, 0x0500, 0xC5C1, 0xC481, 0x0440,
    0xCC01, 0x0CC0, 0x0D80, 0xCD41, 0x0F00, 0xCFC1, 0xCE81, 0x0E40,
    0x0A00, 0xCAC1, 0xCB81, 0x0B40, 0xC901, 0x09C0, 0x0880, 0xC841,
    0xD801, 0x18C0, 0x1980, 0xD941, 0x1B00, 0xDBC1, 0xDA81, 0x1A40,
    0x1E00, 0xDEC1, 0xDF81, 0x1F40, 0xDD01, 0x1DC0, 0x1C80, 0xDC41,
    0x1400, 0xD4C1, 0xD581, 0x1540, 0xD701, 0x17C0, 0x1680, 0xD641,
    0xD201, 0x12C0, 0x1380, 0xD341, 0x1100, 0xD1C1, 0xD081, 0x1040,
    0xF001, 0x30C0, 0x3180, 0xF141, 0x3300, 0xF3C1, 0xF281, 0x3240,
    0x3600, 0xF6C1, 0xF781, 0x3740, 0xF501, 0x35C0, 0x3480, 0xF441,
    0x3C00, 0xFCC1, 0xFD81, 0x3D40, 0xFF01, 0x3FC0, 0x3E80, 0xFE41,
    0xFA01, 0x3AC0, 0x3B80, 0xFB41, 0x3900, 0xF9C1, 0xF881, 0x3840,
    0x2800, 0xE8C1, 0xE981, 0x2940, 0xEB01, 0x2BC0, 0x2A80, 0xEA41,
    0xEE01, 0x2EC0, 0x2F80, 0xEF41, 0x2D00, 0xEDC1, 0xEC81, 0x2C40,
    0xE401, 0x24C0, 0x2580, 0xE541, 0x2700, 0xE7C1, 0xE681, 0x2640,
    0x2200, 0xE2C1, 0xE381, 0x2340, 0xE101, 0x21C0, 0x2080, 0xE041,
    0xA001, 0x60C0, 0x6180, 0xA141, 0x6300, 0xA3C1, 0xA281, 0x6240,
    0x6600, 0xA6C1, 0xA781, 0x6740, 0xA501, 0x65C0, 0x6480, 0xA441,
    0x6C00, 0xACC1, 0xAD81, 0x6D40, 0xAF01, 0x6FC0, 0x6E80, 0xAE41,
    0xAA01, 0x6AC0, 0x6B80, 0xAB41, 0x6900, 0XA9C1, 0XA881, 0X6840,
    0x7800, 0xB8C1, 0xB981, 0x7940, 0xBB01, 0x7BC0, 0x7A80, 0xBA41,
    0xBE01, 0x7EC0, 0x7F80, 0xBF41, 0x7D00, 0xBDC1, 0xBC81, 0x7C40,
    0xB401, 0x74C0, 0x7580, 0xB541, 0x7700, 0xB7C1, 0xB681, 0x7640,
    0x7200, 0xB2C1, 0xB381, 0x7340, 0xB101, 0x71C0, 0x7080, 0xB041,
    0x5000, 0x90C1, 0x9181, 0x5140, 0x9301, 0x53C0, 0x5280, 0x9241,
    0x9601, 0x56C0, 0x5780, 0x9741, 0x5500, 0x95C1, 0x9481, 0x5440,
    0x9C01, 0x5CC0, 0x5D80, 0x9D41, 0x5F00, 0x9FC1, 0x9E81, 0x5E40,
    0x5A00, 0x9AC1, 0x9B81, 0x5B40, 0x9901, 0x59C0, 0x5880, 0x9841,
    0x8801, 0x48C0, 0x4980, 0x8941, 0x4B00, 0x8BC1, 0x8A81, 0x4A40,
    0x4E00, 0x8EC1, 0x8F81, 0x4F40, 0x8D01, 0x4DC0, 0x4C80, 0x8C41,
    0x4400, 0x84C1, 0x8581, 0x4540, 0x8701, 0x47C0, 0x4680, 0x8641,
    0x8201, 0x42C0, 0x4380, 0x8341, 0x4100, 0x81C1, 0x8081, 0x4040,
]

def calculate_crc(data_bytes):
    crc = 0xFFFF
    for byte in data_bytes:
        nTemp = (byte ^ crc) & 0xFF
        crc = (crc >> 8) & 0xFFFF
        crc ^= wCRCTable[nTemp]
        crc &= 0xFFFF
    return crc

def clip(x, lo, hi):
    return max(lo, min(hi, x))

# ========================= DRIVER CLASS =========================

class UONAMRDriver:
    def __init__(self, port, baudrate=DEFAULT_MOTOR_BPS, wheel_radius=0.07, wheel_sep=0.4605):
        self.port = port
        self.baudrate = baudrate
        self.wheel_radius = wheel_radius
        self.wheel_sep = wheel_sep
        
        self.ser = None
        self.connected = False
        self.running = False
        
        # State Variables
        self.is_init_done = False
        self.init_scheduler = 0
        self.comm_scheduler = 0
        
        # Control Variables (Thread Safe)
        self.lock = threading.Lock()
        self.target_v = 0.0
        self.target_w = 0.0
        self.Target_LeftRPM = 0
        self.Target_RightRPM = 0
        
        # SAFETY: Watchdog Timer
        self.last_command_time = time.time()
        self.watchdog_timeout = 0.5  # Stop if no command for 500ms

        # Parameters
        self.max_lin_vel = 3.0
        self.max_ang_vel = 1.7472

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
            self.connected = True
            
            # Start the background scheduler thread
            self.running = True
            self.thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.thread.start()
            
            print(f"[Driver] Connected to {self.port}. Scheduler started.")
        except Exception as e:
            print(f"[Driver] Connect failed: {e}")
            self.connected = False

    def disconnect(self):
        print("[Driver] Stopping motors...")
        # 1. Force Stop Target
        with self.lock:
            self.target_v = 0.0
            self.target_w = 0.0
            self.Target_LeftRPM = 0
            self.Target_RightRPM = 0
        
        # 2. Wait for scheduler to send the STOP command (2 cycles)
        time.sleep(0.1)

        # 3. Stop Thread
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        
        # 4. Hardware Stop (Extra Safety)
        if self.ser and self.ser.is_open:
            try:
                self._set_word(MotorDriver_ID, WR_WORD_STOP)
                self._set_word(MotorDriver_ID_R, WR_WORD_STOP)
                self.ser.close()
            except: pass
            
        self.connected = False
        print("[Driver] Disconnected.")

    # ------------------------------------------------------------------
    # Public API for LeRobot
    # ------------------------------------------------------------------
    def send_velocity(self, v, w):
        """
        Updates the target velocity and resets the watchdog timer.
        """
        with self.lock:
            self.target_v = clip(v, -self.max_lin_vel, self.max_lin_vel)
            self.target_w = clip(w, -self.max_ang_vel, self.max_ang_vel)
            # Update watchdog timestamp
            self.last_command_time = time.time()

    # ------------------------------------------------------------------
    # Background Scheduler Loop
    # ------------------------------------------------------------------
    def _scheduler_loop(self):
        while self.running and self.ser:
            try:
                # 1. Process Incoming Packets
                self._packet_process()

                # 2. Run Scheduler Logic
                if not self.is_init_done:
                    self._run_init_scheduler()
                else:
                    self._run_comm_scheduler()

                # 3. Update RPM Calculation & CHECK WATCHDOG
                self._update_rpm_targets()

                # 4. Sleep (Mimic 5ms timer)
                time.sleep(0.005) 
                
            except Exception as e:
                print(f"[Driver] Scheduler Error: {e}")
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Logic Blocks
    # ------------------------------------------------------------------
    def _update_rpm_targets(self):
        with self.lock:
            # SAFETY CHECK: Watchdog
            if time.time() - self.last_command_time > self.watchdog_timeout:
                self.target_v = 0.0
                self.target_w = 0.0
                # print("[Driver] Watchdog triggered! Stopping robot.") # Uncomment for debug

            v = self.target_v
            w = self.target_w

        L = self.wheel_sep
        R = self.wheel_radius

        # Kinematics
        TPL = (v + w * L / 2.0) / R
        TPR = (v - w * L / 2.0) / R

        # Rad/s -> RPM
        rpm_l = TPL / (2.0 * math.pi) * 60.0
        rpm_r = TPR / (2.0 * math.pi) * 60.0

        # Sign convention (Left +, Right -)
        self.Target_LeftRPM = int(clip(rpm_l, -MAX_SPD, MAX_SPD))
        self.Target_RightRPM = int(clip(-rpm_r, -MAX_SPD, MAX_SPD))

    def _run_init_scheduler(self):
        step = self.init_scheduler
        
        if step == 0: self._set_comm_offline_time(MotorDriver_ID, 500)
        elif step == 1: self._set_vel_mode(MotorDriver_ID)
        elif step == 2: self._set_acc_time(MotorDriver_ID, DEFAULT_ACC_TIME)
        elif step == 3: self._set_dacc_time(MotorDriver_ID, DEFAULT_DACC_TIME)
        elif step == 4: self._set_word(MotorDriver_ID, WR_WORD_QUICKSTOP)
        
        elif step == 5: self._set_comm_offline_time(MotorDriver_ID_R, 500)
        elif step == 6: self._set_vel_mode(MotorDriver_ID_R)
        elif step == 7: self._set_acc_time(MotorDriver_ID_R, DEFAULT_ACC_TIME)
        elif step == 8: self._set_dacc_time(MotorDriver_ID_R, DEFAULT_DACC_TIME)
        elif step == 9: self._set_word(MotorDriver_ID_R, WR_WORD_QUICKSTOP)
        
        elif step == 10: self._set_word(MotorDriver_ID, WR_WORD_CLEARFAULT)
        elif step == 11: self._set_word(MotorDriver_ID_R, WR_WORD_CLEARFAULT)
        
        elif step == 12:
            self._set_word(MotorDriver_ID, WR_WORD_ENABLE)
            time.sleep(0.01)
            self._set_word(MotorDriver_ID_R, WR_WORD_ENABLE)
            self.is_init_done = True
            print("[Driver] Init Sequence Complete. Motors Enabled.")
        
        if step <= 12:
            self.init_scheduler += 1
            time.sleep(0.02) 

    def _run_comm_scheduler(self):
        """
        0: Write Left -> 1: Write Right -> 2: Read Left -> 3: Read Right
        """
        step = self.comm_scheduler
        
        if step == 0: # Write Left
            self._set_rpm(MotorDriver_ID, self.Target_LeftRPM)
        elif step == 1: # Write Right
            self._set_rpm(MotorDriver_ID_R, self.Target_RightRPM)
        elif step == 2: # Read Left 
            self._read_bulk(MotorDriver_ID)
        elif step == 3: # Read Right
            self._read_bulk(MotorDriver_ID_R)
            
        self.comm_scheduler = (self.comm_scheduler + 1) % 4
        time.sleep(0.002) 

    def _packet_process(self):
        if self.ser.in_waiting > 0:
            self.ser.read(self.ser.in_waiting)

    # ------------------------------------------------------------------
    # Packet Builders
    # ------------------------------------------------------------------
    def _send_packet(self, payload):
        pkt = list(payload)
        crc = calculate_crc(pkt)
        pkt.append(crc & 0xFF)
        pkt.append((crc >> 8) & 0xFF)
        try:
            self.ser.write(bytes(pkt))
            self.ser.flush()
        except: pass

    def _set_rpm(self, motor_id, rpm):
        val = int(rpm)
        payload = [motor_id, WRITE_CMD_S, WRITE_SPD_H, WRITE_SPD_L, (val >> 8) & 0xFF, val & 0xFF]
        self._send_packet(payload)

    def _read_bulk(self, motor_id):
        payload = [motor_id, READ_CMD, READ_BULK_H, READ_BULK_L, 0x00, 0x03]
        self._send_packet(payload)

    def _set_word(self, motor_id, cmd):
        payload = [motor_id, WRITE_CMD_S, WR_WORD_H, WR_WORD_L, 0x00, cmd]
        self._send_packet(payload)
        
    def _set_comm_offline_time(self, motor_id, val):
        payload = [motor_id, WRITE_CMD_S, WR_WORD_H, WR_OFFLINE_TIME_L, (val >> 8) & 0xFF, val & 0xFF]
        self._send_packet(payload)
        
    def _set_vel_mode(self, motor_id):
        payload = [motor_id, WRITE_CMD_S, WR_MODE_H, WR_MODE_L, 0x00, WR_MODE_VEL]
        self._send_packet(payload)
        
    def _set_acc_time(self, motor_id, val):
        payload = [motor_id, WRITE_CMD_S, WRITE_ACC_TIME_H, WRITE_ACC_TIME_L, (val >> 8) & 0xFF, val & 0xFF]
        self._send_packet(payload)

    def _set_dacc_time(self, motor_id, val):
        payload = [motor_id, WRITE_CMD_S, WRITE_DACC_TIME_H, WRITE_DACC_TIME_L, (val >> 8) & 0xFF, val & 0xFF]
        self._send_packet(payload)