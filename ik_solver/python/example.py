import sys
import os
import math
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QPushButton, QLabel, QMessageBox, QSlider, QRadioButton)
from PyQt6.QtCore import Qt, QTimer
from ik_solver import ik_solver

from typing import Any, Dict, Tuple

# ------------------------------------------------------------------
# 공유 메모리 클래스
# ------------------------------------------------------------------
from multiprocessing import shared_memory, resource_tracker
import atexit
class SharedMemory:
    def __init__(self, name: str, fields_config: Dict[str, Tuple[int, Any]]):
        self.name = name
        self.fields = {}
        offset = 0
        for f_name, (count, dtype) in fields_config.items():
            item_size = 1 if dtype == str else np.dtype(dtype).itemsize
            byte_size = count * item_size
            self.fields[f_name] = {'offset': offset, 'count': count, 'dtype': dtype, 'byte_size': byte_size}
            offset += byte_size
        self.total_size = offset

        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.total_size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name)

        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except: pass
        atexit.register(self.close)

    def set(self, field_name: str, value: Any) -> None:
        f = self.fields[field_name]
        if f['dtype'] == str:
            encoded = str(value).encode('utf-8')[:f['byte_size']]
            self.shm.buf[f['offset'] : f['offset'] + len(encoded)] = encoded
        else:
            arr = np.ndarray((f['count'],), dtype=f['dtype'], buffer=self.shm.buf, offset=f['offset'])
            arr[:] = value

    def close(self) -> None:
        if hasattr(self, 'shm'): self.shm.close()



fields_config = {'joint': (6, np.float32), 'status': (20, str)}
shm = SharedMemory(name='movej', fields_config=fields_config)



class SimpleIkUi(QWidget):
    def __init__(self):
        super().__init__()
        self.solver = None

        # 제어 데이터 (목표값)
        self.target_q_rad = [0.0] * 6
        self.target_pose = [0.0] * 6 # [x, y, z, r, p, y] (m, rad)

        self.j_sliders = []
        self.j_slider_labels = []
        self.c_sliders = []
        self.c_slider_labels = []
        self.cart_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]

        self.current_mode = "movej"
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timer_callback)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('IK Solver - Seamless Controller')
        main_layout = QVBoxLayout()

        # 초기화 영역
        self.path_input = QLineEdit(self)
        self.path_input.setPlaceholderText("URDF 파일 경로 입력")
        self.btn_init = QPushButton('Initialize Solver', self)
        self.btn_init.clicked.connect(self.run_init)

        self.status_label = QLabel('Status: Ready')
        self.joint_label = QLabel('Current Status: None')

        main_layout.addWidget(self.path_input)
        main_layout.addWidget(self.btn_init)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.joint_label)

        # 모드 선택
        mode_layout = QHBoxLayout()
        self.radio_movej = QRadioButton("MoveJ (Joint)")
        self.radio_movel = QRadioButton("MoveL (Cartesian)")
        self.radio_movej.setChecked(True)
        self.radio_movej.toggled.connect(self.on_mode_switched)
        mode_layout.addWidget(self.radio_movej)
        mode_layout.addWidget(self.radio_movel)
        main_layout.addLayout(mode_layout)

        # MoveJ 슬라이더 (J1~J6)
        main_layout.addWidget(QLabel("[Joint Control - deg]"))
        for i in range(6):
            h = QHBoxLayout()
            lbl = QLabel(f"J{i+1}: 0")
            lbl.setFixedWidth(60)
            sd = QSlider(Qt.Orientation.Horizontal)
            sd.setRange(-360, 360)
            sd.setEnabled(False)
            sd.valueChanged.connect(self.on_j_slider_changed)
            h.addWidget(lbl)
            h.addWidget(sd)
            main_layout.addLayout(h)
            self.j_slider_labels.append(lbl)
            self.j_sliders.append(sd)

        # MoveL 슬라이더 (X,Y,Z,R,P,Y)
        main_layout.addWidget(QLabel("[Cartesian Control - mm, deg]"))
        ranges = [(-1000, 1000), (-1000, 1000), (-1000, 1000), (-180, 180), (-180, 180), (-180, 180)]
        for i, (name, (mn, mx)) in enumerate(zip(self.cart_names, ranges)):
            h = QHBoxLayout()
            lbl = QLabel(f"{name}: 0")
            lbl.setFixedWidth(80)
            sd = QSlider(Qt.Orientation.Horizontal)
            sd.setRange(mn, mx)
            sd.setEnabled(False)
            sd.valueChanged.connect(self.on_c_slider_changed)
            h.addWidget(lbl)
            h.addWidget(sd)
            main_layout.addLayout(h)
            self.c_slider_labels.append(lbl)
            self.c_sliders.append(sd)

        self.setLayout(main_layout)

    def run_init(self):
        path = self.path_input.text().strip()
        if not os.path.exists(path): return
        try:
            self.solver = ik_solver(path)
            if self.solver.init():
                self.solver.set_joint([0.0]*6)
                self.sync_all_targets_to_current() # 현재 위치로 모든 타겟 동기화
                self.on_mode_switched()
                self.timer.start(20) # 50Hz 제어
                self.status_label.setText("Status: Initialized")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def sync_all_targets_to_current(self):
        """로봇의 실제 현재 위치를 내부 타겟 변수에 즉시 동기화"""
        curr_q_deg = self.solver.get_curr_joint_deg()
        self.target_q_rad = np.deg2rad(curr_q_deg).tolist()
        self.target_pose = self.get_current_pose_from_tf()

    def on_mode_switched(self):
        if not self.solver: return
        is_j = self.radio_movej.isChecked()
        self.current_mode = "movej" if is_j else "movel"

        # 모드 전환 시 타겟을 현재 위치로 재동기화하여 점프 방지
        self.sync_all_targets_to_current()

        # 슬라이더 활성화 상태 제어
        for s in self.j_sliders: s.setEnabled(is_j)
        for s in self.c_sliders: s.setEnabled(not is_j)

    def on_j_slider_changed(self):
        if self.current_mode != "movej": return
        vals = [s.value() for s in self.j_sliders]
        self.target_q_rad = np.deg2rad(vals).tolist()

    def on_c_slider_changed(self):
        if self.current_mode != "movel": return
        v = [s.value() for s in self.c_sliders]
        self.target_pose = [v[0]/1000.0, v[1]/1000.0, v[2]/1000.0,
                            math.radians(v[3]), math.radians(v[4]), math.radians(v[5])]

    def timer_callback(self):
        if not self.solver: return

        # 제어 명령 송신
        if self.current_mode == "movej":
            self.solver.movej(self.target_q_rad)
        else:
            self.solver.movel(self.target_pose)

        # UI 및 반대쪽 모드 데이터 실시간 동기화
        self.sync_ui_and_inactive_targets()

        shm.set('joint', self.solver.get_curr_joint_deg().astype(np.float32))
        shm.set('status', "run")

    def sync_ui_and_inactive_targets(self):
        """현재 로봇 상태를 읽어와서 라벨을 갱신하고, 조작 중이지 않은 슬라이더들을 추종하게 만듦"""
        curr_q_deg = self.solver.get_curr_joint_deg()
        curr_pose = self.get_current_pose_from_tf()

        # 공통 상태 표시
        self.joint_label.setText(f"Joints(deg): [" + ", ".join([f"{q:.1f}" for q in curr_q_deg]) + "]")

        # 1. MoveJ 모드일 때: Cartesian 슬라이더들이 로봇을 따라가게 함
        if self.current_mode == "movej":
            c_vals = [curr_pose[0]*1000, curr_pose[1]*1000, curr_pose[2]*1000,
                      math.degrees(curr_pose[3]), math.degrees(curr_pose[4]), math.degrees(curr_pose[5])]
            self.target_pose = curr_pose # 타겟 포즈도 현재 위치로 계속 갱신
            for i, s in enumerate(self.c_sliders):
                val = int(round(c_vals[i]))
                s.blockSignals(True)
                s.setValue(val)
                s.blockSignals(False)
                self.c_slider_labels[i].setText(f"{self.cart_names[i]}: {val}")
            # 현재 활성 라벨 업데이트
            for i, s in enumerate(self.j_sliders): self.j_slider_labels[i].setText(f"J{i+1}: {s.value()}")

        # 2. MoveL 모드일 때: Joint 슬라이더들이 로봇을 따라가게 함
        else:
            self.target_q_rad = np.deg2rad(curr_q_deg).tolist() # 타겟 각도도 현재 각도로 갱신
            for i, s in enumerate(self.j_sliders):
                val = int(round(curr_q_deg[i]))
                s.blockSignals(True)
                s.setValue(val)
                s.blockSignals(False)
                self.j_slider_labels[i].setText(f"J{i+1}: {val}")
            # 현재 활성 라벨 업데이트
            for i, s in enumerate(self.c_sliders): self.c_slider_labels[i].setText(f"{self.cart_names[i]}: {s.value()}")

    def get_current_pose_from_tf(self):
        """C++ 바인딩된 Transform 객체의 메서드를 직접 사용하여 Pose 추출"""

        # self.solver(파이썬 래퍼) 내부의 solver(C++ 객체)에서 Transform 객체 획득
        tf = self.solver.solver.get_curr_tcp_tf()

        # C++ Transform 클래스의 내장 함수 사용 (가장 빠르고 정확함)
        pos = tf.translation() # [x, y, z]
        rpy = tf.rpy()         # [roll, pitch, yaw]

        return [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SimpleIkUi()
    ex.show()
    sys.exit(app.exec())