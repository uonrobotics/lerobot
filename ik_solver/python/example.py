import os
import sys
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

# ik_solver 모듈 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ik_solver import ik_solver, SharedMemory

class RobotControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uon Robotics - IK Solver Controller")
        self.root.geometry("1000x750")

        # 엔진 및 상태 관리
        self.solver = None
        self.is_running = True
        self.lock = threading.Lock()

        # 제어 목표값 및 설정값 (쓰레드 안전을 위해 별도 관리)
        self.target_q = [0.0] * 6
        self.target_pose = [0.0] * 6
        self.control_mode = tk.IntVar(value=0) # 0: MoveJ, 1: MoveL
        self.solver_mode_val = tk.IntVar(value=1) # 1: Strict, 0: Relax

        # 초기 경로 설정
        self.urdf_path = tk.StringVar(value=os.path.abspath(os.path.join(current_dir, "../contents/dsr_m1013.urdf")))

        self._setup_ui()

        # 연산 스레드 분리 (100Hz 목표)
        self.calc_thread = threading.Thread(target=self._solver_loop, daemon=True)
        self.calc_thread.start()

        # UI 업데이트 타이머 (30Hz)
        self._update_ui_periodically()

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 상단: URDF 로드 ---
        top_frame = ttk.LabelFrame(main_frame, text="Robot Setup", padding="5")
        top_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top_frame, text="URDF:").pack(side=tk.LEFT)
        ttk.Entry(top_frame, textvariable=self.urdf_path, width=70).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Load Robot", command=self._cb_load_robot).pack(side=tk.LEFT)

        body_frame = ttk.Frame(main_frame)
        body_frame.pack(fill=tk.BOTH, expand=True)

        # --- 왼쪽: 로봇 제어 ---
        ctrl_frame = ttk.LabelFrame(body_frame, text="Robot Control", padding="10")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        mode_frame = ttk.Frame(ctrl_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(mode_frame, text="MoveJ", variable=self.control_mode, value=0, command=self._cb_mode_changed).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="MoveL", variable=self.control_mode, value=1, command=self._cb_mode_changed).pack(side=tk.LEFT, padx=10)

        # Joint Control
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(ctrl_frame, text="Joint Control (deg)", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.j_vars = []
        self.j_vel_labels = []
        for i in range(6):
            f = ttk.Frame(ctrl_frame)
            f.pack(fill=tk.X)
            ttk.Label(f, text=f"J{i+1}:", width=4).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0.0)
            self.j_vars.append(var)
            s = tk.Scale(f, from_=-360, to=360, variable=var, orient=tk.HORIZONTAL, resolution=0.01, showvalue=False, length=250, command=lambda e: self._cb_ui_input())
            s.pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=var, width=8).pack(side=tk.LEFT, padx=5)
            v_lbl = ttk.Label(f, text="v: 0.0", foreground="gray")
            v_lbl.pack(side=tk.LEFT)
            self.j_vel_labels.append(v_lbl)

        # Task Control
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(ctrl_frame, text="Task Control (m / deg)", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.t_vars = []
        self.t_vel_labels = []
        labels = ["X:", "Y:", "Z:", "R:", "P:", "Y:"]
        ranges = [(-1.5, 1.5), (-1.5, 1.5), (0, 1.5), (-360, 360), (-360, 360), (-360, 360)]
        for i in range(6):
            f = ttk.Frame(ctrl_frame)
            f.pack(fill=tk.X)
            ttk.Label(f, text=labels[i], width=4).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0.0)
            self.t_vars.append(var)
            s = tk.Scale(f, from_=ranges[i][0], to=ranges[i][1], variable=var, orient=tk.HORIZONTAL, resolution=0.001 if i<3 else 0.1, showvalue=False, length=250, command=lambda e: self._cb_ui_input())
            s.pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=var, width=8).pack(side=tk.LEFT, padx=5)
            v_lbl = ttk.Label(f, text="v: 0.0", foreground="gray")
            v_lbl.pack(side=tk.LEFT)
            self.t_vel_labels.append(v_lbl)

        # --- 오른쪽: 시스템 설정 ---
        set_frame = ttk.LabelFrame(body_frame, text="System Settings", padding="10")
        set_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        # Speed & Safety
        ttk.Label(set_frame, text="Speed & Safety", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.set_tcp_speed = tk.DoubleVar(value=1.0)
        sf1 = ttk.Frame(set_frame); sf1.pack(fill=tk.X)
        tk.Scale(sf1, from_=0, to=2, variable=self.set_tcp_speed, orient=tk.HORIZONTAL, resolution=0.1, showvalue=False).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(sf1, textvariable=self.set_tcp_speed, width=5).pack(side=tk.LEFT)

        self.set_safety = tk.DoubleVar(value=1.2)
        ttk.Label(set_frame, text="Safety Scale (1.0~2.0):").pack(anchor=tk.W, pady=(5,0))
        sf2 = ttk.Frame(set_frame); sf2.pack(fill=tk.X)
        tk.Scale(sf2, from_=1, to=2, variable=self.set_safety, orient=tk.HORIZONTAL, resolution=0.05, showvalue=False).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(sf2, textvariable=self.set_safety, width=5).pack(side=tk.LEFT)

        # End Effector Offset
        ttk.Separator(set_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(set_frame, text="End Effector Offset", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.ee_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        for i, l in enumerate(["X (m):", "Y (m):", "Z (m):"]):
            ef = ttk.Frame(set_frame); ef.pack(fill=tk.X)
            ttk.Label(ef, text=l, width=8).pack(side=tk.LEFT)
            ttk.Entry(ef, textvariable=self.ee_vars[i], width=10).pack(side=tk.LEFT)
        ttk.Button(set_frame, text="Apply EE Offset", command=self._cb_apply_ee).pack(fill=tk.X, pady=5)

        # Solver Mode (Relax/Strict)
        ttk.Separator(set_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(set_frame, text="Solver Mode", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        ttk.Radiobutton(set_frame, text="RELAX PATH (Speed)", variable=self.solver_mode_val, value=0, command=self._cb_apply_mode).pack(anchor=tk.W)
        ttk.Radiobutton(set_frame, text="STRICT PATH (Accuracy)", variable=self.solver_mode_val, value=1, command=self._cb_apply_mode).pack(anchor=tk.W)

        ttk.Button(set_frame, text="Apply All Settings", command=self._cb_apply_settings).pack(fill=tk.X, pady=20)

    # --- Callbacks ---
    def _cb_load_robot(self):
        try:
            self.solver = ik_solver(self.urdf_path.get())
            if self.solver.init():
                messagebox.showinfo("Success", "Robot Engine Loaded.")
                q = self.solver.get_current_joints()
                with self.lock:
                    self.target_q = q.tolist()
                for i, val in enumerate(q):
                    self.j_vars[i].set(round(val, 2))
                self._cb_apply_settings()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _cb_ui_input(self):
        with self.lock:
            self.target_q = [v.get() for v in self.j_vars]
            self.target_pose = [v.get() for v in self.t_vars]

    def _cb_mode_changed(self):
        if not self.solver: return
        with self.lock:
            if self.control_mode.get() == 1: # MoveL 전환
                tf = self.solver.get_tcp_tf()
                pos = tf[:3, 3]
                rpy = self.solver.get_tcp_rpy()
                for i in range(3): self.target_pose[i] = pos[i]; self.t_vars[i].set(round(pos[i], 3))
                for i in range(3): self.target_pose[i+3] = rpy[i]; self.t_vars[i+3].set(round(rpy[i], 1))
            else: # MoveJ 전환
                q = self.solver.get_current_joints()
                for i in range(6): self.target_q[i] = q[i]; self.j_vars[i].set(round(q[i], 2))

    def _cb_apply_ee(self):
        if not self.solver: return
        self.solver.set_end_effector(self.ee_vars[0].get(), self.ee_vars[1].get(), self.ee_vars[2].get(), 0, 0, 0)

    def _cb_apply_mode(self):
        if not self.solver: return
        if self.solver_mode_val.get() == 1: self.solver.set_strict_mode()
        else: self.solver.set_relax_mode()

    def _cb_apply_settings(self):
        if not self.solver: return
        self.solver.set_tcp_max_speed(self.set_tcp_speed.get())
        if hasattr(self.solver, 'solver'):
            self.solver.solver.set_safety_scale(self.set_safety.get())
        self._cb_apply_mode()
        self._cb_apply_ee()

    # --- Core Loops ---
    def _solver_loop(self):
        fields_config = {'joint': (6, np.float32), 'status': (20, str)}
        shm = None
        try: shm = SharedMemory(name='movej', fields_config=fields_config)
        except: pass

        target_dt = 1.0/30.0 # 100Hz
        while self.is_running:
            loop_start = time.perf_counter()

            if self.solver and self.solver.is_initialized:
                with self.lock:
                    mode = self.control_mode.get()
                    q_in = list(self.target_q)
                    p_in = list(self.target_pose)

                try:
                    if mode == 0:
                        self.solver.movej(np.radians(q_in))
                    else:
                        # p_in: [x, y, z, r, p, y] - r, p, y are degrees in UI.
                        # ik_solver.movel expects radians for rotation.
                        p_rad = list(p_in)
                        p_rad[3:] = np.radians(p_rad[3:])
                        self.solver.movel(p_rad)

                    if shm:
                        shm.set('joint', self.solver.get_current_joints().astype(np.float32))
                        shm.set('status', "run")
                except Exception as e:
                    print(f"Solver Error: {e}")

            # 정밀한 주기 유지를 위한 동적 수면
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, target_dt - elapsed))

    def _update_ui_periodically(self):
        if self.solver and self.solver.is_initialized:
            try:
                j_vel = self.solver.get_current_jvel()
                t_vel = self.solver.get_current_tcp_speed()
                curr_q = self.solver.get_current_joints()
                tf = self.solver.get_tcp_tf()
                pos = tf[:3, 3]
                rpy = self.solver.get_tcp_rpy()

                for i in range(6):
                    self.j_vel_labels[i].config(text=f"v: {j_vel[i]:.1f}")
                    self.t_vel_labels[i].config(text=f"v: {t_vel[i]:.3f}")

                if self.control_mode.get() == 1: # MoveL 중일 때 Joint UI 역갱신
                    for i in range(6): self.j_vars[i].set(round(curr_q[i], 2))
                else: # MoveJ 중일 때 Task UI 역갱신
                    for i in range(3): self.t_vars[i].set(round(pos[i], 3))
                    for i in range(3): self.t_vars[i+3].set(round(rpy[i], 1))
            except Exception: pass

        if self.is_running:
            self.root.after(33, self._update_ui_periodically)

    def on_close(self):
        self.is_running = False
        time.sleep(0.1)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotControlApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()