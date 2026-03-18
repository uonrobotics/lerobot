import sys
import os

# 두산 라이브러리 링크
lib_dir = os.path.join(os.path.dirname(__file__), '../drfl_wrapper')
sys.path.append(lib_dir)

import doosan_controller_py
import time
import numpy as np


def main():
    # 로봇 IP 설정
    robot = doosan_controller_py.DoosanController("192.168.1.30")

    print("로봇 연결함")
    robot.connect()
    robot.stop()

    time.sleep(1)

    print("로봇 서보 ON")
    robot.servo_on()

    time.sleep(3)

    print("로봇 Move Idle")
    idle = [90.0, -25.0, 120.0, 9.0, 50.0, 0.0]
    robot.movej(idle, 3.0)

    time.sleep(3)
    print("로봇 RT Start")
    robot.start_rt(idle)

    try:
        while True:
            print("loop")
            joint = robot.get_current_joint()
            print(", ".join([f"{j:.2f}" for j in joint]))
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping robot...")
        robot.stop()
        robot.disconnect()

if __name__ == "__main__":
    main()


import rerun as rr
import numpy as np
import time

# 1. Rerun 초기화
rr.init("rerun_example_spiral", spawn=True)

num_points = 100
frames = 100

print("Rerun 시뮬레이션 시작...")

for t in range(frames):
    rr.set_time("frame_idx", sequence=t)

    # 수정 1: Rerun 예약어인 "log_time" 대신 "sim_time" 등 커스텀 이름 사용
    rr.set_time("sim_time", duration=t * 0.1)

    # 2. 데이터 생성
    angle = np.linspace(0, 4 * np.pi, num_points) + (t * 0.1)
    z = np.linspace(0, 5, num_points)
    x = np.cos(angle)
    y = np.sin(angle)
    points = np.vstack([x, y, z]).T

    # 색상 (0~255 범위의 RGB)
    colors = np.zeros((num_points, 3), dtype=np.uint8)
    colors[:, 0] = int(((np.sin(t * 0.1) + 1) / 2) * 255)
    colors[:, 1] = 100
    colors[:, 2] = 200

    # 3. 데이터 로그
    # rr.log("world/spiral", rr.Points3D(points, colors=colors, radii=0.05))

    # 수정 2: rr.Scalar -> rr.Scalars 로 복수형 사용
    rr.log("plots/sin_wave", rr.Scalars(np.sin(t * 0.2)))

    time.sleep(0.05)

# 데이터를 다 그리고 난 후 '다음 프레임(100)'으로 시간을 맞춤
rr.set_time("frame_idx", sequence=frames)
rr.set_time("sim_time", duration=frames * 0.1)

# 이제 화면(plots 하위)의 데이터를 비움
rr.log("plots", rr.Clear(recursive=True))

time.sleep(1) # 잠시 대기

# ==========================================
# 아예 새로운 세션으로 초기화 (화면이 싹 비워짐)
# ==========================================
rr.init("rerun_example_part2", spawn=True)

for t in range(50):
    rr.set_time("frame_idx", sequence=t)
    # 새로운 데이터 기록 (이제 깨끗한 화면에 이것만 보임)
    rr.log("plots/cos_wave", rr.Scalars(np.cos(t * 0.2)))
    time.sleep(0.05)

print("시뮬레이션 완료!")
