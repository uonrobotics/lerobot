import time
import os
import cv2
import math
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pynput import keyboard
import threading
import yaml
from multiprocessing import Process, Manager, Event, Queue, Value


# ros2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray

# my code
from timer.precision_timer import PrecisionTimer                                    # 정밀 타이머
from lerobot_dataset.communicator import Communicator
from lerobot_dataset.data_coverter import (convert_compressedImage_to_numpy,        # 데이터 컨버터
                                           convert_jointState_to_numpy_list,
                                           convert_jointTrajectory_to_numpy_list)

# ==================================================================
# 전역 변수
# ==================================================================
HF_REPO_ID = "user1/dsr1"                      # Repo_io (Dir)
IS_RECORDING = False
START_TIME = 0
TASK_DESCRIPTION = "default task"

STATUS = 'ready'
JOINT_ORDER = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
]

TF_NAMES = [f"tf_{r}{c}" for r in range(4) for c in range(4)]
# 최종 17개 이름: TF(16개) + Gripper(1개)
STATE_NAMES = TF_NAMES + ['gripper']

FEATURES = {
    'observation.images.cam_wrist': {
        'dtype': 'video',
        'shape': (480, 848, 3),
        'names': ['height', 'width', 'channels']
    },
    'observation.state': {
        'dtype': 'float32',
        'shape': (17,),
        'names': STATE_NAMES # 여기서 정의한 17개 이름 사용
    },
    'action': {
        'dtype': 'float32',
        'shape': (17,),
        'names': STATE_NAMES
    },
}


TIMER = PrecisionTimer(30) # 30hz

# ==================================================================
# Rerun
# ==================================================================
def init_rerun():
    rr.init("doosan_robot_monitoring", spawn=True)

    # 1. 그리퍼 축 범위 설정 (0 to 3500)
    gripper_axis = rrb.ScalarAxis(range=(0, 3500))

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            # 3D 뷰에서 로봇 본체와 궤적을 함께 봅니다.
            rrb.Spatial3DView(
                origin="/",
                name="Robot & Trajectory Monitor",
                # 궤적이 너무 많아질 경우 특정 경로만 끄고 켤 수 있도록 트리 구조 유지
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/camera/wrist", name="Wrist Camera"),
                rrb.TimeSeriesView(
                    origin="/robot/joints/gripper",
                    name="Gripper Status",
                    axis_y=gripper_axis
                )
            )
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)


# ==================================================================
# Rerun 로깅 수정
# ==================================================================
def log_to_rerun(msgs):
    if not msgs: return

    # Rerun 타임라인 기준점 설정
    current_time = time.time()
    rr.set_time_seconds("display_time", current_time)

    # --- 팔로워암 (Joints) ---
    dsr_follower_msg = msgs.get('follower')
    if dsr_follower_msg is not None:
        try:
            joint_map = dict(zip(dsr_follower_msg.name, dsr_follower_msg.position))
            for name in JOINT_ORDER:
                if name in joint_map:
                    pos = joint_map[name]
                    rr.log(f"/robot/joints/{name}/follower", rr.Scalars(math.degrees(pos)))
        except Exception as e:
            print(f"[에러] Follower Joint 로깅 실패: {e}")

    # --- 손목캠 (Camera) ---
    cam_msg = msgs.get('cam_wrist')
    if cam_msg is not None:
        try:
            np_arr = np.frombuffer(cam_msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                rr.log("/camera/wrist", rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        except Exception as e:
            print(f"[에러] Camera 로깅 실패: {e}")

    # --- 로봇 TCP (Axis & Trajectory) ---
    tf_msg = msgs.get('tcp_tf')
    if tf_msg is not None:
        try:
            # 1. 데이터 복원 (4x4 Matrix)
            tf_matrix = np.array(tf_msg.data).reshape(4, 4)
            translation = tf_matrix[:3, 3]      # 위치 (x, y, z)
            rotation_matrix = tf_matrix[:3, :3] # 회전 (3x3)

            # 2. 실시간 현재 위치: 축(Axis) 표현
            rr.log(
                "/robot/tcp",
                rr.Transform3D(
                    translation=translation,
                    mat3x3=rotation_matrix,
                    axis_length=0.1
                )
            )

            # 3. 궤적 잔상: static=True를 사용하여 점을 해당 위치에 고정
            # 경로 이름에 타임스탬프(ns)를 붙여 각 점을 고유한 엔티티로 만듭니다.
            # 이렇게 하면 타임라인을 옮겨도 과거의 궤적이 사라지지 않고 그대로 남습니다.
            rr.log(
                f"/robot/trajectory/point_{time.time_ns()}",
                rr.Points3D(
                    translation,
                    radii=0.003,          # 점 크기 (3mm)
                    colors=[255, 0, 0]    # 빨간색
                ),
                static=True # 타임라인 흐름과 상관없이 화면에 계속 유지
            )

        except Exception as e:
            print(f"[에러] TCP 3D 로깅 실패: {e}")

    # --- 그리퍼 (Gripper) ---
    gripper_msg = msgs.get('gripper')
    if gripper_msg is not None:
        try:
            rr.log("/robot/joints/gripper", rr.Scalars(gripper_msg.data))
        except Exception as e:
            print(f"[에러] Gripper 로깅 실패: {e}")





# ==================================================================
# 키 리스너
# ==================================================================
KEY_STATUS = {'page_down': False, 'delete': False, 'end': False}
def key_press_callback(key):
    global STATUS

    if key == keyboard.Key.page_down:
        if not KEY_STATUS['page_down']:
            KEY_STATUS['page_down'] = True
            print(f'\n[Info ] 키입력: Page Down')
            if STATUS == 'ready': STATUS = 'record'
            elif STATUS == 'record': STATUS = 'save'

    if key == keyboard.Key.end:
        if not KEY_STATUS['end']:
            KEY_STATUS['end'] = True
            print(f'\n[Info ] 키입력: End')
            if STATUS == 'ready': STATUS = 'done'

    if key == keyboard.Key.delete:
        if not KEY_STATUS['delete']:
            KEY_STATUS['delete'] = True
            print(f'\n[Info ] 키입력: Delete')
            if STATUS == 'record': STATUS = 'cancel'

def key_release_callback(key):
    if key == keyboard.Key.page_down: KEY_STATUS['page_down'] = False
    if key == keyboard.Key.end: KEY_STATUS['end'] = False
    if key == keyboard.Key.delete: KEY_STATUS['delete'] = False


# ==================================================================
# 멀티프로세싱: 커뮤니케이터
# ==================================================================
IS_COMM_FAILED = False
from multiprocessing import Process, Manager, Event
def communication_process(shared_msgs, config_path, init_event):
    import rclpy
    if not rclpy.ok():
        rclpy.init()

    node = Communicator(config_path, shared_data=shared_msgs)

    if not node.init():
        print("\n[Error] Communicator 초기화 실패. 프로세스를 종료합니다.")
        node.destroy_node()
        rclpy.shutdown()
        return

    # 초기화 성공 시에만 set
    init_event.set()

    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"Comm Process Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ==================================================================
# 멀티프로세싱: 데이터셋 저장
# ==================================================================
def dataset_writer_process(queue, repo_id, features, fps, robot_type, mode, is_saving_val):
    from lerobot_dataset.lerobot_dataset import create_lerobot_dataset

    dataset = create_lerobot_dataset(repo_id=repo_id,
                                     features=features,
                                     fps=fps,
                                     robot_type=robot_type,
                                     mode=mode)
    if dataset is None:
        return

    while True:
        try:
            msg = queue.get()
            if msg is None: break

            cmd = msg.get('cmd')

            if cmd == 'add_frame':
                dataset.add_frame(msg['data'])
            elif cmd == 'save':
                # 저장 시작: True (1)
                is_saving_val.value = 1
                print(f'\n[Dataset Process] 에피소드 저장 중...')
                dataset.save_episode()
                # 저장 완료: False (0)
                is_saving_val.value = 0
                print(f'[Dataset Process] 에피소드 저장 완료')
            elif cmd == 'cancel':
                dataset.clear_episode_buffer()
                print(f'[Info ] [Dataset Process] 에피소드 버퍼 초기화 완료.')
            elif cmd == 'done':
                dataset.finalize()
                print(f'[Info ] [Dataset Process] 데이터셋 finalize complete.')
                break
        except Exception as e:
            print(f"[Dataset Process Error] {e}")
            is_saving_val.value = 0 # 에러 발생 시 상태 초기화







# ==================================================================
# Main 함수
# ==================================================================
def main():
    global STATUS, IS_RECORDING, START_TIME, BARCODE_PUBLISHER

    # ------------------------------------------------------------------
    # 커뮤니케이터 멀티프로세싱 설정 및 실행
    # ------------------------------------------------------------------
    init_event = Event()

    manager = Manager()
    shared_msgs = manager.dict({
        'cam_wrist'       : None,
        'follower'        : None,
        'tcp_tf'          : None,
        'gripper'         : None
    })

    comm_proc = Process(target=communication_process, args=(shared_msgs, 'config/comm.config.yaml', init_event))
    comm_proc.daemon = True
    comm_proc.start()

    # wait 1
    print("[Info ] Communicator 초기화 확인 중...")
    if not init_event.wait(timeout=3.0):
        comm_proc.terminate()
        comm_proc.join()
        return

    print("[Info ] Communicator 초기화 성공.")


    # -----------------------------------------------------------
    # 데이터셋 멀티프로세싱 설정 및 실행
    # ------------------------------------------------------------------
    print("\n[Query] 데이터셋 저장 방식을 선택하세요.")
    ans = input("1: 기존 데이터 이어서 쓰기 (Resume)\n2: 삭제 후 덮어쓰기 (Overwrite)\n선택 (1 또는 2): ")
    dataset_mode = 'resume' if ans == '1' else 'overwrite'

    is_saving_val = Value('i', 0)

    dataset_queue = Queue()
    dataset_proc = Process(target=dataset_writer_process,
                           args=(dataset_queue, HF_REPO_ID, FEATURES, 30, 'omy_f3m', dataset_mode, is_saving_val))
    dataset_proc.daemon = False
    dataset_proc.start()


    # ------------------------------------------------------------------
    # ros2 및 rerun 초기화
    # ------------------------------------------------------------------
    if not rclpy.ok():
        rclpy.init()

    barcode_node = rclpy.create_node('barcode_publisher_node')
    BARCODE_PUBLISHER = barcode_node.create_publisher(Bool, '/barcode/reset', 10)

    init_rerun()
    time.sleep(0.5)


    # ------------------------------------------------------------------
    # 키보드 리스너 시작
    # ------------------------------------------------------------------
    listener = keyboard.Listener(on_press=key_press_callback, on_release=key_release_callback)
    listener.start()


    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    print(f'[Info ] 루프 시작 (30Hz) - Multi Processed Mode')
    print(f'-- Page Down: 녹화/저장 | Delete: 취소 | End: 종료')

    prev_loop_time = time.time()

    try:
        while rclpy.ok():
            TIMER.sleep()
            loop_start = time.time()

            dt = loop_start - prev_loop_time
            prev_loop_time = loop_start

            saving_status = bool(is_saving_val.value)

            # [수정] Q 대신 Saving 상태 표시
            print(f'\r[Debug] dt: {dt:.4f}s | Hz: {1.0/dt if dt > 0 else 0:.3f} | Saving: {saving_status} | Status: {STATUS}', end='')

            msgs = dict(shared_msgs)

            # --- 데이터 유효성 체크 ---
            missing_keys = [k for k, v in shared_msgs.items() if v is None]
            if missing_keys:
                print(f"[Warn ] 토픽 데이터 없음: {missing_keys}")
                continue


            # --- 데이터 변환 ---
            # 이미지
            img_wrist         = convert_compressedImage_to_numpy(msgs['cam_wrist'])

            # 팔로워 조인트 (사용 않함)
            # joint_follower    = convert_jointState_to_numpy_list(msgs['follower'], JOINT_ORDER)

            # TCP tf
            tf_flatten = np.array(msgs['tcp_tf'].data, dtype=np.float32).flatten()


            # 그리퍼
            gripper_val = msgs['gripper'].data

            # 4. TF(16)와 Gripper(1) 결합 -> (17,)
            state_tf_with_gripper = np.concatenate([tf_flatten, [gripper_val]]).astype(np.float32)



            # --- 상태 머신 로직 ---
            if STATUS == 'record':
                if not IS_RECORDING:
                    IS_RECORDING = True
                    START_TIME = time.time()

                frame_data = {
                    'observation.images.cam_wrist'  : img_wrist,
                    'observation.state'             : state_tf_with_gripper,
                    'action'                        : state_tf_with_gripper,
                    'task'                          : TASK_DESCRIPTION,
                }


                # 메인 루프에서 직접 추가하지 않고 Queue를 통해 전달
                dataset_queue.put({'cmd': 'add_frame', 'data': frame_data})

                recording_time = time.time() - START_TIME

                log_to_rerun(msgs)
                print(f' | Recording: {recording_time:.2f}s', end='')

            elif STATUS == 'save':
                print(f'\n[Info ] 데이터셋 프로세스에 저장(Save) 명령 전송...')
                IS_RECORDING = False
                dataset_queue.put({'cmd': 'save'})
                STATUS = 'ready'

            elif STATUS == 'cancel':
                print(f'\n[Info ] 데이터셋 프로세스에 취소(Cancel) 명령 전송...')
                IS_RECORDING = False
                dataset_queue.put({'cmd': 'cancel'})
                STATUS = 'ready'

            elif STATUS == 'done':
                print(f'\n[Info ] 데이터셋 프로세스에 완료(Done) 명령 전송...')
                IS_RECORDING = False
                dataset_queue.put({'cmd': 'done'})
                STATUS = 'ready'

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 사용자에 의해 종료됨")
    finally:
        # 종료 시 프로세스 정리
        dataset_queue.put({'cmd': 'done'})

        comm_proc.join(timeout=2.0)
        if comm_proc.is_alive():
            comm_proc.terminate()

        dataset_proc.join(timeout=2.0)
        if dataset_proc.is_alive():
            dataset_proc.terminate()

        listener.stop()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()