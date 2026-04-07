import os
import argparse
import torch
import cv2
from pathlib import Path
from multiprocessing import Process, Manager, Event, Queue
from pynput import keyboard
import time

# ros2
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from custom_interfaces.srv import LatestSensors

# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame

# my code
from lerobot_dataset.communicator import Communicator
from timer.precision_timer import PrecisionTimer
from lerobot_dataset.data_coverter import (
    convert_compressedImage_to_numpy,
    convert_jointState_to_numpy_list,
)
# ==================================================================
# globals
# ==================================================================
DEFAULT_SAVE_ROOT_PATH = Path.home() / ".cache/huggingface/lerobot"
TIMER = PrecisionTimer(16)  # 60hz
TOP_CAM_RESIZE_TO = (848, 480)  # (width, height)

MODEL_PATH = "/nas/Dataset/dualarm_ckpts/0325_groot_proj_diffusion_base_lr5e5_clip10_b4/030000/pretrained_model"
DATASET_PATH = "/nas/Dataset/dualarm_data/0317_dualarm_cashier/20260325_dual_dataset_for_groot_scan"

TASK_DESCRIPTION = "pick the object, scan the barcode, and return it"
ROBOT_TYPE = "omy_f3m"

JOINT_ORDER_R = [
    "robot1/joint1",
    "robot1/joint2",
    "robot1/joint3",
    "robot1/joint4",
    "robot1/joint5",
    "robot1/joint6",
    "robot1/rh_r1_joint",
]

JOINT_ORDER_L = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "rh_r1_joint",
]
infer_t = 0


def _stamp_to_ns(msg):
    if not hasattr(msg, "header") or msg.header is None:
        return None
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def communication_process(shared_msgs, config_path, init_event, send_queue):
    """별도 프로세스에서 실행될 ROS2 노드 루틴"""
    if not rclpy.ok():
        rclpy.init()

    node = Communicator(config_path, shared_data=shared_msgs, send_queue=send_queue)

    if not node.init():
        if rclpy.ok():
            rclpy.shutdown()
        return

    init_event.set()  # 초기화 완료 알림

    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"Comm Process Error: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class LatestSensorClient(Node):
    def __init__(self):
        super().__init__("latest_sensor_client")
        self.cli = self.create_client(LatestSensors, "/comm/get_latest_sensors")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("서비스 대기 중: /comm/get_latest_sensors")
        self.req = LatestSensors.Request()
        self.req.request_latest = True

    def request_latest(self, timeout_sec=0.03):
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if future.done():
            return future.result()
        return None


# ==================================================================
# key listener
# ==================================================================
OBSERVATION_STAGE = 0
KEY_STATUS = {"space": False, "up": False, "down": False}


def key_press_callback(key):
    global OBSERVATION_STAGE

    if key == keyboard.Key.space:
        if not KEY_STATUS["space"]:
            KEY_STATUS["space"] = True
            OBSERVATION_STAGE = 0
            print(f"[Info ] Observation Stage: {OBSERVATION_STAGE}")

    if key == keyboard.Key.up:
        if not KEY_STATUS["up"]:
            KEY_STATUS["up"] = True
            OBSERVATION_STAGE = OBSERVATION_STAGE + 1
            print(f"[Info ] Observation Stage: {OBSERVATION_STAGE}")

    if key == keyboard.Key.down:
        if not KEY_STATUS["down"]:
            KEY_STATUS["down"] = True
            OBSERVATION_STAGE = OBSERVATION_STAGE - 1


def key_release_callback(key):
    if key == keyboard.Key.space:
        KEY_STATUS["space"] = False
    if key == keyboard.Key.up:
        KEY_STATUS["up"] = False
    if key == keyboard.Key.down:
        KEY_STATUS["down"] = False


def main():
    # ------------------------------------------------------------------
    # parse args
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/comm.config.yaml")
    parser.add_argument("--config", type=str, default=default_config)
    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------
    # keyboard listener
    # ------------------------------------------------------------------
    listener = keyboard.Listener(on_press=key_press_callback, on_release=key_release_callback)
    listener.start()

    # ------------------------------------------------------------------
    # multiprocessing resources
    # ------------------------------------------------------------------
    manager = Manager()
    shared_msgs = manager.dict(
        {
            "cam_top": None,
            "cam_wristR": None,
            "cam_wristL": None,
            "followerR": None,
            "followerL": None,
            "scan_pulse": None,
            "scan_done": None,
        }
    )
    send_queue = Queue()
    init_event = Event()

    # ------------------------------------------------------------------
    # communicator process
    # ------------------------------------------------------------------
    comm_proc = Process(target=communication_process, args=(shared_msgs, args.config, init_event, send_queue))
    comm_proc.daemon = True
    comm_proc.start()

    print("[Info] Communicator 초기화 대기 중...")
    if not init_event.wait(timeout=5.0):
        print("[Error] 초기화 실패")
        comm_proc.terminate()
        return

    # ------------------------------------------------------------------
    # GROOT model load
    # ------------------------------------------------------------------
    device = torch.device("cuda")
    model = GrootPolicy.from_pretrained(MODEL_PATH, strict=False, low_cpu_mem_usage=False, device_map=None)
    model.to(device)
    model.eval()

    dataset_metadata = LeRobotDatasetMetadata(DATASET_PATH)
    preprocess, postprocess = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=MODEL_PATH,
        dataset_stats=dataset_metadata.stats,
    )

    print("[Info] GROOT 모델 불러옴")
    print("[Info] Inference 모듈 시작")

    if not rclpy.ok():
        rclpy.init()
    latest_sensor_client = LatestSensorClient()
    last_warn_time = 0.0

    try:
        while True:
            TIMER.sleep()

            latest_resp = latest_sensor_client.request_latest(timeout_sec=0.03)
            if latest_resp is None:
                now = time.time()
                if now - last_warn_time > 1.0:
                    print("[Warn ] latest sensor 서비스 응답 타임아웃")
                    last_warn_time = now
                continue

            if not latest_resp.success:
                now = time.time()
                if now - last_warn_time > 1.0:
                    print(f"[Warn ] latest sensor 서비스 실패: {latest_resp.reason}")
                    last_warn_time = now
                continue

            msgs = {
                "cam_top": latest_resp.cam_top,
                "cam_wristR": latest_resp.cam_wrist_r,
                "cam_wristL": latest_resp.cam_wrist_l,
                "followerR": latest_resp.follower_r,
                "followerL": latest_resp.follower_l,
                "scan_pulse": latest_resp.scan_pulse,
                "scan_done": latest_resp.scan_done,
            }

            # --- convert incoming data ---
            img_top = convert_compressedImage_to_numpy(msgs["cam_top"])
            img_wristL = convert_compressedImage_to_numpy(msgs["cam_wristL"])
            img_wristR = convert_compressedImage_to_numpy(msgs["cam_wristR"])

            # Match top camera resolution to training-time setting.
            # OpenCV expects size as (width, height).
            img_top = cv2.resize(img_top, TOP_CAM_RESIZE_TO, interpolation=cv2.INTER_AREA)
            # img_wristL = cv2.resize(img_wristL, TOP_CAM_RESIZE_TO, interpolation=cv2.INTER_AREA)
            # img_wristR = cv2.resize(img_wristR, TOP_CAM_RESIZE_TO, interpolation=cv2.INTER_AREA)

            followerR_numpy = convert_jointState_to_numpy_list(msgs["followerR"], JOINT_ORDER_R)
            followerL_numpy = convert_jointState_to_numpy_list(msgs["followerL"], JOINT_ORDER_L)

            barcode_pulse = float(msgs["scan_pulse"].data)
            barcode_done = float(msgs["scan_done"].data)

            # --- raw observation ---
            obs = {
                "cam_top": img_top,
                "cam_wristL": img_wristL,
                "cam_wristR": img_wristR,
                "robot1/joint1": followerR_numpy[0],
                "robot1/joint2": followerR_numpy[1],
                "robot1/joint3": followerR_numpy[2],
                "robot1/joint4": followerR_numpy[3],
                "robot1/joint5": followerR_numpy[4],
                "robot1/joint6": followerR_numpy[5],
                "robot1/rh_r1_joint": followerR_numpy[6],
                "joint1": followerL_numpy[0],
                "joint2": followerL_numpy[1],
                "joint3": followerL_numpy[2],
                "joint4": followerL_numpy[3],
                "joint5": followerL_numpy[4],
                "joint6": followerL_numpy[5],
                "rh_r1_joint": followerL_numpy[6],
                "scan_pulse": barcode_pulse,
                "scan_done": barcode_done,
            }

            # --- build frame + preprocess (required for GROOT) ---
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_metadata.features,
                device=device,
                task=TASK_DESCRIPTION,
                robot_type=ROBOT_TYPE,
            )
            obs_processed = preprocess(obs_frame)

            # --- inference ---
            infer_start_ns = time.time_ns()
            with torch.inference_mode():
                action = model.select_action(obs_processed)
            action = postprocess(action)
            infer_end_ns = time.time_ns()
            infer_ms = (infer_end_ns - infer_start_ns) / 1_000_000.0

            # --- timing / sensor freshness log ---
            top_stamp_ns = _stamp_to_ns(msgs["cam_top"])
            wrist_l_stamp_ns = _stamp_to_ns(msgs["cam_wristL"])
            wrist_r_stamp_ns = _stamp_to_ns(msgs["cam_wristR"])
            follower_r_stamp_ns = _stamp_to_ns(msgs["followerR"])
            follower_l_stamp_ns = _stamp_to_ns(msgs["followerL"])

            def _age_ms(stamp_ns):
                if stamp_ns is None:
                    return -1.0
                return (infer_start_ns - stamp_ns) / 1_000_000.0

            print(
                "[Timing] "
                f"infer={infer_ms:.2f}ms "
                f"snapshot_seq={latest_resp.snapshot_seq} "
                f"cam_top_age={_age_ms(top_stamp_ns):.1f}ms "
                f"cam_wristL_age={_age_ms(wrist_l_stamp_ns):.1f}ms "
                f"cam_wristR_age={_age_ms(wrist_r_stamp_ns):.1f}ms "
                f"followerR_age={_age_ms(follower_r_stamp_ns):.1f}ms "
                f"followerL_age={_age_ms(follower_l_stamp_ns):.1f}ms"
            )

            # --- extract action ---
            action_np = action.squeeze().cpu().numpy()
            robot1_action = action_np[-7:].tolist()
            robot2_action = action_np[:7].tolist()

            # --- right robot control (Robot 1) ---
            joint_msg = JointTrajectory()
            joint_msg.joint_names = JOINT_ORDER_L
            point = JointTrajectoryPoint()
            point.positions = robot1_action
            point.time_from_start = Duration(sec=0, nanosec=33000000)
            joint_msg.points = [point]

            # --- left robot control (Robot 2) ---
            joint_msg_L = JointTrajectory()
            joint_msg_L.joint_names = JOINT_ORDER_R
            point_L = JointTrajectoryPoint()
            point_L.positions = robot2_action
            point_L.time_from_start = Duration(sec=0, nanosec=33000000)
            joint_msg_L.points = [point_L]

            send_queue.put(("robot1", joint_msg_L))
            send_queue.put(("robot2", joint_msg))

    except KeyboardInterrupt:
        print("종료 중...")
    finally:
        if "latest_sensor_client" in locals():
            latest_sensor_client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
