#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.uon_amr.uon_amr import UONAMR, UONAMRConfig
from lerobot.cameras.opencv import OpenCVCameraConfig


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check(condition: bool, ok_msg: str, fail_msg: str):
    if condition:
        print(f"[OK]   {ok_msg}")
    else:
        print(f"[FAIL] {fail_msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic script for UONAMR robot integration before running lerobot-record."
    )
    parser.add_argument("--motor_port", default="/dev/ttyUSB11")
    parser.add_argument("--lidar_port", default="/dev/ttyUSB12")
    parser.add_argument("--lidar_baudrate", type=int, default=460800)
    parser.add_argument("--camera", default="/dev/video6", help="Camera device path, e.g. /dev/video6")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=10, help="Number of observation frames to grab")
    parser.add_argument(
        "--save_frame",
        action="store_true",
        help="Save first camera frame as 'uon_amr_debug_frame.jpg'",
    )
    parser.add_argument(
        "--motion_test",
        action="store_true",
        help="Send a small motion command (use only if robot is in a safe environment).",
    )

    args = parser.parse_args()

    print_header("1. Building UONAMRConfig")

    cfg = UONAMRConfig(
        motor_port=args.motor_port,
        lidar_port=args.lidar_port,
        lidar_baudrate=args.lidar_baudrate,
        cameras={
            "front_cam": OpenCVCameraConfig(
                index_or_path=args.camera,
                width=args.width,
                height=args.height,
                fps=30,
            )
        },
    )

    print("[INFO] UONAMRConfig created:")
    print(f"       motor_port   = {cfg.motor_port}")
    print(f"       lidar_port   = {cfg.lidar_port}")
    print(f"       camera       = {cfg.cameras['front_cam'].index_or_path}")
    print(f"       resolution   = {cfg.camera_w}x{cfg.camera_h}")

    print_header("2. Instantiating UONAMR robot")

    try:
        robot = UONAMR(cfg)
        print("[OK]   UONAMR instance created.")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate UONAMR: {e}")
        return

    # -------------------------------------------------------------------------
    # Connect
    # -------------------------------------------------------------------------
    print_header("3. Connecting robot (motor, lidar, camera)")

    try:
        robot.connect()
        time.sleep(2.0)  # warm-up for camera and lidar
    except Exception as e:
        print(f"[FAIL] robot.connect() raised an exception: {e}")
        return

    check(robot.is_connected, "robot.is_connected == True", "robot.is_connected is False after connect()")

    # -------------------------------------------------------------------------
    # Check feature definitions
    # -------------------------------------------------------------------------
    print_header("4. Checking observation_features and action_features")

    obs_features = robot.observation_features
    act_features = robot.action_features

    print("[INFO] observation_features keys:", list(obs_features.keys()))
    print("[INFO] action_features keys:", list(act_features.keys()))

    expected_obs_keys = {
        "observation.images.front_cam",
        "observation.state",
        "observation.lidar",
    }
    check(
        expected_obs_keys.issubset(set(obs_features.keys())),
        f"observation_features contain {expected_obs_keys}",
        f"observation_features missing at least one of {expected_obs_keys}",
    )

    check(
        "action" in act_features,
        "action_features contain key 'action'",
        "action_features missing key 'action'",
    )

    # -------------------------------------------------------------------------
    # Get observations repeatedly
    # -------------------------------------------------------------------------
    print_header("5. Grabbing observations")

    first_img_saved = False
    lidar_nonzero_total = 0

    for i in range(args.num_frames):
        try:
            obs = robot.get_observation()
        except Exception as e:
            print(f"[FAIL] get_observation() failed on iteration {i}: {e}")
            break

        # 5.1 Camera
        img = obs.get("observation.images.front_cam", None)
        if img is None:
            print(f"[FAIL] iteration {i}: 'observation.images.front_cam' not in observation dict")
        else:
            if isinstance(img, np.ndarray):
                print(
                    f"[INFO] iteration {i}: image shape={img.shape}, dtype={img.dtype}, "
                    f"min={img.min()}, max={img.max()}"
                )
                if img.ndim == 3 and img.shape[2] == 3:
                    # Check that it is not all zeros and has some variation
                    nonzero = np.count_nonzero(img)
                    std_per_channel = img.reshape(-1, 3).std(axis=0)
                    print(
                        f"       nonzero_pixels={nonzero}, "
                        f"std_per_channel={std_per_channel}"
                    )
                    if i == 0 and args.save_frame:
                        # save for manual inspection
                        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out_path = Path("uon_amr_debug_frame.jpg")
                        cv2.imwrite(str(out_path), bgr)
                        print(f"[OK]   Saved first frame to {out_path.resolve()}")
                        first_img_saved = True
                else:
                    print(f"[WARN] iteration {i}: unexpected image shape, expected (H,W,3)")
            else:
                print(f"[FAIL] iteration {i}: image is not a numpy array (type={type(img)})")

        # 5.2 State
        state = obs.get("observation.state", None)
        if state is None:
            print(f"[FAIL] iteration {i}: 'observation.state' missing")
        else:
            print(f"[INFO] iteration {i}: state={state}, shape={np.shape(state)}")

        # 5.3 LiDAR
        lidar = obs.get("observation.lidar", None)
        if lidar is None:
            print(f"[WARN] iteration {i}: 'observation.lidar' missing (LIDAR may be disabled)")
        else:
            if isinstance(lidar, np.ndarray):
                nonzero = int(np.count_nonzero(lidar))
                lidar_nonzero_total += nonzero
                print(
                    f"[INFO] iteration {i}: lidar shape={lidar.shape}, "
                    f"nonzero count={nonzero}, min={lidar.min()}, max={lidar.max()}"
                )
            else:
                print(f"[FAIL] iteration {i}: lidar is not a numpy array (type={type(lidar)})")

        time.sleep(0.1)

    # -------------------------------------------------------------------------
    # Test send_action()
    # -------------------------------------------------------------------------
    print_header("6. Testing send_action() at zero velocity")

    try:
        dummy_action = np.array([0.0, 0.0], dtype=np.float32)
        result = robot.send_action(dummy_action)
        print(f"[OK]   send_action([0.0, 0.0]) succeeded, returned: {result}")
    except Exception as e:
        print(f"[FAIL] send_action([0.0, 0.0]) raised an exception: {e}")

    if args.motion_test:
        print_header("7. Motion test (small command)")
        print(
            "[WARN] This will send a small non-zero velocity command to the robot. "
            "Ensure the environment is safe."
        )
        try:
            dummy_action = np.array([0.1, 0.0], dtype=np.float32)
            robot.send_action(dummy_action)
            print("[INFO] Sent small forward command for 1 second...")
            time.sleep(1.0)
            robot.send_action(np.array([0.0, 0.0], dtype=np.float32))
            print("[OK]   Motion test completed and stopped.")
        except Exception as e:
            print(f"[FAIL] Motion test failed: {e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_header("8. Summary")

    if first_img_saved:
        print("Camera:")
        print("  - Image frames returned without exceptions.")
        print("  - First frame saved as 'uon_amr_debug_frame.jpg' for manual inspection.")
    else:
        print("Camera:")
        print("  - Frames were captured (see logs above).")
        print("  - No frame saved (use --save_frame to save one).")

    if lidar_nonzero_total > 0:
        print("LiDAR:")
        print(f"  - Non-zero measurements observed over {args.num_frames} frames: {lidar_nonzero_total}")
    else:
        print("LiDAR:")
        print("  - No non-zero measurements observed (check lidar connection/port/baudrate).")

    print("\nIf camera frames look correct and no [FAIL] messages remain above,")
    print("you are ready to try `lerobot-record` with robot.type=uon_amr.")


if __name__ == "__main__":
    main()
