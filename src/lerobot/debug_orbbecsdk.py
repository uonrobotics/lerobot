#!/usr/bin/env python3

import os
import time
import cv2
import numpy as np
from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
    OBPropertyID,
)

OUT_DIR = "orbbec_sdk_test"
os.makedirs(OUT_DIR, exist_ok=True)

WIDTH  = 640
HEIGHT = 480
FPS    = 30
MIN_MM = 200
MAX_MM = 3000


def try_color_profile(pipeline, config):
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

    for fmt in (OBFormat.RGB, OBFormat.MJPG):
        try:
            profile = profiles.get_video_stream_profile(WIDTH, HEIGHT, fmt, FPS)
            config.enable_stream(profile)
            print(f"✅ COLOR enabled: {fmt}")
            return fmt
        except Exception as e:
            print(f"❌ COLOR format failed: {fmt} ({e})")

    raise RuntimeError("No usable COLOR format (RGB or MJPG)")


def try_depth_profile(pipeline, config):
    profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    for (w, h) in ((640, 400), (WIDTH, HEIGHT)):
        try:
            profile = profiles.get_video_stream_profile(w, h, OBFormat.Y16, FPS)
            config.enable_stream(profile)
            print(f"✅ DEPTH enabled: {w}x{h} Y16")
            return (w, h)
        except Exception:
            pass

    raise RuntimeError("No usable DEPTH profile")


def _set_bool_property_legacy(sensor, prop_id, enable: bool) -> bool:
    """
    Your pyorbbecsdk binding does not have set_bool_property().
    Many legacy bindings expose set_property(prop_id, value) where booleans are 1/0.
    Returns True if a setter succeeded, False otherwise.
    """
    val = 1 if enable else 0

    # Most common legacy API
    if hasattr(sensor, "set_property"):
        sensor.set_property(prop_id, val)
        return True

    # Some bindings expose set_int_property
    if hasattr(sensor, "set_int_property"):
        sensor.set_int_property(prop_id, val)
        return True

    return False


def enable_camera_options(pipeline):
    """
    Enable COLOR auto-exposure and DEPTH laser/emitter.
    Because bindings vary, we try:
      1) sensor.set_property(...)
      2) device.set_property(...) (fallback)
    """
    device = pipeline.get_device()

    # --- COLOR AE ---
    try:
        color_sensor = device.get_sensor(OBSensorType.COLOR_SENSOR)
        ok = _set_bool_property_legacy(
            color_sensor,
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
            True,
        )
        if ok:
            print("✅ Color auto-exposure enabled (sensor)")
        else:
            raise AttributeError("No compatible setter on sensor")
    except Exception as e1:
        # Fallback: some bindings set properties directly on device
        try:
            if hasattr(device, "set_property"):
                device.set_property(
                    OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, 1
                )
                print("✅ Color auto-exposure enabled (device)")
            else:
                print("⚠️ Failed to enable color AE (no setter):", e1)
        except Exception as e2:
            print("⚠️ Failed to enable color AE:", e2)

    # --- DEPTH LASER ---
    try:
        depth_sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)
        ok = _set_bool_property_legacy(
            depth_sensor,
            OBPropertyID.OB_PROP_LASER_BOOL,
            True,
        )
        if ok:
            print("✅ Depth laser enabled (sensor)")
        else:
            raise AttributeError("No compatible setter on sensor")
    except Exception as e1:
        # Fallback: device-level property
        try:
            if hasattr(device, "set_property"):
                device.set_property(OBPropertyID.OB_PROP_LASER_BOOL, 1)
                print("✅ Depth laser enabled (device)")
            else:
                print("⚠️ Failed to enable depth laser (no setter):", e1)
        except Exception as e2:
            print("⚠️ Failed to enable depth laser:", e2)


def process_color_frame(frame):
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()

    # Make buffer contiguous (your binding can return non-contiguous views)
    buf = np.asarray(frame.get_data(), dtype=np.uint8)

    print("Color frame actual format:", fmt)
    print("Color buffer size:", buf.size, "expected:", w * h * 3)

    if fmt == OBFormat.RGB:
        # SDK gives RGB
        return buf.reshape((h, w, 3))

    if fmt == OBFormat.MJPG:
        # MJPG decodes to BGR then convert to RGB
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("MJPG decode failed (cv2.imdecode returned None)")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise RuntimeError(f"Unsupported COLOR format: {fmt}")


def process_depth_frame(frame):
    w = frame.get_width()
    reported_h = frame.get_height()
    scale = float(frame.get_depth_scale())

    # Make buffer contiguous
    buf = np.asarray(frame.get_data(), dtype=np.uint16)

    # Infer true height from buffer length (handles padded frames)
    h = buf.size // w
    if h * w != buf.size:
        raise RuntimeError(
            f"Depth buffer size {buf.size} is not divisible by width {w}"
        )

    depth_u16 = buf.reshape((h, w))
    print(f"Depth buffer inferred shape: {h}x{w} (reported h={reported_h})")

    # If padded (e.g., 800x640 buffer but reported 400x640), crop to reported height
    if h != reported_h:
        depth_u16 = depth_u16[:reported_h, :]

    # Convert to mm
    if scale < 0.01:
        depth_mm = (depth_u16.astype(np.float32) * scale * 1000.0)
    else:
        depth_mm = (depth_u16.astype(np.float32) * scale)

    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    # Visualization
    invalid = (depth_mm == 0) | (depth_mm == 65535)
    clipped = np.clip(depth_mm.astype(np.int32), MIN_MM, MAX_MM).astype(np.uint16)

    depth_8u = cv2.normalize(
        clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    depth_8u[invalid] = 0

    depth_vis_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    depth_vis_rgb = cv2.cvtColor(depth_vis_bgr, cv2.COLOR_BGR2RGB)

    return depth_mm, depth_vis_rgb


def main():
    pipeline = Pipeline()
    config = Config()

    _ = try_color_profile(pipeline, config)
    _ = try_depth_profile(pipeline, config)

    pipeline.start(config)

    # Enable AE/laser if the binding supports property setting
    enable_camera_options(pipeline)

    # Let AE + laser stabilize
    time.sleep(1.5)

    print("\nWarming up...")
    for _ in range(30):
        pipeline.wait_for_frames(100)

    frames = pipeline.wait_for_frames(1000)
    if frames is None:
        pipeline.stop()
        raise RuntimeError("No frames received")

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    print("Color frame present:", color_frame is not None)
    print("Depth frame present:", depth_frame is not None)

    if color_frame:
        cbuf = np.asarray(color_frame.get_data(), dtype=np.uint8)
        print("Color min/max:", int(cbuf.min()), int(cbuf.max()))

    if depth_frame:
        dbuf = np.asarray(depth_frame.get_data(), dtype=np.uint16)
        print("Depth min/max:", int(dbuf.min()), int(dbuf.max()))

    if color_frame is None or depth_frame is None:
        pipeline.stop()
        raise RuntimeError("Missing color or depth frame")

    rgb = process_color_frame(color_frame)
    depth_raw, depth_vis = process_depth_frame(depth_frame)

    cv2.imwrite(
        os.path.join(OUT_DIR, "sdk_color_rgb.png"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(OUT_DIR, "sdk_depth_raw.png"), depth_raw)
    cv2.imwrite(
        os.path.join(OUT_DIR, "sdk_depth_vis.png"),
        cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR),
    )

    print("\nSaved outputs:")
    print("  sdk_color_rgb.png")
    print("  sdk_depth_raw.png")
    print("  sdk_depth_vis.png")

    print("\nStats:")
    print("  RGB mean:", rgb.mean(axis=(0, 1)))
    print("  Depth min/max:", int(depth_raw.min()), int(depth_raw.max()))

    pipeline.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
