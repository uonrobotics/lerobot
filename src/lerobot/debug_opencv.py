import cv2
import numpy as np
import time

# --- CONFIGURATION ---
RGB_INDEX = 6    
DEPTH_INDEX = 0  

def main():
    print("🚀 Initializing Cameras...")

    # ==============================
    # 1. Capture RGB Snapshot
    # ==============================
    print(f"📷 Opening RGB Camera (Index {RGB_INDEX})...")
    cap_rgb = cv2.VideoCapture(RGB_INDEX)
    cap_rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap_rgb.isOpened():
        print(f"❌ Failed to open RGB index {RGB_INDEX}")
        return

    # Warmup RGB
    for _ in range(30): cap_rgb.read()
    ret_rgb, frame_rgb = cap_rgb.read()
    cap_rgb.release()

    if not ret_rgb:
        print("❌ RGB Capture failed.")
        return
    print("   ✅ RGB Frame Captured.")

    # ==============================
    # 2. Capture Depth Snapshot
    # ==============================
    print(f"\n🌑 Opening Depth Camera (Index {DEPTH_INDEX})...")
    cap_depth = cv2.VideoCapture(DEPTH_INDEX)
    
    # Force Resolution & Disable RGB Conversion (Critical)
    cap_depth.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_depth.set(cv2.CAP_PROP_FRAME_HEIGHT, 400) 
    cap_depth.set(cv2.CAP_PROP_CONVERT_RGB, 0) # RAW Mode

    if not cap_depth.isOpened():
        print(f"❌ Failed to open Depth index {DEPTH_INDEX}")
        return

    # Warmup Depth
    for _ in range(30): cap_depth.read()
    ret_depth, frame_depth = cap_depth.read()
    cap_depth.release()

    if not ret_depth:
        print("❌ Depth Capture failed.")
        return
    print(f"   ✅ Depth Frame Captured (Shape: {frame_depth.shape}, Dtype: {frame_depth.dtype})")

    # ==============================
    # 3. Process Depth Data
    # ==============================
    
    # A) Decode Raw Bytes (The "Snow" Fix)
    if frame_depth.dtype == np.uint8:
        # Reinterpret raw bytes as 16-bit integers
        raw_16bit = frame_depth.view(np.uint16)
        
        # Reshape to 640x400
        # Sometimes buffer has padding, so we might need to flatten and crop
        try:
            raw_16bit = raw_16bit.reshape(400, 640)
        except ValueError:
            print("   ⚠️ Shape mismatch, forcing reshape...")
            raw_16bit = raw_16bit.flatten()[:400*640].reshape(400, 640)
    else:
        raw_16bit = frame_depth

    # B) Normalize for Visualization (0-3000mm -> 0-255)
    # Clip max distance to 3 meters for contrast
    depth_viz = np.clip(raw_16bit, 0, 3000)
    depth_viz = cv2.normalize(depth_viz, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # C) Apply Color Map (Jet: Blue=Close, Red=Far)
    depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

    # D) Resize Depth to match RGB Height (400 -> 480)
    depth_resized = cv2.resize(depth_color, (640, 480))

    # ==============================
    # 4. Combine & Save
    # ==============================
    combined = np.hstack((frame_rgb, depth_resized))
    
    filename = "combined_snapshot.jpg"
    cv2.imwrite(filename, combined)
    print(f"\n✅ Success! Side-by-side snapshot saved to '{filename}'")
    print("   Download this file to your laptop to check alignment.")

if __name__ == "__main__":
    main()