import cv2
import numpy as np

DEVICE_PATH = "/dev/video6"

print(f"Opening camera: {DEVICE_PATH}")
cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)

if not cap.isOpened():
    print("CAP_V4L2 failed, trying default backend...")
    cap = cv2.VideoCapture(DEVICE_PATH)

if not cap.isOpened():
    print("Camera could not be opened at all")
    exit(1)

# Optional: set resolution, but do NOT set FOURCC initially
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Warm-up
for i in range(5):
    try:
        ret, frame = cap.read()
        print(f"Warm-up {i} ret={ret}")
    except cv2.error as e:
        print("OpenCV error during warm-up:", e)
        cap.release()
        exit(1)

# Actual frame capture
try:
    ret, frame = cap.read()
except cv2.error as e:
    print("OpenCV error during capture:", e)
    cap.release()
    exit(1)

print("ret:", ret)
if not ret or frame is None:
    print("Read failed, frame is None")
    cap.release()
    exit(1)

print("frame.shape:", frame.shape, "dtype:", frame.dtype)

if frame.ndim == 3:
    print("per-channel mean (raw):", frame.mean(axis=(0, 1)))
else:
    print("mean (raw):", frame.mean())

# If the frame has 2 channels, assume YUYV and convert to BGR
if frame.ndim == 3 and frame.shape[2] == 2:
    print("Detected 2-channel frame; assuming YUYV/YUY2. Converting to BGR...")
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
else:
    frame_bgr = frame

# Save what OpenCV sees
cv2.imwrite("opencv_test_bgr.jpg", frame_bgr)

# Also save as if it were RGB (for comparison)
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
cv2.imwrite("opencv_test_rgb_as_bgr.jpg", frame_rgb)

cap.release()
print("Saved opencv_test_bgr.jpg and opencv_test_rgb_as_bgr.jpg")
