import cv2
import numpy as np

# On Gemini 336, /dev/video0 or /dev/video2 are usually IR/Depth
for path in ["/dev/video0", "/dev/video2"]:
    print(f"Testing {path} for raw IR...")
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    
    # IR is often 16-bit (Y16), but OpenCV might see it as 8-bit GREY
    # Try setting a typical IR resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    ret, frame = cap.read()
    if ret and frame is not None:
        # If it's 16-bit data, normalize so we can see it
        norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(f"raw_v4l2_ir_{path.split('/')[-1]}.jpg", norm)
        print(f"SUCCESS! Saved raw_v4l2_ir_{path.split('/')[-1]}.jpg")
    else:
        print(f"Failed to read from {path}")
    cap.release()