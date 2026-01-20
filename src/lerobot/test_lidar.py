from rplidar import RPLidar

PORT = '/dev/ttyUSB12'
BAUDRATE = 460800

try:
    lidar = RPLidar(PORT, baudrate=BAUDRATE)
    info = lidar.get_info()
    print(f"✅ LiDAR 찾음! 모델: {info}")
    lidar.stop()
    lidar.disconnect()
except Exception as e:
    print(f"❌ LiDAR 아님 (또는 사용 중): {e}")