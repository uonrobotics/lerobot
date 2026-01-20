import time
from robots.uon_amr.uon_amr_driver import UONAMRDriver

PORT = "/dev/ttyUSB11"   # use the port that works in C++

if __name__ == "__main__":
    drv = UONAMRDriver(PORT, debug=True)
    drv.connect()

    print("\n[TEST] Turn in place")
    drv.test_turn_in_place(rpm=30, duration=2.0)

    drv.send_velocity(0.0, 0.0)
    time.sleep(0.5)

    drv.disconnect()
