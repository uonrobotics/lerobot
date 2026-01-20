import serial
import time
import threading

# --- 현재 포트 설정 ---
RC_PORT = "/dev/ttyUSB14"    # RC 조종기
MOTOR_PORT = "/dev/ttyUSB11" # 모터 드라이버

# 보드레이트 (각각 설정)
RC_BAUD = 115200 
MOTOR_BAUD = 115200

running = True

def read_rc(ser):
    """RC 데이터 읽기 (1초에 한번씩 생존신고)"""
    while running:
        try:
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                # 데이터가 들어오면 '.' 출력
                print(".", end="", flush=True)
            time.sleep(0.01)
        except Exception as e:
            print(f"\n[RC] ❌ 끊김! Error: {e}")
            break

def drive_motor(ser):
    """모터에 계속 명령 보내기"""
    while running:
        try:
            # 상태 읽기 명령 (Ping)
            # Packet: [02 03 20 2C 00 01 CRC CRC] (속도 읽기)
            pkt = b'\x02\x03\x20\x2C\x00\x01\x4E\x30'
            ser.write(pkt)
            time.sleep(0.1)
            
            if ser.in_waiting:
                ser.read(ser.in_waiting) # 버퍼 비우기
                print("^", end="", flush=True) # 응답 받음 표시
                
        except Exception as e:
            print(f"\n[Motor] ❌ 끊김! Error: {e}")
            break

def main():
    global running
    print(f"--- 듀얼 포트 테스트 시작 ---")
    print(f"RC: {RC_PORT}, Motor: {MOTOR_PORT}")
    
    try:
        # 1. RC 연결
        rc_ser = serial.Serial(RC_PORT, RC_BAUD, timeout=0.1)
        print(f"✅ RC 포트 열림")
        
        # 2. 모터 연결
        motor_ser = serial.Serial(MOTOR_PORT, MOTOR_BAUD, timeout=0.1)
        print(f"✅ 모터 포트 열림")
        
        # 3. 동시 실행
        print("테스트 중... (RC='.', Motor='^')")
        t1 = threading.Thread(target=read_rc, args=(rc_ser,))
        t2 = threading.Thread(target=drive_motor, args=(motor_ser,))
        
        t1.start()
        t2.start()
        
        # 10초 동안 유지
        time.sleep(10)
        running = False
        
        t1.join()
        t2.join()
        print("\n\n테스트 종료: 둘 다 살아있나요?")
        
    except Exception as e:
        print(f"\n❌ 포트 열기 실패: {e}")
        print("팁: 하나의 포트가 죽었거나, 권한이 없거나, 다른 프로그램이 사용 중입니다.")

if __name__ == "__main__":
    main()