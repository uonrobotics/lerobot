import time

class PrecisionTimer:
    """
    고정밀 루프 실행을 위한 타이머 클래스
    Drift Correction(오차 누적 방지) 및 Hybrid Waiting(Sleep + Busy-wait) 기법 적용
    """
    def __init__(self, target_hz: float, busy_threshold: float = 0.002):
        self.target_hz = target_hz
        self.interval = 1.0 / target_hz
        self.busy_threshold = busy_threshold
        self.start_time = None
        self.prev_time = None
        self.loop_count = 0
        self._reset()

    def _reset(self):
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time
        self.loop_count = 0

    def sleep(self) -> tuple[float, float]:
        """
        다음 루프 주기까지 정밀하게 대기하고 dt와 error를 반환합니다.
        """
        self.loop_count += 1
        target_time = self.start_time + (self.loop_count * self.interval)

        # 1. 하이브리드 대기 (Hybrid Waiting)
        while True:
            remaining = target_time - time.perf_counter()
            if remaining <= 0:
                break
            if remaining > self.busy_threshold:
                time.sleep(remaining - self.busy_threshold)
            else:
                pass # Busy-waiting

        # 2. 시간 측정 및 결과 반환
        current_time = time.perf_counter()
        dt = current_time - self.prev_time
        error = current_time - target_time
        self.prev_time = current_time

        # 3. 지연 발생 시 기준점 재설정 (Drift Reset)
        if current_time > target_time + self.interval:
            self._reset()

        return dt, error

# 전역 변수로 단일 인스턴스 관리 (간편 호출용)
_global_sleeper = None

def precision_sleep(target_hz: float = None, busy_threshold: float = 0.002) -> tuple[float, float]:
    """
    `time.sleep()`처럼 간단하게 호출하면서도 고정밀도를 유지하는 함수입니다.

    :param target_hz: 목표 주파수 (Hz). 첫 호출 시 반드시 지정해야 합니다.
    :param busy_threshold: busy-waiting 임계 시간 (초).
    :return: (dt, error) 튜플. dt는 실제 경과 시간, error는 목표 시간과의 오차입니다.
    """
    global _global_sleeper

    # target_hz가 제공되거나 아직 초기화되지 않은 경우 새로 생성
    if target_hz is not None:
        if _global_sleeper is None or _global_sleeper.target_hz != target_hz:
            _global_sleeper = PrecisionTimer(target_hz, busy_threshold)

    if _global_sleeper is None:
        raise ValueError("첫 호출 시 target_hz를 지정하여 초기화해야 합니다.")

    return _global_sleeper.sleep()

# --- 사용 예제 ---
if __name__ == "__main__":
    print("10Hz 주기로 정밀 루프를 시작합니다. (3초간 실행)")
    print("-" * 50)

    start_test = time.perf_counter()
    try:
        while True:
            # target_hz를 지정하여 호출 (최초 1회만 지정해도 됨)
            dt, error = precision_sleep(target_hz=30)

            print(f"실행 dt: {dt:.6f}s | 오차: {error:+.6f}s")

            # --- 여기에 실행할 작업 작성 ---
            time.sleep(0.3) # 작업 부하 시뮬레이션
            # ----------------------------

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 종료")
    finally:
        pass
    print("-" * 50)
    print("테스트 종료.")
