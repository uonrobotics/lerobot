import numpy as np
from multiprocessing import shared_memory, resource_tracker
import atexit

class SharedMemory:
    def __init__(self, name, fields_config):
        self.name = name
        self.fields = {}
        offset = 0

        # 필드 설정 계산 (기존 유지)
        for f_name, (count, dtype) in fields_config.items():
            if dtype == str:
                item_size = 1
                byte_size = count
            else:
                item_size = np.dtype(dtype).itemsize
                byte_size = count * item_size

            self.fields[f_name] = {
                'offset': offset,
                'count': count,
                'dtype': dtype,
                'byte_size': byte_size
            }
            offset += byte_size

        self.total_size = offset

        try:
            # 메모리 연결 시도
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.total_size)
            print(f"[{name}] 신규 생성 완료 ({self.total_size} bytes)")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name)
            print(f"[{name}] 기존 메모리 연결 완료")

        # --- 핵심 추가 부분: Resource Tracker에서 해당 메모리 unregister ---
        # 이 코드가 있어야 클라이언트 종료 시 OS가 메모리를 강제로 지우지 않습니다.
        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
            print(f"[{name}] Resource Tracker 추적 해제 완료")
        except Exception as e:
            print(f"[{name}] Tracker 해제 실패 (이미 해제되었을 수 있음): {e}")
        # -------------------------------------------------------------

        atexit.register(self.close)

    def set(self, field_name, value):
        f = self.fields[field_name]

        if f['dtype'] == str:
            encoded = str(value).encode('utf-8')
            if len(encoded) > f['byte_size']:
                raise ValueError(f"'{field_name}'의 크기가 너무 큽니다. (최대 {f['byte_size']}바이트)")

            self.shm.buf[f['offset'] : f['offset'] + len(encoded)] = encoded
            if len(encoded) < f['byte_size']:
                self.shm.buf[f['offset'] + len(encoded) : f['offset'] + f['byte_size']] = b'\x00' * (f['byte_size'] - len(encoded))
        else:
            arr = np.ndarray((f['count'],), dtype=f['dtype'], buffer=self.shm.buf, offset=f['offset'])
            if f['count'] > 1:
                arr[:] = value
            else:
                arr[0] = value

    def get(self, field_name):
        f = self.fields[field_name]

        if f['dtype'] == str:
            raw_bytes = self.shm.buf[f['offset'] : f['offset'] + f['byte_size']].tobytes()
            return raw_bytes.split(b'\x00')[0].decode('utf-8')
        else:
            arr = np.ndarray((f['count'],), dtype=f['dtype'], buffer=self.shm.buf, offset=f['offset'])
            result = arr.copy() if f['count'] > 1 else arr[0]
            return result

    def close(self):
        """연결된 버퍼를 안전하게 해제 (데이터는 유지됨)"""
        if hasattr(self, 'shm'):
            self.shm.close()

    def cleanup(self):
        """시스템에서 완전히 제거 (생성자 쪽에서만 호출 권장)"""
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass