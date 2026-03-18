#pragma once
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <string>
#include <type_traits>
#include <cstring>

using namespace boost::interprocess;

template <typename T>
class SharedMemory {
private:
    std::string name_;
    shared_memory_object shm_;
    mapped_region region_;
    bool is_initialized_ = false;

    static_assert(std::is_trivially_copyable_v<T>, "T must be a trivially copyable type.");

public:
    // 1. 생성자: 이름만 저장하고 실제 로직은 실행하지 않음
    SharedMemory(const std::string& name) : name_(name), is_initialized_(false) {}

    bool init()
    {
        try {
            // 공유 메모리 열기 또는 생성
            shm_ = shared_memory_object(open_or_create, name_.c_str(), read_write);

            // 메모리 크기 설정
            shm_.truncate(sizeof(T));

            // 주소 공간 매핑
            region_ = mapped_region(shm_, read_write);

            is_initialized_ = true;

            std::cout << "[Info ] [SHM] 공유 메모리 생성: " << name_ << std::endl;
            return true;
        } catch (const interprocess_exception& e) {
            is_initialized_ = false;
            std::cout << "[Error] [SHM] 공유 메모리 초기화 실패: " << e.what() << std::endl;
            return false;
        }
    }

    // 데이터 쓰기
    void set(const T& data)
    {
        if (!is_initialized_) return;
        std::memcpy(region_.get_address(), &data, sizeof(T));
    }

    // 데이터 읽기 (초기화되지 않았을 경우에 대한 처리가 필요할 수 있음)
    T& get()
    {
        return *static_cast<T*>(region_.get_address());
    }

    // 초기화 여부 확인
    bool is_ready() const
    {
        return is_initialized_;
    }

    static void unlink(const std::string& name)
    {
        shared_memory_object::remove(name.c_str());
    }
};