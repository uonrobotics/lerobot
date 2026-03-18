#pragma once
#include "DRFLEx.h"

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <thread>

#define DOOSAN_ROBOT_VERSION         "v1.0"
#define DOOSAN_ROBOT_VELOCITY_RT     0.1
#define DOOTAN_ROBOT_ACCELERATION_RT 0.1

using namespace std;
using namespace DRAFramework;


enum RooSanState {
    DooSanState_Init = 0,

};


class DoosanController
{
private:
    unique_ptr<CDRFLEx> robot;
    string ip;

    bool RT_MODE;
    float cmd_joint[6];                 // 내부 명령 제어용
    float last_vel[6];                  // 조인트 속도
    float last_error[6];                // 조인트 위치 에러

    std::unique_ptr<std::thread> doosan_thread;

    // ------------------------------------------------------------------
    // 제어 파라미터
    // ------------------------------------------------------------------
    const float KP = 7.0f;              // P Gain (반응성)
    const float KD = 0.00001f;         // D Gain (뎀핑)
    const float target_time = 0.043f;  // 로봇 내부 제어 시간  (약 30hz)

public:
    DoosanController(string ip);
    ~DoosanController();

    // ------------------------------------------------------------------
    // 기본 기능
    // ------------------------------------------------------------------
    bool connect();                                 // 로봇 연결
    bool disconnect();                              // 연결 끊기
    bool servo_on();                                // 로봇 서보 On
    bool servo_off();                               // 로봇 서보 Off
    void movej(const float (&q)[6], float time);    // MoveJ [deg], [sec]
    bool stop();                                    // 로봇 정지 (rt stop 겸용)

    // ------------------------------------------------------------------
    // Realtime 제어
    // ------------------------------------------------------------------
    bool start_rt(const float (&q)[6]);             // Realtime 제어 모드 시작 (현재 조인트값 입력)[deg]
    bool is_rt_mode() { return RT_MODE; }           // rt 모드 인지?

    void movej_rt(const float (&q)[6], float dt);   // dt: 실제 호출 시간 권장. [deg] [sec]

    // ------------------------------------------------------------------
    // Getter
    // ------------------------------------------------------------------
    vector<float> get_current_joint();              // 실제 로봇 조인트 값 [deg]
};