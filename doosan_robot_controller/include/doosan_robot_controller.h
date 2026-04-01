#pragma once
#include "../lib/include/DRFLEx.h"

#include <iostream>
#include <array>
#include <memory>
#include <cstring>

#define DOOSAN_ROBOT_VERSION         "v1.0"
#define DOOSAN_ROBOT_VELOCITY_RT     0.1
#define DOOTAN_ROBOT_ACCELERATION_RT 0.1

using namespace std;
using namespace DRAFramework;


class DoosanController
{
private:
    std::unique_ptr<CDRFLEx> robot;
    std::string ip;

    bool RT_MODE;
    float cmd_joint[6];                                                  // 내부 명령 제어용
    float last_vel[6];                                                   // 조인트 속도
    float last_error[6];                                                 // 조인트 위치 에러

    // ------------------------------------------------------------------
    // 제어 파라미터
    // ------------------------------------------------------------------
    float KP = 7.0f;
    float KD = 0.1f;
    float target_time = 0.033f;



public:
    DoosanController(const std::string& ip);
    ~DoosanController();

    // ------------------------------------------------------------------
    // 기본 기능
    // ------------------------------------------------------------------
    bool connect();                                                      // 로봇 연결
    bool disconnect();                                                   // 연결 끊기

    bool servo_on();                                                     // 로봇 서보 On
    bool servo_off();                                                    // 로봇 서보 Off

    void movej(const float (&q)[6], float time);                         // MoveJ [deg], [sec]
    bool stop();                                                         // 로봇 정지 (rt stop 겸용)


    // ------------------------------------------------------------------
    // Realtime 제어
    // ------------------------------------------------------------------
    bool start_rt(const float (&q)[6]);                                  // Realtime 제어 모드 시작 (현재 조인트값 입력)[deg]
    bool stop_rt();                                                      // Realtime 제어 모드 종료
    void movej_rt(const float (&q)[6], float dt);                        // dt: 실제 호출 시간 권장. [deg] [sec]
    bool is_rt_mode() { return RT_MODE; }                                // rt 모드 인지?


    // ------------------------------------------------------------------
    // Getter
    // ------------------------------------------------------------------
    int                   get_robot_state();                             // 두산 로봇의 상태 ROBOT_STATE
    std::string           get_robot_state_name(ROBOT_STATE state);
    std::array<float, 6>  get_curr_jpos();                               // 현재 조인트 위치 [rad]
    std::array<float, 6>  get_curr_jvel();                               // 현재 조인트 속도 [rad/s]
    std::array<float, 6>  get_curr_tcp_pos();                            // 현재 tcp 위치   [m, deg]
    std::array<float, 6>  get_curr_tcp_vel();                            // 현재 tcp 속도   [m/s, rad/s]


    // ------------------------------------------------------------------
    // Setter
    // ------------------------------------------------------------------
    void set_kp_gain(double kp) { KP = kp; }                             // Kp 기본값 7.0
    void set_kd_gain(double kd) { KD = kd; }                             // kd 기본값 0.1
    void set_target_time(double time) { target_time = time; }            // target_time 기본값 0.033 (인퍼런스 할때 0.3)
};