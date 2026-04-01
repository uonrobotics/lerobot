#include "../include/doosan_robot_controller.h"

#include <cmath>

#ifndef DEG_TO_RAD
#define DEG_TO_RAD(x) ((x) * 0.017453292519943295769236907684886)
#endif
#ifndef RAD_TO_DEG
#define RAD_TO_DEG(x) ((x) * 57.29577951308232087679815481410518)
#endif


DoosanController::DoosanController(const std::string& ip): ip(ip), RT_MODE(false)
{
    robot = std::make_unique<CDRFLEx>();
    memset(cmd_joint, 0, sizeof(cmd_joint));
    memset(last_vel, 0, sizeof(last_vel));
    memset(last_error, 0, sizeof(last_error));
}

DoosanController::~DoosanController()
{
    stop();
}

bool DoosanController::connect()
{
    if (!robot->open_connection(ip)) {
        cout << "[Error] [DSR] 두산 로봇 연결 실패: " << ip << endl;
        return false;
    }

    robot->ManageAccessControl(MANAGE_ACCESS_CONTROL_FORCE_REQUEST);    // 접근 제어 강제 요청

    cout << "[Info ] [DSR] 두산 로봇 연결 성공: " << ip << endl;

    return true;
}

bool DoosanController::disconnect()
{
    cout << "[Info ] [DSR] 두산 로봇 컨트롤러 종료" << endl;

    if (RT_MODE)
        robot->disconnect_rt_control();

    robot->close_connection();

    return true;
}

bool DoosanController::servo_on()
{
    robot->SetRobotSystem(ROBOT_SYSTEM_REAL);                           // 실제 로봇 제어
    robot->SetRobotMode(ROBOT_MODE_MANUAL);                             // 메뉴얼 모드

    if (RT_MODE)
        robot->disconnect_rt_control();

    if (!robot->set_robot_control(CONTROL_SERVO_ON)) {
        cout << "[Error] [DSR] 서보 온 실패" << endl;
        return false;
    }

    cout << "[Info ] [DSR] 서보 온" << endl;
    return true;
}

bool DoosanController::servo_off()
{
    if (!robot->servo_off(STOP_TYPE_QUICK)) {
        cout << "[Error] [DSR] 서보 오프 실패" << endl;
        return false;
    }

    cout << "[Info ] [DSR] 서보 오프" << endl;
    return true;
}

bool DoosanController::stop()
{
    robot->stop();
    if (RT_MODE) {
        RT_MODE = false;
        robot->stop_rt_control();
    }

    return true;;
}

bool DoosanController::start_rt(const float (&q)[6])
{
    // 전달받은 q(현재 관절각)로 명령 위치 초기화 (급발진 방지)
    memcpy(cmd_joint, q, sizeof(float) * 6);

    // Realtime 제어기 연결
    if (!robot->connect_rt_control(ip)) {
        cout << "[Error] [DSR] Realtime 제어기 연결 실패: " << ip << endl;
        return false;
    }

    // 출력 데이터 주기 설정
    robot->set_rt_control_output(DOOSAN_ROBOT_VERSION, 0.001, 10);

    // RT 제어 시작 서비스 호출
    if (!robot->start_rt_control()) {
        cout << "[Error] [DSR] Realtime 제어 시작 실패" << endl;
        return false;
    }

    // 속도 및 가속도 설정
    float vel[6] = {DOOSAN_ROBOT_VELOCITY_RT,};
    float acc[6] = {DOOTAN_ROBOT_ACCELERATION_RT,};
    robot->set_velj_rt(vel);
    robot->set_accj_rt(acc);

    // 로봇 모드 설정
    robot->set_robot_mode(ROBOT_MODE_AUTONOMOUS);
    robot->set_safety_mode(SAFETY_MODE_AUTONOMOUS, SAFETY_MODE_EVENT_MOVE);

    RT_MODE = true;
    cout << "[Info ] [DSR] 두산 로봇 Realtime 제어 시작" << endl;
    return true;
}

bool DoosanController::stop_rt()
{
    if (RT_MODE) {
        robot->stop_rt_control();
        RT_MODE = false;
        return true;
    }
    return false;
}

void DoosanController::movej(const float (&q)[6], float time)
{
    // 두산 로봇 내부에서  val, acc를 자동으로 보간함
    float acc[6] = {0,};
    float vel[6] = {0,};
    robot->movej((float*)q, vel, acc, time);
}

void DoosanController::movej_rt(const float (&q)[6], float dt)
{
    if (!RT_MODE || dt <= 1e-6f)
        return;

    // T: 도달하고자 하는 목표 고정 시간 (예: 0.033f 또는 설정된 target_time)
    float T = target_time;

    float send_vel[6] = {0.0f,};
    float send_acc[6] = {0.0f,};

    for (int i = 0; i < 6; i++) {
        // 현재 오차 계산 (실시간으로 변하는 q[i] 대응)
        float error = q[i] - cmd_joint[i];

        // 5차 다항식 기반의 실시간 속도 생성 (Velocity Scheduling)
        float ref_vel = (error / T) * 1.875f; // 1.875는 5차 다항식의 피크 속도 계수 활용

        // PD 제어
        float error_dot = ref_vel - last_vel[i];
        float current_vel = (error * KP) + (error_dot * KD);

        //  가속도 제한 (로봇 보호를 위한 Clipping)
        float current_acc = (current_vel - last_vel[i]) / dt;

        // 상태 업데이트
        cmd_joint[i] += current_vel * dt;

        send_vel[i] = current_vel;
        send_acc[i] = current_acc;

        last_vel[i] = current_vel;
        last_error[i] = error;
    }

    // 6. 로봇으로 명령 전송 (내부 target_time 사용)
    robot->servoj_rt(cmd_joint, send_vel, send_acc, target_time);
}

int DoosanController::get_robot_state()
{
    return robot->get_robot_state();
}

std::string DoosanController::get_robot_state_name(ROBOT_STATE state)
{
    switch (state) {
        case STATE_INITIALIZING:   return "INITIALIZING";
        case STATE_STANDBY:        return "STANDBY";
        case STATE_MOVING:         return "MOVING";
        case STATE_SAFE_OFF:       return "SAFE_OFF";
        case STATE_TEACHING:       return "TEACHING";
        case STATE_SAFE_STOP:      return "SAFE_STOP";
        case STATE_EMERGENCY_STOP: return "EMERGENCY_STOP";
        case STATE_HOMMING:        return "HOMMING";
        case STATE_RECOVERY:       return "RECOVERY";
        case STATE_SAFE_STOP2:     return "SAFE_STOP2";
        case STATE_SAFE_OFF2:      return "SAFE_OFF2";
        case STATE_RESERVED1:      return "RESERVED1";
        case STATE_RESERVED2:      return "RESERVED2";
        case STATE_RESERVED3:      return "RESERVED3";
        case STATE_RESERVED4:      return "RESERVED4";
        case STATE_NOT_READY:      return "NOT_READY";
        case STATE_LAST:           return "LAST";
        default:                   return "UNKNOWN_STATE";
    }
}

std::array<float, 6> DoosanController::get_curr_jvel()
{
    auto data = robot->read_data_rt();

    std::array<float, 6> vel;
    vel[0] = DEG_TO_RAD(data->actual_joint_velocity[0]); // [deg/s]
    vel[1] = DEG_TO_RAD(data->actual_joint_velocity[1]); // [deg/s]
    vel[2] = DEG_TO_RAD(data->actual_joint_velocity[2]); // [deg/s]
    vel[3] = DEG_TO_RAD(data->actual_joint_velocity[3]); // [deg/s]
    vel[4] = DEG_TO_RAD(data->actual_joint_velocity[4]); // [deg/s]
    vel[5] = DEG_TO_RAD(data->actual_joint_velocity[5]); // [deg/s]

    return vel;
}

std::array<float, 6> DoosanController::get_curr_tcp_pos()
{
    auto data = robot->read_data_rt();

    std::array<float, 6> pos;
    pos[0] = data->actual_tcp_position[0] * 0.001;     // x [m]
    pos[1] = data->actual_tcp_position[1] * 0.001;     // y [m]
    pos[2] = data->actual_tcp_position[2] * 0.001;     // z [m]
    pos[3] = DEG_TO_RAD(data->actual_tcp_position[3]); // roll   [rad]   z
    pos[4] = DEG_TO_RAD(data->actual_tcp_position[4]); // pitch  [rad]   y
    pos[5] = DEG_TO_RAD(data->actual_tcp_position[5]); // yaw    [rad]   z

    return pos;
}

std::array<float, 6> DoosanController::get_curr_tcp_vel()
{
    auto data = robot->read_data_rt();

    std::array<float, 6> vel;
    vel[0] = data->actual_tcp_velocity[0] * 0.001;    // x [m/s]
    vel[1] = data->actual_tcp_velocity[1] * 0.001;    // y [m/s]
    vel[2] = data->actual_tcp_velocity[2] * 0.001;    // z [m/s]
    vel[3] = DEG_TO_RAD(data->actual_tcp_velocity[3]); // roll [rad/s]    z
    vel[4] = DEG_TO_RAD(data->actual_tcp_velocity[4]); // pitch [rad/s]   y
    vel[5] = DEG_TO_RAD(data->actual_tcp_velocity[5]); // yaw [rad/s]     z

    return vel;
}

std::array<float, 6> DoosanController::get_curr_jpos()
{
    auto data = robot->read_data_rt();

    std::array<float, 6> pos;
    pos[0] = DEG_TO_RAD(data->actual_joint_position[0]); // [rad]
    pos[1] = DEG_TO_RAD(data->actual_joint_position[1]); // [rad]
    pos[2] = DEG_TO_RAD(data->actual_joint_position[2]); // [rad]
    pos[3] = DEG_TO_RAD(data->actual_joint_position[3]); // [rad]
    pos[4] = DEG_TO_RAD(data->actual_joint_position[4]); // [rad]
    pos[5] = DEG_TO_RAD(data->actual_joint_position[5]); // [rad]

    return pos;
}
