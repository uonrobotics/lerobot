#include "../include/doosan_controller.h"

DoosanController::DoosanController(string ip): ip(ip), RT_MODE(false)
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

    robot->SetRobotSystem(ROBOT_SYSTEM_REAL);                           // 실제 로봇 제어
    robot->SetRobotMode(ROBOT_MODE_MANUAL);                             // 메뉴얼 모드
    robot->ManageAccessControl(MANAGE_ACCESS_CONTROL_FORCE_REQUEST);    // 접근 제어 강제 요청

    cout << "[Info ] [DSR] 두산 로봇 연결 성공: " << ip << endl;

    return true;
}

bool DoosanController::disconnect()
{
    cout << "[Info ] [DSR] 두산 로봇 컨트롤러 종료됨" << endl;
    robot->close_connection();
    if (RT_MODE)
        robot->disconnect_rt_control();
    return true;
}

bool DoosanController::servo_on()
{
    if (!robot->SetRobotControl(CONTROL_SERVO_ON)) {
        cout << "[Warn ] [DSR] 서보 온 실패" << endl;
        return false;
    }

    return true;
}

bool DoosanController::servo_off()
{
    robot->SetRobotControl(CONTROL_ENABLE_OPERATION);
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
        cout << "[Error] 두산 로봇 Realtime 제어기 연결 실패: " << ip << endl;
        return false;
    }

    // RT 제어 설정
    robot->set_rt_control_output(DOOSAN_ROBOT_VERSION, target_time, 10);

    // RT 제어 시작 서비스 호출
    if (!robot->start_rt_control()) {
        cout << "[Error] [DSR] 두산 로봇 RT 제어 시작 명령 실패" << endl;
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
    cout << "[Info ] [DSR] 두산 로봇 Realtime 제어 모드 시작" << endl;
    return true;
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

    float send_vel[6] = {0.0f,};
    float send_acc[6] = {0.0f,};

    for (int i = 0; i < 6; i++) {
        float error = q[i] - cmd_joint[i];              // P
        float error_dot = (error - last_error[i]) / dt; // D

        float current_vel = (error * KP) + (error_dot * KD);    // 속도 계산
        float current_acc = (current_vel - last_vel[i]) / dt;   // 가속도 계산

        // Joint 값 업데이트
        cmd_joint[i] += current_vel * dt;

        // 로봇 API 전달용 배열 채우기
        send_vel[i] = current_vel;
        send_acc[i] = current_acc;

        // 상태 저장
        last_vel[i] = current_vel;
        last_error[i] = error;
    }

    // Servoj rt
    robot->servoj_rt(cmd_joint, send_vel, send_acc, target_time);
}

vector<float> DoosanController::get_current_joint()
{
    vector<float> joints(6);

    const auto val  = robot->get_current_posj();
    for (int i = 0; i < 6; i++) {
        joints[i] = val->_fPosition[i];
    }

    return joints;
}

