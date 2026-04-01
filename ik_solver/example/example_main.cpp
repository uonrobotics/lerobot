#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>
#include <fstream>


// my code
#include "imgui.h"
#include "implot3d.h"
#include "ik_solver.h"
#include "draw.h"
#include "shard_memory.hpp"


// ==============================
// 정의
// ==============================
#define SHARED_MEMORY_ID    "movej"
#define CONFIG_FILE         std::string(ROOT_PATH) + "/config/robot.yaml"
#define CONTENT_PATH        std::string(ROOT_PATH) + "/contents"
#define SLEEP(x)            std::this_thread::sleep_for(std::chrono::milliseconds(x))


#define DEFAULT_TCP_SPEED       1.0
#define DEFAULT_JFACTOR         0.8
#define DEFAULT_WORKSPACE_MIN_X -1.0
#define DEFAULT_WORKSPACE_MAX_X 1.0
#define DEFAULT_WORKSPACE_MIN_Y -1.0
#define DEFAULT_WORKSPACE_MAX_Y 1.0
#define DEFAULT_WORKSPACE_MIN_Z 0.0
#define DEFAULT_WORKSPACE_MAX_Z 1.0

#define END_EFFECTOR_COLOR        \
ImVec4(1.00f, 0.65f, 0.65f, 1.0f),\
ImVec4(0.65f, 1.00f, 0.65f, 1.0f),\
ImVec4(0.65f, 0.85f, 1.00f, 1.0f)

// ==============================
// 정밀 루프 타이머 매크로
// ==============================
#define INIT_LOOP_TIMER(interval_ms) \
auto _target_interval = std::chrono::microseconds(static_cast<long long>((interval_ms) * 1000.0)); \
auto _next_wakeup = std::chrono::steady_clock::now(); \
auto _last_time = _next_wakeup;

// dt 계산
#define UPDATE_LOOP_TIMER() \
auto _now = std::chrono::steady_clock::now(); \
std::chrono::duration<double, std::milli> _dt = _now - _last_time; \
_last_time = _now;

// 슬립
#define SLEEP_LOOP_TIMER() \
_next_wakeup += _target_interval; \
std::this_thread::sleep_until(_next_wakeup);


using namespace std;




// ==============================
// 공유 메모리
// ==============================
struct ShmMoveJ {
    float joint[6];
    char status[20];
};

enum ControlMode_ {
    ControlMode_MoveJ  = 0,
    ControlMode_MoveL  = 1,
    ControlMode_MoveX  = 2,
};


// ==============================
// 객체
// ==============================

std::string URDF_FILE = "";
std::unique_ptr<IkSolver> SOLVER;
std::unique_ptr<SharedMemory<ShmMoveJ>> SHM;



// ==============================
// UI 변수
// ==============================
// ui 표시용
float J1 = 0, J2 = 0, J3 = 0, J4 = 0, J5 = 0, J6 = 0;   // j1 ~ j6
float X  = 0, Y  = 0, Z  = 0, A  = 0, B  = 0, C  = 0;   // x, y, z, roll, pitch, yaw

// 설정값
int  CONTROL_MODE = 0;                   // 제어 모드
int  SOLVER_MODE = 1;                    // 솔버 모드
bool SHOW_MANIPULABILITY = false;        // 매니풀러빌리티 표시
bool UI_IMMEDIATE_MOVE = false;          // 즉시 움직일지 여부

float UI_WS_MIN[3] = {0, 0, 0};          // 작업 영역 최소 값
float UI_WS_MAX[3] = {0, 0, 0};          // 작업 영역 최대 값

float UI_JOINT_LIMIT[6][2] = {0, };      // joint 제한 범위
float UI_JOINT_VELOCITY_LIMIT[6] = {0,}; // joint 최대 속도
float UI_TCP_SPEED = 0;                  // tcp 최대 속도
float UI_EE_TF[6] = {0, 0, 0, 0, 0, 0};  // [x, y, z, roll, pitch, yaw]
bool  UI_EE_REALTIME = false;            // 실시간 적용 여부
float UI_JOINT_VSCALE = 1.0;             // joint 최대 제한 속도 비율

float UI_MOVEL_ECLIPSE = 0.0;


// ==============================
// 그래프 그리기용 Axis
// ==============================
Axis3D AXIS3D[8] = {
    Axis3D(0.12),       // world
    Axis3D(0.12),       // j0
    Axis3D(0.12),       // j1
    Axis3D(0.12),       // j2
    Axis3D(0.12),       // j3
    Axis3D(0.12),       // j4
    Axis3D(0.12),       // j5
    Axis3D(0.12)        // ee
};


void init()
{
    // --- 공유 메모리 ---
    SHM = std::make_unique<SharedMemory<ShmMoveJ>>(SHARED_MEMORY_ID);
    if (!SHM->init()) {
        return;
    }

    // --- 솔버 생성 ---
    SOLVER = std::make_unique<IkSolver>(URDF_FILE);
    if (!SOLVER->init()) {
        SOLVER.reset();
        return;
    }

    // 워크스페이스 제한
    UI_WS_MIN[0] = DEFAULT_WORKSPACE_MIN_X; // min x
    UI_WS_MAX[0] = DEFAULT_WORKSPACE_MAX_X; // max x
    UI_WS_MIN[1] = DEFAULT_WORKSPACE_MIN_Y; // min y
    UI_WS_MAX[1] = DEFAULT_WORKSPACE_MAX_Y; // max y
    UI_WS_MIN[2] = DEFAULT_WORKSPACE_MIN_Z; // min z
    UI_WS_MAX[2] = DEFAULT_WORKSPACE_MAX_Z; // max z

    // 조인트 제한
    UI_JOINT_LIMIT[0][0] = RAD_TO_DEG(SOLVER->get_joint_info(0).limits.lower); // joint 0 min
    UI_JOINT_LIMIT[0][1] = RAD_TO_DEG(SOLVER->get_joint_info(0).limits.upper); // joint 0 max
    UI_JOINT_LIMIT[1][0] = RAD_TO_DEG(SOLVER->get_joint_info(1).limits.lower); // joint 1 min
    UI_JOINT_LIMIT[1][1] = RAD_TO_DEG(SOLVER->get_joint_info(1).limits.upper); // joint 1 max
    UI_JOINT_LIMIT[2][0] = RAD_TO_DEG(SOLVER->get_joint_info(2).limits.lower); // joint 2 min
    UI_JOINT_LIMIT[2][1] = RAD_TO_DEG(SOLVER->get_joint_info(2).limits.upper); // joint 2 max
    UI_JOINT_LIMIT[3][0] = RAD_TO_DEG(SOLVER->get_joint_info(3).limits.lower); // joint 3 min
    UI_JOINT_LIMIT[3][1] = RAD_TO_DEG(SOLVER->get_joint_info(3).limits.upper); // joint 3 max
    UI_JOINT_LIMIT[4][0] = RAD_TO_DEG(SOLVER->get_joint_info(4).limits.lower); // joint 4 min
    UI_JOINT_LIMIT[4][1] = RAD_TO_DEG(SOLVER->get_joint_info(4).limits.upper); // joint 4 max
    UI_JOINT_LIMIT[5][0] = RAD_TO_DEG(SOLVER->get_joint_info(5).limits.lower); // joint 5 min
    UI_JOINT_LIMIT[5][1] = RAD_TO_DEG(SOLVER->get_joint_info(5).limits.upper); // joint 5 max

    // 조인트 속도 제한
    UI_JOINT_VELOCITY_LIMIT[0] = RAD_TO_DEG(SOLVER->get_joint_info(0).limits.velocity); // joint 1 max angle velocity
    UI_JOINT_VELOCITY_LIMIT[1] = RAD_TO_DEG(SOLVER->get_joint_info(1).limits.velocity); // joint 2 max angle velocity
    UI_JOINT_VELOCITY_LIMIT[2] = RAD_TO_DEG(SOLVER->get_joint_info(2).limits.velocity); // joint 3 max angle velocity
    UI_JOINT_VELOCITY_LIMIT[3] = RAD_TO_DEG(SOLVER->get_joint_info(3).limits.velocity); // joint 4 max angle velocity
    UI_JOINT_VELOCITY_LIMIT[4] = RAD_TO_DEG(SOLVER->get_joint_info(4).limits.velocity); // joint 5 max angle velocity
    UI_JOINT_VELOCITY_LIMIT[5] = RAD_TO_DEG(SOLVER->get_joint_info(5).limits.velocity); // joint 6 max angle velocity

    Transform ee_tf = SOLVER->get_end_effector_offset();
    UI_EE_TF[0] = ee_tf.translation().x();
    UI_EE_TF[1] = ee_tf.translation().y();
    UI_EE_TF[2] = ee_tf.translation().z();
    UI_EE_TF[3] = RAD_TO_DEG(ee_tf.rpy()[0]);
    UI_EE_TF[4] = RAD_TO_DEG(ee_tf.rpy()[1]);
    UI_EE_TF[5] = RAD_TO_DEG(ee_tf.rpy()[2]);


    // TCP 속도 제한
    UI_TCP_SPEED    = DEFAULT_TCP_SPEED;
    UI_JOINT_VSCALE = DEFAULT_JFACTOR;



    // --- 조인트 한계 값 설정 ---
    // SOLVER->set_joint_limit(0, DEG_TO_RAD(JLIMITS[0][0]), DEG_TO_RAD(JLIMITS[0][1])); // J1
    // SOLVER->set_joint_limit(1, DEG_TO_RAD(JLIMITS[1][0]), DEG_TO_RAD(JLIMITS[1][1])); // J2
    // SOLVER->set_joint_limit(2, DEG_TO_RAD(JLIMITS[2][0]), DEG_TO_RAD(JLIMITS[2][1])); // J3
    // SOLVER->set_joint_limit(3, DEG_TO_RAD(JLIMITS[3][0]), DEG_TO_RAD(JLIMITS[3][1])); // J4
    // SOLVER->set_joint_limit(4, DEG_TO_RAD(JLIMITS[4][0]), DEG_TO_RAD(JLIMITS[4][1])); // J5
    // SOLVER->set_joint_limit(5, DEG_TO_RAD(JLIMITS[5][0]), DEG_TO_RAD(JLIMITS[5][1])); // J6

    // --- 조인트 최대 속도 값 설정 ---
    // SOLVER->set_joint_vlimit(0, DEG_TO_RAD(VLIMIT[0]));                               // J1
    // SOLVER->set_joint_vlimit(1, DEG_TO_RAD(VLIMIT[1]));                               // J2
    // SOLVER->set_joint_vlimit(2, DEG_TO_RAD(VLIMIT[2]));                               // J3
    // SOLVER->set_joint_vlimit(3, DEG_TO_RAD(VLIMIT[3]));                               // J4
    // SOLVER->set_joint_vlimit(4, DEG_TO_RAD(VLIMIT[4]));                               // J5
    // SOLVER->set_joint_vlimit(5, DEG_TO_RAD(VLIMIT[5]));                               // J6

    // --- 작업 영역 한계 값 설정 ---
    SOLVER->set_workspace_limitX(UI_WS_MIN[0], UI_WS_MAX[0]);                                  // -x, x
    SOLVER->set_workspace_limitY(UI_WS_MIN[1], UI_WS_MAX[1]);                                  // -y, y
    SOLVER->set_workspace_limitZ(UI_WS_MIN[2], UI_WS_MAX[2]);                                  // -z, z

    // --- TCP 속도 제한 ---
    SOLVER->set_tcp_max_speed(UI_TCP_SPEED);                                                // tcp speed
    SOLVER->set_joint_velocity_limit_scale(UI_JOINT_VSCALE);                                             // joint velocity fector


    // -- SOLVER 초기 위치 설정 ---
    vec<6> q = {0, 0, 0, 0, 0, 0};
    SOLVER->movej(q);

    // --- Ui TCP 위치 업데이트 ---
    Transform tcp = SOLVER->get_curr_tcp_tf();
    X = tcp.translation().x();
    Y = tcp.translation().y();
    Z = tcp.translation().z();

    vec3 xyz = tcp.rpy();
    A = RAD_TO_DEG(xyz[0]);
    B = RAD_TO_DEG(xyz[1]);
    C = RAD_TO_DEG(xyz[2]);

    // -- UI joint 업데이트 ---
    const vec<6> q_deg = SOLVER->get_curr_joint_deg();
    J1 = q_deg[0];
    J2 = q_deg[1];
    J3 = q_deg[2];
    J4 = q_deg[3];
    J5 = q_deg[4];
    J6 = q_deg[5];

    // --- 제어 모드 초기화 ---
    CONTROL_MODE = ControlMode_MoveJ;

    // --- 시각화 좌표 초기 위치 설정 ---
    AXIS3D[1].set_transform(SOLVER->get_curr_tf(0));                           // J1
    AXIS3D[2].set_transform(SOLVER->get_curr_tf(1));                           // J2
    AXIS3D[3].set_transform(SOLVER->get_curr_tf(2));                           // J3
    AXIS3D[4].set_transform(SOLVER->get_curr_tf(3));                           // J4
    AXIS3D[5].set_transform(SOLVER->get_curr_tf(4));                           // J5
    AXIS3D[6].set_transform(SOLVER->get_curr_tf(5));                           // J6
    AXIS3D[7].set_transform(SOLVER->get_curr_tcp_tf());                        // flange
    AXIS3D[7].set_colors(END_EFFECTOR_COLOR);
}

void draw_control()
{
    ImGui::Begin("로봇 제어");

    static std::string current_urdf_file = URDF_FILE;
    const std::string urdf_dir = CONTENT_PATH;
    std::vector<std::string> urdf_files;

    if (std::filesystem::exists(urdf_dir) && std::filesystem::is_directory(urdf_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(urdf_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".urdf") {
                urdf_files.push_back(entry.path().filename().string());
            }
        }
    }

    if (ImGui::BeginCombo("##Select URDF", current_urdf_file.empty() ? "URDF 파일을 선택하세요." : current_urdf_file.c_str())) {
        if (urdf_files.empty()) {
            ImGui::Selectable("No URDF files found", false);
        } else {
            for (const auto& file : urdf_files) {
                bool is_selected = (current_urdf_file == file);
                if (ImGui::Selectable(file.c_str(), is_selected)) {
                    current_urdf_file = file;
                    URDF_FILE = urdf_dir + "/" + current_urdf_file;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
        }
        ImGui::EndCombo();
    }

    ImGui::SameLine();

    if (ImGui::Button("불러오기")) {
        current_urdf_file = URDF_FILE;
        init();
    }



    float v[6] = {0, }; // 조인트 속도
    float V[6] = {0, }; // tcp 속도

    if (SOLVER) {
        const vec<6> jvel = SOLVER->get_curr_joint_velocity_deg();
        v[0] = jvel[0];
        v[1] = jvel[1];
        v[2] = jvel[2];
        v[3] = jvel[3];
        v[4] = jvel[4];
        v[5] = jvel[5];

        // Tcp 현재 속도
        const vec<6> tcp_speed = SOLVER->get_curr_tcp_speed();
        V[0] = tcp_speed(0);
        V[1] = tcp_speed(1);
        V[2] = tcp_speed(2);
        V[3] = RAD_TO_DEG(tcp_speed(3));
        V[4] = RAD_TO_DEG(tcp_speed(4));
        V[5] = RAD_TO_DEG(tcp_speed(5));
    }


    // -----------------------------
    // 두산 로봇 제어 UI
    // -----------------------------
    ImGui::SeparatorText("제어 모드");

    // 모드 설정 라디오 버튼
    ImGui::RadioButton("MoveJ", &CONTROL_MODE, ControlMode_::ControlMode_MoveJ); ImGui::SameLine();
    ImGui::RadioButton("MoveL", &CONTROL_MODE, ControlMode_::ControlMode_MoveL); ImGui::SameLine();
    ImGui::RadioButton("MoveX", &CONTROL_MODE, ControlMode_::ControlMode_MoveX);

    // movel 연산 시간 평균화
    static const int MOVING_AVG_SIZE = 100;
    static float eclipse_history[MOVING_AVG_SIZE] = {0};
    static int eclipse_index = 0;
    static float eclipse_sum = 0.0f;

    eclipse_sum -= eclipse_history[eclipse_index];
    eclipse_history[eclipse_index] = UI_MOVEL_ECLIPSE / 1000.0f;
    eclipse_sum += eclipse_history[eclipse_index];
    eclipse_index = (eclipse_index + 1) % MOVING_AVG_SIZE;

    float eclipse_avg = eclipse_sum / MOVING_AVG_SIZE;

    ImGui::Text("연산 시간: %.3f ms (평균: %.3f ms)", UI_MOVEL_ECLIPSE / 1000.0, eclipse_avg);
    ImGui::Dummy(ImVec2(0, 10));


    ImGui::PushItemWidth(150);

    // 조인트 설정 슬라이더
    ImGui::SeparatorText("조인트 제어");
    ImGui::DragFloat("Joint1", &J1, 0.1f, -360, 360, "%.3f deg"); // j1
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[0]);
    ImGui::DragFloat("Joint2", &J2, 0.1f, -360, 360, "%.3f deg"); // j2
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[1]);
    ImGui::DragFloat("Joint3", &J3, 0.1f, -360, 360, "%.3f deg"); // j3
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[2]);
    ImGui::DragFloat("Joint4", &J4, 0.1f, -360, 360, "%.3f deg"); // j4
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[3]);
    ImGui::DragFloat("Joint5", &J5, 0.1f, -360, 360, "%.3f deg"); // j5
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[4]);
    ImGui::DragFloat("Joint6", &J6, 0.1f, -360, 360, "%.3f deg"); // j6
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[5]);
    ImGui::Dummy(ImVec2(0, 10));

    // 테스크 제어 슬라이더
    ImGui::SeparatorText("테스크 제어");

    ImGui::DragFloat("X [m]   ", &X, 0.005f, -2.3f, 2.3f);            // x
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vx: %.3f m/s", V[0]);
    ImGui::DragFloat("Y [m]   ", &Y, 0.005f, -2.3f, 2.3f);
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vy: %.3f m/s", V[1]);    // y
    ImGui::DragFloat("Z [m]   ", &Z, 0.005f, -2.3f, 2.3f);
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vz: %.3f m/s", V[2]);    // z
    ImGui::DragFloat("\u03C6 [deg]", &A, 0.1f, -360.f, 360.f);
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Wx: %.1f d/s", V[3]);    // roll
    ImGui::DragFloat("\u03B8 [deg]", &B, 0.1f, -360.f, 360.f);
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Wy: %.1f d/s", V[4]);    // pitch
    ImGui::DragFloat("\u03C8 [deg]", &C, 0.1f, -360.f, 360.f);
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Wz: %.1f d/s", V[5]);    // yaw

    ImGui::PopItemWidth();
    ImGui::End();
}

void draw_setting() {
    bool change_ws = false;
    bool change_jl = false;
    bool change_vl = false;
    bool change_ts = false;
    bool change_ee = false;
    bool change_sc = false;

    ImGui::Begin("설정");

    // --- 매니퓰러빌리티 표시 ---
    ImGui::Checkbox("Show Manipulability", &SHOW_MANIPULABILITY);
    ImGui::Dummy(ImVec2(0, 10));


    // --- 작업 영역 설정 ---
    ImGui::SeparatorText("작업 영역 설정 (Box)");
    change_ws |= ImGui::SliderFloat3("최소 [m]", UI_WS_MIN, -1.0f, 1.0f);
    change_ws |= ImGui::SliderFloat3("최대 [m]", UI_WS_MAX , -1.0f, 1.0f);
    ImGui::Dummy(ImVec2(0, 10));


    // --- 조인트 범위 설정 ---
    ImGui::SeparatorText("조인트 범위 설정");
    change_jl |= ImGui::SliderFloat2("J0 [deg]", UI_JOINT_LIMIT[0], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J1 [deg]", UI_JOINT_LIMIT[1], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J2 [deg]", UI_JOINT_LIMIT[2], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J3 [deg]", UI_JOINT_LIMIT[3], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J4 [deg]", UI_JOINT_LIMIT[4], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J5 [deg]", UI_JOINT_LIMIT[5], -360.f, 360.f);
    ImGui::Dummy(ImVec2(0, 10));


    // --- 조인트 속도 제한 설정 ---
    ImGui::SeparatorText("조인트 속도 제한 설정");
    change_vl |= ImGui::SliderFloat("J0 [deg/s²]", &UI_JOINT_VELOCITY_LIMIT[0], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J1 [deg/s²]", &UI_JOINT_VELOCITY_LIMIT[1], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J2 [deg/s²]", &UI_JOINT_VELOCITY_LIMIT[2], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J3 [deg/s²]", &UI_JOINT_VELOCITY_LIMIT[3], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J4 [deg/s²]", &UI_JOINT_VELOCITY_LIMIT[4], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J5 [deg/s²]", &UI_JOINT_VELOCITY_LIMIT[5], -360.f, 360.f);
    ImGui::Dummy(ImVec2(0, 5));
    change_sc |= ImGui::SliderFloat("속도 펙터", &UI_JOINT_VSCALE, 0.0f, 1.0f);
    ImGui::Dummy(ImVec2(0, 10));


    // --- TCP 속도 제한 설정 ---
    ImGui::SeparatorText("TCP 설정");
    change_ts |= ImGui::SliderFloat("속도 제한 [m/s]", &UI_TCP_SPEED, 0.0f, 1.0f);
    ImGui::Dummy(ImVec2(0, 10));


    // -- End-Effector 설정 ---
    ImGui::SeparatorText("End Effector 설정");
    ImGui::SliderFloat3("Position [m]", &UI_EE_TF[0], -1.0f, 1.0f);
    ImGui::SliderFloat3("Rotation [deg]", &UI_EE_TF[3], -360.0f, 360.0f); // Roll, Pitch, Yaw 추가

    if (ImGui::Button("적용")) {
        change_ee = true;
    }
    ImGui::SameLine();
    ImGui::Checkbox("실시간 적용", &UI_EE_REALTIME);
    ImGui::Dummy(ImVec2(0, 10));


    ImGui::End();


    // --------------------


    if (!SOLVER)
        return;

    if (change_ws) {
        SOLVER->set_workspace_limitX(UI_WS_MIN[0], UI_WS_MAX[0]);
        SOLVER->set_workspace_limitY(UI_WS_MIN[1], UI_WS_MAX[1]);
        SOLVER->set_workspace_limitZ(UI_WS_MIN[2], UI_WS_MAX[2]);
    }

    if (change_jl) {
        SOLVER->set_joint_limit(0, DEG_TO_RAD(UI_JOINT_LIMIT[0][0]), DEG_TO_RAD(UI_JOINT_LIMIT[0][1]));
        SOLVER->set_joint_limit(1, DEG_TO_RAD(UI_JOINT_LIMIT[1][0]), DEG_TO_RAD(UI_JOINT_LIMIT[1][1]));
        SOLVER->set_joint_limit(2, DEG_TO_RAD(UI_JOINT_LIMIT[2][0]), DEG_TO_RAD(UI_JOINT_LIMIT[2][1]));
        SOLVER->set_joint_limit(3, DEG_TO_RAD(UI_JOINT_LIMIT[3][0]), DEG_TO_RAD(UI_JOINT_LIMIT[3][1]));
        SOLVER->set_joint_limit(4, DEG_TO_RAD(UI_JOINT_LIMIT[4][0]), DEG_TO_RAD(UI_JOINT_LIMIT[4][1]));
        SOLVER->set_joint_limit(5, DEG_TO_RAD(UI_JOINT_LIMIT[5][0]), DEG_TO_RAD(UI_JOINT_LIMIT[5][1]));
    }

    if (change_vl) {
        SOLVER->set_joint_velocity_limit(0, DEG_TO_RAD(UI_JOINT_VELOCITY_LIMIT[0]));
        SOLVER->set_joint_velocity_limit(1, DEG_TO_RAD(UI_JOINT_VELOCITY_LIMIT[1]));
        SOLVER->set_joint_velocity_limit(2, DEG_TO_RAD(UI_JOINT_VELOCITY_LIMIT[2]));
        SOLVER->set_joint_velocity_limit(3, DEG_TO_RAD(UI_JOINT_VELOCITY_LIMIT[3]));
        SOLVER->set_joint_velocity_limit(4, DEG_TO_RAD(UI_JOINT_VELOCITY_LIMIT[4]));
        SOLVER->set_joint_velocity_limit(5, DEG_TO_RAD(UI_JOINT_VELOCITY_LIMIT[5]));
    }

    if (change_ts) {
        SOLVER->set_tcp_max_speed(UI_TCP_SPEED);
    }

    if (change_ee || UI_EE_REALTIME) {
        Transform tf = Transform::make_tf(
                UI_EE_TF[0], UI_EE_TF[1], UI_EE_TF[2],
                DEG_TO_RAD(UI_EE_TF[3]),
                DEG_TO_RAD(UI_EE_TF[4]),
                DEG_TO_RAD(UI_EE_TF[5]));
        SOLVER->set_end_effector_offset(tf);
    }

    if (change_sc) {
        SOLVER->set_joint_velocity_limit_scale(UI_JOINT_VSCALE);
    }

}

void draw_3d()
{
    ImGui::Begin("3D");

    if (ImPlot3D::BeginPlot("Robot View", ImVec2(-1, -1), ImPlot3DFlags_Equal | ImPlot3DFlags_NoLegend)) {
        ImPlot3D::SetupAxesLimits(-1.3, 1.3, -1.3, 1.3, 0, 1.4, ImPlot3DCond_Always);
        ImPlot3D::SetupAxes("X", "Y", "Z");

        // --- 바닥 그리기 ---
        Draw::draw_plane(-1.3f, 1.3f, -1.3f, 1.3f);


        // --- 링크 그리기 ---
        {
            ImPlot3D::PushStyleVar(ImPlot3DStyleVar_LineWeight, 3.0f);
            ImPlot3D::PushStyleColor(ImPlot3DCol_Line, ImVec4(0.7f, 0.7f, 0.7f, 1.0f)); // 밝은 회색

            for (int i = 0; i < 6; ++i) {
                Eigen::Vector3d p1 = AXIS3D[i].get_transform();
                Eigen::Vector3d p2 = AXIS3D[i+1].get_transform();

                float xs[2] = { (float)p1.x(), (float)p2.x() };
                float ys[2] = { (float)p1.y(), (float)p2.y() };
                float zs[2] = { (float)p1.z(), (float)p2.z() };

                // ##Link_i 형태로 ID를 주어 겹치지 않게 함
                std::string link_id = "##Link_" + std::to_string(i);
                ImPlot3D::PlotLine(link_id.c_str(), xs, ys, zs, 2);
            }

            // End Effector가 있는 경우 추가 링크 그리기
            if (SOLVER) {
                // 마지막 조인트(Flange) 위치
                Eigen::Vector3d p_flange = AXIS3D[6].get_transform();
                // 엔드 이펙터(EE) 위치
                Eigen::Vector3d p_ee = AXIS3D[7].get_transform();

                float xs_ee[2] = { (float)p_flange.x(), (float)p_ee.x() };
                float ys_ee[2] = { (float)p_flange.y(), (float)p_ee.y() };
                float zs_ee[2] = { (float)p_flange.z(), (float)p_ee.z() };

                ImPlot3D::PushStyleColor(ImPlot3DCol_Line, ImVec4(0.4f, 0.8f, 0.4f, 1.0f)); // EE 링크는 연한 녹색
                ImPlot3D::PlotLine("##Link_EE", xs_ee, ys_ee, zs_ee, 2);
                ImPlot3D::PopStyleColor();
            }

            if (SOLVER) {
                ImVec4 col_color = ImVec4(1.0f, 0.5f, 0.0f, 0.4f); // 오렌지색 반투명

                for (int i = 0; i < 6; ++i) {
                    // 해당 조인트의 정보 가져오기
                    JointInfo info = SOLVER->get_joint_info(i);
                    // 해당 조인트의 현재 월드 Transform 가져오기
                    Transform joint_tf = SOLVER->get_curr_tf(i);

                    // 조인트에 딸린 모든 콜리전 루프
                    for (const auto& col : info.collisions) {
                        Draw::draw_collision(col, joint_tf, col_color);
                    }
                }
            }

            ImPlot3D::PopStyleColor();
            ImPlot3D::PopStyleVar();
        }


        // --- 좌표계 그리기 ---
        {
            // 월드 원점
            AXIS3D[0].draw();

            // 조인트 좌표계
            AXIS3D[1].draw();
            AXIS3D[2].draw();
            AXIS3D[3].draw();
            AXIS3D[4].draw();
            AXIS3D[5].draw();
            AXIS3D[6].draw();

            // End Effector
            if (SOLVER) {
                AXIS3D[7].set_transform(SOLVER->get_curr_tcp_tf());
                AXIS3D[7].draw();
            }
        }


        // --- 매니폴러빌리티 그리기 ---
        if (SOLVER && SHOW_MANIPULABILITY) {
            vec3 center = SOLVER->get_curr_tcp_tf().translation();

            // 위치 타원체 그리기
            mat<3, 6> j_pos = SOLVER->get_curr_jacobian().block<3, 6>(0, 0);
            Draw::draw_manipulability(j_pos, center, 0.3f, ImVec4(0, 1, 1, 0.2f), false);

            // 회전 타원체 그리기
            mat<3, 6> j_ori = SOLVER->get_curr_jacobian().bottomRows<3>();
            Draw::draw_manipulability(j_ori, center, 0.4f, ImVec4(1, 0, 1, 0.2f), true);
        }


        // --- 경계 박스 그리기 ---
        const vec3 min = vec3{UI_WS_MIN[0], UI_WS_MIN[1], UI_WS_MIN[2]};
        const vec3 max = vec3{UI_WS_MAX[0], UI_WS_MAX[1], UI_WS_MAX[2]};
        Draw::draw_box(min, max);


        ImPlot3D::EndPlot();
    }

    ImGui::End();
}


int main(int argc, char *argv[]) {

    INIT_LOOP_TIMER(1.0); // 1ms


    ImGui::start("Uon Robotics", ImVec2(1280, 720));

    auto start_time = std::chrono::system_clock::now();
    double dt = 0;
    while (ImGui::isRunning())
    {
        UPDATE_LOOP_TIMER();

        // ---------------------------------------------
        // ImGui 컨텍스트
        // ---------------------------------------------
        ImGui::context([&]()
        {
            draw_setting();
            draw_control();
            draw_3d();
        });

        if (!SOLVER)
            continue;

        // ---------------------------------------------
        // 제어 알고리즘 실행
        // ---------------------------------------------

        auto t1 = std::chrono::system_clock::now();
        if (CONTROL_MODE == ControlMode_::ControlMode_MoveJ) {

            // ui의 값으로 target을 만듬
            vec<6> target;
            target[0] = J1;
            target[1] = J2;
            target[2] = J3;
            target[3] = J4;
            target[4] = J5;
            target[5] = J6;

            SOLVER->movej(DEG_TO_RAD(target));

            // ui의 task space값 업데이트
            Transform tcp = SOLVER->get_curr_tcp_tf();
            X = (float)tcp.x();
            Y = (float)tcp.y();
            Z = (float)tcp.z();

            vec3 rpy = tcp.rpy();
            A = (float)RAD_TO_DEG(rpy[0]);
            B = (float)RAD_TO_DEG(rpy[1]);
            C = (float)RAD_TO_DEG(rpy[2]);
        }
        else {
            // ui의 task space 값으로 target을 만듬
            Transform target = Transform::make_tf(X, Y, Z, DEG_TO_RAD(A), DEG_TO_RAD(B), DEG_TO_RAD(C));

            // MoveL/MoveX 모드
            if (CONTROL_MODE == ControlMode_::ControlMode_MoveL)
                SOLVER->movel(target);
            else
                SOLVER->movex(target);

            // ui의 joint값 업데이트
            const vec<6> q_deg = SOLVER->get_curr_joint_deg();
            J1 = q_deg[0];
            J2 = q_deg[1];
            J3 = q_deg[2];
            J4 = q_deg[3];
            J5 = q_deg[4];
            J6 = q_deg[5];
        }

        auto t2 = std::chrono::system_clock::now();
        UI_MOVEL_ECLIPSE = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        // ---------------------------------------------
        // 그래프 시각화 값 업데이트
        // ---------------------------------------------
        AXIS3D[1].set_transform(SOLVER->get_curr_tf(0));
        AXIS3D[2].set_transform(SOLVER->get_curr_tf(1));
        AXIS3D[3].set_transform(SOLVER->get_curr_tf(2));
        AXIS3D[4].set_transform(SOLVER->get_curr_tf(3));
        AXIS3D[5].set_transform(SOLVER->get_curr_tf(4));
        AXIS3D[6].set_transform(SOLVER->get_curr_tf(5));

        // ---------------------------------------------
        // 공유 메모리에 데이터 쓰기 (블렌더 시각화)
        // ---------------------------------------------
        const vec<6> q_current_deg = SOLVER->get_curr_joint_deg(); // 실제 현재 각도 (deg)
        ShmMoveJ data;
        data.joint[0] = q_current_deg[0];
        data.joint[1] = q_current_deg[1];
        data.joint[2] = q_current_deg[2];
        data.joint[3] = q_current_deg[3];
        data.joint[4] = q_current_deg[4];
        data.joint[5] = q_current_deg[5];
        strcpy(data.status, "run");
        SHM->set(data);

        // ---------------------------------------------
        // 타이머 대기
        // ---------------------------------------------
        SLEEP_LOOP_TIMER();
    }




    ImGui::stop();

    return 0;
}