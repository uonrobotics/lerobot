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
#include "yaml.hpp"
#include "shard_memory.hpp"


// ==============================
// 정의
// ==============================
#define SHARED_MEMORY_ID    "movej"
#define CONFIG_FILE         std::string(ROOT_PATH) + "/config/robot.yaml"
#define CONTENT_PATH        std::string(ROOT_PATH) + "/contents"
#define SLEEP(x)            std::this_thread::sleep_for(std::chrono::milliseconds(x))


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
};


// ==============================
// 객체
// ==============================

std::string URDF_FILE = "";
std::unique_ptr<IkSolver> SOLVER;
std::unique_ptr<Yaml> CONFIG;
std::unique_ptr<SharedMemory<ShmMoveJ>> SHM;



// ==============================
// UI 변수
// ==============================
float J1 = 0;
float J2 = 0;
float J3 = 0;
float J4 = 0;
float J5 = 0;
float J6 = 0;

float X = 0;
float Y = 0;
float Z = 0;
float RX = 0;
float RY = 0;
float RZ = 0;

int CONTROL_MODE = 0;                   // 제어 모드
int SOLVER_MODE = 1;                    // 솔버 모드
bool SHOW_MANIPULABILITY = false;       // 매니풀러빌리티 표시

float WS_MIN[3] = {0, 0, 0};            // 작업 영역 최소 값
float WS_MAX[3] = {0, 0, 0};            // 작업 영역 최대 값

float JLIMITS[6][2] = {0, };            // joint 제한 범위
float VLIMIT[6] = {0, 0, 0, 0, 0, 0};   // joint 최대 속도
float TCP_SPEED = 0;                    // tcp 최대 속도
float EE_TF[3] = {0, 0, 0};             // end effector 위치 [x, y, z]
float JFACTOR = 1.0;                    // joint 최대 제한 속도 비율




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
    CONFIG = std::make_unique<Yaml>(CONFIG_FILE);
    if (!CONFIG->init()) {
        return;
    }
    else {
        // ws 설정 가져오기
        std::vector<float> ws_min = CONFIG->get<std::vector<float>>("workspace.min", {-1.0, -1.0, 0.0});
        std::vector<float> ws_max = CONFIG->get<std::vector<float>>("workspace.max", {1.0, 1.0, 1.0});

        for (int i=0; i<3; ++i) {
            WS_MIN[i] = ws_min[i];
            WS_MAX[i] = ws_max[i];
        }


        // 조인트 제한
        std::vector<float> joint_min = CONFIG->get<std::vector<float>>("joint.lower", {-360, -360, -160, -360, -360, -360});
        std::vector<float> joint_max = CONFIG->get<std::vector<float>>("joint.upper", {360, 360, 160, 360, 360, 360});

        for(int i=0; i<6; ++i) {
            JLIMITS[i][0] = joint_min[i];
            JLIMITS[i][1] = joint_max[i];
        }


        // 조인트 속도 제한
        std::vector<float> joint_vlimit = CONFIG->get<std::vector<float>>("joint.velocity", {120, 120, 180, 255, 255, 255});
        for(int i=0; i<6; ++i) {
            VLIMIT[i] = joint_vlimit[i];
        }


        // TCP 속도 제한
        TCP_SPEED = CONFIG->get<float>("tcp.speed", 1.0);
        JFACTOR = CONFIG->get<float>("joint.factor", 1.0);


        // 엔드 이펙터 설정
        std::vector<float> ee_tf = CONFIG->get<std::vector<float>>("tcp.tf", {0,0,0});
        EE_TF[0] = ee_tf[0];
        EE_TF[1] = ee_tf[1];
        EE_TF[2] = ee_tf[2];

        CONFIG->save();
    }


    // --- 공유 메모리 ---
    SHM = std::make_unique<SharedMemory<ShmMoveJ>>(SHARED_MEMORY_ID);
    if (!SHM->init()) {
        return;
    }

    if (URDF_FILE.empty())
        return;

    // --- 솔버 생성 ---
    SOLVER = std::make_unique<IkSolver>(URDF_FILE);
    if (!SOLVER->init()) {
        return;
    }


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
    SOLVER->set_workspace_limitX(WS_MIN[0], WS_MAX[0]);                               // -x, x
    SOLVER->set_workspace_limitY(WS_MIN[1], WS_MAX[1]);                               // -y, y
    SOLVER->set_workspace_limitZ(WS_MIN[2], WS_MAX[2]);                               // -z, z

    // --- TCP 속도 제한 ---
    SOLVER->set_tcp_max_speed(TCP_SPEED);                                           // tcp speed
    SOLVER->set_safety_scale(JFACTOR);                                                 // joint velocity fector

    // --- End Effecotr 설정 ---
    Transform tf = Transform::make_tf(EE_TF[0], EE_TF[1], EE_TF[2], 0, 0, 0);
    SOLVER->set_end_effector_tf(tf);

    // -- SOLVER 초기 위치 설정 ---
    double q[6] = {0, 0, 0, 0, 0, 0};
    SOLVER->movej(q);

    // --- Ui TCP 위치 업데이트 ---
    Transform tcp = SOLVER->get_tcp_tf();
    X = tcp.translation().x();
    Y = tcp.translation().y();
    Z = tcp.translation().z();

    vec3 zyz = tcp.rpy();
    RX = RAD_TO_DEG(zyz[0]);
    RY = RAD_TO_DEG(zyz[1]);
    RZ = RAD_TO_DEG(zyz[2]);

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
    AXIS3D[1].set_transform(SOLVER->get_tf(0));                           // J1
    AXIS3D[2].set_transform(SOLVER->get_tf(1));                           // J2
    AXIS3D[3].set_transform(SOLVER->get_tf(2));                           // J3
    AXIS3D[4].set_transform(SOLVER->get_tf(3));                           // J4
    AXIS3D[5].set_transform(SOLVER->get_tf(4));                           // J5
    AXIS3D[6].set_transform(SOLVER->get_tf(5));                           // J6
    AXIS3D[7].set_transform(SOLVER->get_tcp_tf());                        // flange
    AXIS3D[7].set_colors(
        ImVec4(1.00f, 0.65f, 0.65f, 1.0f), // 파스텔 레드 (X축)
        ImVec4(0.65f, 1.00f, 0.65f, 1.0f), // 파스텔 그린 (Y축)
        ImVec4(0.65f, 0.85f, 1.00f, 1.0f)  // 파스텔 블루 (Z축)
    );
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
        const vec<6> jvel = SOLVER->get_curr_jvel_deg();
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
    ImGui::RadioButton("MoveL", &CONTROL_MODE, ControlMode_::ControlMode_MoveL);
    ImGui::Dummy(ImVec2(0, 10));

    ImGui::PushItemWidth(150);

    // 조인트 설정 슬라이더
    ImGui::SeparatorText("조인트 제어");
    ImGui::DragFloat("Joint1", &J1, 0.1f, -360, 360, "%.3f deg");  ImGui::SameLine();  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[0]);
    ImGui::DragFloat("Joint2", &J2, 0.1f, -360, 360, "%.3f deg");  ImGui::SameLine();  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[1]);
    ImGui::DragFloat("Joint3", &J3, 0.1f, -360, 360, "%.3f deg");  ImGui::SameLine();  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[2]);
    ImGui::DragFloat("Joint4", &J4, 0.1f, -360, 360, "%.3f deg");  ImGui::SameLine();  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[3]);
    ImGui::DragFloat("Joint5", &J5, 0.1f, -360, 360, "%.3f deg");  ImGui::SameLine();  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[4]);
    ImGui::DragFloat("Joint6", &J6, 0.1f, -360, 360, "%.3f deg");  ImGui::SameLine();  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vel: %.1f d/s", v[5]);
    ImGui::Dummy(ImVec2(0, 10));

    // 테스크 제어 슬라이더
    ImGui::SeparatorText("테스크 제어");
    ImGui::DragFloat("X [m]   ", &X, 0.005f, -2.3f, 2.3f);      ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vx: %.3f m/s", V[0]); // x
    ImGui::DragFloat("Y [m]   ", &Y, 0.005f, -2.3f, 2.3f);      ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vy: %.3f m/s", V[1]); // y
    ImGui::DragFloat("Z [m]   ", &Z, 0.005f, -2.3f, 2.3f);      ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Vz: %.3f m/s", V[2]); // z
    ImGui::DragFloat("\u03C6 [deg]", &RX, 0.1f, -360.f, 360.f); ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Wx: %.1f d/s", V[3]); // roll
    ImGui::DragFloat("\u03B8 [deg]", &RY, 0.1f, -360.f, 360.f); ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Wy: %.1f d/s", V[4]); // pitch
    ImGui::DragFloat("\u03C8 [deg]", &RZ, 0.1f, -360.f, 360.f); ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Wz: %.1f d/s", V[5]); // yaw

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
    bool change_sm = false;


    ImGui::Begin("설정");

    // --- 매니퓰러빌리티 표시 ---
    ImGui::Checkbox("Show Manipulability", &SHOW_MANIPULABILITY);
    ImGui::Dummy(ImVec2(0, 10));


    // --- 작업 영역 설정 ---
    ImGui::SeparatorText("작업 영역 설정 (Box)");
    change_ws |= ImGui::SliderFloat3("최소 [m]", WS_MIN, -1.0f, 1.0f);
    change_ws |= ImGui::SliderFloat3("최대 [m]", WS_MAX , -1.0f, 1.0f);
    ImGui::Dummy(ImVec2(0, 10));


    // --- 조인트 범위 설정 ---
    ImGui::SeparatorText("조인트 범위 설정");
    change_jl |= ImGui::SliderFloat2("J0 [deg]", JLIMITS[0], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J1 [deg]", JLIMITS[1], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J2 [deg]", JLIMITS[2], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J3 [deg]", JLIMITS[3], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J4 [deg]", JLIMITS[4], -360.f, 360.f);
    change_jl |= ImGui::SliderFloat2("J5 [deg]", JLIMITS[5], -360.f, 360.f);
    ImGui::Dummy(ImVec2(0, 10));


    // --- 조인트 속도 제한 설정 ---
    ImGui::SeparatorText("조인트 속도 제한 설정");
    change_vl |= ImGui::SliderFloat("J0 [deg/s²]", &VLIMIT[0], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J1 [deg/s²]", &VLIMIT[1], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J2 [deg/s²]", &VLIMIT[2], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J3 [deg/s²]", &VLIMIT[3], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J4 [deg/s²]", &VLIMIT[4], -360.f, 360.f);
    change_vl |= ImGui::SliderFloat("J5 [deg/s²]", &VLIMIT[5], -360.f, 360.f);
    ImGui::Dummy(ImVec2(0, 5));
    change_sc |= ImGui::SliderFloat("속도 펙터", &JFACTOR, 1.0f, 2.0f);
    ImGui::Dummy(ImVec2(0, 10));


    // --- TCP 속도 제한 설정 ---
    ImGui::SeparatorText("TCP 설정");
    change_ts |= ImGui::SliderFloat("속도 제한 [m/s]", &TCP_SPEED, 0.0f, 1.0f);
    ImGui::Dummy(ImVec2(0, 10));


    ImGui::SeparatorText("End Effector 설정");
    ImGui::SliderFloat3("x, y, z", &EE_TF[0], -1.0f, 1.0f); ImGui::SameLine();
    if (ImGui::Button("적용")) {
        change_ee = true;
    }
    ImGui::Dummy(ImVec2(0, 10));


    // --- 솔버 모드 설정 ---
    ImGui::SeparatorText("Solver 모드 설정");
    change_sm |= ImGui::RadioButton("RELAX PATH", &SOLVER_MODE, 0); ImGui::SameLine();
    change_sm |= ImGui::RadioButton("STRICT PATH", &SOLVER_MODE, 1);
    ImGui::Dummy(ImVec2(0, 10));


    ImGui::End();


    // --------------------


    if (!SOLVER)
        return;

    if (change_ws) {
        SOLVER->set_workspace_limitX(WS_MIN[0], WS_MAX[0]);
        SOLVER->set_workspace_limitY(WS_MIN[1], WS_MAX[1]);
        SOLVER->set_workspace_limitZ(WS_MIN[2], WS_MAX[2]);
    }

    if (change_jl) {
        SOLVER->set_joint_limit(0, DEG_TO_RAD(JLIMITS[0][0]), DEG_TO_RAD(JLIMITS[0][1]));
        SOLVER->set_joint_limit(1, DEG_TO_RAD(JLIMITS[1][0]), DEG_TO_RAD(JLIMITS[1][1]));
        SOLVER->set_joint_limit(2, DEG_TO_RAD(JLIMITS[2][0]), DEG_TO_RAD(JLIMITS[2][1]));
        SOLVER->set_joint_limit(3, DEG_TO_RAD(JLIMITS[3][0]), DEG_TO_RAD(JLIMITS[3][1]));
        SOLVER->set_joint_limit(4, DEG_TO_RAD(JLIMITS[4][0]), DEG_TO_RAD(JLIMITS[4][1]));
        SOLVER->set_joint_limit(5, DEG_TO_RAD(JLIMITS[5][0]), DEG_TO_RAD(JLIMITS[5][1]));
    }

    if (change_vl) {
        SOLVER->set_joint_vlimit(0, DEG_TO_RAD(VLIMIT[0]));
        SOLVER->set_joint_vlimit(1, DEG_TO_RAD(VLIMIT[1]));
        SOLVER->set_joint_vlimit(2, DEG_TO_RAD(VLIMIT[2]));
        SOLVER->set_joint_vlimit(3, DEG_TO_RAD(VLIMIT[3]));
        SOLVER->set_joint_vlimit(4, DEG_TO_RAD(VLIMIT[4]));
        SOLVER->set_joint_vlimit(5, DEG_TO_RAD(VLIMIT[5]));
    }

    if (change_ts) {
        SOLVER->set_tcp_max_speed(TCP_SPEED);
    }

    if (change_ee) {
        Transform tf = Transform::make_tf(EE_TF[0], EE_TF[1], EE_TF[2], 0, 0, 0);
        SOLVER->set_end_effector_tf(tf);
    }

    if (change_sc) {
        SOLVER->set_safety_scale(JFACTOR);
    }

    if (change_sm) {
        if (SOLVER_MODE == IkSolver::LimitMode::RELAX_MODE)
            SOLVER->set_relax_mode();
        else if (SOLVER_MODE == IkSolver::LimitMode::STRICT_MODE)
            SOLVER->set_strict_mode();
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
                AXIS3D[7].set_transform(SOLVER->get_tcp_tf());
                AXIS3D[7].draw();
            }
        }


        // --- 매니폴러빌리티 그리기 ---
        if (SOLVER && SHOW_MANIPULABILITY) {
            vec3 center = SOLVER->get_tcp_tf().translation();

            // 위치 타원체 그리기
            mat<3, 6> j_pos = SOLVER->get_jacobian().block<3, 6>(0, 0);
            Draw::draw_manipulability(j_pos, center, 0.3f, ImVec4(0, 1, 1, 0.2f), false);

            // 회전 타원체 그리기
            mat<3, 6> j_ori = SOLVER->get_jacobian().bottomRows<3>();
            Draw::draw_manipulability(j_ori, center, 0.4f, ImVec4(1, 0, 1, 0.2f), true);
        }


        // --- 경계 박스 그리기 ---
        const vec3 min = vec3{WS_MIN[0], WS_MIN[1], WS_MIN[2]};
        const vec3 max = vec3{WS_MAX[0], WS_MAX[1], WS_MAX[2]};
        Draw::draw_box(min, max);


        ImPlot3D::EndPlot();
    }

    ImGui::End();
}


int main(int argc, char *argv[]) {

    INIT_LOOP_TIMER(1.0); // 1ms


    ImGui::start("Uon Robotics", ImVec2(1280, 720));
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
        if (CONTROL_MODE == ControlMode_::ControlMode_MoveJ) {
            // --- 솔버 업데이트 --------------------------
            double q[6];
            q[0] = DEG_TO_RAD(J1);
            q[1] = DEG_TO_RAD(J2);
            q[2] = DEG_TO_RAD(J3);
            q[3] = DEG_TO_RAD(J4);
            q[4] = DEG_TO_RAD(J5);
            q[5] = DEG_TO_RAD(J6);
            SOLVER->movej(q);

            // --- 위치 ----------------------------------
            Transform tcp = SOLVER->get_tcp_tf();
            X = tcp.translation().x();
            Y = tcp.translation().y();
            Z = tcp.translation().z();

            // --- 회전 ----------------------------------
            vec3 rpy = tcp.rpy();
            RX = RAD_TO_DEG(rpy[0]); // rpy[0]은 Roll (X)
            RY = RAD_TO_DEG(rpy[1]); // rpy[1]은 Pitch (Y)
            RZ = RAD_TO_DEG(rpy[2]); // rpy[2]은 Yaw (Z)
        }
        else if (CONTROL_MODE == ControlMode_::ControlMode_MoveL) {
            double x, y, z, roll, pitch, yaw;
            x = X;
            y = Y;
            z = Z;
            yaw = DEG_TO_RAD(RZ);
            pitch = DEG_TO_RAD(RY);
            roll = DEG_TO_RAD(RX);

            Transform tf = Transform::make_tf(x, y, z, roll, pitch, yaw);

            auto t1 = std::chrono::high_resolution_clock::now();
            SOLVER->movel(tf);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << "MoveL Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
        }


        // ---------------------------------------------
        // UI Joint 값 업데이트
        // ---------------------------------------------
        const vec<6> q = SOLVER->get_curr_joint_deg();
        J1 = q[0];
        J2 = q[1];
        J3 = q[2];
        J4 = q[3];
        J5 = q[4];
        J6 = q[5];


        // ---------------------------------------------
        // 그래프 시각화 값 업데이트
        // ---------------------------------------------
        AXIS3D[1].set_transform(SOLVER->get_tf(0));
        AXIS3D[2].set_transform(SOLVER->get_tf(1));
        AXIS3D[3].set_transform(SOLVER->get_tf(2));
        AXIS3D[4].set_transform(SOLVER->get_tf(3));
        AXIS3D[5].set_transform(SOLVER->get_tf(4));
        AXIS3D[6].set_transform(SOLVER->get_tf(5));

        // ---------------------------------------------
        // 공유 메모리에 데이터 쓰기 (블렌더 시각화)
        // ---------------------------------------------
        ShmMoveJ data;
        data.joint[0] = q[0];
        data.joint[1] = q[1];
        data.joint[2] = q[2];
        data.joint[3] = q[3];
        data.joint[4] = q[4];
        data.joint[5] = q[5];
        strcpy(data.status, "run");
        SHM->set(data);

        // ---------------------------------------------
        // 실행 주기 보장을 위한 정밀 대기
        // ---------------------------------------------
        SLEEP_LOOP_TIMER();
    }




    ImGui::stop();

    return 0;
}
