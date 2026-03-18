#pragma once
#include "kinematics.h"
#include "transform.hpp"
#include <chrono>
#include <iostream>

class IkSolver {
public:
    // ==================================================================
    // QP Solver Output Struct
    // ==================================================================
    struct QPProblem {
        mat<6, 6> H;                                                     // hessian
        vec<6> g;                                                        // gradiant
        vec<6> lb;                                                       // lower bound
        vec<6> ub;                                                       // upper bound
    };

    // ==================================================================
    // 제한 모드
    // ==================================================================
    enum LimitMode {
        RELAX_MODE = 0,                                                  // 경로 이탈 허용 (싱귤려 모션에서 관절 속도만 조절함)
        STRICT_MODE = 1,                                                 // 경로 엄격 유지
    };

private:
    // ------------------------------------------------------------------
    // 제어 파라미터
    // ------------------------------------------------------------------
    const int MAX_IK_ITER = 100;                                         // IK 반복 횟수
    const int MAX_QP_ITER = 20;                                          // QP 반복 횟수
    const double STEP_SIZE = 1.0;                                        // Alpha
    const double LAMBDA = 0.0001;                                        // Damping factor

    const double POS_ERROR_THRESHOLD = 0.00001;                          // 위치 오차 [m]
    const double ROT_ERROR_THRESHOLD = DEG_TO_RAD(0.001);                // 회전 오차 [rad]

    const double WS_LIMIT_MARGIN = 0.05;                                 // 범위 제한이 적용이 시작 되는 값 [m]
    const double WS_P_GAIN = 5.0;                                        // 범위 제한 복원 이득 (클수록 강하게 안으로 밀어넣음)

    // ---
    vec3 ws_limit_min;                                                   // 범위 제한을 위한 최소값 [m]
    vec3 ws_limit_max;                                                   // 범위 제한을 위한 최대값 [m]
    double tcp_max_speed;                                                // 최대 TCP 속도 [m/s] (0이면 제한 없음)
    double safety_factor;                                                // 각속도 제한 factor

    Transform flange_offset;                                             // joint6에서 Flange까지의 고정 오프셋
    Transform end_effector;                                              // End-Effector Transform
    Transform end_effector_inverse;                                      // End-Effector Transform's inverse
    std::array<Kinematics, 6> ki;                                        // Kinematics
    std::array<Transform, 6> fk_cache;                                   // FK cache
    LimitMode limit_mode;                                                // Limit Mode

    mat<6, 6> J;                                                         // Jacobian

    vec6 cur_q;                                                          // 현재 조인트 각도 [rad]
    vec6 joint_vel;                                                      // 현재 조인트 각속도 [rad/s]
    vec6 tcp_vel;                                                        // 현재 끝단 속도 [m/s]

    // ---
    std::string urdf_path;
    std::chrono::steady_clock::time_point lastTime_;

public:
    IkSolver(const std::string& urdf_path);
    bool init();

    // ------------------------------------------------------------------
    // Getter
    // ------------------------------------------------------------------
    vec<6> get_curr_joint_rad() const { return cur_q;                  } // [rad]
    vec<6> get_curr_joint_deg() const { return RAD_TO_DEG(cur_q);      } // [deg]
    vec<6> get_curr_jvel_rad()  const { return joint_vel;              } // [rad]
    vec<6> get_curr_jvel_deg()  const { return RAD_TO_DEG(joint_vel);  } // [deg]
    vec<6> get_curr_tcp_speed() const { return tcp_vel;                } // [m]
    mat<6, 6> get_jacobian()    const { return J;                      } // [Matrix 6x6]

    // ------------------------------------------------------------------
    // tcp, end effector
    // ------------------------------------------------------------------
    Transform get_tcp_tf() const;                                        // Get TCP Transform
    Transform get_tf(int index) const;                                   // Get FK Transform

    void set_end_effector_tf(const Transform& tf);                       // End-Effector 설정
    bool has_end_effector() const;                                       // ee가 설정되었는지 확인
    void remove_end_effector();                                          // ee 제거


    // ------------------------------------------------------------------
    // 모드 설정
    // ------------------------------------------------------------------
    void set_strict_mode()    { limit_mode = STRICT_MODE; }              // 경로 정확도 우선
    void set_relax_mode()     { limit_mode = RELAX_MODE;  }              // 조인트 속도 우선

    // ------------------------------------------------------------------
    // 제어 함수
    // ------------------------------------------------------------------
    void movej(const double (&q)[6]);                                    // MoveJ [rad]
    void movel(const Transform& tf);                                     // MoveL

    // ------------------------------------------------------------------
    // 제한값 설정
    // ------------------------------------------------------------------
    void set_joint_limit(int index, double min, double max);             // 조인트 범위 제한 [rad]
    void set_joint_vlimit(int index, double max);                        // 조인트 최대 속도 제한 [rad/s]
    void set_workspace_limitX(double min, double max);                   // x축 작업 범위 제한 [m]
    void set_workspace_limitY(double min, double max);                   // y축 작업 범위 제한 [m]
    void set_workspace_limitZ(double min, double max);                   // z축 작업 범위 제한 [m]
    void set_tcp_max_speed(double max);                                  // TCP 최대 속도 제한 [m/s]
    void set_safety_scale(double scale);                                 // 각 조인트 속도 제한 비율 [1.0~2.0]

private:
    // ------------------------------------------------------------------
    // 알고리즘
    // ------------------------------------------------------------------
    Transform fk(const vec<6>& q) const;                                                               // Forward Kinematics
    vec<6>    ik(const Transform& tar_tf, double dt);                                                  // Inverse Kinematics
    vec<6>    ik_core(const Transform& tar_flange,double dt);                                          // Ik core
    Transform fk_and_jacobian(const vec<6>& q);                                                        // FK와 Jacobian 동시 계산 (최적화용)

    QPProblem setup_QP(const mat<6, 6>& J, const vec<6>& dx, const vec<6>& cur_q, double dt) const;    // QP에 필요한 요소 계산
    vec<6>    solve_QP(const QPProblem& qp) const;                                                     // Quadratic Programming Solver

    vec<6>    limit_workspace(const vec3& cur_pos, const vec<6>& dx) const;                            // TCP 작업 영역 제한
    vec<6>    limit_joint_velocity(const vec<6>& tar_q, double dt) const;                              // 각 조인트 최대 속도 제한
    Transform limit_tcp_speed(const Transform& cur_tf, const Transform& tar_tf, double dt) const;      // TCP 속도 제한한 TF 계산

    void      _movej(double q0, double q1, double q2, double q3, double q4, double q5, double dt);
    void      _movel(const Transform& target, double dt);

    // ------------------------------------------------------------------
    // 유틸리티
    // ------------------------------------------------------------------
    double dt();
};


