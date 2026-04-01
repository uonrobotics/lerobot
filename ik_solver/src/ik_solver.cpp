#include "ik_solver.h"

#include <filesystem>
#include <iostream>
#include <fstream>

#define IK_INFO  cout << "[Info ] [Ik Solver] "
#define IK_WARN  cout << "[Warn ] [Ik Solver] "
#define IK_ERROR cout << "[Error] [Ik Solver] "

using namespace std;

IkSolver::IkSolver(const std::string& urdf_path) : urdf_path(urdf_path)
{
    // 변수 초기화
    cur_q.setZero();
    joint_vel.setZero();
    tcp_vel.setZero();
    J.setZero();

    // Workspace Limit 설정
    ws_min << -1.0, -1.0, 0.0; // [m]
    ws_max << 1.0, 1.0, 1.0;   // [m]

    // tcp 속도 제한
    tcp_max_speed = 1.0; // [m/s]
    joint_velocity_scale = 1.0;

    end_effector = Transform();
    end_effector_inverse = Transform();
    flange_offset = Transform();

    lastTime_ = std::chrono::steady_clock::now();
}

bool IkSolver::init()
{
    // urdf 파일 로드
    UrdfPtr urdf = Kinematics::load_urdf_file(urdf_path);
    if (!urdf) {
        IK_ERROR << "URDF 파일 로드 실패: " << urdf_path << endl;
        return false;
    }

    // print version
    version();

    // Kinematics 설정
    Kinematics::print_header();
    ki[0] = Kinematics::build_from_urdf(urdf, "joint1");
    ki[1] = Kinematics::build_from_urdf(urdf, "joint2");
    ki[2] = Kinematics::build_from_urdf(urdf, "joint3");
    ki[3] = Kinematics::build_from_urdf(urdf, "joint4");
    ki[4] = Kinematics::build_from_urdf(urdf, "joint5");
    ki[5] = Kinematics::build_from_urdf(urdf, "joint6");

    // flange
    auto ki_f = Kinematics::build_from_urdf(urdf, "flange");
    flange_offset = ki_f.get_origin_tf(); // flange 설정

    // end-effector
    auto ki_e = Kinematics::build_from_urdf(urdf, "end_effector");
    set_end_effector_offset(ki_e.get_origin_tf()); // end-effector 설정

    
    // 초기 자세 설정 및 캐시/상태 초기화
    _movej(0, 0, 0, 0, 0, 0, 0);

    
    filesystem::path p(urdf_path);
    IK_INFO << "초기화 완료: " << p.filename().string() << endl;
    return true;
}

void IkSolver::movel(const Transform& tf)
{
    _movel(tf, safe_dt());
}

void IkSolver::movex(const Transform& tf)
{
    vec<6> temp_q = cur_q; // ik 계산중에 업데이트 하지 않기 위함

    // flange
    Transform tar_flange = tf * end_effector_inverse;
    vec<6> target_q = temp_q;

    // --- 내부 IK Solver 루프 (상태 업데이트 없이 계산만 수행) ---
    int iter = MAX_IK_ITER;
    while (iter-- > 0) {
        Transform cur_flange = fk(target_q);

        // Twist 계산 (Cartesian Error)
        vec<6> dx = Transform::twist(cur_flange, tar_flange);

        // 에러 체크
        if (dx.head<3>().norm() < POS_TOL && dx.tail<3>().norm() < ROT_TOL)
            break;

        // Jacobian 계산
        fk_and_jacobian(target_q);

        const QPProblem qp = setup_QP(J, dx, target_q, 0.1); // 가상의 dt 사용
        const vec<6> dq = solve_QP(qp);

        target_q += dq * STEP_SIZE;
    }

    // 기존 movej 대신 동기화 버전 호출
    _movej_sync(target_q[0], target_q[1], target_q[2], target_q[3], target_q[4], target_q[5], safe_dt());
}

vec<6> IkSolver::get_curr_joint_rad() const
{
    return cur_q;
}

vec<6> IkSolver::get_curr_joint_deg() const
{
    return RAD_TO_DEG(cur_q);
}

vec<6> IkSolver::get_curr_joint_velocity_rad() const
{
    return joint_vel;
}

vec<6> IkSolver::get_curr_joint_velocity_deg() const
{
    return RAD_TO_DEG(joint_vel);
}

vec<6> IkSolver::get_curr_tcp_speed() const
{
    return tcp_vel;
}

Transform IkSolver::get_curr_tcp_tf() const
{
    return fk_cache[5] * flange_offset * end_effector;
}

Transform IkSolver::get_curr_tf(int index) const
{
    return fk_cache[index];
}

mat<6, 6> IkSolver::get_curr_jacobian() const
{
    return J;
}

JointInfo IkSolver::get_joint_info(int index) const
{
    return ki[index].get_joint_info();
}

Transform IkSolver::get_end_effector_offset() const
{
    return end_effector;
}

void IkSolver::set_joint(const vec<6> &joint)
{
    _movej(joint[0], joint[1], joint[2], joint[3], joint[4], joint[5],0.0);
}

void IkSolver::set_joint_limit(int index, double min, double max)
{
    ki[index].set_joint_limit(min, max);
}

void IkSolver::set_workspace_limitX(double min, double max)
{
    ws_min(0) = min;
    ws_max(0) = max;
}

void IkSolver::set_workspace_limitY(double min, double max)
{
    ws_min(1) = min;
    ws_max(1) = max;
}

void IkSolver::set_workspace_limitZ(double min, double max)
{
    ws_min(2) = min;
    ws_max(2) = max;
}

void IkSolver::set_tcp_max_speed(double max)
{
    tcp_max_speed = max;
}

void IkSolver::set_joint_velocity_limit_scale(double scale)
{
    if (scale > 1e-6) {
        joint_velocity_scale = 1.0 / scale;
    } else {
        joint_velocity_scale = 1.0;
    }
}

void IkSolver::set_joint_velocity_limit(int index, double max)
{
    ki[index].set_joint_vlimit(max);
}

void IkSolver::set_end_effector_offset(const Transform& tf)
{
    end_effector = tf;
    end_effector_inverse = tf.inverse();
}

void IkSolver::movej(const vec<6>& q)
{
    vec<6> tar_q = { q[0], q[1], q[2], q[3], q[4], q[5] };
    double _dt = safe_dt();

    // 관절 범위 적용
    for (int i = 0; i < 6; ++i) {
        double min_q = ki[i].get_joint_limits().lower;
        double max_q = ki[i].get_joint_limits().upper;
        tar_q[i] = std::clamp(tar_q[i], min_q, max_q);
    }

    // 관절 속도 제한 적용
    if (_dt > 1e-6) {
        tar_q = limit_joint_velocity(tar_q, _dt);
        _movej(tar_q[0], tar_q[1], tar_q[2], tar_q[3], tar_q[4], tar_q[5], _dt);
    }
}

Transform IkSolver::fk(const vec<6>& q) const
{
    return Kinematics::fk_kernel<double>(ki, q) * flange_offset;
}

Transform IkSolver::fk_and_jacobian(const vec<6>& q)
{
    Transform flange;
    Transform t_accum = Transform();

    vec3 p_joints[6];
    vec3 z_joints[6];

    // fk 계산
    for (int i = 0; i < 6; ++i) {
        const Kinematics& k = ki[i];
        const JointInfo& info = k.get_joint_info();

        Transform tf_joint_origin_base = t_accum * k.get_origin_tf();
        p_joints[i] = tf_joint_origin_base.translation();
        z_joints[i] = tf_joint_origin_base.rotation() * info.axis;

        Transform tf_joint_motion;
        if (info.type == JointType::REVOLUTE || info.type == JointType::CONTINUOUS) {
            tf_joint_motion = Transform(AngleAxis(q[i], info.axis), vec3::Zero());
        }
        else if (info.type == JointType::PRISMATIC) {
            tf_joint_motion = Transform(quat::Identity(), info.axis * q[i]);
        }
        else {
            tf_joint_motion = Transform(); // Identity for fixed
        }

        t_accum = tf_joint_origin_base * tf_joint_motion;
    }


    // tcp는 flange
    t_accum = t_accum * flange_offset;
    flange = t_accum;
    const vec3 p_e = t_accum.translation(); // flange의 위치


    // jacobian 계산
    for (int i = 0; i < 6; ++i) {
        const JointType type = ki[i].get_joint_info().type;
        if (type == JointType::REVOLUTE || type == JointType::CONTINUOUS) {
            J.block<3, 1>(0, i) = z_joints[i].cross(p_e - p_joints[i]);
            J.block<3, 1>(3, i) = z_joints[i];
        }
        else if (type == JointType::PRISMATIC) {
            J.block<3, 1>(0, i) = z_joints[i];
            J.block<3, 1>(3, i) = vec3::Zero();
        }

        for (auto& col : ki[i].get_joint_info().collisions) {
            col.world_origin = t_accum * col.origin;
        }
    }

    return flange;
}

void IkSolver::_movej(double q0, double q1, double q2, double q3, double q4, double q5, double dt)
{
    vec<6> tar_q = {q0, q1, q2, q3, q4, q5};

    // 조인트 속도(rad/s) 계산
    if (dt > 1e-6) {
        joint_vel = (tar_q - cur_q) / dt;
    } else {
        joint_vel.setZero();
    }

    // FK 및 Jacobian 업데이트
    fk_and_jacobian(tar_q);
    cur_q = tar_q;

    // TCP 속도(m/s, rad/s) 계산
    tcp_vel = J * joint_vel;

    // 키네마틱스 업데이트
    ki[0].set_joint_value(cur_q[0]);
    ki[1].set_joint_value(cur_q[1]);
    ki[2].set_joint_value(cur_q[2]);
    ki[3].set_joint_value(cur_q[3]);
    ki[4].set_joint_value(cur_q[4]);
    ki[5].set_joint_value(cur_q[5]);

    // fk 캐시 업데이트
    fk_cache[0] = ki[0].get_local_tf();
    fk_cache[1] = fk_cache[0] * ki[1].get_local_tf();
    fk_cache[2] = fk_cache[1] * ki[2].get_local_tf();
    fk_cache[3] = fk_cache[2] * ki[3].get_local_tf();
    fk_cache[4] = fk_cache[3] * ki[4].get_local_tf();
    fk_cache[5] = fk_cache[4] * ki[5].get_local_tf();
}

void IkSolver::_movel(const Transform &target, double dt)
{
    const vec<6> q = ik(target, dt);
    _movej(q[0], q[1], q[2], q[3], q[4], q[5], dt);
}

double IkSolver::safe_dt()
{
    const auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = now - lastTime_;
    lastTime_ = now;

    double delta = diff.count();

    if (delta > 0.1) {
        return 0.001;
    }

    return delta;
}

vec<6> IkSolver::ik(const Transform& tar_tf, double dt)
{
    // dt가 너무 작으면 연산하지 않고 현재 위치 반환
    if (dt <= 1e-6) {
        IK_WARN << "dt가 너무 작아서 IK 연산 무시됨: dt:" << dt << endl;
        return cur_q;
    }


    // flange 위치 계산
    Transform tar_flange = tar_tf * end_effector_inverse; // flange = target * inv(end effector)
    Transform cur_flange = fk(cur_q);

    // TCP 속도 제한 적용
    tar_flange = limit_tcp_speed(cur_flange, tar_flange, dt); // 갈 수 있는 최대 거리가 제한됨


    // Inverse Kinematics
    vec<6> new_q = ik_core(tar_flange, dt);

    // Cartesian Time-Scaling 적용 (TCP 경로 철저히 유지)
    const vec<6> error_q = new_q - cur_q;
    double max_scale = 1.0;

    // 조인트 최대 속도를 넘는지 체크
    for (int i = 0; i < 6; ++i) {
        const double required_vel = std::abs(error_q[i] / dt);
        const double limit = ki[i].get_joint_limits().velocity / joint_velocity_scale;

        if (limit > 1e-6 && required_vel > limit) {
            const double current_scale = required_vel / limit;
            if (current_scale > max_scale) {
                max_scale = current_scale;
            }
        }
    }

    // 어느 한 조인트라도 속도 한계를 초과했다면
    if (max_scale > 1.0 + 1e-6) {
        const vec3 start_pos = cur_flange.translation();
        const vec3 target_pos = tar_flange.translation();
        const quat start_rot = cur_flange.quaternion();
        const quat target_rot = tar_flange.quaternion();

        // max_scale 만큼 속도를 늦춰야 하므로 이동 비율을 1 / max_scale 로 줄임
        const double ratio = 1.0 / max_scale;

        const vec3 scaled_pos = start_pos + (target_pos - start_pos) * ratio;
        const quat scaled_rot = start_rot.slerp(ratio, target_rot);

        Transform scaled_target_tf(scaled_rot, scaled_pos);

        // 축소된 타겟으로 IK 재수행 (경로 이탈 방지)
        new_q = ik_core(scaled_target_tf, dt);
    }

    // 각도 제한 값이 초과 될 경우 모든 조인트 감속됨
    return limit_joint_velocity(new_q, dt);
}

vec<6> IkSolver::ik_core(const Transform& tar_flange, double dt)
{
    vec<6> q = cur_q;

    int i = MAX_IK_ITER;
    while (i --> 0) {
        Transform cur_flange = fk_and_jacobian(q);

        Transform curr_tcp_tf = cur_flange * end_effector;
        vec<6> dx = Transform::twist(cur_flange, tar_flange);

        const double p_error = dx.head<3>().norm();
        const double r_error = dx.tail<3>().norm();

        if (p_error < POS_TOL && r_error < ROT_TOL)
            break;

        const vec3 curr_tcp_pos = curr_tcp_tf.translation();
        dx = limit_workspace(curr_tcp_pos, dx);

        const QPProblem qp = setup_QP(J, dx, q, dt);
        const vec<6> dq = solve_QP(qp);

        q += dq * STEP_SIZE;
    }

    return q;
}

IkSolver::QPProblem IkSolver::setup_QP(const mat<6, 6>& J, const vec<6>& dx, const vec<6>& cur_q, double dt) const
{
    QPProblem qp;
    const mat<6, 6> J_transpose = J.transpose();

    // Hessian Matrix 계산: H = JᵀJ + λI
    qp.H = J_transpose * J + LAMBDA * mat<6, 6>::Identity();

    // Gradient Vector 계산: g = -Jᵀdx
    qp.g = -J_transpose * dx;

    // Joint Limits (Box Constraints) 설정: lb ≤ dq ≤ ub
    for (int i = 0; i < 6; ++i) {
        // 위치 기반의 최대 이동 허용량
        double max_dq_pos = ki[i].get_joint_limits().upper - cur_q[i];
        double min_dq_pos = ki[i].get_joint_limits().lower - cur_q[i];

        // 속도 기반의 최대 이동 허용량 (dt 동안 이동 가능한 거리)
        double max_vel = ki[i].get_joint_limits().velocity / joint_velocity_scale;
        double max_dq_vel = max_vel * dt;
        double min_dq_vel = -max_dq_vel;

        // 위치 한계와 속도 한계 중 더 빡빡한(안전한) 값을 선택
        qp.ub(i) = std::min(max_dq_pos, max_dq_vel);
        qp.lb(i) = std::max(min_dq_pos, min_dq_vel);
    }

    return qp;
}

vec<6> IkSolver::solve_QP(const QPProblem& qp) const
{
    vec<6> x = vec<6>::Zero();

    const mat<6, 6> H = qp.H;
    const vec<6> g = qp.g;
    const vec<6> lb = qp.lb;
    const vec<6> ub = qp.ub;

    // QP 반복 최적화
    int iter = MAX_QP_ITER;
    while (iter --> 0) {
        // 각 변수에 대해 순차적으로 업데이트
        for (int i = 0; i < 6; ++i) {
            // Step 1: 다른 변수들의 영향(σᵢ) 계산
            const double sigma = H.row(i).dot(x) - H(i, i) * x(i);

            // Step 2: 무제약 최적해 계산
            const double x_unconstrained = (-g(i) - sigma) / H(i, i);

            // Step 3: Box 제약 조건 적용 (투영)
            const double x_constrained = std::clamp(x_unconstrained, lb(i), ub(i));

            // 업데이트
            x(i) = x_constrained;
        }
    }

    return x;
}

vec<6> IkSolver::limit_joint_velocity(const vec<6>& tar_q, double dt) const
{
    // dt가 너무 작으면 계산 불가하므로 현재 각도 유지 혹은 목표 반환
    if (dt <= 1e-6)
        return cur_q;

    const vec<6> error_q = tar_q - cur_q;
    double scale = 1.0;

    // 모든 관절 중 속도 제한을 가장 많이 초과한 비율(scale)을 찾음
    for (int i = 0; i < 6; ++i) {
        const double required_vel = std::abs(error_q[i] / dt);
        const double limit = ki[i].get_joint_limits().velocity / joint_velocity_scale;

        // 제한값이 유효하고, 필요 속도가 제한을 초과한 경우
        if (limit > 1e-6 && required_vel > limit) {
            const double current_scale = required_vel / limit;
            if (current_scale > scale) {
                scale = current_scale;
            }
        }
    }

    // 가장 많이 초과한 비율만큼 모든 관절의 증분(error_q)을 감쇄시켜 적용
    if (scale > 1.0) {
        return cur_q + (error_q / scale);
    }

    return tar_q;
}

void IkSolver::_movej_sync(double q0, double q1, double q2, double q3, double q4, double q5, double dt)
{
    vec<6> tar_q = { q0, q1, q2, q3, q4, q5 };

    vec<6> diff_q = tar_q - cur_q;
    double max_scale = 1.0;

    // 1. 모든 관절 중 속도 제한을 가장 많이 초과한 '비율'을 찾음 (동기화의 기준)
    for (int i = 0; i < 6; ++i) {
        double limit_vel = ki[i].get_joint_limits().velocity / joint_velocity_scale;
        if (limit_vel > 1e-6) {
            double required_vel = std::abs(diff_q[i]) / dt;
            if (required_vel > limit_vel) {
                double current_scale = required_vel / limit_vel;
                if (current_scale > max_scale) max_scale = current_scale;
            }
        }
    }

    // 2. 가장 느린 관절의 속도 비율(max_scale)에 맞춰 모든 관절의 이동량을 동일하게 스케일링
    // 이렇게 하면 모든 관절이 동시에 시작해서 '최종 목적지'에 동시에 도착하게 됨
    vec<6> next_q = cur_q + (diff_q / max_scale);

    // 3. 실제 상태 업데이트 및 FK/Jacobian 갱신
    _movej(next_q[0], next_q[1], next_q[2], next_q[3], next_q[4], next_q[5], dt);
}

Transform IkSolver::limit_tcp_speed(const Transform& cur_tf, const Transform& tar_tf, double dt) const
{
    // 속도 제한이 설정되어 있지 않거나(0) dt가 유효하지 않으면 원래 목표 반환
    if (tcp_max_speed == 0 || dt <= 1e-6) {
        return tar_tf;
    }

    const vec3 start_pos = cur_tf.translation();
    const vec3 target_pos = tar_tf.translation();
    const vec3 diff = target_pos - start_pos;
    const double dist = diff.norm();

    // 이번 스텝(dt) 동안 갈 수 있는 최대 거리 계산
    const double max_dist = tcp_max_speed * dt;

    // 목표 거리가 최대 허용 거리보다 멀 경우에만 보간 수행
    if (dist > max_dist) {
        // 위치 보간: 방향은 유지하되 길이를 max_dist로 제한
        const vec3 clamped_pos = start_pos + diff * (max_dist / dist);

        // 회전 보간: Slerp를 사용하여 max_dist 비율만큼만 회전
        const quat start_rot(cur_tf.rotation());
        const quat target_rot(tar_tf.rotation());
        const quat clamped_rot = start_rot.slerp(max_dist / dist, target_rot);

        return Transform(clamped_rot, clamped_pos);
    }

    return tar_tf;
}

vec<6> IkSolver::limit_workspace(const vec3& cur_pos, const vec<6>& dx) const
{
    vec<6> limited_dx = dx;

    for (int i = 0; i < 3; ++i) { // 위치 x, y, z
        // --- 최소값(Min) 체크 및 보정 ---
        const double dist_min = cur_pos(i) - ws_min(i);
        if (dist_min < 0.0) {
            // 경계 침범: 강제 복원 속도 계산
            const double recovery_vel = -dist_min * WS_P_GAIN;
            if (limited_dx(i) < recovery_vel) limited_dx(i) = recovery_vel;
        }
        else if (dist_min < WS_MARGIN) {
            // 경고 마진: 밖으로 나가는 속도 감쇄
            if (limited_dx(i) < 0) {
                const double ratio = dist_min / ((WS_MARGIN > 1e-6) ? WS_MARGIN : 1.0);
                limited_dx(i) *= ratio;
            }
        }

        // --- 최대값(Max) 체크 및 보정 ---
        const double dist_max = ws_max(i) - cur_pos(i);
        if (dist_max < 0.0) {
            // 경계 침범: 강제 복원 속도 계산
            const double recovery_vel = dist_max * WS_P_GAIN;
            if (limited_dx(i) > recovery_vel) limited_dx(i) = recovery_vel;
        }
        else if (dist_max < WS_MARGIN) {
            // 경고 마진: 밖으로 나가는 속도 감쇄
            if (limited_dx(i) > 0) {
                const double ratio = dist_max / ((WS_MARGIN > 1e-6) ? WS_MARGIN : 1.0);
                limited_dx(i) *= ratio;
            }
        }
    }

    return limited_dx;
}