#pragma once
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/src/AutoDiff/AutoDiffScalar.h>

// ==================================================================
// 매크로 및 상수 정의
// ==================================================================
#ifndef DEG_TO_RAD
#define DEG_TO_RAD(x) ((x) * 0.017453292519943295769236907684886)
#endif
#ifndef RAD_TO_DEG
#define RAD_TO_DEG(x) ((x) * 57.29577951308232087679815481410518)
#endif

// ==================================================================
// Transform 클래스: 순수 기하학적 변환만 담당
// ==================================================================

template<typename Scalar>
class TransformT {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Eigen 내부 타입 정의
    using Isometry_t = Eigen::Transform<Scalar, 3, Eigen::Isometry>;
    using Matrix3_t = Eigen::Matrix<Scalar, 3, 3>;
    using Matrix4_t = Eigen::Matrix<Scalar, 4, 4>;
    using Vector3_t = Eigen::Matrix<Scalar, 3, 1>;
    using Vector6_t = Eigen::Matrix<Scalar, 6, 1>;
    using AngleAxis_t = Eigen::AngleAxis<Scalar>;
    using Quat_t = Eigen::Quaternion<Scalar>;

private:
    Isometry_t T_;

public:
    TransformT() : T_(Isometry_t::Identity())
    {
    }

    explicit TransformT(const Isometry_t& iso) : T_(iso)
    {
    }

    explicit TransformT(const Matrix4_t& H) : T_(H)
    {
    }

    TransformT(const Quat_t& q, const Vector3_t& t)
    {
        T_.linear() = q.toRotationMatrix();
        T_.translation() = t;
    }

    TransformT(const AngleAxis_t& r, const Vector3_t& t)
    {
        T_.linear() = r.toRotationMatrix();
        T_.translation() = t;
    }

    // operator
    TransformT& operator=(const Matrix4_t& m)
    {
        this->T_ = m;
        return *this;
    }

    TransformT operator*(const TransformT& other) const
    {
        return TransformT(this->T_ * other.T_);
    }

    friend std::ostream& operator<<(std::ostream& os, const TransformT& rt)
    {
        os << rt.matrix();
        return os;
    }

    // Setter
    void translate(const Vector3_t& t)   { T_.translation() = t;              }
    void rotate(const AngleAxis_t& r)    { T_.linear() = r.toRotationMatrix();}

    // Getter
    const Matrix4_t& matrix()      const { return T_.matrix();                }
    const Matrix3_t  rotation()    const { return T_.rotation();              }
    const Vector3_t  translation() const { return T_.translation();           }
    const Quat_t     quaternion()  const { return Quat_t(T_.linear());        }
    TransformT       inverse()     const { return TransformT(T_.inverse());   }

    const Vector3_t rpy() const
    {
        Matrix3_t R = T_.rotation();
        Vector3_t rpy;

        // X-Y-Z Euler angles (Roll, Pitch, Yaw)
        // R = Rx(roll) * Ry(pitch) * Rz(yaw)
        // R = [ cy*cp          -sy*cp           sp    ]
        //     [ sy*cr+cy*sp*sr  cy*cr-sy*sp*sr -cp*sr ]
        //     [ sy*sr-cy*sp*cr  cy*sr+sy*sp*cr  cp*cr ]

        rpy[1] = std::asin(std::clamp(R(0, 2), -1.0, 1.0)); // Pitch (theta)

        if (std::abs(R(0, 2)) < 0.99999) {
            // 일반적인 경우
            rpy[0] = std::atan2(-R(1, 2), R(2, 2)); // Roll (phi)
            rpy[2] = std::atan2(-R(0, 1), R(0, 0)); // Yaw (psi)
        } else {
            // 짐벌 락 상황
            rpy[0] = std::atan2(R(2, 1), R(1, 1));
            rpy[2] = 0;
        }
        return rpy;
    }


    // Twist (Error) 계산
    static Vector6_t twist(const TransformT& tf_cur, const TransformT& tf_tar)
    {
        Vector6_t error_twist;

        // 위치 오차: Δp = p_tar - p_cur
        error_twist.template head<3>() = tf_tar.translation() - tf_cur.translation();

        // 회전 오차: 현재 좌표계 기준 상대 회전 (Local Frame)
        // ΔR = R_cur^T * R_tar
        Matrix3_t R_rel = tf_cur.rotation().transpose() * tf_tar.rotation();

        // Angle-Axis 표현: ΔR → ω = θ * axis
        AngleAxis_t aa(R_rel);
        Vector3_t w_local = aa.axis() * aa.angle();

        // Local Frame 오차를 Global Frame으로 변환: ω_global = R_cur * ω_local
        error_twist.template tail<3>() = tf_cur.rotation() * w_local;

        return error_twist;
    }

    // Transform 생성 함수
    static TransformT make_tf(double x, double y, double z, double roll, double pitch, double yaw)
    {
        // 로봇 베이스 기준 회전 (Extrinsic ZYX, i.e., Base Z -> Base Y -> Base X)
        // 수학적으로는 R = Rx(roll) * Ry(pitch) * Rz(yaw) 순서로 곱해짐
        const Quat_t t_rot = Quat_t(Eigen::AngleAxisd(roll, Vector3_t::UnitX()) *
                                    Eigen::AngleAxisd(pitch, Vector3_t::UnitY()) *
                                    Eigen::AngleAxisd(yaw, Vector3_t::UnitZ()));

        const Vector3_t t_pos(x, y, z);
        return TransformT<double>(t_rot, t_pos);
    }
};

// ==================================================================
// 편의성 타입 정의
// ==================================================================

template<int Rows, int Cols>
using mat = Eigen::Matrix<double, Rows, Cols>;
template<int Rows>
using vec = Eigen::Matrix<double, Rows, 1>;

using mat4 = Eigen::Matrix<double, 4, 4>;
using mat3 = Eigen::Matrix<double, 3, 3>;
using vec3 = Eigen::Matrix<double, 3, 1>;
using vec6 = Eigen::Matrix<double, 6, 1>;
using quat = Eigen::Quaterniond;
using AngleAxis = Eigen::AngleAxisd;

using Transform = TransformT<double>;
using ADScalar = Eigen::AutoDiffScalar<Eigen::Matrix<double, 6, 1> >;
using ADTransform = TransformT<ADScalar>;

template<int Rows>
using ADVec = Eigen::Matrix<ADScalar, Rows, 1>;
template<int Rows, int Cols>
using ADMat = Eigen::Matrix<ADScalar, Rows, Cols>;
