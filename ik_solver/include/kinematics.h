#pragma once
#include "transform.hpp"
#include <urdf_parser/urdf_parser.h>

using UrdfPtr = urdf::ModelInterfaceSharedPtr;

struct CollisionInfo {
    enum class GeometryType { NONE, BOX, CYLINDER, SPHERE, MESH };

    GeometryType type = GeometryType::NONE;
    Transform origin = Transform();

    // 이 부분을 mutable로 변경합니다.
    mutable Transform world_origin = Transform();

    vec3 size = vec3::Zero();
    std::string mesh_path = "";
    vec3 mesh_scale = vec3::Ones();
};

// ==================================================================
// // URDF Joint Type
// ==================================================================
enum class JointType {
    FIXED      = 0,               // 움직임이 없는 고정 조인트
    REVOLUTE   = 1,               // 한 축을 중심으로 회전하는 조인트 (각도 제한 있음)
    PRISMATIC  = 2,               // 한 축을 따라 직선 운동하는 조인트
    CONTINUOUS = 3,               // Revolute와 같으나 각도 제한이 없는 회전 조인트
    PLANAR     = 4,               // 평면상의 XY 이동과 회전이 가능한 조인트
    FLOATING   = 5,               // 6자유도(XYZ 이동 + 회전) 모두 가능한 조인트
    UNKNOWN    = 6                // 알 수 없는 타입의 조인트
};


// ==================================================================
// Joint Limits
// ==================================================================
struct JointLimits {
    double lower    = 0.0;        //  하한값 [rad 또는 m]
    double upper    = 0.0;        //  상한값 [rad 또는 m]
    double effort   = 0.0;        //  최대 토크 또는 힘 [Nm 또는 N]
    double velocity = 0.0;        //  최대 속도 [rad/s 또는 m/s]

    JointLimits() = default;
    JointLimits(double l, double u, double e, double v)
        : lower(l), upper(u), effort(e), velocity(v) {}
};

    
// ==================================================================
// Joint Info
// ==================================================================
struct JointInfo {
    std::string name = "";        // 조인트 이름
    std::string parent_link = ""; // 부모 링크 이름
    std::string child_link = "";  // 자식 링크 이름
    JointType type;               // 조인트 타입
    vec3 axis = vec3::Zero();     // 회전/이동 축 (정규화됨)
    JointLimits limits;           // 조인트 제한 정보
    double current_value = 0.0;   // 현재 조인트 값 [rad 또는 m]

    std::string mesh_path = "";   // 시각화용 메시 파일 경로 (STL, OBJ, DAE 등)
    vec3 mesh_scale = vec3::Ones(); // 메시 스케일 (URDF의 <scale> 대응)

    std::vector<CollisionInfo> collisions;

    JointInfo() : type(JointType::UNKNOWN) {}
};


// ==================================================================
// Kinematics
//  - URDF에서 파싱한 Joint 정보를 관리
// ==================================================================
class Kinematics {
private:
    JointInfo joint_info_;                                                             // 조인트 정보
    Transform origin_tf_;                                                              // 부모 링크에서 조인트 원점까지의 변환 (offset)
    mutable Transform local_tf_;                                                       // 현재 조인트 값을 반영한 변환 (local)
    bool has_joint_info_ = false;                                                      // 조인트 정보 설정 여부
    mutable bool transform_dirty_ = true;                                              // Transform 재계산 필요 여부

    
public:
    Kinematics() : origin_tf_(Transform()), local_tf_(Transform()) {}

    // ------------------------------------------------------------------
    // URDF loader and builder
    // ------------------------------------------------------------------
    static UrdfPtr    load_urdf_file(const std::string& urdf_file);                    // URDF 파일 로드
    static Kinematics build_from_urdf(const UrdfPtr& urdf, const std::string& name);   // urdf 정보 파싱

    // ------------------------------------------------------------------
    // Factory method
    // ------------------------------------------------------------------
    static Kinematics make(const std::string& name, JointType type, const vec3& axis,  const Transform& origin_tf, const JointLimits& limits);

    // ------------------------------------------------------------------
    // Setter
    // ------------------------------------------------------------------
    void set_joint_info(const JointInfo& info);                                        // 조인트 정보 설정
    void set_joint_value(double value);                                                // 현재 조인트 값 설정 [rad]. 설정된 Kinematics의 joint를 움직임
    void set_joint_limit(double lower, double upper);                                  // 조인트 제한 설정 [rad]
    void set_joint_vlimit(double velocity);                                            // 조인트 최대 속도 제한 설정 [rad/s]
    void set_origin_tf(const Transform& origin);                                       // 원점 변환 설정 (부모 링크에서 조인트 원점까지)

    // ------------------------------------------------------------------
    // Getter
    // ------------------------------------------------------------------
    const JointInfo&   get_joint_info()   const { return joint_info_;                } // 조인트의 Kinematics 정보
    const JointLimits& get_joint_limits() const { return joint_info_.limits;         } // 조인트 제한 정보 [rad]
    const double       get_joint_value()  const { return joint_info_.current_value;  } // 현재 조인트 값 [rad]
    const Transform&   get_origin_tf()    const { return origin_tf_;                 } // 원점 Transform
    const Transform&   get_local_tf()     const { _update_tf(); return local_tf_;    } // 현재 Transform

    // ------------------------------------------------------------------
    // 유틸리티
    // ------------------------------------------------------------------
    static void print_header();                                                         // 정보 헤더 출력
    void print_info() const;                                                            // Kinematics 객체의 조인트 정보 출력

    // ------------------------------------------------------------------
    // Forward Kinematics Kernel
    // ------------------------------------------------------------------
    template<typename T, typename KinematicsContainer, typename ValueContainer>
    static TransformT<T> fk_kernel(const KinematicsContainer& chain, const ValueContainer& joint_values)
    {
        // 시작은 Identity Matrix (Global Base)
        TransformT<T> t_accumulated;

        auto kit = std::begin(chain);
        auto vit = std::begin(joint_values);
        auto kit_end = std::end(chain);
        auto vit_end = std::end(joint_values);

        for (; kit != kit_end && vit != vit_end; ++kit, ++vit) {
            // --- URDF의 고정 정보 (double) 가져오기 ---
            const Transform& origin_dbl = kit->get_origin_tf(); // 부모->현재 조인트 원점
            const JointInfo& info = kit->get_joint_info();
            const vec3 axis_dbl = info.axis;

            // --- double -> T (AutoDiff) 캐스팅 ---
            // Origin 변환 행렬 캐스팅
            typename TransformT<T>::Matrix4_t mat_T = origin_dbl.matrix().cast<T>();
            TransformT<T> tf_origin(mat_T);

            // 회전축 캐스팅
            typename TransformT<T>::Vector3_t axis_T = axis_dbl.cast<T>();

            // --- 조인트 움직임 계산 (Joint Transform) ---
            TransformT<T> tf_joint; // Identity
            T q_val = *vit;

            if (info.type == JointType::REVOLUTE || info.type == JointType::CONTINUOUS) {
                // 회전 관절: 축(axis_T)을 중심으로 q_val 만큼 회전
                Eigen::AngleAxis<T> rot(q_val, axis_T);
                typename TransformT<T>::Vector3_t zero_vec = TransformT<T>::Vector3_t::Zero();
                tf_joint = TransformT<T>(rot, zero_vec);
            } else if (info.type == JointType::PRISMATIC) {
                // 직선 관절: 축 방향으로 q_val 만큼 이동
                typename TransformT<T>::Vector3_t trans = axis_T * q_val;
                tf_joint = TransformT<T>(Eigen::Quaternion<T>::Identity(), trans);
            }
            // FIXED인 경우 tf_joint는 Identity 그대로 유지

            // --- 변환 누적: Current = Prev * Origin * JointMotion ---
            t_accumulated = t_accumulated * tf_origin * tf_joint;
        }

        return t_accumulated;
    }


private:
    void _update_tf() const; // 현재 조인트 값을 기반으로 Transform 재계산

};
