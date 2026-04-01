#include "kinematics.h"
#include <string>
#include <iostream>
#include <filesystem>

#define INFO cout << "[Info ] [Kinematics] "
#define WARN cout << "[Warn ] [Kinematics] "
#define ERROR cout << "[Error] [Kinematics] "

using namespace std;
namespace fs = std::filesystem;


JointType urdfJointTypeToEnum(unsigned int urdf_type)
{
    switch (urdf_type) {
        case urdf::Joint::FIXED:      return JointType::FIXED;
        case urdf::Joint::REVOLUTE:   return JointType::REVOLUTE;
        case urdf::Joint::PRISMATIC:  return JointType::PRISMATIC;
        case urdf::Joint::CONTINUOUS: return JointType::CONTINUOUS;
        case urdf::Joint::PLANAR:     return JointType::PLANAR;
        case urdf::Joint::FLOATING:   return JointType::FLOATING;
        default: return JointType::UNKNOWN;
    }
}

std::string jointTypeToString(JointType type)
{
    switch (type) {
        case JointType::FIXED:      return "fixed";
        case JointType::REVOLUTE:   return "revolute";
        case JointType::PRISMATIC:  return "prismatic";
        case JointType::CONTINUOUS: return "continuous";
        case JointType::PLANAR:     return "planar";
        case JointType::FLOATING:   return "floating";
        default: return "unknown";
    }
}

void Kinematics::_update_tf() const
{
    if (!transform_dirty_)
        return;

    if (!has_joint_info_) {
        local_tf_ = origin_tf_;
        transform_dirty_ = false;
        return;
    }

    const Transform joint_transform = origin_tf_;


    // Fixed Joint
    if (joint_info_.type == JointType::FIXED) {
        local_tf_ = joint_transform;
    }

    // Revolute Joint
    else if (joint_info_.type == JointType::REVOLUTE || joint_info_.type == JointType::CONTINUOUS) {
        AngleAxis rotation(joint_info_.current_value, joint_info_.axis);
        Transform rotation_transform(rotation, vec3::Zero());
        local_tf_ = joint_transform * rotation_transform;
    }

    // Prismatic Joint
    else if (joint_info_.type == JointType::PRISMATIC) {
        vec3 translation = joint_info_.axis * joint_info_.current_value;
        Transform translation_transform(quat::Identity(), translation);
        local_tf_ = joint_transform * translation_transform;
    }

    // Other Joint Types
    else {
        local_tf_ = joint_transform;
    }

    transform_dirty_ = false;
}

UrdfPtr Kinematics::load_urdf_file(const std::string& urdf_file)
{
    if (urdf_file.empty()) {
        ERROR << "URDF 파일 경로가 비어있습니다." << endl;
        return nullptr;
    }

    fs::path full_path = fs::absolute(urdf_file);

    // 파일 체크
    if (!fs::exists(full_path)) {
        ERROR << "URDF 파일을 찾을 수 없습니다: " << full_path.string() << endl;
        return nullptr;
    }

    // urdf 파싱
    urdf::ModelInterfaceSharedPtr urdf = urdf::parseURDFFile(urdf_file);

    if (!urdf) {
        ERROR << "URDF 파싱 실패 (내부 형식 오류): " << full_path.string() << endl;
        return nullptr;
    }

    // 성공
    INFO << "URDF 파일 로드: " <<  fs::canonical(full_path).string() << endl;
    return urdf;
}

Kinematics Kinematics::build_from_urdf(const UrdfPtr& urdf, const std::string& joint_name)
{
    auto urdf_joint = urdf->getJoint(joint_name);

    if (!urdf_joint) {
        WARN << "URDF에 Joint 속성이 없음: " << joint_name << endl;
        return Kinematics();
    }

    Kinematics k;
    k.joint_info_.name = urdf_joint->name;
    k.joint_info_.parent_link = urdf_joint->parent_link_name;
    k.joint_info_.child_link = urdf_joint->child_link_name;

    // 조인트 타입 변환
    k.joint_info_.type = urdfJointTypeToEnum(urdf_joint->type);

    // 축 정보 설정 (정규화)
    if (k.joint_info_.type != JointType::FIXED)
    {
        k.joint_info_.axis = vec3(
            urdf_joint->axis.x,
            urdf_joint->axis.y,
            urdf_joint->axis.z
        ).normalized();
    }

    // 제한 정보 설정
    if (urdf_joint->limits)
    {
        k.joint_info_.limits.lower = urdf_joint->limits->lower;
        k.joint_info_.limits.upper = urdf_joint->limits->upper;
        k.joint_info_.limits.effort = urdf_joint->limits->effort;
        k.joint_info_.limits.velocity = urdf_joint->limits->velocity;
    }

    // 원점 변환 설정
    const urdf::Pose& origin = urdf_joint->parent_to_joint_origin_transform;
    vec3 pos(origin.position.x, origin.position.y, origin.position.z);
    quat rot(origin.rotation.w, origin.rotation.x, origin.rotation.y, origin.rotation.z);
    k.origin_tf_ = Transform(rot, pos);


    auto child_link = urdf->getLink(urdf_joint->child_link_name);
    if (child_link && child_link->visual && child_link->visual->geometry) {
        if (child_link->visual->geometry->type == urdf::Geometry::MESH) {
            auto mesh = std::static_pointer_cast<urdf::Mesh>(child_link->visual->geometry);
            k.joint_info_.mesh_path = mesh->filename;
            k.joint_info_.mesh_scale = vec3(mesh->scale.x, mesh->scale.y, mesh->scale.z);
        }
    }

    if (child_link) {
        // --- Collision 정보 파싱 시작 ---
        for (const auto& col : child_link->collision_array) {
            if (!col || !col->geometry) continue;

            CollisionInfo info;

            // 1. Origin 파싱
            const urdf::Pose& p = col->origin;
            quat q(p.rotation.w, p.rotation.x, p.rotation.y, p.rotation.z);
            vec3 t(p.position.x, p.position.y, p.position.z);
            info.origin = Transform(q, t);

            // 2. Geometry 파싱
            if (col->geometry->type == urdf::Geometry::BOX) {
                auto box = std::static_pointer_cast<urdf::Box>(col->geometry);
                info.type = CollisionInfo::GeometryType::BOX;
                info.size = vec3(box->dim.x, box->dim.y, box->dim.z);
            }
            else if (col->geometry->type == urdf::Geometry::CYLINDER) {
                auto cyl = std::static_pointer_cast<urdf::Cylinder>(col->geometry);
                info.type = CollisionInfo::GeometryType::CYLINDER;
                info.size = vec3(cyl->radius, cyl->length, 0.0);
            }
            else if (col->geometry->type == urdf::Geometry::SPHERE) {
                auto sph = std::static_pointer_cast<urdf::Sphere>(col->geometry);
                info.type = CollisionInfo::GeometryType::SPHERE;
                info.size = vec3(sph->radius, 0.0, 0.0);
            }
            else if (col->geometry->type == urdf::Geometry::MESH) {
                auto mesh = std::static_pointer_cast<urdf::Mesh>(col->geometry);
                info.type = CollisionInfo::GeometryType::MESH;
                info.mesh_path = mesh->filename;
                info.mesh_scale = vec3(mesh->scale.x, mesh->scale.y, mesh->scale.z);
            }

            k.joint_info_.collisions.push_back(info);
        }
    }

    k.has_joint_info_ = true;
    k.transform_dirty_ = true;

    k.print_info();

    return k;
}

Kinematics Kinematics::make(const std::string& name, JointType type, const vec3& axis,  const Transform& origin_tf, const JointLimits& limits)
{
    Kinematics k;

    k.joint_info_.name = name;
    k.joint_info_.type = type;

    if (type != JointType::FIXED) {
        k.joint_info_.axis = axis.normalized();
    } else {
        k.joint_info_.axis = axis; // Fixed joint axis might be irrelevant, but keeping it
    }

    k.joint_info_.limits = limits;
    k.origin_tf_ = origin_tf;

    k.has_joint_info_ = true;
    k.transform_dirty_ = true;

    k.print_info();

    return k;
}

void Kinematics::set_joint_info(const JointInfo& info)
{
    joint_info_ = info;
    has_joint_info_ = true;
    transform_dirty_ = true;
}

void Kinematics::set_joint_value(double value)
{
    joint_info_.current_value = value;
    transform_dirty_ = true;
}

void Kinematics::set_origin_tf(const Transform &origin)
{
    origin_tf_ = origin;
    transform_dirty_ = true;
}

void Kinematics::set_joint_limit(double lower, double upper)
{
    joint_info_.limits.lower = lower; // [rad]
    joint_info_.limits.upper = upper; // [rad]
}

void Kinematics::set_joint_vlimit(double velocity)
{
    joint_info_.limits.velocity = velocity; // [rad/s] or [m/s]
}

void Kinematics::print_header()
{
    cout << string(195, '-') << endl;
    cout << left << setw(15) << "Name"
         << setw(12) << "Type"
         << setw(18) << "Parent/Child"
         << setw(28) << "Position (x,y,z)"
         << setw(18) << "Axis (x,y,z)"
         << setw(22) << "Limits (Deg/m)"
         << setw(18) << "V-Limit(deg/s)"
         << setw(15) << "Value(deg/m)"
         << setw(20) << "Scale (x,y,z)"
         << "Mesh Path"
         << endl;
    cout << string(195, '-') << endl;
}

void Kinematics::print_info() const
{
    if (!has_joint_info_) {
        WARN <<"조인트 정보 없음\n";
        return;
    }

    // 1. 기본 정보 (Name, Type, Parent/Child)
    cout << left << setw(15) << (joint_info_.name.empty() ? "None" : joint_info_.name);
    cout << setw(12) << jointTypeToString(joint_info_.type);

    string pc = joint_info_.parent_link + ">" + joint_info_.child_link;
    if (pc.length() > 17) pc = pc.substr(0, 14) + "...";
    cout << setw(18) << pc;

    // 2. Position (x,y,z) - setw(28)에 맞춤
    const vec3& pos = get_local_tf().translation();
    char pos_buf[30];
    snprintf(pos_buf, sizeof(pos_buf), "%.3f, %.3f, %.3f", pos.x(), pos.y(), pos.z());
    cout << setw(28) << pos_buf;

    // 3. Axis (x,y,z) - setw(18)에 맞춤
    if (joint_info_.type != JointType::FIXED) {
        char axis_buf[20];
        snprintf(axis_buf, sizeof(axis_buf), "%.2f,%.2f,%.2f",
                 joint_info_.axis(0), joint_info_.axis(1), joint_info_.axis(2));
        cout << setw(18) << axis_buf;
    } else {
        cout << setw(18) << "N/A (Fixed)";
    }

    // 4. Limits - setw(22)에 맞춤
    char limit_buf[30];
    if (joint_info_.type == JointType::REVOLUTE || joint_info_.type == JointType::CONTINUOUS) {
        snprintf(limit_buf, sizeof(limit_buf), "[%.1f, %.1f]",
                 RAD_TO_DEG(joint_info_.limits.lower),
                 RAD_TO_DEG(joint_info_.limits.upper));
    } else {
        snprintf(limit_buf, sizeof(limit_buf), "[%.2f, %.2f]",
                 joint_info_.limits.lower,
                 joint_info_.limits.upper);
    }
    cout << setw(22) << limit_buf;

    // 5. Velocity Limit - setw(18)에 맞춤
    double v_limit = joint_info_.limits.velocity;
    if (joint_info_.type == JointType::REVOLUTE || joint_info_.type == JointType::CONTINUOUS) {
        v_limit = RAD_TO_DEG(v_limit);
    }
    cout << fixed << setprecision(2) << setw(18) << v_limit;

    // 6. Current Value - setw(15)에 맞춤
    double q_val = joint_info_.current_value;
    if (joint_info_.type == JointType::REVOLUTE || joint_info_.type == JointType::CONTINUOUS) {
        q_val = RAD_TO_DEG(q_val);
    }
    cout << fixed << setprecision(4) << setw(15) << q_val;

    // 7. Mesh Scale - setw(20)에 맞춤
    char scale_buf[25];
    snprintf(scale_buf, sizeof(scale_buf), "%.2f,%.2f,%.2f",
             joint_info_.mesh_scale.x(), joint_info_.mesh_scale.y(), joint_info_.mesh_scale.z());
    cout << setw(20) << scale_buf;

    // 8. Mesh Path
    if (joint_info_.mesh_path.empty() || joint_info_.mesh_path == "None") {
        cout << "N/A" << endl;
    } else {
        cout << joint_info_.mesh_path << endl;
    }
}


