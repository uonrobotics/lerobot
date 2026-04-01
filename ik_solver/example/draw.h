#pragma once
#include "implot3d.h"
#include "transform.hpp"
#include "kinematics.h"

class Axis3D {
private:
    Transform tf_;
    float length_ = 1.0f;

    ImVec4 color_x = ImVec4(1, 0, 0, 1);
    ImVec4 color_y = ImVec4(0, 1, 0, 1);
    ImVec4 color_z = ImVec4(0, 0, 1, 1);

public:
    Axis3D(float length = 1.0f) : length_(length) {}

    // transform 설정
    void set_transform(const Transform& tf);
    vec3 get_transform() const;

    void set_colors(ImVec4 x, ImVec4 y, ImVec4 z) {
        color_x = x; color_y = y; color_z = z;
    }

    // axis 그리기
    void draw();

private:
    void draw_axis_line(const char* id, const Eigen::Vector3d& start, const Eigen::Vector3d& end, const ImVec4& color);
};


namespace Draw
{
    void draw_manipulability(const mat<3, 6>& J_position, const vec3& center, float scale, const ImVec4& color, bool is_rot);
    void draw_box(const vec3& min_pos, const vec3& max_pos);
    void draw_plane(float min_x, float max_x, float min_y, float max_y, float z = 0.0f);

    void draw_collision(const CollisionInfo& col, const Transform& joint_world_tf, const ImVec4& color);
    void draw_sphere(const vec3& center, float radius, const ImVec4& color);
    void draw_cylinder(const vec3& center, const mat3& rotation, float radius, float height, const ImVec4& color);
}
