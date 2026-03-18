#include "draw.h"

void Axis3D::set_transform(const Transform& tf)
{
    tf_ = tf;
}

vec3 Axis3D::get_transform() const
{
    return tf_.translation();
}

void Axis3D::draw()
{
    vec3 origin = tf_.translation();
    mat3 rotation = tf_.rotation();

    vec3 x_end = origin + rotation.col(0) * length_;
    vec3 y_end = origin + rotation.col(1) * length_;
    vec3 z_end = origin + rotation.col(2) * length_;

    // --- 선 그리기 ------------------------------------
    ImPlot3D::PushStyleVar(ImPlot3DStyleVar_LineWeight, 5.0f); // 선 굵기

    draw_axis_line("##X", origin, x_end, color_x);   // x
    draw_axis_line("##Y", origin, y_end, color_y);   // y
    draw_axis_line("##Z", origin, z_end, color_z);   // z

    ImPlot3D::PopStyleVar(); // LineWeight 복구

    // --- 중심점 그리기 ----------------------------------
    ImPlot3D::PushStyleVar(ImPlot3DStyleVar_MarkerSize, 5.0f); // 점 크기

    float ox = (float)origin.x(), oy = (float)origin.y(), oz = (float)origin.z(); // 점 위치

    ImPlot3D::PushStyleColor(ImPlot3DCol_MarkerFill, ImVec4(1, 1, 1, 1));    // 흰색 채우기
    ImPlot3D::PushStyleColor(ImPlot3DCol_MarkerOutline, ImVec4(0, 0, 0, 1)); // 검정 테두리

    ImPlot3D::PlotScatter("##Origin", &ox, &oy, &oz, 1);

    ImPlot3D::PopStyleColor(2);
    ImPlot3D::PopStyleVar();
}

void Axis3D::draw_axis_line(const char* id, const Eigen::Vector3d& start, const Eigen::Vector3d& end, const ImVec4& color)
{
    float xs[2] = { (float)start.x(), (float)end.x() };
    float ys[2] = { (float)start.y(), (float)end.y() };
    float zs[2] = { (float)start.z(), (float)end.z() };

    ImPlot3D::PushStyleColor(ImPlot3DCol_Line, color);
    ImPlot3D::PlotLine(id, xs, ys, zs, 2);
    ImPlot3D::PopStyleColor();
}

void Draw::draw_manipulability(const mat<3, 6>& J_position, const vec3& center, float scale, const ImVec4& color, bool is_rot)
{
    // 로봇의 위치 자코비안 (J_position: 3x6 행렬)을 사용하여
    // 조작성 행렬 M (3x3)을 계산
    mat3 M = J_position * J_position.transpose();


    // 조작성 행렬 M에 대해 고유값 분해
    Eigen::SelfAdjointEigenSolver<mat3> eigensolver(M);

    if (eigensolver.info() != Eigen::Success)
        return;

    vec3 eigenvalues = eigensolver.eigenvalues();   // 고유값: 타원체 축 길이의 제곱
    mat3 eigenvectors = eigensolver.eigenvectors(); // 고유 벡터: 타원체의 주축 방향
    vec3 radii = eigenvalues.cwiseSqrt() * scale;   // 타원체의 각 주축 길이(반지름)를 계산

    if (is_rot) {
        float max_val = eigenvalues.maxCoeff();

        for (int i = 0; i < 3; ++i) {
            float norm = eigenvalues[i] / (max_val + 1e-7f);
            radii[i] = std::pow(norm, 2.0f) * scale;
        }
    }


    const int lat_segments = 16; // 위도 분할 개수
    const int lon_segments = 16; // 경도 분할 개수
    std::vector<float> sphere_x, sphere_y, sphere_z;


    // 위도(theta)와 경도(phi)에 따라 표면 점들을 반복하며 계산
    for (int i = 0; i <= lat_segments; ++i) {
        float theta = M_PI * i / lat_segments;
        float sin_theta = std::sin(theta);
        float cos_theta = std::cos(theta);

        for (int j = 0; j <= lon_segments; ++j) {
            float phi = 2.0f * M_PI * j / lon_segments;
            float sin_phi = std::sin(phi);
            float cos_phi = std::cos(phi);

            vec3 unit_sphere(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
            vec3 ellipsoid_point = eigenvectors * (radii.asDiagonal() * unit_sphere);
            ellipsoid_point += center;

            sphere_x.push_back(ellipsoid_point.x());
            sphere_y.push_back(ellipsoid_point.y());
            sphere_z.push_back(ellipsoid_point.z());
        }
    }

    ImPlot3D::PushStyleColor(ImPlot3DCol_Line, color);


    // Latitudinal lines
    for (int i = 0; i <= lat_segments; ++i) {
        std::vector<float> line_x, line_y, line_z;
        for (int j = 0; j <= lon_segments; ++j) {
            int idx = i * (lon_segments + 1) + j;
            line_x.push_back(sphere_x[idx]);
            line_y.push_back(sphere_y[idx]);
            line_z.push_back(sphere_z[idx]);
        }
        ImPlot3D::PlotLine("Manipulability", line_x.data(), line_y.data(), line_z.data(), line_x.size());
    }



    // Longitudinal lines
    for (int j = 0; j <= lon_segments; ++j) {
        std::vector<float> line_x, line_y, line_z;
        for (int i = 0; i <= lat_segments; ++i) {
            int idx = i * (lon_segments + 1) + j;
            line_x.push_back(sphere_x[idx]);
            line_y.push_back(sphere_y[idx]);
            line_z.push_back(sphere_z[idx]);
        }
        ImPlot3D::PlotLine("Manipulability", line_x.data(), line_y.data(), line_z.data(), line_x.size());
    }


    ImPlot3D::PopStyleColor();
}

void Draw::draw_box(const vec3& min_pos, const vec3& max_pos)
{
    ImPlot3D::SetNextLineStyle(ImVec4(1, 0, 0, 0.6f), 2.0f);

    float x1 = min_pos.x(), x2 = max_pos.x();
    float y1 = min_pos.y(), y2 = max_pos.y();
    float z1 = min_pos.z(), z2 = max_pos.z();

    // 바닥 (Z1)
    float bx[] = {x1, x2, x2, x1, x1};
    float by[] = {y1, y1, y2, y2, y1};
    float bz[] = {z1, z1, z1, z1, z1};
    ImPlot3D::PlotLine("BoxBot", bx, by, bz, 5);

    // 천장 (Z2)
    float tx[] = {x1, x2, x2, x1, x1};
    float ty[] = {y1, y1, y2, y2, y1};
    float tz[] = {z2, z2, z2, z2, z2};
    ImPlot3D::PlotLine("BoxTop", tx, ty, tz, 5);

    // 기둥 4개
    float p1x[] = {x1, x1}, p1y[] = {y1, y1}, p1z[] = {z1, z2};
    float p2x[] = {x2, x2}, p2y[] = {y1, y1}, p2z[] = {z1, z2};
    float p3x[] = {x2, x2}, p3y[] = {y2, y2}, p3z[] = {z1, z2};
    float p4x[] = {x1, x1}, p4y[] = {y2, y2}, p4z[] = {z1, z2};

    ImPlot3D::PlotLine("P1", p1x, p1y, p1z, 2);
    ImPlot3D::PlotLine("P2", p2x, p2y, p2z, 2);
    ImPlot3D::PlotLine("P3", p3x, p3y, p3z, 2);
    ImPlot3D::PlotLine("P4", p4x, p4y, p4z, 2);
}

void Draw::draw_plane(float min_x, float max_x, float min_y, float max_y, float z)
{
    ImPlot3DPoint vertices[4] = {
        ImPlot3DPoint(min_x, min_y, z), // 0: 좌하단 (Min, Min)
        ImPlot3DPoint(max_x, min_y, z), // 1: 우하단 (Max, Min)
        ImPlot3DPoint(max_x, max_y, z), // 2: 우상단 (Max, Max)
        ImPlot3DPoint(min_x, max_y, z)  // 3: 좌상단 (Min, Max)
    };

    // 인덱스 배열
    static const unsigned int indices[] = {
        0, 1, 2,
        0, 2, 3
    };

    // 스타일 설정 (색상 및 투명도)
    ImPlot3D::PushStyleColor(ImPlot3DCol_Fill, ImVec4(0.3f, 0.7f, 1.0f, 0.2f));

    ImPlot3D::PlotMesh("CustomPlane", vertices, indices, 4, 6, 0);

    ImPlot3D::PopStyleColor();
}
