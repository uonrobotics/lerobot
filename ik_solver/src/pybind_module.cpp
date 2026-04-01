#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // NumPy 지원 추가
#include <vector>
#include <stdexcept>
#include <ik_solver.h>
#include <transform.hpp>

namespace py = pybind11;

PYBIND11_MODULE(ik_solver_py, m) {
    // ------------------------------------------------------------------
    // Transform 클래스 바인딩 (이미 Eigen을 사용하여 NumPy와 호환됨)
    // ------------------------------------------------------------------
    py::class_<Transform>(m, "Transform")
        .def(py::init<>())
        .def(py::init([](const Eigen::Ref<const Eigen::Matrix4d>& mat) {
            return new Transform(mat);
        }))
        .def("matrix", &Transform::matrix)
        .def("inverse", &Transform::inverse)
        .def("translation", &Transform::translation)
        .def("rotation", &Transform::rotation)
        .def("rpy", [](const Transform& self) {
            vec3 rpy = self.rpy();
            return py::array_t<double>({3}, {sizeof(double)}, rpy.data());
        })
        .def_static("make_tf", &Transform::make_tf);

    // ------------------------------------------------------------------
    // IkSolver 클래스 바인딩
    // ------------------------------------------------------------------
    py::class_<IkSolver> ik_solver(m, "IkSolver");

    ik_solver
        .def(py::init<const std::string &>())
        .def("init", &IkSolver::init)

        // ==================================================================
        // Setter / Control (NumPy Array -> Eigen::Matrix)
        // ==================================================================
        .def("set_joint", [](IkSolver& self, py::array_t<double> q) {
            auto r = q.unchecked<1>();
            if (r.size() != 6) throw std::invalid_argument("Joint array must have 6 elements.");
            vec<6> eigen_q;
            for(int i=0; i<6; ++i) eigen_q(i) = r(i);
            self.set_joint(eigen_q);
        }, "각 Joint를 즉시 설정합니다. [rad]", py::arg("joint_values"))

        .def("movej", [](IkSolver& self, py::array_t<double> q) {
            auto r = q.unchecked<1>();
            if (r.size() != 6) throw std::invalid_argument("Joint array must have 6 elements.");
            vec<6> eigen_q;
            for(int i=0; i<6; ++i) eigen_q(i) = r(i);
            self.movej(eigen_q);
        }, "목표 각도로 이동합니다. [rad]", py::arg("q"))

        .def("movel", &IkSolver::movel, "TCP 이동 (Transform 사용)", py::arg("tf"))

        // ==================================================================
        // Getter (Eigen::Matrix -> NumPy Array)
        // ==================================================================
        .def("get_curr_joint_deg", [](const IkSolver& self) {
            vec<6> q = self.get_curr_joint_deg();
            return py::array_t<double>({6}, {sizeof(double)}, q.data());
        })

        .def("get_curr_joint_rad", [](const IkSolver& self) {
            vec<6> q = self.get_curr_joint_rad();
            return py::array_t<double>({6}, {sizeof(double)}, q.data());
        })

        .def("get_curr_joint_velocity_deg", [](const IkSolver& self) {
            vec<6> v = self.get_curr_joint_velocity_deg();
            return py::array_t<double>({6}, {sizeof(double)}, v.data());
        })

        .def("get_curr_joint_velocity_rad", [](const IkSolver& self) {
            vec<6> v = self.get_curr_joint_velocity_rad();
            return py::array_t<double>({6}, {sizeof(double)}, v.data());
        })

        .def("get_curr_tcp_speed", [](const IkSolver& self) {
            vec<6> v = self.get_curr_tcp_speed();
            return py::array_t<double>({6}, {sizeof(double)}, v.data());
        })

        // Jacobian: 6x6 NumPy Matrix로 변환 (복사 발생 방지를 위해 Eigen::Ref나 직접 생성)
        .def("get_curr_jacobian", [](const IkSolver& self) {
            mat<6, 6> J = self.get_curr_jacobian();
            // 2차원 NumPy 배열 생성 (Row-major 기준)
            return py::array_t<double>(
                {6, 6},                       // shape
                {6 * sizeof(double), sizeof(double)}, // strides
                J.data()                      // data pointer
            );
        })

        .def("get_curr_tcp_tf", &IkSolver::get_curr_tcp_tf)
        .def("get_curr_tf", &IkSolver::get_curr_tf, py::arg("index"))

        // ==================================================================
        // Setting limits
        // ==================================================================
        .def("set_end_effector_offset", &IkSolver::set_end_effector_offset)
        .def("set_joint_limit", &IkSolver::set_joint_limit)
        .def("set_joint_velocity_limit", &IkSolver::set_joint_velocity_limit)
        .def("set_workspace_limitX", &IkSolver::set_workspace_limitX)
        .def("set_workspace_limitY", &IkSolver::set_workspace_limitY)
        .def("set_workspace_limitZ", &IkSolver::set_workspace_limitZ)
        .def("set_tcp_max_speed", &IkSolver::set_tcp_max_speed)
        .def("set_joint_velocity_limit_scale", &IkSolver::set_joint_velocity_limit_scale);
}