#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <ik_solver.h>
#include <transform.hpp>

namespace py = pybind11;

PYBIND11_MODULE(ik_solver_py, m) {
    // ------------------------------------------------------------------
    // Transform 클래스 바인딩 (유지)
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
    // IkSolver 클래스 바인딩 (모든 기능 추가)
    // ------------------------------------------------------------------
    py::class_<IkSolver> ik_solver(m, "IkSolver");

    // LimitMode Enum 바인딩
    py::enum_<IkSolver::LimitMode>(ik_solver, "LimitMode")
        .value("RELAX_MODE", IkSolver::LimitMode::RELAX_MODE)
        .value("STRICT_MODE", IkSolver::LimitMode::STRICT_MODE)
        .export_values();

    ik_solver
        .def(py::init<const std::string &>())
        .def("init", &IkSolver::init)

        // --- 이동 제어 ---
        .def("movel", &IkSolver::movel, "Move TCP in a straight line to target transform", py::arg("tf"))
        .def("movej", [](IkSolver &self, std::vector<double> q) {
            if (q.size() != 6) throw std::runtime_error("Joint array must have size 6");
            double q_arr[6];
            std::copy(q.begin(), q.end(), q_arr);
            self.movej(q_arr);
        }, "Move joints directly to target angles (6 joints)")

        // --- 상태 조회 ---
        .def("get_tcp_tf", &IkSolver::get_tcp_tf, "Get current TCP transform")
        .def("get_tf", &IkSolver::get_tf, "Get transform of a specific frame index", py::arg("index"))
        .def("get_curr_joint_deg", &IkSolver::get_curr_joint_deg)
        .def("get_curr_joint_rad", &IkSolver::get_curr_joint_rad)
        .def("get_curr_jvel_deg", &IkSolver::get_curr_jvel_deg)
        .def("get_curr_jvel_rad", &IkSolver::get_curr_jvel_rad)
        .def("get_curr_tcp_speed", &IkSolver::get_curr_tcp_speed)
        .def("get_jacobian", &IkSolver::get_jacobian, "Get 6x6 Jacobian matrix")

        // --- End-Effector 설정 ---
        .def("set_end_effector_tf", &IkSolver::set_end_effector_tf, "Set end-effector transform offset", py::arg("tf"))
        .def("has_end_effector", &IkSolver::has_end_effector, "Check if end-effector is set")
        .def("remove_end_effector", &IkSolver::remove_end_effector, "Remove end-effector offset")

        // --- 모드 설정 ---
        .def("set_strict_mode", &IkSolver::set_strict_mode, "Set solver to strict mode (path priority)")
        .def("set_relax_mode", &IkSolver::set_relax_mode, "Set solver to relax mode (velocity priority)")

        // --- 제한 제어 및 안전 설정 ---
        .def("set_joint_limit", &IkSolver::set_joint_limit, "Set joint range limits [rad]", py::arg("index"), py::arg("min"), py::arg("max"))
        .def("set_joint_vlimit", &IkSolver::set_joint_vlimit, "Set maximum joint velocity [rad/s]", py::arg("index"), py::arg("max_vel"))
        .def("set_workspace_limitX", &IkSolver::set_workspace_limitX, "Set X-axis workspace limits [m]", py::arg("min"), py::arg("max"))
        .def("set_workspace_limitY", &IkSolver::set_workspace_limitY, "Set Y-axis workspace limits [m]", py::arg("min"), py::arg("max"))
        .def("set_workspace_limitZ", &IkSolver::set_workspace_limitZ, "Set Z-axis workspace limits [m]", py::arg("min"), py::arg("max"))
        .def("set_tcp_speed_limit", &IkSolver::set_tcp_max_speed, "Set maximum TCP linear velocity [m/s]", py::arg("max_speed"))
        .def("set_tcp_max_speed", &IkSolver::set_tcp_max_speed, "Set maximum TCP linear velocity [m/s]", py::arg("max_speed"))
        .def("set_safety_scale", &IkSolver::set_safety_scale, "Set safety scale factor for joint velocity (1.0~2.0)", py::arg("scale"));
}