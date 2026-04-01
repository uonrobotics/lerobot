#include "../include/doosan_robot_controller.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(doosan_robot_controller_py, m) {
  m.doc() = "Python binding for DoosanController";

  py::class_<DoosanController>(m, "DoosanController")
      .def(py::init<std::string>(), py::arg("ip"))
      .def("connect",           &DoosanController::connect,           "로봇 연결")
      .def("disconnect",        &DoosanController::disconnect,        "연결 종료")
      .def("servo_on",          &DoosanController::servo_on,          "서보 on")
      .def("servo_off",         &DoosanController::servo_off,         "서보 off")
      .def("stop",              &DoosanController::stop,              "정지")
      .def("is_rt_mode",        &DoosanController::is_rt_mode,        "RealTime 제어 모드 여부")
      .def("get_current_joint", &DoosanController::get_curr_jpos,     "로봇의 현재 joint 위치 (degrees)")
      .def("set_kp_gain",       &DoosanController::set_kp_gain,       "Kp 제어기 설정")
      .def("set_kd_gain",       &DoosanController::set_kd_gain,       "Kd 제어기 설정")
      .def("set_target_time",   &DoosanController::set_target_time,   "타겟 시간 설정")

      .def(
          "movej",
          [](DoosanController &self, py::array_t<float> q, float time)
          {
            py::buffer_info buf = q.request();
            float* ptr = static_cast<float *>(buf.ptr);
            float joint_array[6];
            for (int i = 0; i < 6; ++i)
              joint_array[i] = ptr[i];
            self.movej(joint_array, time);
          },
          py::arg("q"), py::arg("time"), "Move robot joints (degrees)")
      .def(
          "start_rt",
          [](DoosanController &self, py::array_t<float> q)
          {
            py::buffer_info buf = q.request();
            if (buf.size != 6) {
              throw std::runtime_error("Joint array must have 6 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            float joint_array[6];
            for (int i = 0; i < 6; ++i)
              joint_array[i] = ptr[i];
            return self.start_rt(joint_array);
          },
          py::arg("q"), "Start Realtime mode")
      .def(
          "movej_rt",
          [](DoosanController &self, py::array_t<float> q, float dt)
          {
            py::buffer_info buf = q.request();
            if (buf.size != 6) {
              throw std::runtime_error("Joint array must have 6 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            float joint_array[6];
            for (int i = 0; i < 6; ++i)
              joint_array[i] = ptr[i];
            self.movej_rt(joint_array, dt);
          },
          py::arg("q"), py::arg("dt"), "RT move robot joints (degrees)");
}
