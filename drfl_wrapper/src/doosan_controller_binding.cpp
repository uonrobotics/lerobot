#include "../include/doosan_controller.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(doosan_controller_py, m) {
  m.doc() = "Python binding for DoosanController";

  py::class_<DoosanController>(m, "DoosanController")
      .def(py::init<std::string>(), py::arg("ip"))
      .def("connect", &DoosanController::connect,                     "Connect to the robot")
      .def("disconnect", &DoosanController::disconnect,               "Disconnect from the robot")
      .def("servo_on", &DoosanController::servo_on,                   "Turn on robot servos")
      .def("servo_off", &DoosanController::servo_off,                 "Turn off robot servos")
      .def("stop", &DoosanController::stop,                           "Stop robot movement")
      .def("is_rt_mode", &DoosanController::is_rt_mode,               "Check if in RT mode")
      .def("get_current_joint", &DoosanController::get_current_joint, "Get current joint angles")



      .def(
          "movej",
          [](DoosanController &self, py::array_t<float> q, float time) {
            py::buffer_info buf = q.request();
            if (buf.size != 6) {
              throw std::runtime_error("Joint array must have 6 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            float joint_array[6];
            for (int i = 0; i < 6; ++i)
              joint_array[i] = ptr[i];
            self.movej(joint_array, time);
          },
          py::arg("q"), py::arg("time"), "Move robot joints (degrees)")
      .def(
          "start_rt",
          [](DoosanController &self, py::array_t<float> q) {
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
          [](DoosanController &self, py::array_t<float> q, float dt) {
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
