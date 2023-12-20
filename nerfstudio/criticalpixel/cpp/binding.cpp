#include <torch/torch.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <camera/Camera.h>
#include <pybind11/functional.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Undistort", &Undistort);
  m.def("Distort", &Distort);

    py::enum_<CameraModel>(m, "CameraModel")
    .value("SimpleRadial", CameraModel::SimpleRadial)
    .value("Pinhole", CameraModel::Pinhole)
    .value("OpenCV", CameraModel::OpenCV)
    .value("Fisheye", CameraModel::Fisheye)
    .value("Panoramic", CameraModel::Panoramic)
    .export_values();
}
