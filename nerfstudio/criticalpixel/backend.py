import os
from pathlib import Path
from torch.utils.cpp_extension import load
from nerfstudio.criticalpixel.utils.file import get_all_files
from pkg_resources import parse_version
import torch
import subprocess
import re

PROJECT_DIR = Path(__file__).absolute().parent


def min_supported_compute_capability(cuda_version):
    if cuda_version >= parse_version("12.0"):
        return 50
    else:
        return 20


def max_supported_compute_capability(cuda_version):
    if cuda_version < parse_version("11.0"):
        return 75
    elif cuda_version < parse_version("11.1"):
        return 80
    elif cuda_version < parse_version("11.8"):
        return 86
    else:
        return 90


assert torch.cuda.is_available()

major, minor = torch.cuda.get_device_capability()
compute_capabilities = [major * 10 + minor]
print(f"Obtained compute capability {compute_capabilities[0]} from PyTorch")

cpp_standard = 14

# Get CUDA version and make sure the targeted compute capability is compatible
nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode()
cuda_version = re.search(r"release (\S+),", nvcc_out)

if cuda_version:
    cuda_version = parse_version(cuda_version.group(1))
    print(f"Detected CUDA version {cuda_version}")
    if cuda_version >= parse_version("11.0"):
        cpp_standard = 17

    supported_compute_capabilities = [
        cc
        for cc in compute_capabilities
        if cc >= min_supported_compute_capability(cuda_version)
        and cc <= max_supported_compute_capability(cuda_version)
    ]

    if not supported_compute_capabilities:
        supported_compute_capabilities = [max_supported_compute_capability(cuda_version)]

    if supported_compute_capabilities != compute_capabilities:
        print(
            f"WARNING: Compute capabilities {compute_capabilities} are not all supported by the installed CUDA version {cuda_version}. Targeting {supported_compute_capabilities} instead."
        )
        compute_capabilities = supported_compute_capabilities

min_compute_capability = min(compute_capabilities)


print(f"Targeting C++ standard {cpp_standard}")

base_nvcc_flags = [
    f"-std=c++{cpp_standard}",
    "--extended-lambda",
    "--expt-relaxed-constexpr",
    # The following definitions must be undefined
    # to support half-precision operation.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

if os.name == "posix":
    base_cflags = [f"-std=c++{cpp_standard}"]
    base_nvcc_flags += ["-Xcompiler=-Wno-float-conversion", "-Xcompiler=-fno-strict-aliasing", "-O3"]
elif os.name == "nt":
    base_cflags = [f"/std:c++{cpp_standard}"]


source_dir = PROJECT_DIR

proj_cu_files = get_all_files(source_dir / "cpp" / "camera", "*.cu")
proj_cpp_files = (
    get_all_files(source_dir / "cpp" / "camera", "*.cpp")
    + get_all_files(source_dir / "cpp" / "utils", "*.cpp")
    + [source_dir / "cpp" / "binding.cpp"]
)

proj_src_files = proj_cu_files + proj_cpp_files
proj_src_files = [str(f.absolute()) for f in proj_src_files]

include_dirs = [
    f"{PROJECT_DIR}/cpp",
    f"{PROJECT_DIR}/dependencies",
    f"{PROJECT_DIR}/dependencies/eigen",
    f"{PROJECT_DIR}/dependencies/json/include",
    f"{PROJECT_DIR}/dependencies/pybind11_json/include",
]


base_definitions = []


c_backend = load(
    name="criticalpixel_cpp",
    extra_cflags=base_cflags,
    extra_cuda_cflags=base_nvcc_flags,
    sources=proj_src_files,
    extra_include_paths=include_dirs,
    verbose=False,
    with_cuda=True,
)

print('Load CPP Extension')

__all__ = ["c_backend"]
