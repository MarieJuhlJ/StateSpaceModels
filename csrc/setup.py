import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# If you're in a Conda environment, set the CUDA_HOME from the Conda environment path
if "CONDA_PREFIX" in os.environ:
    conda_env_path = os.environ["CONDA_PREFIX"]
    cuda_home = os.path.join(conda_env_path, "lib", "python3.x", "site-packages", "torch", "lib", "include", "cuda")
    
    # Set CUDA_HOME inline if Conda is being used
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = os.path.join(cuda_home, "bin") + ":" + os.environ["PATH"]
    print("CUDA_HOME set to:", os.environ["CUDA_HOME"])

# Check if CUDA_HOME is set, otherwise raise error
if "CUDA_HOME" not in os.environ:
    raise EnvironmentError("CUDA_HOME is not set. Please install CUDA or set CUDA_HOME manually.")

# Example CUDA flags (you can keep these or modify them)
cc_flag = [
    "-gencode", "arch=compute_53,code=sm_53",
    "-gencode", "arch=compute_62,code=sm_62",
    "-gencode", "arch=compute_70,code=sm_70",
    "-gencode", "arch=compute_72,code=sm_72",
    "-gencode", "arch=compute_80,code=sm_80",
    "-gencode", "arch=compute_87,code=sm_87",
]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v",
        "-lineinfo",
    ] + cc_flag,
}

ext_modules = [
    CUDAExtension(
        name="selective_scan_cuda",
        sources=[
            "selective_scan/selective_scan.cpp",
            "selective_scan/selective_scan_fwd_fp32.cu",
            "selective_scan/selective_scan_fwd_fp16.cu",
            "selective_scan/selective_scan_fwd_bf16.cu",
            "selective_scan/selective_scan_bwd_fp32_real.cu",
            "selective_scan/selective_scan_bwd_fp32_complex.cu",
            "selective_scan/selective_scan_bwd_fp16_real.cu",
            "selective_scan/selective_scan_bwd_fp16_complex.cu",
            "selective_scan/selective_scan_bwd_bf16_real.cu",
            "selective_scan/selective_scan_bwd_bf16_complex.cu",
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=[Path(os.path.dirname(os.path.abspath(__file__))) / "selective_scan"],
    )
]

setup(
    name="selective_scan_cuda",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch"],
)
