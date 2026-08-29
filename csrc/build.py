import os
import sys
import subprocess
import torch
from torch.utils.cpp_extension import load, CppExtension, CUDAExtension

# Localização das ferramentas de compilação no Windows
MSVC_BIN = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"

# Garante que cl.exe, nvcc.exe e ninja.exe estejam no PATH
os.environ["PATH"] = f"{MSVC_BIN};{CUDA_BIN};{os.environ['PATH']}"
os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_SRC = os.path.join(CURRENT_DIR, "dcfr_bindings.cpp")
CU_SRC1 = os.path.join(CURRENT_DIR, "dcfr_kernels.cu")
CU_SRC2 = os.path.join(CURRENT_DIR, "dcfr_graph_runner.cu")

print("======================================================================")
print(" [COMPILANDO] EXTENSAO CUDA NATIVA C++ (D-CFR FAST KERNELS)")
print("======================================================================")
print(f"  CUDA Sources: {CU_SRC1}, {CU_SRC2}")
print(f"  C++ Source:   {CPP_SRC}")
print("=" * 70)

# Flags de otimização máxima para arquitetura Ampere (RTX 3060: sm_86)
extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_86,code=sm_86",
    "-Xcompiler", "/O2",
    "-Xcompiler", "/fp:fast",
    "-Xcompiler", "/wd4819",
]

extra_cflags = [
    "/O2",
    "/fp:fast",
    "/std:c++17",
]

try:
    dcfr_cuda_ext = load(
        name="dcfr_cuda_ext",
        sources=[CPP_SRC, CU_SRC1, CU_SRC2],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True
    )
    print("\n[SUCESSO] EXTENSAO CUDA COMPILADA E CARREGADA COM SUCESSO!")
    print("Modulos exportados:", dir(dcfr_cuda_ext))
except Exception as e:
    print(f"\n[ERRO] Erro durante a compilacao: {e}")
    sys.exit(1)
