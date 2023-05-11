"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
from .builder import TorchCPUOpBuilder


class CPUAdagradBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAGRAD"
    NAME = "cpu_adagrad"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adagrad.{self.NAME}_op'

    def sources(self):
        return ['csrc/adagrad/cpu_adagrad.cpp', 'csrc/common/custom_cuda_kernel.cu']

    def include_paths(self):
        import torch
        if not self.is_rocm_pytorch():
            CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        else:
            CUDA_INCLUDE = [
                os.path.join(torch.utils.cpp_extension.ROCM_HOME,
                             "include"),
                os.path.join(torch.utils.cpp_extension.ROCM_HOME,
                             "include",
                             "rocrand"),
                os.path.join(torch.utils.cpp_extension.ROCM_HOME,
                             "include",
                             "hiprand"),
            ]
        return ['csrc/includes'] + CUDA_INCLUDE
