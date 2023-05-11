import time
from pynvml import *
import torch
import torch.distributed as dist
import numpy
from deepspeed.runtime.constants import *


class EnergyProfiler(object):
    def __init__(self, grid, config):
        # Initialize NVML
        nvmlInit()
        self.device_cnt = nvmlDeviceGetCount()
        self.grid = grid
        self.stage_id = self.grid.stage_id
        self.config = config
        self.handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

        if self.config["gpu"] == ENVPIPE_GPU_V100:
            sm_freq_filter_max = V100_SM_FREQ_FILTER_MAX
            sm_freq_filter_min = V100_SM_FREQ_FILTER_MIN
            sm_freq_granularity = V100_SM_FREQ_GRANULARITY
            mem_freq_idx = V100_MEM_FREQ_IDX

        elif self.config["gpu"] == ENVPIPE_GPU_RTX3090:
            sm_freq_filter_max = RTX3090_SM_FREQ_FILTER_MAX
            sm_freq_filter_min = RTX3090_SM_FREQ_FILTER_MIN
            sm_freq_granularity = RTX3090_SM_FREQ_GRANULARITY
            mem_freq_idx = RTX3090_MEM_FREQ_IDX

        else:
            raise RuntimeError(
                f'{self.__class__.__name__} gpu not registered.\
                     {self.config["gpu"]}')

        # (RTX3090) Get supported gpu clocks for P2 state
        # (NVIDIA forces P2 State for CUDA programs in RTX3090)
        # (V100) Get supported gpu clocks for P0 state
        mem_clock = nvmlDeviceGetSupportedMemoryClocks(self.handle)[
            mem_freq_idx]

        # Get supported GPU clocks
        supported_gpu_clocks = nvmlDeviceGetSupportedGraphicsClocks(
            self.handle, mem_clock)

        # Filter out clocks for profiling efficiency
        self.gpu_clocks = list(
            filter(lambda x: x % sm_freq_granularity == 0 and
                   x <= sm_freq_filter_max and
                   x >= sm_freq_filter_min, supported_gpu_clocks))

        if self.config["type"] != ENVPIPE_TYPE_UNIFORM:
            # Profile is not needed for last stage because it is already critical path.
            # Just setup with maximum freq.
            if self.grid.is_last_stage:
                self.gpu_clocks = [self.gpu_clocks[0]] * len(self.gpu_clocks)

        self.is_profiling = (
            self.config["type"] != ENVPIPE_TYPE_BASELINE)
        self.profile_clock_idx = 0
        self.profile_cnt = 0
        self.profile_start_energy = 0
        self.raw_energy_consumption = []
        self.profile_energy = [0] * len(self.gpu_clocks)
        self.min_gpu_clock = self.gpu_clocks[0]
        self.min_clock_idx = 0

        self.raw_forward_execution_time = []
        self.raw_backward_execution_time = []
        self.profile_forward_execution_time = [0] * len(self.gpu_clocks)
        self.profile_backward_execution_time = [0] * len(self.gpu_clocks)
        self.forward_execution_event_start = []
        self.forward_execution_event_end = []
        self.backward_execution_event_start = []
        self.backward_execution_event_end = []

        self.activation_size = 0
        self.activation_size_start = 0

        # Initialize with max clock frequency
        nvmlDeviceSetGpuLockedClocks(
            self.handle, self.gpu_clocks[0], self.gpu_clocks[0])

    def start_profile(self):
        """Start profiling.
        """
        assert self.is_profiling

        nvmlDeviceSetGpuLockedClocks(self.handle, self.gpu_clocks[self.profile_clock_idx],
                                     self.gpu_clocks[self.profile_clock_idx])

        assert self.profile_start_energy == 0

        torch.cuda.synchronize()
        self.profile_start_energy = nvmlDeviceGetTotalEnergyConsumption(
            self.handle)

    def end_profile(self):
        """Stop profiling.
        """
        assert self.is_profiling
        assert self.profile_start_energy != 0

        torch.cuda.synchronize()
        avg_forward_time, avg_backward_time = self.measure_elapsed_time()
        energy_consumption = nvmlDeviceGetTotalEnergyConsumption(
            self.handle) - self.profile_start_energy

        assert energy_consumption != 0

        self.raw_energy_consumption.append(energy_consumption)
        self.raw_forward_execution_time.append(avg_forward_time)
        self.raw_backward_execution_time.append(avg_backward_time)
        self.profile_cnt += 1
        self.profile_start_energy = 0

        # Profile next clock frequency
        if self.profile_cnt > PROFILE_ITER_LIMIT:
            self.profile_energy[self.profile_clock_idx] = self.average_without_outliers(
                self.raw_energy_consumption)
            self.profile_forward_execution_time[self.profile_clock_idx] = \
                self.average_without_outliers(self.raw_forward_execution_time)
            self.profile_backward_execution_time[self.profile_clock_idx] = \
                self.average_without_outliers(self.raw_backward_execution_time)

            self.raw_energy_consumption = []
            self.raw_forward_execution_time = []
            self.raw_backward_execution_time = []
            self.profile_cnt = 0
            self.profile_clock_idx += 1

            if dist.get_rank() == 0:
                print("[ENVPIPE] Profiling {progress:.0f}%".format(
                    progress=(self.profile_clock_idx * 100 / len(self.gpu_clocks))))

            # End profiling
            if self.profile_clock_idx == len(self.gpu_clocks):
                # For consistency between data parallel groups, just need
                # one data parallel rank for each pipeline stage
                src = self.grid.stage_to_global(
                    self.stage_id) - self.grid.get_data_parallel_rank()
                dist.broadcast_object_list(self.profile_energy,
                                           src=src,
                                           group=self.grid.get_data_parallel_group())
                dist.broadcast_object_list(self.profile_forward_execution_time,
                                           src=src,
                                           group=self.grid.get_data_parallel_group())
                dist.broadcast_object_list(self.profile_backward_execution_time,
                                           src=src,
                                           group=self.grid.get_data_parallel_group())
                self.min_clock_idx = self.profile_energy.index(
                    min(self.profile_energy))
                self.min_gpu_clock = self.gpu_clocks[self.min_clock_idx]
                self.is_profiling = False

                self.memory_constraint_limit = max(int(torch.cuda.mem_get_info()[
                    0] / self.activation_size), 0)

                if self.grid.get_data_parallel_rank() == 0:
                    print("[ENVPIPE] Minimum sm freq for stage",
                          self.stage_id, ":", self.min_gpu_clock)
                    print("Forward: {forward_max} {forward_min} Backward {backward_max} {backward_min}".format(
                        forward_max=self.profile_forward_execution_time[0],
                        forward_min=self.profile_forward_execution_time[self.min_clock_idx],
                        backward_max=self.profile_backward_execution_time[0],
                        backward_min=self.profile_backward_execution_time[self.min_clock_idx]))

    def get_max_clock(self):
        return self.gpu_clocks[0]

    def get_min_clock(self):
        return self.min_gpu_clock

    def average_without_outliers(self, input):
        if len(input) <= 3:
            return numpy.mean(input)

        mean = numpy.mean(input)
        sd = numpy.std(input)
        output = [x for x in input if (x > mean - sd)]
        output = [x for x in output if (x < mean + sd)]

        return numpy.mean(output)

    def start_event_record(self, is_forward):
        start = torch.cuda.Event(enable_timing=True)
        start.record()

        if is_forward:
            self.forward_execution_event_start.append(start)

            # calculate activation size after one forward pass
            if self.activation_size == 0 and len(self.forward_execution_event_start) == 2:
                self.activation_size_start = torch.cuda.memory_allocated()
        else:
            self.backward_execution_event_start.append(start)

    def end_event_record(self, is_forward):
        end = torch.cuda.Event(enable_timing=True)
        end.record()

        if is_forward:
            self.forward_execution_event_end.append(end)
            if self.activation_size == 0 and len(self.forward_execution_event_start) == 2:
                assert self.activation_size_start != 0
                self.activation_size = torch.cuda.memory_allocated() - \
                    self.activation_size_start
        else:
            self.backward_execution_event_end.append(end)

    def measure_elapsed_time(self):
        assert len(self.forward_execution_event_start) == len(
            self.forward_execution_event_end)
        assert len(self.backward_execution_event_start) == len(
            self.backward_execution_event_end)

        forward_execution_time = []
        for i in range(len(self.forward_execution_event_start)):
            start = self.forward_execution_event_start[i]
            end = self.forward_execution_event_end[i]
            forward_execution_time.append(start.elapsed_time(end))
        avg_forward_time = self.average_without_outliers(
            forward_execution_time)

        backward_execution_time = []
        for i in range(len(self.backward_execution_event_start)):
            start = self.backward_execution_event_start[i]
            end = self.backward_execution_event_end[i]
            backward_execution_time.append(start.elapsed_time(end))
        avg_backward_time = self.average_without_outliers(
            backward_execution_time)

        self.forward_execution_event_start = []
        self.forward_execution_event_end = []
        self.backward_execution_event_start = []
        self.backward_execution_event_end = []

        return avg_forward_time, avg_backward_time

    def get_reschedule_forward_cnt(self, micro_batches):
        if self.is_profiling or self.grid.is_last_stage:
            return 0

        if (self.config["scheduling"] == ENVPIPE_SCHEDULING_1F1B or
                self.config["type"] == ENVPIPE_TYPE_BASELINE):
            return 0

        max_backward_time = self.profile_backward_execution_time[0]
        min_forward_time = self.profile_forward_execution_time[self.min_clock_idx]

        reschedule_cnt = int(max_backward_time *
                             (self.grid.get_pipe_parallel_world_size() -
                              self.stage_id - 1) / min_forward_time)

        reschedule_cnt = min(reschedule_cnt, self.memory_constraint_limit)

        total_micro_batch_limit = micro_batches - \
            (self.grid.get_pipe_parallel_world_size() - self.stage_id)
        reschedule_cnt = min(total_micro_batch_limit, reschedule_cnt)

        return reschedule_cnt
