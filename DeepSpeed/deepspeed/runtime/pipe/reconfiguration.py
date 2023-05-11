import torch
import torch.distributed as dist
from deepspeed.runtime.constants import *
from pynvml import *


class PipelineExecutionGrid(object):
    """Implements a grid object that stores pipeline parallel execution stages
    corresponding to each execution (forward pass or backward pass)
    Needed for finding the critical path and get track of the current component's
    gpu clock
    """

    def __init__(self, stage_id, stages, micro_batches, pipe_grid, profiler, envpipe_config):
        self.pipe_grid = pipe_grid
        self.grid = [[] for _ in range(stages)]
        self.tag_to_grid_idx = [{} for _ in range(stages)]
        self.profiler = profiler
        self.stage_id = stage_id
        self.stages = stages
        self.micro_batches = micro_batches
        self.envpipe_config = envpipe_config
        self.initialized = False
        self.critical_path = []
        if self.envpipe_config["reconfig"] == ENVPIPE_RECONFIGURE_DEFAULT:
            self.is_reconfiguring = [False for _ in range(stages)]
        else:
            self.is_reconfiguring = [True for _ in range(stages)]
        self.reconfigure_finish_cnt = 0
        self.reconfigure_threshold = 0
        self.reconfigure_threshold_scale = int(
            self.envpipe_config["reconfigure_threshold_scale"])

        if self.envpipe_config["gpu"] == ENVPIPE_GPU_V100:
            self.reconfigure_granularity = V100_RECONFIGURE_GRANULARITY
        elif self.envpipe_config["gpu"] == ENVPIPE_GPU_RTX3090:
            self.reconfigure_granularity = RTX3090_RECONFIGURE_GRANULARITY
        else:
            raise RuntimeError(
                f'{self.__class__.__name__} gpu not registered.\
                     {self.config["gpu"]}')

    def add_execution_component(self, micro_batch_id, is_forward):
        if self.envpipe_config["type"] == ENVPIPE_TYPE_BASELINE:
            gpu_clock = self.profiler.get_max_clock()

        elif self.envpipe_config["type"] == ENVPIPE_TYPE_UNIFORM:
            gpu_clock = self.profiler.get_min_clock()

        elif self.envpipe_config["type"] == ENVPIPE_TYPE_ENVELOPE:
            if self.stage_id == self.stages - 1:
                gpu_clock = self.profiler.get_max_clock()

            # 1st micro-batch forward
            elif micro_batch_id == 0 and is_forward:
                gpu_clock = self.profiler.get_max_clock()

            # Last micro-batch backward
            elif micro_batch_id == self.micro_batches - 1 and not is_forward:
                gpu_clock = self.profiler.get_max_clock()

            else:
                gpu_clock = self.profiler.get_min_clock()

        else:
            raise RuntimeError(
                f'{self.__class__.__name__} invalid envpipe configuration \
                        {self.envpipe_config.type}')

        component = PipelineExecutionComponent(self.stage_id, micro_batch_id,
                                               is_forward, gpu_clock)
        self.grid[self.stage_id].append(component)

        tag = self.get_tag(micro_batch_id, is_forward)
        self.tag_to_grid_idx[self.stage_id][tag] = len(
            self.grid[self.stage_id]) - 1

    def get_reconfigure_threshold(self):
        if self.reconfigure_threshold != 0:
            return self.reconfigure_threshold

        for component in self.grid[-1]:
            if component.recv_time == 0:
                continue

            if self.reconfigure_threshold == 0:
                self.reconfigure_threshold = component.recv_time

            elif component.recv_time < self.reconfigure_threshold:
                self.reconfigure_threshold = component.recv_time

        assert self.reconfigure_threshold != 0

        self.reconfigure_threshold *= self.reconfigure_threshold_scale

        print("threshold", self.reconfigure_threshold)

        return self.reconfigure_threshold

    def __str__(self):
        result = ""

        if len(self.grid) == 0:
            return result

        for stage_id in range(self.stages):
            result += "Stage {stage_id} | ".format(stage_id=stage_id)
            for component in self.grid[stage_id]:
                result += str(component)
            result += "\n"

        return result

    def get_tag(self, micro_batch_id, is_forward):
        if is_forward:
            tag = micro_batch_id + 1
        else:
            tag = -(micro_batch_id + 1)

        return tag

    def get_component(self, micro_batch_id, is_forward):
        tag = self.get_tag(micro_batch_id, is_forward)
        return self.grid[self.stage_id][self.tag_to_grid_idx[self.stage_id][tag]]

    def get_gpu_clock(self, micro_batch_id, is_forward):
        return self.get_component(micro_batch_id, is_forward).gpu_clock

    def start_event_record(self, micro_batch_id, is_forward, is_send):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        if is_send:
            self.get_component(
                micro_batch_id, is_forward).send_event_start = start
        else:
            self.get_component(
                micro_batch_id, is_forward).recv_event_start = start

    def end_event_record(self, micro_batch_id, is_forward, is_send):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if is_send:
            self.get_component(micro_batch_id, is_forward).send_event_end = end
        else:
            self.get_component(micro_batch_id, is_forward).recv_event_end = end

    def measure_elapsed_time(self):
        torch.cuda.synchronize()
        for component in self.grid[self.stage_id]:
            component.measure_elapsed_time()

    def gather_grid(self):
        dist.all_gather_object(self.grid, self.grid[self.stage_id],
                               group=self.pipe_grid.get_pipe_parallel_group())

    def broadcast_grid(self):
        dist.broadcast_object_list(self.grid, src=0)
        dist.broadcast_object_list(self.is_reconfiguring, src=0)

    def need_to_reconfigure(self):
        assert len(self.critical_path) != 0

        # Check if the critical path is the outer envelope
        for component in self.critical_path:
            if component.stage_id == self.stages - 1:
                continue
            elif component.micro_batch_id == 0 and component.is_forward:
                continue
            elif (component.micro_batch_id == self.micro_batches - 1
                  and not component.is_forward):
                continue
            else:
                return True

        if not self.finish_reconfigure():
            self.reconfigure_finish_cnt += 1

            if self.reconfigure_finish_cnt > ENVPIPE_RECONFIGURE_FINISH_LIMIT:
                self.is_reconfiguring[0] = False

        return False

    def finish_reconfigure(self):
        return not self.is_reconfiguring[0]

    # Note: The implementation to find the critical path differs from 
    # the paper's explanation which compares Bubble Slack Time and 
    # Dependency Delay. In the paper, this was done to give a better understanding
    # which pipeline unit affects the delay of the next unit in the critical path. 
    # However, in our actual implementation, we compare the recv activation 
    # time (Bubble Slack Time) to a threshold, which is the average recv 
    # activation time when there is no bubble. This approach allows us to 
    # more accurately identify delays in our system. 
    def find_critical_path(self):
        # Initialize critical path. Need to find critical path every step
        self.critical_path = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                self.grid[i][j].is_critical_path = False

        # Critical path starts from last backward pass of first stage
        cur_stage = 0
        cur_idx = len(self.grid[0]) - 1

        current = self.grid[cur_stage][cur_idx]
        current.is_critical_path = True
        self.critical_path.append(current)

        # End condition is when current is first forward of first stage
        while not (cur_stage == 0 and cur_idx == 0):
            assert cur_stage >= 0 and cur_stage < self.stages
            assert cur_idx >= 0 and cur_idx < len(self.grid[cur_stage])

            if current.is_forward:
                # If recv activation time is less than threshold, extend critical
                # path to current stage. Else, extend critical path to prev stage
                if current.recv_time < self.get_reconfigure_threshold():
                    cur_idx -= 1
                else:
                    cur_stage -= 1
                    tag = self.get_tag(
                        current.micro_batch_id, current.is_forward)
                    cur_idx = self.tag_to_grid_idx[cur_stage][tag]

            else:
                # if recv gradient time is less than threshold, extend critical
                # path to current stage. Else, extend critical path to next stage
                if current.recv_time < self.get_reconfigure_threshold():
                    cur_idx -= 1
                else:
                    cur_stage += 1
                    tag = self.get_tag(
                        current.micro_batch_id, current.is_forward)
                    cur_idx = self.tag_to_grid_idx[cur_stage][tag]

            current = self.grid[cur_stage][cur_idx]
            current.is_critical_path = True
            self.critical_path.append(current)

    def reconfigure_greedy(self):
        reconfigure_success = False

        # Increase the SM frequency from the back of the critical path. 
        # (self.critical path is already reversed)
        for component in self.critical_path:
            if component.gpu_clock >= self.profiler.get_max_clock():
                continue

            # Mutiplied to match the number of reconfigurations in a single step
            # with that of balanced method. Balanced method increases the minimum 
            # value(s) in the critical path which may be more than a single unit.
            component.gpu_clock += self.reconfigure_granularity * 3
            reconfigure_success = True
            break

        # If all execution component in critical path is max clock, reconfig finish
        if reconfigure_success:
            self.reconfigure_finish_cnt = 0

        else:
            self.reconfigure_finish_cnt += 1

            if self.reconfigure_finish_cnt > ENVPIPE_RECONFIGURE_FINISH_LIMIT:
                self.is_reconfiguring[0] = False

    def reconfigure_balanced(self):
        reconfigure_success = False
        min_gpu_clock = self.profiler.get_max_clock()

        # Find minimum gpu clock in critical path
        for component in self.critical_path:
            if component.gpu_clock < min_gpu_clock:
                min_gpu_clock = component.gpu_clock

        # After experimenting with various balancing strategies, we found that starting from 
        # the front of the critical path provides the best energy savings.
        if min_gpu_clock != self.profiler.get_max_clock():
            for component in reversed(self.critical_path):
                if component.gpu_clock == min_gpu_clock:
                    component.gpu_clock += self.reconfigure_granularity
                    reconfigure_success = True

        if reconfigure_success:
            self.reconfigure_finish_cnt = 0

        else:
            # If all execution component in critical path is max clock, reconfig finish
            self.reconfigure_finish_cnt += 1

            if self.reconfigure_finish_cnt > ENVPIPE_RECONFIGURE_FINISH_LIMIT:
                self.is_reconfiguring[0] = False

    def reconfigure(self):
        self.find_critical_path()

        if self.need_to_reconfigure():
            if self.envpipe_config["reconfig"] == ENVPIPE_RECONFIGURE_GREEDY:
                self.reconfigure_greedy()

            elif self.envpipe_config["reconfig"] == ENVPIPE_RECONFIGURE_BALANCED:
                self.reconfigure_balanced()

            elif self.envpipe_config["reconfig"] == ENVPIPE_RECONFIGURE_DEFAULT:
                return

            else:
                raise RuntimeError(
                    f'{self.__class__.__name__} invalid envpipe reconfig policy \
                        {self.envpipe_config["reconfig"]}')


class PipelineExecutionComponent(object):
    """Implements one execution component which overall consists to a
    PipelineParallelExecutionGrid
    """

    def __init__(self, stage_id, micro_batch_id, is_forward, default_gpu_clock):
        self.stage_id = stage_id
        self.micro_batch_id = micro_batch_id
        self.is_forward = is_forward
        self.gpu_clock = default_gpu_clock
        self.is_critical_path = False
        self.send_event_start = None
        self.send_event_end = None
        self.send_time = 0
        self.recv_event_start = None
        self.recv_event_end = None
        self.recv_time = 0
        self.start_time = 0

    def __str__(self):
        if self.is_critical_path:
            return "[{forward}{micro_batch_id} {gpu_clock} {recv_time:.1f}]".format(
                forward="F" if self.is_forward else "B",
                micro_batch_id=self.micro_batch_id,
                gpu_clock=self.gpu_clock,
                send_time=self.send_time,
                recv_time=self.recv_time)

        else:
            return " {forward}{micro_batch_id} {gpu_clock} {recv_time:.1f} ".format(
                forward="F" if self.is_forward else "B",
                micro_batch_id=self.micro_batch_id,
                gpu_clock=self.gpu_clock,
                send_time=self.send_time,
                recv_time=self.recv_time)

    def measure_elapsed_time(self):
        if self.send_event_start != None:
            assert self.send_event_end != None
            self.send_time = self.send_event_start.elapsed_time(
                self.send_event_end)
            self.send_event_start = None
            self.send_event_end = None

        if self.recv_event_start != None:
            assert self.recv_event_end != None
            self.recv_time = self.recv_event_start.elapsed_time(
                self.recv_event_end)
            self.recv_event_start = None
            self.recv_event_end = None
