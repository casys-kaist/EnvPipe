'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import sys
import types
from typing import Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from packaging import version as pkg_version

from . import ops
from . import module_inject

from .runtime.engine import DeepSpeedEngine, DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable
from .runtime.engine import ADAM_OPTIMIZER, LAMB_OPTIMIZER
from .runtime.pipe.engine import PipelineEngine
from .inference.engine import InferenceEngine

from .runtime.lr_schedules import add_tuning_arguments
from .runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from .runtime.activation_checkpointing import checkpointing
from .ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from .module_inject import replace_transformer_layer, revert_transformer_layer

from .utils import log_dist, OnDevice
from .comm.comm import init_distributed

from .runtime import zero
from .runtime import DeepSpeedOptimizer, ZeROOptimizer

from .pipe import PipelineModule

from .git_version_info import version, git_hash, git_branch


def _parse_version(version_str):
    '''Parse a version string and extract the major, minor, and patch versions.'''
    ver = pkg_version.parse(version_str)
    return ver.major, ver.minor, ver.micro


# Export version information
__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(
    __version__)
__git_hash__ = git_hash
__git_branch__ = git_branch


def initialize(args=None,
               model: torch.nn.Module = None,
               optimizer: Optional[Union[Optimizer,
                                         DeepSpeedOptimizerCallable]] = None,
               model_parameters: Optional[torch.nn.Module] = None,
               training_data: Optional[torch.utils.data.Dataset] = None,
               lr_scheduler: Optional[Union[_LRScheduler,
                                            DeepSpeedSchedulerCallable]] = None,
               mpu=None,
               dist_init_required: Optional[bool] = None,
               collate_fn=None,
               config=None,
               config_params=None):
    """Initialize the DeepSpeed Engine.

    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and returns a Scheduler object.
            The scheduler object should define a get_lr(), step(), state_dict(), and load_state_dict() methods

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        dist_init_required: Optional: None will auto-initialize torch distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

        config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
            as an argument instead, as a path or a dictionary.

        config_params: Optional: Same as `config`, kept for backwards compatibility.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.

        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.

        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
        __version__,
        __git_hash__,
        __git_branch__),
        ranks=[0])

    # Disable zero.Init context if it's currently enabled
    zero.partition_parameters.shutdown_init_context()

    assert model is not None, "deepspeed.initialize requires a model"

    if not isinstance(model, PipelineModule):
        engine = DeepSpeedEngine(args=args,
                                 model=model,
                                 optimizer=optimizer,
                                 model_parameters=model_parameters,
                                 training_data=training_data,
                                 lr_scheduler=lr_scheduler,
                                 mpu=mpu,
                                 dist_init_required=dist_init_required,
                                 collate_fn=collate_fn,
                                 config=config,
                                 config_params=config_params)
    else:
        assert mpu is None, "mpu must be None with pipeline parallelism"
        engine = PipelineEngine(args=args,
                                model=model,
                                optimizer=optimizer,
                                model_parameters=model_parameters,
                                training_data=training_data,
                                lr_scheduler=lr_scheduler,
                                mpu=model.mpu(),
                                dist_init_required=dist_init_required,
                                collate_fn=collate_fn,
                                config=config,
                                config_params=config_params)

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler
    ]
    return tuple(return_items)


def _add_core_arguments(parser):
    r"""Helper (internal) function to update an argument parser with an argument group of the core DeepSpeed arguments.
        The core set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.

        This is a helper function to the public add_config_arguments()

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    group = parser.add_argument_group('DeepSpeed', 'DeepSpeed configurations')

    group.add_argument(
        '--deepspeed',
        default=False,
        action='store_true',
        help='Enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepspeed_config',
                       default=None,
                       type=str,
                       help='DeepSpeed json configuration file.')

    group.add_argument(
        '--deepscale',
        default=False,
        action='store_true',
        help='Deprecated enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)'
    )

    group.add_argument('--deepscale_config',
                       default=None,
                       type=str,
                       help='Deprecated DeepSpeed json configuration file.')

    group.add_argument(
        '--deepspeed_mpi',
        default=False,
        action='store_true',
        help="Run via MPI, this will attempt to discover the necessary variables to initialize torch "
        "distributed from the MPI environment")

    return parser


def add_config_arguments(parser):
    r"""Update the argument parser to enabling parsing of DeepSpeed command line arguments.
        The set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    parser = _add_core_arguments(parser)

    return parser


def init_inference(model,
                   triangular_masking=True,
                   mp_size=1,
                   training_mp_size=1,
                   mpu=None,
                   ep_group=None,
                   expert_mp_group=None,
                   checkpoint=None,
                   dtype=None,
                   injection_policy=None,
                   replace_method='auto',
                   quantization_setting=None,
                   replace_with_kernel_inject=False,
                   return_tuple=True,
                   ep_size=1,
                   moe=False,
                   moe_experts=1,
                   moe_type='standard',
                   args=None,
                   enable_cuda_graph=False,
                   save_mp_checkpoint_path=None,
                   base_dir=""):
    """Initialize the DeepSpeed InferenceEngine.

    Arguments:
        model: Required: nn.module class before apply any wrappers

        triangular_masking: Required: this shows the type of masking for attention scores in transformer layer
            note that the masking is application specific.

        mp_size: Optional: Desired model parallel size, default is 1 meaning no
            model parallelism.

        training_mp_size: Optional: if loading a checkpoint this is the mp size that it was trained with,
            it may be different than what the mp size that you want to use during inference.

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        checkpoint: Optional: Path to deepspeed compatible checkpoint or path to
            JSON with load policy.

        dtype: Optional: Desired model data type, will convert model to this type.
            Supported target types: torch.half, torch.int8, torch.float

        injection_policy: Optional: Dictionary mapping a client nn.Module to its corresponding
            injection policy. e.g., {BertLayer : deepspeed.inference.HFBertLayerPolicy}

        replace_method: Optional: If 'auto' DeepSpeed will automatically try and replace
            model modules with its optimized versions. If an injection_policy is set this will
            override the automatic replacement behavior.

        quantization_setting: Optional: Quantization settings used for quantizing your model using the MoQ.
            The setting can be one element or a tuple. If one value is passed in, we consider it as the number
            of groups used in quantization. A tuple is passed in if we want to mention that there is extra-grouping
            for the MLP part of a Transformer layer (e.g. (True, 8) shows we quantize the model using 8 groups for
            all the network except the MLP part that we use 8 extra grouping).
        replace_with_kernel_inject: this flag need to be set to true to inject inference kernels for models such as, Bert, GPT2, GPT-Neo and GPT-J. Otherwise,
            the injection_dict provides the names of two linear layers as a tuple: (attention_output projection, transformer output projection)
        return_tuple: Specify whether or not the transformer layers need to return a tuple or a Tensor. It is set to True by default (returning a tuple).
        ep_size: The expert-parallelism size which is used for partitioning the experts across the GPUs in the expert-parallel group.
        moe: Specify if the type of Transformer is MoE. It is set to False by default.
        moe_experts: The global number of experts used in an MoE layer.
        moe_type: Specify the type of MoE layer. We have two types of MoE layer: 'Standard' and 'Residual'. It is set to 'Standard' type by default.
        args: All the arguments used for launching the inference api that can be useful at the inference-engine for injecting the optimizations.
        enable_cuda_graph: use this flag for capturing the CUDA-Graph of the inference ops, so that it can run faster using the graph replay method,
            this is set to False by default
        save_mp_checkpoint_path: The path for which we want to save the loaded model with a checkpoint. This feature is used for adjusting the
            parallelism degree to help alleviate the model loading overhead. It does not save any new checkpoint if no path is passed.
        base_dir: This shows the root directory under which all the checkpoint files exists. This can be passed through the json config too.

    Returns:
        A deepspeed.InferenceEngine wrapped model.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
        __version__,
        __git_hash__,
        __git_branch__),
        ranks=[0])

    engine = InferenceEngine(model,
                             triangular_masking,
                             mp_size,
                             training_mp_size,
                             ep_size,
                             mpu,
                             ep_group,
                             expert_mp_group,
                             checkpoint,
                             dtype,
                             injection_policy,
                             return_tuple,
                             replace_method,
                             quantization_setting,
                             replace_with_kernel_inject,
                             moe,
                             moe_experts,
                             moe_type,
                             args,
                             enable_cuda_graph,
                             save_mp_checkpoint_path,
                             base_dir)

    return engine
