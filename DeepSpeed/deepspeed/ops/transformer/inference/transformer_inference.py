'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import json
import math
import torch
from torch.autograd import Function
from ... import op_builder
import torch.nn as nn
from deepspeed import comm as dist
from deepspeed.utils.logging import log_dist
from deepspeed.utils.types import ActivationFuncType

# Cuda modules will be imported if needed
inference_cuda_module = None
minus_inf = -10000.0


class TransformerConfig():
    def __init__(self, hidden_size, intermediate_size, heads, num_hidden_layers):
        self.layer_id = -1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.heads = heads
        self.num_hidden_layers = num_hidden_layers


class DeepSpeedInferenceConfig(TransformerConfig):
    """Initialize the DeepSpeed Transformer Config.
        Arguments:
            hidden_size: The hidden size of the transformer layer
            intermediate_size: The intermediate size of the feed-forward part of transformer layer
            heads: The number of heads in the self-attention of the transformer layer
            num_hidden_layers: The number of transformer layers
            layer_norm_eps: The epsilon value for the layer norm
            local_rank: Optional: The rank of GPU running the transformer kernel, it is not required
                to use if the model already set the current device, otherwise need to set it
                so that the transformer kernel can work on the right device
            mp_size (optional): This argument is mainly used to create the parameters on the kernel side
                using model-parallel architecture. If the client model already takes care of this, there is no
                need to pass this argument.
            fp16: Enable half-precision computation
            pre_layer_norm: Select between Pre-LN or Post-LN transformer architecture
            stochastic_mode:  Enable for high performance, please note that this flag has some level of
                non-determinism and can produce different results on different runs.  However, we have seen
                that by enabling it, the pretraining tasks such as BERT are not affected and can obtain
                a high accuracy level. On the other hand, for the downstream tasks, such as fine-tuning, we recommend
                to turn it off in order to be able to reproduce the same result through the regular kernel execution.

            scale_attention: If true, both q and k are scaled by 1/sqrt(attention_heads) before attention computation.
            return_tuple: if True, returns the transformer output as a tuple, otherwise returns as a tensor
            bigscience_bloom: This flag is added temporarily for supporting the BLOOM-176B model architecture.
    """
    def __init__(self,
                 hidden_size=-1,
                 intermediate_size=-1,
                 heads=-1,
                 num_hidden_layers=-1,
                 layer_norm_eps=1e-12,
                 local_rank=-1,
                 mp_size=1,
                 fp16=False,
                 q_int8=False,
                 pre_layer_norm=True,
                 stochastic_mode=False,
                 scale_attention=True,
                 triangular_masking=True,
                 local_attention=False,
                 window_size=256,
                 rotary_dim=-1,
                 rotate_half=False,
                 rotate_every_two=True,
                 return_tuple=True,
                 mlp_after_attn=True,
                 mlp_act_func_type=ActivationFuncType.GELU,
                 training_mp_size=1,
                 bigscience_bloom=False):
        super(DeepSpeedInferenceConfig,
              self).__init__(
                  hidden_size,
                  (intermediate_size if intermediate_size > 0 else 4 * hidden_size),
                  heads,
                  num_hidden_layers)
        self.fp16 = fp16
        self.pre_layer_norm = pre_layer_norm
        self.local_rank = local_rank
        self.stochastic_mode = stochastic_mode
        self.epsilon = layer_norm_eps
        self.mp_size = mp_size
        self.q_int8 = q_int8
        self.scale_attention = scale_attention
        self.triangular_masking = triangular_masking
        self.local_attention = local_attention
        self.window_size = window_size
        self.rotary_dim = rotary_dim
        self.rotate_half = rotate_half
        self.rotate_every_two = rotate_every_two
        self.return_tuple = return_tuple
        self.mlp_after_attn = mlp_after_attn
        self.mlp_act_func_type = mlp_act_func_type
        self.specialized_mode = False
        self.training_mp_size = training_mp_size
        self.bigscience_bloom = bigscience_bloom

    @classmethod
    def from_dict(cls, json_object):
        config = DeepSpeedInferenceConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))


class DeepSpeedSelfAttentionFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                input_mask,
                head_mask,
                layer_past,
                get_present,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
                norm_w,
                norm_b,
                config,
                attn_qkvw,
                attn_qkvb,
                num_attention_heads_per_partition,
                norm_factor,
                hidden_size_per_partition,
                attn_ow,
                attn_ob,
                mp_group,
                q_scales,
                q_groups,
                merge_count,
                qkv_merging,
                score_context_func,
                alibi):
        def _transpose_for_scores(x, key=False, reshape=False):
            attention_head_size = x.shape[-1] // num_attention_heads_per_partition
            new_x_shape = x.size()[:-1] + (num_attention_heads_per_partition,
                                           attention_head_size)
            x_1 = x.view(*new_x_shape)
            if key:
                x_1 = x_1.permute(0, 2, 3, 1)
            else:
                x_1 = x_1.permute(0, 2, 1, 3)
            if reshape:
                return x_1.reshape(x.shape)
            return x_1.contiguous()

        def _transpose_for_context(x):
            x = x.permute(0, 2, 1, 3).contiguous()
            new_x_layer_shape = x.size()[:-2] + \
                                      (hidden_size_per_partition,)
            return x.view(*new_x_layer_shape).contiguous()

        ########### This part is taken/modified form the HF modeling_bloom.py ################
        # Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py

        def split_tensor_along_last_dim(tensor,
                                        num_partitions,
                                        contiguous_split_chunks=True):
            """Split a tensor along its last dimension.

            Args:
                tensor: ([`torch.tensor`], *required*):
                    input tensor to split
                num_partitions ([`int`], *required*):
                    number of partitions to split the tensor
                contiguous_split_chunks ([`bool`], *optional*, default=`False`)::
                    If True, make each chunk contiguous in memory.
            """
            # Get the size and dimension.
            last_dim = tensor.dim() - 1
            numerator, denominator = tensor.size()[last_dim], num_partitions
            if not (numerator % denominator == 0):
                raise ValueError(f"{numerator} is not divisible by {denominator}")
            last_dim_size = numerator // denominator
            # Split.
            tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
            # Note: torch.split does not create contiguous tensors by default.
            if contiguous_split_chunks:
                return tuple(chunk.contiguous() for chunk in tensor_list)

            return tensor_list

        def backup_attention(mixed_x_layer, layer_past, alibi, input_mask, norm_factor):
            alibi = alibi.to(torch.cuda.current_device())
            head_dim = hidden_size_per_partition // num_attention_heads_per_partition
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                num_attention_heads_per_partition,
                3 * head_dim)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            (query_layer,
             key_layer,
             value_layer) = split_tensor_along_last_dim(mixed_x_layer,
                                                        3)

            # [batch_size, head_dim, q_length, k_length]
            output_size = (query_layer.size(0),
                           query_layer.size(2),
                           query_layer.size(1),
                           key_layer.size(1))
            # [batch_size, q_length, num_heads, head_dim] -> [q_length, batch_size * num_heads, head_dim]
            query_layer = query_layer.transpose(1,
                                                2).reshape(
                                                    output_size[0] * output_size[1],
                                                    output_size[2],
                                                    -1)
            # [batch_size, k_length, num_heads, head_dim] -> [k_length, batch_size * num_heads, head_dim]
            key_layer = key_layer.transpose(1,
                                            2).reshape(output_size[0] * output_size[1],
                                                       output_size[3],
                                                       -1).transpose(-1,
                                                                     -2)
            value_layer = value_layer.transpose(1,
                                                2).reshape(
                                                    output_size[0] * output_size[1],
                                                    output_size[3],
                                                    -1)
            if layer_past is not None:
                past_key, past_value = layer_past
                # concatenate along seq_length dimension -> [batch_size, qk_length, num_heads, head_dim]
                key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=-1)
                value_layer = torch.cat((past_value.type_as(value_layer),
                                         value_layer),
                                        dim=-2)

            presents = (key_layer, value_layer)
            # Raw attention scores. [batch_size * num_heads, q_length, k_length]
            matmul_result = torch.matmul(query_layer, key_layer)
            # change view to [batch_size, num_heads, q_length, k_length]
            attention_scores = matmul_result.view(output_size[0],
                                                  output_size[1],
                                                  output_size[2],
                                                  -1)

            offset = dist.get_rank(
            ) * num_attention_heads_per_partition if dist.is_initialized() else 0
            attention_probs = inference_cuda_module.softmax_fp16(
                attention_scores,
                ((1 - input_mask).half() *
                 minus_inf) if input_mask.dtype == torch.int64 else input_mask,
                alibi,
                (config.triangular_masking and (attention_scores.shape[-2] > 1)),
                False,
                False,
                1,
                False,
                1 / (norm_factor * norm_factor),
                offset,
                config.mp_size)
            # change view [batch_size x num_heads, q_length, k_length]
            attention_probs_reshaped = attention_probs.view(*matmul_result.shape)

            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = torch.bmm(attention_probs_reshaped, value_layer)

            # change view [batch_size, num_heads, q_length, head_dim]
            context_layer = context_layer.view(
                context_layer.size(0) // num_attention_heads_per_partition,
                num_attention_heads_per_partition,
                context_layer.size(1),
                context_layer.shape[-1])

            context_layer = _transpose_for_context(context_layer)

            return context_layer, presents

        ###################### End of HF modeling_bloom addition ########################

        def compute_attention(qkv_out, input_mask):
            no_masking = input_mask is None

            head_size = (qkv_out.shape[-1] // 3 // num_attention_heads_per_partition)
            if no_masking:
                input_mask = torch.empty(1)
            if merge_count > 0 and config.q_int8:
                split_dim = (qkv_out.dim() - 1)
                qkv_split = torch.split(qkv_out,
                                        (qkv_out.shape[-1] // (2**merge_count)),
                                        dim=split_dim)
                qkv_split = [
                    torch.split(s,
                                (s.shape[-1] // 3),
                                dim=split_dim) for s in qkv_split
                ]
                (mixed_query,
                 key_layer,
                 value_layer) = [
                     torch.cat([s[i] for s in qkv_split],
                               axis=-1) for i in range(len(qkv_split[0]))
                 ]

                if config.rotary_dim > 0:
                    mixed_query, key_layer = inference_cuda_module.apply_rotary_pos_emb(
                        mixed_query,
                        key_layer,
                        config.rotary_dim,
                        0 if layer_past is None else layer_past[0].shape[-2],
                        num_attention_heads_per_partition,
                        config.rotate_half,
                        config.rotate_every_two)
                if layer_past is not None:
                    past_key, past_value = layer_past
                    key_layer = torch.cat((past_key.type_as(key_layer),
                                           key_layer),
                                          dim=-2)
                    value_layer = torch.cat((past_value.type_as(value_layer),
                                             value_layer),
                                            dim=-2)
                presents = (key_layer, value_layer)
                mixed_query = _transpose_for_scores(mixed_query, False, True)
                key_layer = _transpose_for_scores(
                    key_layer,
                    True,
                    True) / (norm_factor if config.scale_attention else 1.0)
                value_layer = _transpose_for_scores(value_layer, False, True)
                if layer_past is None:
                    attn_key_value = score_context_func(
                        mixed_query,
                        key_layer,
                        torch.empty(1),
                        ((1 - input_mask).half() *
                         minus_inf) if input_mask.dtype == torch.int64 else input_mask,
                        value_layer,
                        torch.empty(1),
                        num_attention_heads_per_partition,
                        (1 / norm_factor if config.scale_attention else 1.0),
                        (not unfused_mode),  # noqa: F821
                        config.triangular_masking,
                        config.local_attention,
                        config.window_size,
                        no_masking)
                else:
                    attn_key_value = score_context_func(
                        mixed_query,
                        (key_layer if unfused_mode else past_key.type_as(key_layer)),  # noqa: F821
                        key_layer,
                        ((1 - input_mask).half() *
                         minus_inf) if input_mask.dtype == torch.int64 else input_mask,
                        (value_layer
                         if unfused_mode else past_value.type_as(value_layer)),  # noqa: F821
                        value_layer,
                        num_attention_heads_per_partition,
                        (1 / norm_factor if config.scale_attention else 1.0),
                        (not unfused_mode),  # noqa: F821
                        config.triangular_masking,
                        config.local_attention,
                        config.window_size,
                        no_masking)
                if unfused_mode:  # noqa: F821
                    context_layer, _, _ = attn_key_value
                else:
                    context_layer, key_layer, value_layer = attn_key_value

                # Transpose Context
                context_layer = _transpose_for_context(context_layer)

                return context_layer, presents[0], presents[1] # atten_output, key_layer, value_layer
            else:
                # Note: This modification is added for the BLOOM-176B model and will be removed later!
                if config.bigscience_bloom:
                    context_layer, presents = backup_attention(qkv_out, layer_past, alibi, input_mask, norm_factor)
                    return context_layer, presents[0], presents[1] #key_layer, value_layer
                else:
                    if alibi is not None:
                        batch_heads = qkv_out.shape[0] * num_attention_heads_per_partition
                        offset = dist.get_rank() * batch_heads if dist.is_initialized(
                        ) else 0
                        sliced_alibi = alibi[offset:batch_heads + offset, :, :]


#
                    attn_key_value = score_context_func(
                        qkv_out,
                        ((1 - input_mask).to(qkv_out.dype) *
                         minus_inf) if input_mask.dtype == torch.int64 else input_mask,
                        config.rotary_dim,
                        config.rotate_half,
                        config.rotate_every_two,
                        num_attention_heads_per_partition,
                        (1 / norm_factor if config.scale_attention else 1.0),
                        config.triangular_masking,
                        config.local_attention,
                        config.window_size,
                        no_masking,
                        config.layer_id,
                        DeepSpeedTransformerInference.layer_id,
                        sliced_alibi if alibi is not None else torch.empty(1))
                    context_layer, key_layer, value_layer = attn_key_value
                    return context_layer, key_layer, value_layer

        def selfAttention_fp():
            vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if config.fp16 else \
                                    inference_cuda_module.vector_matmul_fp32

            if not config.pre_layer_norm:
                linear_func = inference_cuda_module.linear_layer_fp16 if config.fp16 else \
                                    inference_cuda_module.linear_layer_fp32
                qkv_out = linear_func(input,
                                      attn_qkvw,
                                      attn_qkvb,
                                      attn_qkvb is not None,
                                      False,
                                      False,
                                      num_attention_heads_per_partition,
                                      DeepSpeedTransformerInference.layer_id)
            else:
                qkv_func = inference_cuda_module.qkv_gemm_fp16 if config.fp16 else \
                                    inference_cuda_module.qkv_gemm_fp32
                qkv_out = qkv_func(input,
                                   attn_qkvw,
                                   attn_qkvw.scale,
                                   (attn_qkvb if attn_qkvb is not None else norm_b),
                                   norm_w,
                                   norm_b,
                                   config.epsilon,
                                   (attn_qkvb is not None),
                                   DeepSpeedTransformerInference.layer_id,
                                   config.bigscience_bloom,
                                   config.mp_size,
                                   dist.get_rank() if dist.is_initialized() else 0,
                                   config.q_int8)
            context_layer, key_layer, value_layer = compute_attention(qkv_out[0] if isinstance(qkv_out, list) else qkv_out, input_mask)
            output = vector_matmul_func(context_layer,
                                        attn_ow,
                                        False,
                                        attn_ow.scale,
                                        config.q_int8)

            return output, key_layer, value_layer, context_layer, qkv_out[-1]

        def selfAttention_int8():
            if not config.pre_layer_norm:
                qkv_out = inference_cuda_module.linear_layer_int8(
                    input,
                    attn_qkvw,
                    attn_qkvb,
                    q_scales[0],
                    (q_groups * (3 if qkv_merging else 1) * (2**merge_count)))

            else:
                qkv_out = inference_cuda_module.qkv_gemm_int8(
                    input,
                    attn_qkvw,
                    attn_qkvb,
                    norm_w,
                    norm_b,
                    config.epsilon,
                    q_scales[0],
                    (q_groups * (3 if qkv_merging else 1) * (2**merge_count)),
                    (attn_qkvb is not None))
            context_layer, key_layer, value_layer = compute_attention(qkv_out)
            output = inference_cuda_module.vector_matmul_int8(context_layer,
                                                              attn_ow,
                                                              q_scales[1],
                                                              q_groups,
                                                              (merge_count))
            return output, key_layer, value_layer, context_layer

        if False:  #config.q_int8:
            output, key_layer, value_layer, context_layer = selfAttention_int8()
        else:
            output, key_layer, value_layer, context_layer, inp_norm = selfAttention_fp()
        if config.mlp_after_attn and mp_group is not None and dist.get_world_size(
                group=mp_group) > 1:
            dist.all_reduce(output, group=mp_group)

        return (output, key_layer, value_layer, context_layer, inp_norm)

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2, grad_output3):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedSelfAttention(nn.Module):
    num_layers = 0

    def __init__(self,
                 config,
                 mp_group=None,
                 q_scales=None,
                 q_groups=1,
                 merge_count=1,
                 qkv_merging=False):
        super(DeepSpeedSelfAttention, self).__init__()
        self.config = config
        data_type = torch.int8 if config.q_int8 else torch.half if config.fp16 else torch.float
        data_type_fp = torch.half if config.fp16 else torch.float
        self.config.layer_id = DeepSpeedSelfAttention.num_layers
        DeepSpeedSelfAttention.num_layers = DeepSpeedSelfAttention.num_layers + 1
        device = torch.cuda.current_device() if config.bigscience_bloom else 'cpu'
        qkv_size_per_partition = (self.config.hidden_size // self.config.mp_size) * 3
        self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                  qkv_size_per_partition,
                                                  dtype=data_type,
                                                  device=device),
                                      requires_grad=False)
        self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition,
                                                  dtype=data_type_fp,
                                                  device=device),
                                      requires_grad=False)
        out_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)

        self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)

        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.mp_group = mp_group

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups
        self.merge_count = int(math.log2(merge_count))

        self.norm_factor = math.sqrt(
            math.sqrt(self.config.hidden_size // self.config.heads))
        self.qkv_merging = qkv_merging

        self.score_context_func = inference_cuda_module.softmax_context_fp32 if (not config.fp16) else \
                                    inference_cuda_module.softmax_context_fp16

    def forward(self,
                input,
                input_mask,
                head_mask=None,
                layer_past=None,
                get_present=False,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                norm_w=None,
                norm_b=None,
                alibi=None):
        output = DeepSpeedSelfAttentionFunction.apply(
            input,
            input_mask,
            head_mask,
            layer_past,
            get_present,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            norm_w,
            norm_b,
            self.config,
            self.attn_qkvw,
            self.attn_qkvb,
            self.num_attention_heads_per_partition,
            self.norm_factor,
            self.hidden_size_per_partition,
            self.attn_ow,
            self.attn_ob,
            self.mp_group,
            self.q_scales,
            self.q_groups,
            self.merge_count,
            self.qkv_merging,
            self.score_context_func,
            alibi)

        return output


class DeepSpeedMLPFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                residual,
                residual_norm,
                bias,
                inter_w,
                inter_b,
                attn_nw,
                attn_nb,
                config,
                mp_group,
                output_b,
                output_w,
                q_scales,
                q_groups,
                merge_count,
                mlp_gemm_func,
                fused_gemm_gelu,
                vector_matmul_func,
                bias_residual_func,
                residual_add_func,
                activation_func_type=ActivationFuncType.GELU):

        if attn_nw is None:
            output = fused_gemm_gelu(residual_norm,
                                     inter_w,
                                     inter_b,
                                     output_w,
                                     config.epsilon,
                                     config.pre_layer_norm,
                                     False)
        else:
            output, residual_add = mlp_gemm_func(input,
                                             residual,
                                             bias,
                                             inter_w,
                                             output_w,
                                             inter_b,
                                             attn_nw,
                                             attn_nb,
                                             config.epsilon,
                                             config.pre_layer_norm,
                                             config.mlp_after_attn,
                                             inter_w.scale,
                                             output_w.scale,
                                             config.q_int8,
                                             config.mlp_act_func_type)
        residual = residual if config.pre_layer_norm else residual_add
        residual_add_func(
            output,                # hidden state
            residual,              # residual
            input,                 # attention output
            bias if bias is not None else output_b,
            output_b,
            config.mp_size,         # model parallel size
            config.mlp_after_attn,  # whether mlp is after attention (GPTJ model architecture runs the MLP layer in parallel with attention)
            bias is not None,       # whether bias addition is fused
            config.pre_layer_norm)  # whether the layer norm is applied before attention
        if mp_group is not None and dist.get_world_size(group=mp_group) > 1:
            dist.all_reduce(residual, group=mp_group)
        return residual

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedMLP(nn.Module):
    def __init__(self,
                 config,
                 mp_group=None,
                 q_scales=None,
                 q_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):
        super(DeepSpeedMLP, self).__init__()

        self.config = config
        data_type = torch.int8 if config.q_int8 else torch.half if config.fp16 else torch.float
        data_type_fp = torch.half if config.fp16 else torch.float
        device = torch.cuda.current_device() if config.bigscience_bloom else 'cpu'
        self.attn_nw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
        self.attn_nb = nn.Parameter(torch.empty(self.config.hidden_size,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
        intm_size_per_partition = self.config.intermediate_size // self.config.mp_size
        self.inter_w = nn.Parameter(torch.empty(self.config.hidden_size,
                                                intm_size_per_partition,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.inter_b = nn.Parameter(torch.empty(intm_size_per_partition,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
        self.output_w = nn.Parameter(torch.empty(intm_size_per_partition,
                                                 self.config.hidden_size,
                                                 dtype=data_type,
                                                 device=device),
                                     requires_grad=False)
        self.output_b = nn.Parameter(torch.empty(self.config.hidden_size,
                                                 dtype=data_type_fp,
                                                 device=device),
                                     requires_grad=False)

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))

        self.mp_group = mp_group
        self.mlp_gemm_func = inference_cuda_module.mlp_gemm_fp16 if config.fp16 else \
                                    inference_cuda_module.mlp_gemm_fp32
        self.vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if config.fp16 else \
                                inference_cuda_module.vector_matmul_fp32
        self.fused_gemm_gelu = inference_cuda_module.fused_gemm_gelu_fp16 if config.fp16 else \
                                    inference_cuda_module.fused_gemm_gelu_fp32

        self.bias_residual_func = inference_cuda_module.bias_residual_fp16 if config.fp16 or config.q_int8 else \
                                    inference_cuda_module.bias_residual_fp32

        self.residual_add_func = inference_cuda_module.residual_add_bias_fp16 if config.fp16 or config.q_int8 else \
                                    inference_cuda_module.residual_add_bias_fp32

    def forward(self, input, residual, residual_norm, bias):
        return DeepSpeedMLPFunction.apply(input,
                                          residual,
                                          residual_norm,
                                          bias,
                                          self.inter_w,
                                          self.inter_b,
                                          self.attn_nw,
                                          self.attn_nb,
                                          self.config,
                                          self.mp_group,
                                          self.output_b,
                                          self.output_w,
                                          self.q_scales,
                                          self.q_groups,
                                          self.merge_count,
                                          self.mlp_gemm_func,
                                          self.fused_gemm_gelu,
                                          self.vector_matmul_func,
                                          self.bias_residual_func,
                                          self.residual_add_func)


class DeepSpeedTransformerInference(nn.Module):
    """Initialize the DeepSpeed Transformer Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepSpeedInferenceConfig
            mp_group: Model parallelism group initialized on the modeling side.
            quantize_scales: This argument groups all the layers' scales used for quantization
            quantize_groups: Number of groups used for quantizing the model
            merge_count: Shows the number of model-parallel checkpoints merged before running inference.
                We use this argument to control the quantization scale for the model parameters if a bigger
                quantize-grouping than 1 is used.
            mlp_extra_grouping: This flag is used to show a 2x higher number of groups used for the MLP part
                of a Transformer layer. We use this feature for quantization to reduce the convergence impact
                for specific downstream tasks.
    """
    layer_id = 0

    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False,
                 qkv_merging=False):
        super(DeepSpeedTransformerInference, self).__init__()

        self.config = config
        self.config.layer_id = DeepSpeedTransformerInference.layer_id
        DeepSpeedTransformerInference.layer_id += 1

        data_type = torch.half if config.fp16 else torch.float
        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()

        if DeepSpeedTransformerInference.layer_id == 1:
            log_dist(f"DeepSpeed-Inference config: {self.config.__dict__}", [0])

        self.attention = DeepSpeedSelfAttention(self.config,
                                                mp_group,
                                                quantize_scales,
                                                quantize_groups,
                                                merge_count,
                                                qkv_merging)
        self.mlp = DeepSpeedMLP(self.config,
                                mp_group,
                                quantize_scales,
                                quantize_groups,
                                merge_count,
                                mlp_extra_grouping)

        device = torch.cuda.current_device() if config.bigscience_bloom else 'cpu'
        self.norm_w = nn.Parameter(torch.empty(self.config.hidden_size,
                                               dtype=data_type,
                                               device=device),
                                   requires_grad=False)
        self.norm_b = nn.Parameter(torch.empty(self.config.hidden_size,
                                               dtype=data_type,
                                               device=device),
                                   requires_grad=False)
        self.layer_past = None
        self.allocate_workspace = inference_cuda_module.allocate_workspace_fp32 if (not config.fp16) else \
                                inference_cuda_module.allocate_workspace_fp16

    def forward(
            self,
            input,
            input_mask=None,
            attention_mask=None,
            head_mask=None,
            layer_past=None,
            get_key_value=False,
            get_present=False,
            encoder_output=None,
            enc_dec_attn_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            alibi=None,
            output_attentions=False,
            # TODO(arashb): 'layer_head_mask' and 'past_key_value' are only added to satisfy the OPT models API.
            # This needs to be redesigned later!
            layer_head_mask=None,
            past_key_value=None):
        # Allocate memory only on first layer forward
        if self.config.layer_id == 0:
            self.allocate_workspace(self.config.hidden_size,
                                    input.size()[0],
                                    input.size()[1],
                                    DeepSpeedTransformerInference.layer_id,
                                    self.config.heads,
                                    self.config.mp_size,
                                    self.config.bigscience_bloom,
                                    dist.get_rank() if dist.is_initialized() else 0)

        get_present = (get_present or get_key_value or use_cache)
        input_mask = input_mask if attention_mask is None else attention_mask

        # We set the prev key/value to None when there is a prompt
        if input.shape[1] > 1:
            self.layer_past = None
        layer_past = layer_past if layer_past is not None else self.layer_past
        head_mask = layer_head_mask if layer_head_mask is not None else head_mask

        attn_mask = None
        if isinstance(input, tuple):
            attn_mask = input[1]
            input = input[0]
        input_type = input.dtype

        if (self.config.fp16 or self.config.q_int8) \
            and input.dtype == torch.float:
            input = input.half()

        with torch.no_grad():
            attention_output, key, value, context_outputtn_ctx, inp_norm = \
                                     self.attention(input,
                                              input_mask,
                                              head_mask,
                                              layer_past,
                                              get_present,
                                              encoder_hidden_states,
                                              encoder_attention_mask,
                                              output_attentions,
                                              self.norm_w,
                                              self.norm_b,
                                              alibi)
            presents = (key, value)
            self.layer_past = presents if layer_past is None else None
            output = self.mlp(attention_output, input, inp_norm, self.attention.attn_ob)

            if not self.config.pre_layer_norm:
                ds_layernorm = inference_cuda_module.layer_norm_fp16 if self.config.fp16 or self.config.q_int8 else \
                                        inference_cuda_module.layer_norm_fp32
                output = ds_layernorm(output,
                                      self.norm_w,
                                      self.norm_b,
                                      self.config.epsilon)

            output = output.to(input_type)
        if get_present:
            output = (output, presents)

        if self.config.return_tuple:
            return output if type(output) is tuple else (output, attn_mask)
        else:
            return output
