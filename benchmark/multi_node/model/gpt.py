import copy
import math
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
from pynvml import *


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DatasetSimple(torch.utils.data.Dataset):
    def __init__(self, seq, d_model, size=100):
        self._size = size
        self._inputs = np.random.randn(size, seq, d_model)
        self._labels = np.random.randn(size, seq)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return (torch.tensor(self._inputs[idx], dtype=torch.float32),
                self._labels[idx].astype('float32'))


class DecoderLayerSimple(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x):
        "Follow Figure 1 (right) for connections."
        # We can not handle more than one input, so just generate a random one here
        m = torch.sqrt(1 - x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class GPT2Simple(nn.Module):
    def __init__(self, N=6, d_model=512, h=8, dropout=0.1):
        """ A simplified bert without embedding and language model heads """
        super().__init__()
        c = copy.deepcopy

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        layer = DecoderLayerSimple(d_model, c(attn), c(attn), c(ff), dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.reduce = lambda x: x.sum(-1)

    def forward(self, x):
        """
        Args:
            x (Tensor[batch size, sequence length, d_model]): input after encoder

        Returns:
            out ()
        """
        for layer in self.layers:
            x = layer(x)
        return self.reduce(self.norm(x))

    def join_layers(self):
        return [i for i in self.layers] + [self.norm, self.reduce]


def make_model(*args):
    return GPT2Simple(*args)


def test():
    device = 'cuda:0'

    d_model = 32
    n_layer = 4
    model = make_model(n_layer, d_model).to(device)

    batch_size = 4
    seq_len = 12
    x = torch.randn(batch_size, seq_len, d_model).to(device)

    out = model(x)
    print(f'Input shape: {x.shape}; \n'
          f'Output shape: {out.shape}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=10,
                        help='quit after this many steps')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--local_rank',
                        type=int,
                        default=None,
                        help='local rank passed from distributed launcher.')
    parser.add_argument('--dp',
                        type=int,
                        default=1,
                        help='size of data parallelism')
    parser.add_argument('--pp',
                        type=int,
                        default=4,
                        help='size of pipeline parallelism')
    parser.add_argument('--seed', type=int, default=7777, help='seed')

    # Model config args
    parser.add_argument('-N', type=int, default=24)
    parser.add_argument('--d-model', '-dm', type=int, default=1024)
    parser.add_argument('-H', type=int, default=16)
    parser.add_argument('--seq', type=int, default=512)
    parser.add_argument('--parts',
                        type=str,
                        default='',
                        help='Specify number of layers for each partition; separated by comma like `1,2,2,3`')
    parser.add_argument('--aci',
                        type=int,
                        default=1,
                        help='Activation checkpoint interval')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def init_dist(args):
    deepspeed.init_distributed(args.backend)
    data_parallel_size = args.dp
    pipe_parallel_size = args.pp
    custom_topology = PipeDataParallelTopology(
        pipe_parallel_size, data_parallel_size)

    return {'data_parallel_size': data_parallel_size,
            'pipe_parallel_size': pipe_parallel_size,
            'topo': custom_topology}


def gen_parts(args):
    parts = []
    if args.parts and args.parts != "-":
        parts = [int(p) for p in args.parts.split(',')]
        assert sum(parts) == args.N
        parts[-1] += 2
        parts = [0] + [sum(parts[:i]) + p for i, p in enumerate(parts)]

    return parts


def train():
    args = get_args()
    np.random.seed(args.seed)
    parts = gen_parts(args)
    dist_config = init_dist(args)
    layers = make_model(args.N, args.d_model, args.H).join_layers()
    model = PipelineModule(layers=layers,
                           loss_fn=nn.MSELoss(),
                           num_stages=dist_config['pipe_parallel_size'],
                           partition_method='type:DecoderLayerSimple' if len(
                               parts) == 0 else 'custom',
                           topology=dist_config['topo'],
                           activation_checkpoint_interval=args.aci)

    dataset = DatasetSimple(args.seq, args.d_model)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset)

    # Profiling phase
    while True:
        engine.train_batch()
        if not engine.energy_profiler.is_profiling:
            break

    # Reconfigure phase
    while True:
        engine.train_batch()
        if engine.execution_grid.finish_reconfigure():
            break

    device_count = nvmlDeviceGetCount()
    current_energy = [0] * device_count
    total_energy_consumption = [0] * device_count

    if args.local_rank == 0:
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            current_energy[i] = nvmlDeviceGetTotalEnergyConsumption(handle)

        start_time = time.time()

    for _ in range(args.steps):
        engine.train_batch()

    if args.local_rank == 0:
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            total_energy_consumption[i] = nvmlDeviceGetTotalEnergyConsumption(
                handle) - current_energy[i]

        throughput = (engine.train_batch_size() * args.steps) / \
            (time.time() - start_time)

        print("[RESULT]", round(throughput, 3), ",", round(sum(
            total_energy_consumption) / args.steps, 3))


if __name__ == '__main__':
    train()
