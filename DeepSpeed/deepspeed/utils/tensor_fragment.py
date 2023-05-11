"""
Copyright 2022 The Microsoft DeepSpeed Team
"""

import torch
from dataclasses import dataclass
from deepspeed import comm as dist


@dataclass
class fragment_address:
    numel: int
    start: int


@dataclass
class tensor_fragment:
    lp_fragment: torch.Tensor
    lp_fragment_address: fragment_address
    hp_fragment: torch.Tensor
    hp_fragment_address: fragment_address
    optim_fragment: {}

    def update_hp(self):
        self.hp_fragment.data.copy_(self.lp_fragment.data)

    def update_lp(self):
        self.lp_fragment.data.copy_(self.hp_fragment.data)

    def get_optim_state_fragment(self, key):
        if key in self.optim_fragment:
            return self.optim_fragment[key]
        else:
            raise ValueError(f'{key} not found in optimizer state fragment')

    def get_hp_fragment_address(self):
        return self.hp_fragment_address

    def get_optim_state_keys(self):
        return list(self.optim_fragment.keys())


def get_full_hp_param(self, optim_state_key=None):
    reduce_buffer = torch.zeros_like(self, dtype=torch.float32).flatten()
    if self._hp_mapping is not None:
        lp_frag_address = self._hp_mapping.lp_fragment_address
        reduce_fragment = torch.narrow(reduce_buffer,
                                       0,
                                       lp_frag_address.start,
                                       lp_frag_address.numel)
        if optim_state_key is None:
            hp_fragment = self._hp_mapping.hp_fragment
        else:
            hp_fragment = self._hp_mapping.get_optim_state_fragment(optim_state_key)

        reduce_fragment.data.copy_(hp_fragment.data)
    dist.all_reduce(reduce_buffer, group=self._dp_group)
    return reduce_buffer.reshape_as(self)


def get_hp_fragment_mapping(lp_param,
                            lp_start,
                            flat_hp_partition,
                            partition_start,
                            partition_size,
                            optimizer_state_dict):
    lp_end = lp_param.numel() + lp_start
    hp_start = partition_start
    hp_end = partition_start + partition_size

    fragment_start = max(lp_start, hp_start)
    fragment_end = min(lp_end, hp_end)
    #            print(
    #                f'{self.dp_rank=} {lp_start=} {lp_end-lp_start=} {hp_start=} {hp_end-hp_start=} {fragment_start=} {fragment_end-fragment_start=}'
    #            )
    assert fragment_start < fragment_end, \
        f'fragment start {fragment_start} should be < fragment_end {fragment_end}'

    fragment_numel = fragment_end - fragment_start
    hp_frag_address = fragment_address(start=fragment_start - hp_start,
                                       numel=fragment_numel)
    hp_fragment_tensor = flat_hp_partition.narrow(0,
                                                  hp_frag_address.start,
                                                  hp_frag_address.numel)

    optim_fragment = {
        key: value.narrow(0,
                          hp_frag_address.start,
                          hp_frag_address.numel)
        for key,
        value in optimizer_state_dict.items()
        if torch.is_tensor(value) and value.dim() > 0
    }

    lp_frag_address = fragment_address(start=fragment_start - lp_start,
                                       numel=fragment_numel)
    lp_fragment_tensor = lp_param.flatten().narrow(0,
                                                   lp_frag_address.start,
                                                   lp_frag_address.numel)

    return tensor_fragment(lp_fragment=lp_fragment_tensor,
                           lp_fragment_address=lp_frag_address,
                           hp_fragment=hp_fragment_tensor,
                           hp_fragment_address=hp_frag_address,
                           optim_fragment=optim_fragment)
