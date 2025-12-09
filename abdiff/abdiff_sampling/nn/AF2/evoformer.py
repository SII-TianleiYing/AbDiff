# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
from typing import Optional

from training_config import config
from abdiff.abdiff_sampling.nn.AF2.primitives import Linear, LayerNorm
from abdiff.abdiff_sampling.nn.AF2.dropout import DropoutRowwise
from abdiff.abdiff_sampling.nn.AF2.pair_transition import PairTransition
from abdiff.abdiff_sampling.nn.AF2.triangular_attention import (
    TriangleAttention,
    TriangleAttentionEndingNode
)
from abdiff.abdiff_sampling.nn.AF2.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing
)

from diffusers.models.normalization import AdaLayerNorm


def add(m1, m2, inplace):
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if(not inplace):
        m1 = m1 + m2
    else:
        m1 += m2

    return m1

class OuterProduct(nn.Module):
    def __init__(self, c_s, c_z, c_hidden, eps=1e-3, norm_type='adanorm',) -> None:
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps
        self.norm_type = norm_type

        if norm_type == 'adanorm':
            self.layer_norm = AdaLayerNorm(embedding_dim=c_s, num_embeddings=config.num_train_timesteps)
        else:
            self.layer_norm = nn.LayerNorm(c_s)
        self.linear_1 = Linear(c_s, c_hidden)
        self.linear_2 = Linear(c_s, c_hidden)
        self.linear_out = Linear(c_hidden, c_z, init="final")

    def forward(self, s, timestep):
        # [*, N_res, C_m]
        if self.norm_type == 'adanorm':
            ln = self.layer_norm(s, timestep)
        else:
            ln = self.layer_norm(s)

        # [*, N_res, c_hidden]
        a = self.linear_1(ln)
        b = self.linear_2(ln)

        # [*, N_res, N_res, c_hidden]
        outer = a[..., None, :] * b[..., None, :, :]

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

class PairStack(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_s: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        fuse_projection_weights: bool,
        inf: float,
        norm_type='adanorm',
    ):
        super(PairStack, self).__init__()

        self.norm_type = norm_type
        self.out_prod = OuterProduct(c_s=c_s, c_z=c_z, c_hidden=c_z, norm_type=norm_type)

        if fuse_projection_weights:
            self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = FusedTriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(self,
        z: torch.Tensor,
        s: torch.Tensor,
        timestep,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        pair_trans_mask = pair_mask if _mask_trans else None

        opm = self.out_prod(s, timestep)
        z = z + opm

        if (_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        tmu_update = self.tri_mul_out(
            z,
            timestep,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            timestep,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_start(
                        z,
                        timestep,
                        mask=pair_mask,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_end(
                        z,
                        timestep,
                        mask=pair_mask,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.pair_transition(
                    z, mask=pair_trans_mask, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
        )

        return z
