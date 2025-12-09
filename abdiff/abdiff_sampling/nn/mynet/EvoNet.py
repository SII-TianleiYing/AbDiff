from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import deprecate, is_torch_version, logging
from diffusers.utils.outputs import BaseOutput
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import ImagePositionalEmbeddings, PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle

from abdiff.abdiff_sampling.nn.idpsam.common import AF2_PositionalEmbedding
from abdiff.abdiff_sampling.nn.AF2.evoformer import PairStack


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class EvoNetOutput(BaseOutput):
    """
    The output of [`EvoNet`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, height, width, num_channels)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.Tensor"  # noqa: F821

class EvoNet(ModelMixin, ConfigMixin):
    """
    Evoformer like diffusion model dealing with 2D pair repr.

    Args:
        in_channels (int, optional): The number of channels in input pair repr. Defaults to 128.
        inner_dim (int, optional): The number of channels inner PairStack. Defaults to 128.
        out_channels (int, optional): The number of channels in output pair repr. Defaults to 128.
        hidden_states_dim (int, optional): The number of channels in single repr. Defaults to 384.
        num_layers (int, optional): The number of PairStack layers. Defaults to 1.
    """
    @register_to_config
    def __init__(self,
                in_channels=128,
                inner_dim=128,
                out_channels=128,
                hidden_states_dim=384,
                norm_num_groups=32,
                num_layers=1,):
        super().__init__()
        self.in_channels = in_channels
        self.inner_dim=inner_dim
        self.out_channels=out_channels
        self.hidden_states_dim=hidden_states_dim
        self.norm_num_groups=norm_num_groups
        self.num_layers=num_layers

        self.pos_embed = AF2_PositionalEmbedding(pos_embed_dim=128, dim_order="trajectory")
        self.norm = torch.nn.LayerNorm(in_channels)
        self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                PairStack(
                    c_z=in_channels,
                    c_s=hidden_states_dim,
                    c_hidden_mul=in_channels,
                    c_hidden_pair_att=32,
                    no_heads_pair=4,
                    transition_n=2,
                    pair_dropout=0.25,
                    fuse_projection_weights=False,
                    inf=1e9,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        
        # 1. Input
        residual = hidden_states
        # add pos emb
        pos_emb = self.pos_embed(encoder_hidden_states)
        # pos_emb = pos_emb.permute(0, 3, 1, 2)
        hidden_states = hidden_states + pos_emb
        
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                s=encoder_hidden_states,
                timestep=timestep[0] if len(timestep.shape) == 1 else timestep,
                pair_mask=None,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return EvoNetOutput(sample=output)
    
    def get_time_emb(self, timestep):
        pass

model = EvoNet(
    num_layers=6
)
