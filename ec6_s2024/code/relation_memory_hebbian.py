"""Hebbian / associative relational memory (PyTorch backend).

Ported from the Paddle version (relation_memory_hebbian.py). The forward
math is unchanged; only the backend API differs (paddle.einsum ->
torch.einsum, F.normalize(axis=) -> dim=, bias_attr -> bias, and the memory
state now lives on the same device as the parameters).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from relation_memory_interface import (
    MemoryCapabilities,
    MemoryTrainingRead,
    RelationalMemoryBase,
)


@dataclass
class HebbianMemoryState:
    assoc: Tensor


class HebbianRelationalMemory(RelationalMemoryBase):
    capabilities = MemoryCapabilities(
        reads_x_from_g=True,
        reads_g_from_x_or_gx=False,
        supports_action_query=False,
        online_write=True,
    )

    def __init__(
        self,
        g_dim: int,
        x_dim: int,
        hidden_dim: int,
        eta: float = 0.1,
        decay: float = 0.99,
        use_oja: bool = True,
    ):
        super().__init__()
        self.g_dim = g_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.eta = eta
        self.decay = decay
        self.use_oja = use_oja
        self.to_key = nn.Linear(g_dim, hidden_dim, bias=False)
        self.to_value = nn.Linear(x_dim, hidden_dim, bias=False)
        self.to_x = nn.Linear(hidden_dim, x_dim, bias=False)
        self.key_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.value_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def init_state(self, batch: int, device=None, dtype=None) -> HebbianMemoryState:
        if dtype is None:
            dtype = self.to_key.weight.dtype
        if device is None:
            device = self.to_key.weight.device
        assoc = torch.zeros([batch, self.hidden_dim, self.hidden_dim],
                            dtype=dtype, device=device)
        return HebbianMemoryState(assoc=assoc)

    def encode_key(self, g_proj: Tensor) -> Tensor:
        return F.normalize(self.key_norm(self.to_key(g_proj)), dim=-1)

    def encode_value(self, x_proj: Tensor) -> Tensor:
        return self.value_norm(self.to_value(x_proj))

    def read_x(self, g_proj: Tensor, x_proj: Tensor, state: HebbianMemoryState | None) -> Tensor:
        if state is None:
            return torch.zeros_like(x_proj)
        key = self.encode_key(g_proj)
        hidden = torch.einsum("bth,bhd->btd", key, state.assoc)
        return self.to_x(hidden)

    def write(
        self,
        g_proj: Tensor,
        x_proj: Tensor,
        state: HebbianMemoryState | None,
        detach_mem_state: bool = False,
    ) -> HebbianMemoryState:
        if state is None:
            state = self.init_state(g_proj.shape[0], dtype=g_proj.dtype)

        assoc = state.assoc
        keys = self.encode_key(g_proj)
        values = self.encode_value(x_proj)

        for index in range(keys.shape[1]):
            key = keys[:, index]
            value = values[:, index]
            prediction = torch.einsum("bh,bhd->bd", key, assoc)
            target = value - prediction if self.use_oja else value
            update = torch.einsum("bh,bd->bhd", key, target)
            assoc = self.decay * assoc + self.eta * update

        if detach_mem_state:
            assoc = assoc.detach()
        return HebbianMemoryState(assoc=assoc)

    def read_training(
        self,
        g_proj: Tensor,
        x_proj: Tensor,
        state: HebbianMemoryState | None,
        g_prev_proj: Tensor | None,
        n_actions: int,
    ) -> MemoryTrainingRead:
        batch, seq_len = g_proj.shape[:2]
        device = g_proj.device
        return MemoryTrainingRead(
            x_from_g=self.read_x(g_proj, x_proj, state),
            retrieved_g=torch.zeros([batch, seq_len, g_proj.shape[-1]],
                                    dtype=g_proj.dtype, device=device),
            action_logits=torch.zeros([batch, seq_len, n_actions],
                                      dtype=g_proj.dtype, device=device),
            action_mask=torch.zeros([batch, seq_len], dtype=g_proj.dtype, device=device),
            action_available=False,
        )
