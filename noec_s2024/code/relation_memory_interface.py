"""Interface shared by every relational-memory implementation (PyTorch backend).

Ported from the Paddle version (relation_memory_interface.py); only the backend
API changed (paddle.zeros -> torch.zeros, nn.Layer -> nn.Module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MemoryCapabilities:
    reads_x_from_g: bool = True
    reads_g_from_x_or_gx: bool = False
    supports_action_query: bool = False
    online_write: bool = True


@dataclass
class MemoryTrainingRead:
    x_from_g: Tensor
    retrieved_g: Tensor
    action_logits: Tensor
    action_mask: Tensor
    action_available: bool
    # Readouts that only a joint (m -> m_hat) memory can provide.  ``None`` means the
    # backend cannot produce it; the trainer then skips the corresponding loss instead
    # of silently booking a constant (the Hebbian backend's failure mode).
    g_from_x: Tensor | None = None        # L_x2g: structure from content
    g_corrected: Tensor | None = None     # query [g, x] -> g_hat, feeds error correction
    x_from_joint: Tensor | None = None    # query [g, x] -> x_hat, for sg(||x - x_hat||)


def zeros_memory_training_read(
    g_proj: Tensor,
    x_proj: Tensor,
    n_actions: int,
) -> MemoryTrainingRead:
    batch, seq_len = g_proj.shape[:2]
    action_logits = torch.zeros([batch, seq_len, n_actions], dtype=g_proj.dtype,
                                device=g_proj.device)
    action_mask = torch.zeros([batch, seq_len], dtype=g_proj.dtype,
                              device=g_proj.device)
    return MemoryTrainingRead(
        x_from_g=torch.zeros_like(x_proj),
        retrieved_g=torch.zeros_like(g_proj),
        action_logits=action_logits,
        action_mask=action_mask,
        action_available=False,
    )


class RelationalMemoryBase(nn.Module):
    capabilities = MemoryCapabilities()

    def read_x(self, g_proj: Tensor, x_proj: Tensor, state: Any) -> Tensor:
        raise NotImplementedError

    def write(
        self,
        g_proj: Tensor,
        x_proj: Tensor,
        state: Any,
        detach_mem_state: bool = False,
    ) -> Any:
        raise NotImplementedError

    def read_training(
        self,
        g_proj: Tensor,
        x_proj: Tensor,
        state: Any,
        g_prev_proj: Tensor | None,
        n_actions: int,
    ) -> MemoryTrainingRead:
        return zeros_memory_training_read(g_proj, x_proj, n_actions)
