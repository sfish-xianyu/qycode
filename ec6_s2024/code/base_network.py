"""Encoder / decoder building blocks (PyTorch backend).

Ported from the Paddle version (base_network.py). Only the backend API
has changed: nn.Layer -> nn.Module, paddle.argmax(axis=) -> torch.argmax(dim=),
in-place initializers, and .astype(dtype) -> .float()/.to(dtype).
"""

import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


def xavier_(tensor):
    """In-place Xavier-uniform init (equivalent of pinit.XavierUniform())."""
    nn.init.xavier_uniform_(tensor)


def zero_(tensor):
    """In-place zero init (equivalent of pinit.Constant(0.0))."""
    nn.init.constant_(tensor, 0.0)


XAVIER = xavier_
ZERO = zero_


class LinearEncDec(nn.Module):
    def __init__(self, visual_dim, stoch_size):
        super(LinearEncDec, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(visual_dim, stoch_size),
        )

        self.dec = nn.Sequential(
            nn.Linear(stoch_size, visual_dim),
        )

        self.visual_dim = visual_dim
        self.stoch_size = stoch_size
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                XAVIER(m.weight)
                if m.bias is not None:
                    ZERO(m.bias)

    def forward(self, sense, sample=True):
        logits = self.enc(sense)
        return logits

    def compute_loss(self, sense, stoch_logits=None, return_dec=False, return_acc=True,
                     mask=None, dim=None, use_dec=True):
        if stoch_logits is None:
            stoch = self(sense)
        else:
            stoch = stoch_logits

        data_shape = sense.shape
        if use_dec:
            pred = self.dec(stoch)
        else:
            pred = stoch
        sense_label = torch.argmax(sense, dim=-1).reshape([-1])
        pred_flatten = pred.reshape([-1, sense.shape[-1]])
        loss = F.cross_entropy(pred_flatten, sense_label,
                               reduction="none" if mask is not None else "mean")
        pred_label = torch.argmax(pred, dim=-1).reshape([-1])

        if mask is not None:
            loss = loss.reshape(tuple(data_shape[:-1]))
            acc = (pred_label == sense_label).float().reshape(tuple(data_shape[:-1]))
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            acc = ((acc * mask).sum(dim=dim) + 1e-8) / (mask.sum(dim=dim) + 1e-8)
        else:
            acc = (pred_label == sense_label).float().mean()
        if return_acc:
            if return_dec:
                return loss, pred, acc
            return loss, acc
        else:
            if return_dec:
                return loss, pred
            return loss
