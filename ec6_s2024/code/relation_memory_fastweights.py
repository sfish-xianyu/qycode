"""Relational memory as a meta-MLP with hierarchical fast weights (the paper's design).

Replaces the Hebbian outer-product store of ``relation_memory_hebbian`` with what
mm-TEM actually uses (ICLR 2026 submission, Sec. 3 "Relational Memory Network"):

    m_t = [g_t ; x_t]                      joint structural / sensory code
    k_t, v_t, q_t = W_k m_t, W_v m_t, W_q m_t                     (Eq. 3)
    L(k, v; Theta) = || f_MLP(k; Theta) - v ||^2                  (Eq. 4)
    H_t = eta_t H_{t-1} - beta_t grad_Theta L(k_t, v_t; H_{t-1})  (Eq. 6)
    Theta_t = (1 - alpha_t) Theta_{t-1} + H_t                     (Eq. 5)
    alpha_t, eta_t, beta_t = sigmoid(W_{a,h,b} m_t)

Retrieval is a *joint* reconstruction: f_MLP(q; Theta) -> m_hat = [g_hat ; x_hat], so a
query may carry only the structure (m=[g,0]), only the content (m=[0,x]) or both
(m=[g,x]).  That is what makes the paper's four readouts -- L_gen, L_x2g, L_g2g and the
error-correction query -- the *same* map with different input masks.  The Hebbian
backend this package shipped with is asymmetric (g -> x only) and therefore cannot
express three of them at all.

Deliberate deviations from the paper, both documented in the run report:
  * The inner-loop update is not differentiated through (``Theta`` is detached between
    blocks).  The paper only says Theta is optimised "in the inner loop"; Titans -- the
    stated inspiration -- also treats the surprise update as a detached state update.
    Backpropagating through it would be a second-order meta-gradient and roughly an
    order of magnitude slower per step, which does not fit one 3090.
  * ``n_heads`` is 2 x 64 by default, as in the paper; set it to 1 x d_m for a
    block-diagonal variant with the same parameter count.
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
class FastWeightState:
    """Per-batch fast weights and their momentum buffers."""

    weights: tuple[Tensor, ...]
    momentum: tuple[Tensor, ...]
    n_updates: int = 0


def _split(names):
    return names


class FastWeightRelationalMemory(RelationalMemoryBase):
    capabilities = MemoryCapabilities(
        reads_x_from_g=True,
        reads_g_from_x_or_gx=True,
        supports_action_query=False,
        online_write=True,
    )

    def __init__(
        self,
        g_dim: int,
        x_dim: int,
        hidden_dim: int | None = None,   # per-head width; defaults to m_dim / n_heads
        n_heads: int = 2,
        n_layers: int = 3,
        eta_init: float = 0.9,      # bias of the sigmoid momentum gate
        beta_init: float = 0.1,     # bias of the sigmoid learning-rate gate
        alpha_init: float = 0.1,    # bias of the sigmoid forgetting gate
        gate_init: float = -4.0,    # kept for API symmetry; unused
    ):
        super().__init__()
        self.g_dim = g_dim
        self.x_dim = x_dim
        self.m_dim = g_dim + x_dim
        self.n_heads = n_heads
        self.d_head = hidden_dim if hidden_dim else self.m_dim // n_heads
        assert self.d_head * n_heads == self.m_dim, (
            "the head split must cover the joint code exactly: "
            f"{n_heads} x {self.d_head} != {self.m_dim}")
        assert n_layers == 3, "the paper's meta-MLP is 3 layers"
        self.n_layers = n_layers
        out_dim = n_heads * self.d_head
        # The joint reconstruction lives in R^{m_dim}; a head output of
        # n_heads*hidden_dim is projected back with a *slow* (outer-loop) matrix.
        self.to_m = nn.Linear(out_dim, self.m_dim)

        self.W_k = nn.Linear(self.m_dim, self.m_dim, bias=False)
        self.W_v = nn.Linear(self.m_dim, self.m_dim, bias=False)
        self.W_q = nn.Linear(self.m_dim, self.m_dim, bias=False)
        self.W_alpha = nn.Linear(self.m_dim, 1)
        self.W_eta = nn.Linear(self.m_dim, 1)
        self.W_beta = nn.Linear(self.m_dim, 1)
        for lin, bias in ((self.W_alpha, alpha_init), (self.W_eta, eta_init),
                          (self.W_beta, beta_init)):
            # Small random input weights make the gates *data-dependent* (they vary with
            # m_t, as in the paper) even though they receive no gradient: with the inner
            # loop detached, their only route to the loss is through Theta, and that
            # route is multiplied by (1-alpha)^n_blocks ~ 0.5^32.  Documented deviation.
            nn.init.normal_(lin.weight, 0.0, 0.05)
            nn.init.constant_(lin.bias, bias)
        self._gate_init = gate_init

        # Theta_0: the slow ("meta") initialisation of the fast weights.  The inner loop
        # moves away from it; the outer loop learns where to start from.
        self.theta0 = nn.ParameterDict()
        for name, shape in self._weight_shapes():
            self.theta0[name] = nn.Parameter(torch.zeros(shape))
        with torch.no_grad():
            for name, shape in self._weight_shapes():
                if name.startswith("b"):
                    continue
                fan_in = shape[-1]
                self.theta0[name].normal_(0.0, (1.0 / fan_in) ** 0.5)
                if name == "W3":
                    self.theta0[name].mul_(0.1)   # start near the identity-ish map

    # ------------------------------------------------------------------ shapes
    def _weight_shapes(self):
        d, h = self.d_head, self.n_heads
        return [("W1", (h, d, d)), ("b1", (h, d)),
                ("W2", (h, d, d)), ("b2", (h, d)),
                ("W3", (h, d, d)), ("b3", (h, d))]

    def _names(self):
        return [n for n, _ in self._weight_shapes()]

    # ------------------------------------------------------------------- state
    def init_state(self, batch: int, device=None, dtype=None) -> FastWeightState:
        first = self.theta0["W1"]
        device = device if device is not None else first.device
        dtype = dtype if dtype is not None else first.dtype
        names = self._names()
        weights = tuple(self.theta0[n].to(device=device, dtype=dtype).unsqueeze(0)
                        .expand(batch, *self.theta0[n].shape).clone() for n in names)
        momentum = tuple(torch.zeros_like(w) for w in weights)
        return FastWeightState(weights=weights, momentum=momentum)

    # ------------------------------------------------------- functional forward
    @staticmethod
    def _fwd(params, q):
        """f_MLP(q; Theta) with an explicit weight tuple, one head axis.

        q: [B, ...] x d_head-space already split per head: [B, T, H, d]
        """
        W1, b1, W2, b2, W3, b3 = params
        h = F.gelu(torch.einsum("bthd,bhde->bthe", q, W1) + b1[:, None, :, :])
        h = F.gelu(torch.einsum("bthe,bhef->bthf", h, W2) + b2[:, None, :, :])
        out = torch.einsum("bthf,bhfg->bthg", h, W3) + b3[:, None, :, :]
        return out

    def _project(self, m: Tensor, state: FastWeightState) -> Tensor:
        """W_q m followed by the head split: [B,T,m_dim] -> [B,T,H,d_head]."""
        q = self.W_q(m)
        b, t, _ = q.shape
        return q.reshape(b, t, self.n_heads, -1)

    # ------------------------------------------------------------------- query
    def query(self, state: FastWeightState, g: Tensor | None, x: Tensor | None):
        """Retrieve the joint code for a masked query.

        ``g`` / ``x`` are optional; a missing part is replaced by zeros, which is
        exactly the paper's "m = [g, 0]" / "m = [0, x]" masking.

        Returns (g_hat, x_hat), each [B, T, *_dim].
        """
        ref = g if g is not None else x
        b, t = ref.shape[0], ref.shape[1]
        if g is None:
            g = ref.new_zeros(b, t, self.g_dim)
        if x is None:
            x = ref.new_zeros(b, t, self.x_dim)
        m = torch.cat([g, x], dim=-1)
        q = self._project(m, state)
        out = self._fwd(state.weights, q)                       # [B,T,H,d]
        out = out.reshape(b, t, self.n_heads * self.d_head)
        m_hat = self.to_m(out)                                   # [B,T,m_dim]
        return m_hat[..., : self.g_dim], m_hat[..., self.g_dim:]

    # ------------------------------------------------------------------- write
    def write_block(self, state: FastWeightState, g: Tensor, x: Tensor,
                    n_blocks_between_updates: int = 1):
        """One Titans update over the block's pairs (Eq. 5-6).

        g, x: [B, T, dim] -- T is the block that has accumulated since the last update
        (T == m_b in the paper).  ``n_blocks_between_updates`` is unused here; the
        trainer calls this once per m_b steps, which *is* the update frequency.

        Returns (new_state, l_write).  l_write is Eq. 4 evaluated with the *current*
        fast weights -- it is the only path by which W_k / W_v receive gradient, because
        the adaptive step itself is detached (see the module docstring).  The trainer
        adds it to the relational term.
        """
        m = torch.cat([g, x], dim=-1)                            # [B,T,m_dim]
        k = self.W_k(m)
        v = self.W_v(m)
        b, t, _ = k.shape
        Hd = self.n_heads
        dh = self.d_head
        k = k.reshape(b, t, Hd, dh)
        v = v.reshape(b, t, Hd, dh)
        v_flat = v.reshape(b, t, Hd * dh)

        def sample_loss(params_i, k_i, v_i):
            out = self._fwd(tuple(p.unsqueeze(0) for p in params_i),
                            k_i.unsqueeze(0)).squeeze(0)         # [T,H,dh]
            out = out.reshape(t, Hd * dh)
            return ((out - v_i) ** 2).sum(-1).mean()

        params = state.weights
        # per-sample gradient of the block reconstruction error (Eq. 4)
        grads = torch.func.vmap(
            torch.func.grad(sample_loss), in_dims=(0, 0, 0)
        )(params, k, v_flat)
        # Eq. 4 itself, batched and differentiable, so that W_k / W_v are meta-trained
        out_all = self._fwd(params, k).reshape(b, t, Hd * dh)
        l_write = ((out_all - v_flat) ** 2).sum(-1).mean()

        m_last = m[:, -1]
        # Raw gates are [B, 1]; they are broadcast per-tensor below, so a weight of
        # [B, H, d, d] and a bias of [B, H, d] both get the gate on their first axis.
        a_raw = torch.sigmoid(self.W_alpha(m_last))
        e_raw = torch.sigmoid(self.W_eta(m_last))
        b_raw = torch.sigmoid(self.W_beta(m_last))

        def _bc(t, ndim):
            return t.reshape(t.shape[0], *([1] * (ndim - 1)))

        new_mom, new_w = [], []
        for w, hprev, gr in zip(params, state.momentum, grads):
            a = _bc(a_raw, w.dim())
            e = _bc(e_raw, w.dim())
            be = _bc(b_raw, w.dim())
            h = e * hprev - be * gr
            w = (1.0 - a) * w + h
            new_mom.append(h.detach())
            new_w.append(w.detach())
        return (FastWeightState(weights=tuple(new_w), momentum=tuple(new_mom),
                                n_updates=state.n_updates + 1), l_write)

    def write(self, g_proj: Tensor, x_proj: Tensor, state, detach_mem_state: bool = False):
        if state is None:
            state = self.init_state(g_proj.shape[0], device=g_proj.device,
                                    dtype=g_proj.dtype)
        state, _ = self.write_block(state, g_proj, x_proj)
        return state

    # ------------------------------------------------------------------ legacy
    def read_x(self, g_proj: Tensor, x_proj: Tensor, state) -> Tensor:
        if state is None:
            return torch.zeros_like(x_proj)
        _, x_hat = self.query(state, g_proj, None)
        return x_hat

    def read_training(self, g_proj, x_proj, state, g_prev_proj, n_actions):
        """Readouts used by the trainer when the block loop is *not* driving the model.

        The error-correction path fills these in itself (it needs them per block).
        """
        if state is None:
            return MemoryTrainingRead(
                x_from_g=torch.zeros_like(x_proj),
                retrieved_g=torch.zeros_like(g_proj),
                action_logits=g_proj.new_zeros(g_proj.shape[0], g_proj.shape[1], n_actions),
                action_mask=g_proj.new_zeros(g_proj.shape[0], g_proj.shape[1]),
                action_available=False,
            )
        g_hat_x, x_hat_g = self.query(state, g_proj, None)        # [g,0] -> (g_hat, x_hat)
        g_hat_x2g, _ = self.query(state, None, x_proj)            # [0,x] -> (g_hat, x_hat)
        return MemoryTrainingRead(
            x_from_g=x_hat_g,
            retrieved_g=g_hat_x,
            g_from_x=g_hat_x2g,
            action_logits=g_proj.new_zeros(g_proj.shape[0], g_proj.shape[1], n_actions),
            action_mask=g_proj.new_zeros(g_proj.shape[0], g_proj.shape[1]),
            action_available=False,
        )
