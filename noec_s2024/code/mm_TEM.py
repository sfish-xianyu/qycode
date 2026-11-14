"""Top level mm-TEM model (PyTorch backend).

Ported from the Paddle version (mm_TEM.py). Math is unchanged; only the
backend API differs (nn.Module / sublayers -> modules / bias_attr -> bias /
paddle.tile -> repeat / paddle.concat(axis=) -> torch.cat(dim=)).
"""

import math
import os
from runtime_cfg import CFG

import torch
import torch.nn.functional as F
from torch import nn

from base_network import LinearEncDec
from grid_cell import GridModuleFast
from relation_memory_registry import build_relational_memory


def XAVIER(tensor):
    nn.init.xavier_uniform_(tensor)


def ZERO(tensor):
    nn.init.constant_(tensor, 0.0)


def ORTHOGONAL(tensor):
    nn.init.orthogonal_(tensor)


class mm_TEM(nn.Module):
    def __init__(self, para, prefix=""):
        super(mm_TEM, self).__init__()
        self.par = para
        self.prefix = prefix
        # GridModuleFast：先把整条动作序列映射成转移矩阵，再逐步递推。
        self.grid = GridModuleFast(self.par.n_actions, self.par.g_size)

        # [preset r17] 接线：把 --a2trans_init 真正作用到 a2trans 的最后一层。
        # 本包原先解析了 par.a2trans_init 却从未读取，ORTHOGONAL 只用在 projection_g 上，
        # 所以 --preset r17 会静默退化成 Xavier。这段与 复现代码/mm_TEM.py 一致。
        _a2init = str(getattr(self.par, "a2trans_init", "xavier")).lower()
        if _a2init in ("orthogonal", "orthogonal_scaled"):
            last = [m for m in self.grid.a2trans_layer.modules() if isinstance(m, nn.Linear)][-1]
            ORTHOGONAL(last.weight)
            if _a2init == "orthogonal_scaled":
                with torch.no_grad():
                    fan_in, fan_out = last.weight.shape[1], last.weight.shape[0]
                    target_std = (2.0 / (fan_in + fan_out)) ** 0.5
                    cur = last.weight.data.std().item()
                    if cur > 0:
                        last.weight.data.mul_(target_std / cur)
            print("[preset r17] a2trans_init=%s applied to a2trans_layer[-1] %s"
                  % (_a2init, tuple(last.weight.shape)), flush=True)

        self.visual_encdec = LinearEncDec(self.par.s_size, self.par.s_size_project)

        self.projection_g = nn.Linear(self.par.g_size, self.par.g_size_project, bias=False)
        ORTHOGONAL(self.projection_g.weight)

        self.p2g_mu = nn.Sequential(
            nn.Linear(self.par.g_size_project, 2 * self.par.g_size),
            nn.ELU(),
            nn.Linear(2 * self.par.g_size, self.par.g_size),
        )
        for module in self.p2g_mu.modules():
            if isinstance(module, nn.Linear):
                XAVIER(module.weight)
                if module.bias is not None:
                    ZERO(module.bias)

        self.action_pred_head = nn.Sequential(
            nn.Linear(self.par.g_size_project, 2 * self.par.g_size),
            nn.ELU(),
            nn.Linear(2 * self.par.g_size, self.par.n_actions),
        )
        for module in self.action_pred_head.modules():
            if isinstance(module, nn.Linear):
                XAVIER(module.weight)
                if module.bias is not None:
                    ZERO(module.bias)

        self.rel_memory_impl = getattr(self.par, "rel_memory_impl", "legacy")
        self.memory = build_relational_memory(
            self.par,
            p2g_mu=self.p2g_mu,
            action_pred_head=self.action_pred_head,
        )

        # ------------------------------------------------------------------
        # [EC] 路径积分纠错回路（Hippoformer A.2）:
        #   g_inf = g_gen + alpha * (g_hat - g_gen) * f_delta(g_gen, g_hat, ||x-x_hat||^2)
        # alpha 由 trainer 每 epoch 写入 self._ec_alpha（0 -> 1，ec_alpha_ramp 步）。
        # g_hat 由记忆检索出的内容码 x_hat 经 p2g_mu 映回结构空间（hebbian 后端
        # 只能 g->x 检索；p2g_mu 在本配置里原本是死模块，正好承担"内容->结构"）。
        # 关闭时（默认）前向与改动前逐比特一致。
        # ------------------------------------------------------------------
        self.ec_enabled = bool(getattr(self.par, "use_error_correction", False))
        self.ec_source = CFG.ec_source
        self._ec_alpha = 1.0
        if self.ec_enabled:
            gsz = int(self.par.g_size)
            self.ec_gain = nn.Sequential(
                nn.Linear(2 * gsz + 1, 2 * gsz),
                nn.ELU(),
                nn.Linear(2 * gsz, gsz),
            )
            XAVIER(self.ec_gain[0].weight); ZERO(self.ec_gain[0].bias)
            nn.init.normal_(self.ec_gain[2].weight, std=0.01)
            # 初始增益 g0 可调（EC_GAIN_INIT）；softplus(b) = g0 => b = log(e^g0 - 1)
            g0 = CFG.ec_gain_init
            nn.init.constant_(self.ec_gain[2].bias, math.log(max(math.expm1(g0), 1e-6)))
            print("[EC] 纠错回路启用：g_inf = g + alpha*(g_hat-g)*f_delta(.)，"
                  "锚来源=%s，初始增益=%.3f"
                  % (self.ec_source, g0), flush=True)

    # ------------------------------------------------------------------
    # 块级记忆读写（EC 与非 EC 共用）
    # 只抽"读/写"，不抽"生成"：两条路径的 gen_seq_g 调用粒度不同
    # （EC 每块一次、非 EC 整条一次），合并会改变 a2trans 的 GEMM 形状，
    # 从而破坏逐位可复现性。行为与原实现逐字等价。
    # ------------------------------------------------------------------
    def _read_block(self, g_proj_b, xp, state):
        """读出块内内容码；无记忆状态时返回全零（与旧实现一致）。"""
        if state is not None:
            return self.memory.read_x(g_proj_b, xp, state)
        return torch.zeros([xp.shape[0], xp.shape[1], self.par.s_size_project],
                           dtype=xp.dtype, device=xp.device)

    def _write_block(self, g_proj_b, xp, state, start, context):
        """块写回记忆（仅前 context 步），state 为 None 时由后端自行初始化。"""
        if start < context:
            return self.memory.write(g_proj_b, xp, state, detach_mem_state=False)
        return state

    def _forward_with_ec(self, act_seq, x_seq, last_g, cache, mem_step, context):
        """带纠错的路径积分（Hippoformer A.2）。

        与未纠错版的唯一差别：每 mem_step 步为一个块，块末用记忆检索出的
        内容码 x_hat 推出结构码 g_hat，再按

            g_inf = g_gen + alpha * (g_hat - g_gen) * f_delta(g_gen, g_hat, E)
            E     = ||x_t - x_hat_t||^2   (stop-grad)

        修正后作为下一块的初始状态；纠错后的码也与 x_t 一起写入记忆。
        块内的 g_gen 序列（用于预测损失）保持不变。
        """
        batch, seq_len, _ = act_seq.shape
        g_chunks, xgen_chunks = [], []
        g_state = last_g
        st = cache

        for start in range(0, seq_len, mem_step):
            act_c = act_seq[:, start:start + mem_step]
            xp = x_seq[:, start:start + mem_step]

            g_c = self.grid.gen_seq_g(g_state, act_c)        # (B,L,d) 生成
            gp_c = self.projection_g(g_c)

            x_hat = self._read_block(gp_c, xp, st)           # 记忆检索内容码

            # 只在块边界修正一次（论文的 t+mb 处）
            g_b = g_c[:, -1]
            alpha = float(self._ec_alpha)
            if alpha == 0.0:
                # alpha=0 表示不纠错：保持与未纠错版逐比特一致
                g_state = g_b
            else:
                # EC_SOURCE=obs: 用观测推出的结构码做锚（默认，收缩）
                # EC_SOURCE=ret: 用记忆检索码做锚（自指回路，会放大；仅作对照）
                if self.ec_source == "ret":
                    g_hat = self.p2g_mu(x_hat)               # 内容 -> 结构
                else:
                    g_hat = self.p2g_mu(xp)                  # 观测 -> 结构
                gh_b = g_hat[:, -1]
                err = ((xp[:, -1] - x_hat[:, -1]) ** 2).sum(-1, keepdim=True).detach()
                gain = F.softplus(self.ec_gain(torch.cat([g_b, gh_b, err], dim=-1)))
                g_inf = g_b + alpha * (gh_b - g_b) * gain
                g_state = F.normalize(F.relu(g_inf), dim=-1)

            # 纠错后的码写入记忆（论文：g_inf 与 x_t 一起存）
            gp_w = gp_c.clone()
            gp_w[:, -1] = self.projection_g(g_state)
            st = self._write_block(gp_w, xp, st, start, context)

            g_chunks.append(g_c)
            xgen_chunks.append(x_hat)

        return torch.cat(g_chunks, dim=1), torch.cat(xgen_chunks, dim=1), st

    def get_x2mem(self, x_proj, x_gen=None):
        """What is written into relational memory for one timestep.

        The hebbian memory stores the encoder projection x_proj directly, so this is
        the identity; a backend that wants a different input (e.g. a denoised
        reconstruction) can expose `memory.prepare_x(x_proj, x_gen)` to override it.
        (The Paddle version referenced this method from train_mm_TEM._get_mem_xproj
        without defining it; it was never hit because res always carries "x_projs".)
        """
        prepare = getattr(self.memory, "prepare_x", None)
        if callable(prepare):
            return prepare(x_proj, x_gen)
        return x_proj

    def forward(
        self,
        act_seq,
        sen_seq,
        last_g=None,
        cache=None,
        context=0,
        mem_batch=None,
        x_proj=None,
    ):
        batch, seq_len, _ = act_seq.shape
        if context == 0:
            context = seq_len + 10
        if x_proj is None:
            x_seq = self.visual_encdec(sen_seq)
        else:
            x_seq = x_proj
        if last_g is None:
            # paddle.tile(g0, repeat_times=[batch, 1]) -> torch repeat
            last_g = self.grid.g0.to(x_seq.device).repeat(batch, 1)

        x_gen_arr = []
        mem_step = mem_batch if mem_batch is not None else self.par.chunk_size

        if self.ec_enabled:
            # 纠错版：按 mb 步分块生成，块边界用记忆反馈修正结构码
            g_seq, x_gen_cat, cache = self._forward_with_ec(
                act_seq, x_seq, last_g, cache, mem_step, context)
            x_gen_arr.append(x_gen_cat)
        else:
            # 原版（未纠错）：整条序列一次生成
            g_seq = self.grid.gen_seq_g(last_g, act_seq)
            g_proj_l = self.projection_g(g_seq)
            for start in range(0, seq_len, mem_step):
                xp = x_seq[:, start:start + mem_step]
                g_proj_t = g_proj_l[:, start:start + mem_step]
                x_gen_pred = self._read_block(g_proj_t, xp, cache)
                cache = self._write_block(g_proj_t, xp, cache, start, context)
                x_gen_arr.append(x_gen_pred)

        g_proj = self.projection_g(g_seq)
        g0_proj = self.projection_g(last_g).unsqueeze(1)
        g_prev_proj = torch.cat((g0_proj, g_proj[:, :-1]), dim=1)

        res_dict = {}
        res_dict["g_gen"] = g_seq
        res_dict["g_proj"] = g_proj
        res_dict["x_gen"] = torch.cat(x_gen_arr, dim=1)
        res_dict["x_projs"] = x_seq

        act_pred = torch.zeros_like(act_seq)
        act_pred_available = False
        # ``retr_g`` only means something for backends that can read g back from
        # x/g (``capabilities.reads_g_from_x_or_gx``).  The hebbian_assoc backend
        # shipped in this package returns zeros, so ``l_mem`` (= the gs2g term)
        # degenerates to the constant mean(g_gen^2) and carries no information.
        # The flag lets the trainer report it as unavailable instead of printing a
        # fake loss value (this flag does not touch any tensor).
        caps = getattr(self.memory, "capabilities", None)
        retr_g_available = bool(getattr(caps, "reads_g_from_x_or_gx", False))

        if self.training and cache is not None:
            memory_read = self.memory.read_training(
                g_proj=g_proj,
                x_proj=x_seq,
                state=cache,
                g_prev_proj=g_prev_proj,
                n_actions=self.par.n_actions,
            )
            g2xp = memory_read.x_from_g
            retr_g = memory_read.retrieved_g
            if retr_g.shape[-1] != g_seq.shape[-1]:
                retr_g = torch.zeros_like(g_seq)
                retr_g_available = False
            act_pred = memory_read.action_logits
            act_pred_available = memory_read.action_available
        else:
            g2xp = torch.zeros_like(x_seq)
            retr_g = torch.zeros_like(g_seq)
            retr_g_available = False

        res_dict["retr_g"] = retr_g
        res_dict["retr_g_available"] = retr_g_available
        res_dict["g2xp_rel"] = g2xp
        res_dict["act_pred"] = act_pred
        res_dict["act_pred_available"] = act_pred_available

        return res_dict, cache
