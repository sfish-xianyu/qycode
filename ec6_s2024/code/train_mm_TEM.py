import os
from runtime_cfg import CFG
import sys
import time
import argparse
import random
import pdb

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
from collections import OrderedDict
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass
try:
    from r17_preset import (apply_preset, banner as r17_banner,
                            check_env as r17_check_env, report_fingerprint)
except Exception:  # 只拷走 train_mm_TEM.py 时仍能单独运行

    def apply_preset(argv):
        return list(argv)

    def r17_banner():
        pass

    def r17_check_env(allow=False):
        return True

    def report_fingerprint(epoch, loss):
        pass

from envs.data_utils import *
from mm_TEM import mm_TEM
from grid_cell import draw_grid, capacity_loss
from grid_losses import (grid_isotropy_loss, grid_isotropy_loss_v2, resolve_widths,
                         format_detail as format_iso_detail)
# from config.titan_params import default_params, get_env_params
from config.titan_params_large import default_params, get_env_params
from utils import DotDict, snapshot_run_sources, ensure_run_layout, infer_run_root, validate_args

USE_SINGLE_ENV = False
WIDTH, HEIGHT = None, None

def get_device():
    return _DEFAULT_DEVICE


def to_tensor(x, dtype=torch.float32):
    """numpy / list / tensor -> torch tensor on the active device (like paddle.to_tensor)."""
    if isinstance(x, torch.Tensor):
        return x.to(device=_DEFAULT_DEVICE, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=_DEFAULT_DEVICE)


def scalar_zero(ref=None, dtype=torch.float32):
    """0-dim tensor on the same device as `ref` (paddle zeros([]) for grad accumulation)."""
    device = ref.device if isinstance(ref, torch.Tensor) else _DEFAULT_DEVICE
    return torch.zeros((), dtype=dtype, device=device)


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def sync_env_config(par, width=None, height=None):
    par.env = get_env_params(par, width=width, height=height)
    par.n_actions = par.env.n_actions
    # WALK_VEL=cont: the continuous-velocity arm feeds a 2-D velocity vector instead
    # of an n_actions-wide one-hot.  GridModuleFast builds action_to_features as
    # Linear(n_actions, act_hidden), so overriding here is enough to switch the
    # model's action input dimension; nothing else in the architecture changes.
    if CFG.walk_vel_cont:
        par.n_actions = 2
    return par


def expand_schedule_arg(values, expected_len, arg_name):
    values = list(values)
    if len(values) == 1 and expected_len > 1:
        return values * expected_len
    if len(values) != expected_len:
        raise ValueError(f"{arg_name} must have length 1 or match --train_seq_bucket")
    return values


def tensor_grad_stats(tensor):
    grad = getattr(tensor, "grad", None)
    if grad is None:
        return None
    grad = grad.detach()
    finite = torch.isfinite(grad)
    finite_ratio = finite.float().mean().item() if int(grad.numel()) > 0 else 1.0
    if finite.any():
        finite_grad = grad[finite]
        return {
            "norm": grad.norm().item(),
            "mean_abs": finite_grad.abs().mean().item(),
            "max_abs": finite_grad.abs().max().item(),
            "finite_ratio": finite_ratio,
        }
    return {
        "norm": float("nan"),
        "mean_abs": float("nan"),
        "max_abs": float("nan"),
        "finite_ratio": finite_ratio,
    }


def retain_tensor_grad(tensors):
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            tensor.retain_grad()


class Trainer:
    def __init__(self, prefix="", only_eval=False, par=None, legacy_eval=False, run_layout=None):
        self.par = default_params(width=WIDTH, height=HEIGHT) if par is None else par
        sync_env_config(self.par, width=WIDTH, height=HEIGHT)

        eval_width = max(self.par.env.widths)
        eval_height = max(self.par.env.heights)
        self.eval_par = default_params(width=eval_width, height=eval_height)
        sync_env_config(self.eval_par, width=eval_width, height=eval_height)
        self.eval_use_legacy_batch = legacy_eval

        self.device = _set_torch_device(self.par.device)
        self.dynamic = mm_TEM(self.par, prefix)
        self.dynamic.to(self.device)
        self.grad_debug = bool(getattr(self.par, "grad_debug", False))
        self.grad_debug_interval = max(1, int(getattr(self.par, "grad_debug_interval", 20)))
        self.grad_debug_param_topk = max(0, int(getattr(self.par, "grad_debug_param_topk", 8)))
        if hasattr(self.dynamic.memory, "debug_enabled"):
            self.dynamic.memory.debug_enabled = self.grad_debug

        self.opt_train = torch.optim.Adam(self.dynamic.parameters(), lr=self.par.lr_train)

        # paddle's StepDecay(step_size=500, gamma=0.9) decays the LR every 500 scheduler
        # steps (one per epoch here); torch.optim.lr_scheduler.StepLR is the equivalent.
        # NOTE: in torch the scheduler must be stepped *after* optimizer.step().
        # LR_STEP_SIZE lets a long run keep the *shape* of the schedule instead of
        # decaying to numerically zero: at 2e5 epochs the default 500 would decay 400
        # times (1e-3 * 0.9^400 ~ 5e-22) and training would freeze within a few thousand
        # epochs.  Set LR_STEP_SIZE = total_epochs / 4 to reproduce the original
        # four-decay schedule stretched over the longer horizon.
        _lr_step = CFG.lr_step_size
        self.schedule_train = torch.optim.lr_scheduler.StepLR(self.opt_train, step_size=_lr_step, gamma=0.9)
        
        self.prefix = prefix
        self.run_layout = ensure_run_layout(prefix) if run_layout is None else run_layout
        for path in self.run_layout.values():
            os.makedirs(path, exist_ok=True)

        self.run_dir = self.run_layout["root"]
        self.train_dir = self.run_layout["train"]        
        self.train_log_dir = self.run_layout["logs_train"]
        self.code_dir = self.run_layout["code"]
        
        if not only_eval:
            self.train_writer = SummaryWriter(log_dir=self.train_log_dir)
        else:
            self.train_writer = None

    def _prepare_grad_debug(self, res):
        if not self.grad_debug:
            return
        retain_tensor_grad(
            [
                res.get("x_gen"),
                res.get("g2xp_rel"),
                res.get("retr_g"),
                res.get("g_gen"),
                res.get("g_proj"),
                res.get("x_projs"),
                res.get("act_pred"),
            ]
        )

    def _write_grad_stats(self, prefix, name, tensor, step, lines):
        stats = tensor_grad_stats(tensor)
        if stats is None:
            lines.append(f"{name}: grad=None")
            self.train_writer.add_scalar(f"grad/{prefix}/{name}/has_grad", 0.0, step)
            return

        safe_name = name.replace(".", "/")
        self.train_writer.add_scalar(f"grad/{prefix}/{safe_name}/norm", stats["norm"], step)
        self.train_writer.add_scalar(f"grad/{prefix}/{safe_name}/mean_abs", stats["mean_abs"], step)
        self.train_writer.add_scalar(f"grad/{prefix}/{safe_name}/max_abs", stats["max_abs"], step)
        self.train_writer.add_scalar(f"grad/{prefix}/{safe_name}/finite_ratio", stats["finite_ratio"], step)
        self.train_writer.add_scalar(f"grad/{prefix}/{safe_name}/has_grad", 1.0, step)
        lines.append(
            f"{name}: norm={stats['norm']:.3e}, mean_abs={stats['mean_abs']:.3e}, "
            f"max_abs={stats['max_abs']:.3e}, finite={stats['finite_ratio']:.3f}"
        )

    def _log_grad_debug(self, res, step, total_norm_before_clip):
        if not self.grad_debug or step % self.grad_debug_interval != 0:
            return

        self.train_writer.add_scalar("grad/total_norm_before_clip", float(total_norm_before_clip), step)
        lines = [f"[grad_debug] epoch={step} total_norm_before_clip={float(total_norm_before_clip):.3e}"]

        tensor_items = [
            ("x_gen", res.get("x_gen")),
            ("g2xp_rel", res.get("g2xp_rel")),
            ("retr_g", res.get("retr_g")),
            ("g_gen", res.get("g_gen")),
            ("g_proj", res.get("g_proj")),
            ("x_projs", res.get("x_projs")),
            ("act_pred", res.get("act_pred")),
        ]
        for name, tensor in tensor_items:
            if isinstance(tensor, torch.Tensor):
                self._write_grad_stats("tensor", name, tensor, step, lines)

        debug_tensors = getattr(self.dynamic.memory, "debug_tensors", {})
        for name, tensor in sorted(debug_tensors.items()):
            self._write_grad_stats("hebbian", name, tensor, step, lines)

        param_stats = []
        watched_keywords = ("memory", "rnn", "to_x", "projection_g", "visual_encdec", "p2g_mu")
        for name, param in self.dynamic.named_parameters():
            if param.grad is None:
                if any(keyword in name for keyword in watched_keywords):
                    self.train_writer.add_scalar(f"grad/param/{name.replace('.', '/')}/has_grad", 0.0, step)
                continue
            norm = param.grad.detach().norm().item()
            param_stats.append((norm, name, param))
            if any(keyword in name for keyword in watched_keywords):
                self._write_grad_stats("param", name, param, step, lines)

        for rank, (norm, name, param) in enumerate(sorted(param_stats, reverse=True)[: self.grad_debug_param_topk]):
            self.train_writer.add_scalar(f"grad/top_params/{rank}_norm", norm, step)
            lines.append(f"top{rank}:{name} norm={norm:.3e}")

        print("\n".join(lines))

    def get_train_last_ckpt_path(self):
        return os.path.join(self.train_dir, f"w{self.par.env.widths[0]}S{self.par.seq_len}-mb{self.par.chunk_size}-last.pth")

    def _get_mem_xproj(self, res, sen_seq):
        if "x_projs" in res:
            return res["x_projs"]
        x_proj = self.dynamic.visual_encdec(sen_seq)
        return self.dynamic.get_x2mem(x_proj, res.get("x_gen"))

    def _ramp_at(self, e):
        """Multiplier for the representation weights at absolute epoch ``e``.

        1.0 below ``grid_ramp_start``, linear up to ``grid_ramp_mult`` at
        ``grid_ramp_end``, then constant.  With the defaults (mult=1.0 or
        end<=start) this is the identity, so every earlier run is reproducible.
        """
        mult = float(getattr(self.par, "grid_ramp_mult", 1.0))
        e0 = float(getattr(self.par, "grid_ramp_start", 0))
        e1 = float(getattr(self.par, "grid_ramp_end", 0))
        if mult == 1.0 or e1 <= e0:
            return 1.0
        f = (float(e) - e0) / (e1 - e0)
        f = 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
        return 1.0 + (mult - 1.0) * f

    def _task_ramp_at(self, e):
        """Multiplier for the TASK reconstruction weights at absolute epoch ``e``.

        Same schedule shape as :meth:`_ramp_at`, on its own window
        (``task_ramp_{start,end,mult}``).  It scales ``lx_gen_scale`` and
        ``lx_gen_rel_scale`` only -- i.e. it anneals how strongly the task pulls the
        code, which is the one force still competing with the lattice late in
        training.  It never adds a lattice target; mult=1.0 is the identity.
        """
        mult = float(getattr(self.par, "task_ramp_mult", 1.0))
        e0 = float(getattr(self.par, "task_ramp_start", 0))
        e1 = float(getattr(self.par, "task_ramp_end", 0))
        if mult == 1.0 or e1 <= e0:
            return 1.0
        f = (float(e) - e0) / (e1 - e0)
        f = 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
        return 1.0 + (mult - 1.0) * f

    def _task_warmup_at(self, e):
        """Task-free opening: 0.0 at epoch 0, linear to 1.0 at ``task_warmup_epochs``.

        The point is NOT to regularise but to change *which basin is entered*.  Phase 6
        (cloud_phase6_2026-10-26/DIAGNOSIS.md) showed the grid/place decision is already
        made by e400-1200, and eight arms of loss engineering (phase 7-11) never changed
        the seed ranking -- so the only untouched variable is the very early dynamics.
        With a warmup the representation is organised by the self-supervised capacity +
        capacity + sparsity forces first, and the task pulls on an already-settled code.
        """
        n = float(getattr(self.par, "task_warmup_epochs", 0))
        if n <= 0:
            return 1.0
        f = float(e) / n
        return 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)

    # =====================================================================
    # compute_loss —— 编排层
    # 各 term 方法返回一个 DotDict，由 compute_loss 合并进 ctx 后交给 _assemble。
    # 调用顺序：
    #     ctx -> task -> ec -> capacity -> iso -> assemble
    # SIC 的 sep/inv 已移除；capacity 保留；iso 保留接口（默认权重 0）。
    # 每个 term 方法返回一个 DotDict，由 compute_loss 合并进 ctx 后交给
    # _assemble 统一加权与打包。
    # =====================================================================
    def compute_loss(self, res, sen_seq, s_visited, act_seq, pos_seq=None,
                     lx_gen_scale=1.0, lx_gen_rel_scale=1.0, widths=None):
        ctx = self._loss_ctx(res, sen_seq, s_visited, act_seq, pos_seq, widths,
                             lx_gen_scale, lx_gen_rel_scale)
        ctx.update(self._task_terms(ctx))
        ctx.update(self._ec_terms(ctx))
        ctx.update(self._capacity_term(ctx))
        ctx.update(self._iso_term(ctx))
        return self._assemble(ctx)

    def _loss_ctx(self, res, sen_seq, s_visited, act_seq, pos_seq, widths,
                  lx_gen_scale, lx_gen_rel_scale):
        """本轮输入 + per-epoch 缩放因子。这是唯一一处无条件写入 ctx 的地方。"""

        # ---- late task anneal -------------------------------------------------
        # Phase 8 (cloud_phase8_2026-10-27/REPORT_DECAY.md): the representation
        # weights are non-monotone in strength (x3 worse, x1 best, x0.2 no better), so
        # the lattice decay after e~2200 is not a dose problem; the task loss is the
        # only remaining force that keeps pulling the code away from a lattice.
        # This scales ONLY the reconstruction weights; mult=1.0 is the identity.
        task_ramp = float(getattr(self, "_task_ramp", 1.0))
        task_warm = float(getattr(self, "_task_warmup", 1.0))
        task_ramp = task_ramp * task_warm
        lx_gen_scale = lx_gen_scale * task_ramp
        lx_gen_rel_scale = lx_gen_rel_scale * task_ramp
        if s_visited is None:
            s_visited = torch.ones(sen_seq.shape[:2], dtype=sen_seq.dtype, device=sen_seq.device)
        # 表征权重的 late ramp：原实现由 _sic_term 顺带算出，SIC 移除后移到这里。
        # 现在它只作用于 lg_weight_l1（见 _assemble）。
        ramp = float(getattr(self, "_ramp", 1.0))

        return DotDict(res=res, sen_seq=sen_seq, act_seq=act_seq,
                       ramp=ramp,
                       pos_seq=pos_seq, widths=widths,
                       task_ramp=task_ramp, task_warm=task_warm,
                       lx_gen_scale=lx_gen_scale,
                       lx_gen_rel_scale=lx_gen_rel_scale,
                       s_visited=s_visited)

    def _task_terms(self, ctx):
        """重建型任务损失：记忆读出 (l_mem)、动作预测、x_gen 与 x_gen_rel。"""
        res, sen_seq, s_visited = ctx.res, ctx.sen_seq, ctx.s_visited

        # ---- gs2g (l_mem) -------------------------------------------------
        # Only defined for memory backends that can read g back from x/g.  With
        # hebbian_assoc ``retr_g`` is all zeros, so this term is a *constant*
        # (mean(g_gen^2), ~1/g_size) with zero gradient: computing it changes
        # nothing, and printing it as "gs2g=0.0156" invites reading a dead number
        # as a loss.  So it is skipped when unavailable, with a one-time warning
        # when a non-zero weight was requested -- silently doing nothing is exactly
        # how this stayed unnoticed for three rounds of runs.
        retr_available = bool(res.get("retr_g_available", False))
        mem_scale = float(getattr(self.par, "gs2g_scale", 0.0))
        if 'l_mem' in self.par.which_costs and mem_scale != 0 and not retr_available:
            # A warning, not an error: --gs2g_scale_list defaults to 0.1, so raising
            # would break every run that simply leaves the default in place.  The
            # term is a constant with zero gradient either way, so skipping it
            # cannot change any parameter update -- only the printed "total", which
            # no longer includes a number that looked like a loss but was not one.
            if not getattr(self, "_warned_gs2g_unavailable", False):
                print("[warn] gs2g (l_mem) weight=%g but rel_memory_impl=%r does not "
                      "provide retrieved_g; the term is a constant with zero gradient, "
                      "so it is skipped (pass --gs2g_scale_list 0 to silence this)."
                      % (mem_scale, getattr(self.par, "rel_memory_impl", "?")), flush=True)
                self._warned_gs2g_unavailable = True
        if retr_available:
            mem_loss = F.mse_loss(res["retr_g"], res["g_gen"].detach())
        else:
            mem_loss = torch.zeros((), device=sen_seq.device, dtype=sen_seq.dtype)

        act_pred_scale = getattr(self.par, "act_pred_scale", 0.0)
        act_pred_available = bool(res.get("act_pred_available", False))
        if "act_pred" in res and act_pred_available:
            l_act_pred = F.mse_loss(res["act_pred"], act_seq)
        else:
            l_act_pred = torch.zeros((), device=sen_seq.device, dtype=sen_seq.dtype)

        lx_gen, acc_gen = self.dynamic.visual_encdec.compute_loss(sen_seq, res["x_gen"], mask=s_visited)
        lx_gen_rel, acc_rel = self.dynamic.visual_encdec.compute_loss(sen_seq, res["g2xp_rel"], mask=None)

        return DotDict(mem_loss=mem_loss, mem_scale=mem_scale,
                       retr_available=retr_available,
                       act_pred_scale=act_pred_scale,
                       act_pred_available=act_pred_available,
                       l_act_pred=l_act_pred,
                       lx_gen=lx_gen, acc_gen=acc_gen,
                       lx_gen_rel=lx_gen_rel, acc_rel=acc_rel)

    def _ec_terms(self, ctx):
        """论文 Eq.7 的其余项（只有 EC 路径会产生；Hebbian 后端下恒为零）。"""
        res, sen_seq, act_seq = ctx.res, ctx.sen_seq, ctx.act_seq

        # ---- the paper's remaining terms (mm-TEM Eq. 7) -----------------------
        # Only the error-correction path produces these; with the Hebbian backend they
        # stay at zero, which is exactly the gap this build closes.
        z = torch.zeros((), device=sen_seq.device, dtype=sen_seq.dtype)
        x2g_scale = float(getattr(self.par, "x2g_scale", 0.0))
        con_scale = float(getattr(self.par, "con_scale", 0.0))
        inf_scale = float(getattr(self.par, "inf_scale", 0.0))
        write_scale = float(getattr(self.par, "write_scale", 0.0))
        if "g_x2g" in res and x2g_scale != 0:
            # L_x2g: structure from content, ||g_hat - g||^2 read with m=[0, x]
            l_x2g = F.mse_loss(res["g_x2g"], res["g_gen"].detach())
        else:
            l_x2g = z
        if "g_inf" in res and con_scale != 0:
            # L_con: the corrected code must stay close to the generated one
            l_con = F.mse_loss(res["g_gen"], res["g_inf"])
        else:
            l_con = z
        if "x_inf" in res and inf_scale != 0:
            # L_inf: content still decodable after the memory correction
            lx_inf, acc_inf = self.dynamic.visual_encdec.compute_loss(
                sen_seq, res["x_inf"], mask=s_visited)
        else:
            lx_inf = z
            acc_inf = z
        l_write = res["l_write"] if res.get("l_write") is not None else z
        lg_act_l2 = res["g_gen"].pow(2).mean()
        lg_weight_l1 = torch.zeros((), device=sen_seq.device, dtype=sen_seq.dtype)
        for module in self.dynamic.grid.a2trans_layer:
            if isinstance(module, torch.nn.Linear):
                lg_weight_l1 = lg_weight_l1 + module.weight.abs().mean()

        return DotDict(z=z, x2g_scale=x2g_scale, con_scale=con_scale,
                       inf_scale=inf_scale, write_scale=write_scale,
                       l_x2g=l_x2g, l_con=l_con, lx_inf=lx_inf, acc_inf=acc_inf,
                       l_write=l_write, lg_act_l2=lg_act_l2,
                       lg_weight_l1=lg_weight_l1)

    def _capacity_term(self, ctx):
        """防死单元的容量项（负的平均发放率平方）。

        权重是固定的 ``capacity_scale``。原先的 aux_norm_ratio 重标机制随 SIC
        一起移除——它的目的是让 sep / inv / cap 三项贡献相等，只剩一项时没有意义。
        """
        res, sen_seq = ctx.res, ctx.sen_seq
        capacity_scale = float(getattr(self.par, "capacity_scale", 0.0))
        capacity_n_steps = int(getattr(self.par, "capacity_n_steps", 10))
        if capacity_scale != 0 and "g_gen" in res:
            l_capacity = capacity_loss(res["g_gen"], n_steps=capacity_n_steps)
        else:
            l_capacity = torch.zeros((), device=sen_seq.device, dtype=sen_seq.dtype)
        return DotDict(l_capacity=l_capacity, capacity_scale=capacity_scale,
                       cap_scale_eff=capacity_scale)

    def _iso_term(self, ctx):
        """编码度量各向同性（v1 pooled / v2 per-unit），只由 scale 门控。"""
        res, sen_seq = ctx.res, ctx.sen_seq
        pos_seq, widths = ctx.pos_seq, ctx.widths

        # ---- grid-code isotropy (direction-independent step length) ----------
        # "Moving one cell must change g by the same amount, whichever way you
        # head."  Measured as the variance of ||g_t - g_{t-1}|| over consecutive
        # steps that are a real one-cell move.  grid_losses.py documents the
        # definition, the two caveats (necessary but not sufficient for a 2-D
        # periodic metric; minimised by collapse) and the optional one-sided
        # anchor.  Gated on the *scale* only, not on which_costs: a term that is
        # silently not wired is the single most expensive bug this project has
        # had (see P2 in the project notes), so a non-zero scale is always used.
        iso_scale = float(getattr(self.par, "grid_iso_scale", 0.0))
        iso_version = int(getattr(self.par, "grid_iso_version", 1))
        iso_detail = {}
        if iso_scale != 0 and "g_gen" in res and pos_seq is not None:
            iso_widths = resolve_widths(widths, res["g_gen"].shape[0], self.par)
            if iso_version >= 2:
                l_grid_iso, iso_detail = grid_isotropy_loss_v2(
                    res["g_gen"], pos_seq, widths=iso_widths,
                    k=getattr(self.par, "grid_iso_k", 1),
                    nbins=int(getattr(self.par, "grid_iso_nbins", 8)),
                    min_disp=float(getattr(self.par, "grid_iso_min_disp", 1.0)),
                    agg=getattr(self.par, "grid_iso_agg", "cv"),
                    per_unit=bool(int(getattr(self.par, "grid_iso_perunit", 1))),
                    metric=getattr(self.par, "grid_iso_metric", "within_radius"),
                    target=getattr(self.par, "grid_iso_target", None),
                    anchor=float(getattr(self.par, "grid_iso_anchor", 0.0)))
            else:
                l_grid_iso, iso_detail = grid_isotropy_loss(
                    res["g_gen"], pos_seq, widths=iso_widths,
                    mode=getattr(self.par, "grid_iso_mode", "var"),
                    dist=getattr(self.par, "grid_iso_dist", "l2"),
                    target=getattr(self.par, "grid_iso_target", None),
                    anchor=float(getattr(self.par, "grid_iso_anchor", 0.0)))
        else:
            if iso_scale != 0 and pos_seq is None and not getattr(self, "_warned_iso_no_pos", False):
                print("[warn] grid_iso_scale=%g but no pos_seq was passed: the isotropy "
                      "term needs the trajectory (%s)" % (iso_scale, "skipped"), flush=True)
                self._warned_iso_no_pos = True
            l_grid_iso = torch.zeros((), device=sen_seq.device, dtype=sen_seq.dtype)

        return DotDict(iso_scale=iso_scale, iso_version=iso_version,
                       l_grid_iso=l_grid_iso, iso_detail=iso_detail)

    def _assemble(self, ctx):
        """加权求和并打包成 DotDict（字段名与日志/TensorBoard 保持一致）。"""
        res, sen_seq, act_seq = ctx.res, ctx.sen_seq, ctx.act_seq
        pos_seq, widths = ctx.pos_seq, ctx.widths
        lx_gen, lx_gen_rel = ctx.lx_gen, ctx.lx_gen_rel
        acc_gen, acc_rel = ctx.acc_gen, ctx.acc_rel
        lx_gen_scale, lx_gen_rel_scale = ctx.lx_gen_scale, ctx.lx_gen_rel_scale
        mem_loss, mem_scale = ctx.mem_loss, ctx.mem_scale
        retr_available = ctx.retr_available
        act_pred_scale, act_pred_available = ctx.act_pred_scale, ctx.act_pred_available
        l_act_pred = ctx.l_act_pred
        z = ctx.z
        x2g_scale, con_scale = ctx.x2g_scale, ctx.con_scale
        inf_scale, write_scale = ctx.inf_scale, ctx.write_scale
        l_x2g, l_con, lx_inf, acc_inf = ctx.l_x2g, ctx.l_con, ctx.lx_inf, ctx.acc_inf
        l_write = ctx.l_write
        lg_act_l2, lg_weight_l1 = ctx.lg_act_l2, ctx.lg_weight_l1
        l_capacity, capacity_scale = ctx.l_capacity, ctx.capacity_scale
        cap_scale_eff = ctx.cap_scale_eff
        l_grid_iso, iso_scale, iso_detail = ctx.l_grid_iso, ctx.iso_scale, ctx.iso_detail
        ramp, task_ramp = ctx.ramp, ctx.task_ramp

        losses = DotDict()
        cost_all = 0.0
        losses.lx_gen = lx_gen
        losses.l_mem = mem_loss
        losses.lx_gen_rel = lx_gen_rel
        losses.l_x2g = l_x2g
        losses.l_con = l_con
        losses.lx_inf = lx_inf
        losses.l_write = l_write
        losses.x2g_scale = x2g_scale
        losses.con_scale = con_scale
        losses.inf_scale = inf_scale
        losses.write_scale = write_scale
        losses.acc_inf = acc_inf
        losses.l_act_pred = l_act_pred
        losses.lg_act_l2 = lg_act_l2
        losses.lg_weight_l1 = lg_weight_l1
        losses.l_capacity = l_capacity
        losses.l_grid_iso = l_grid_iso
        losses.grid_iso_scale = iso_scale
        losses.grid_iso_detail = iso_detail
        losses.ramp = ramp
        losses.task_ramp = task_ramp
        losses.acc_gen = acc_gen
        losses.acc_rel = acc_rel
        losses.lx_gen_scale = lx_gen_scale
        losses.lx_gen_rel_scale = lx_gen_rel_scale
        losses.act_pred_scale = act_pred_scale
        losses.capacity_scale = float(cap_scale_eff)
        losses.has_pos = pos_seq is not None
        losses.l_mem_available = retr_available
        losses.act_pred_available = act_pred_available

                    
        if 'lx_gen' in self.par.which_costs:
            cost_all += lx_gen * lx_gen_scale
        if 'lx_gen_rel' in self.par.which_costs:
            cost_all += lx_gen_rel * lx_gen_rel_scale
        if 'l_mem' in self.par.which_costs and retr_available:
            cost_all += mem_loss * mem_scale
        if x2g_scale != 0:
            cost_all += l_x2g * x2g_scale
        if con_scale != 0:
            cost_all += l_con * con_scale
        if inf_scale != 0:
            cost_all += lx_inf * inf_scale
        if write_scale != 0:
            cost_all += l_write * write_scale
        if 'lg_act_l2' in self.par.which_costs:
            cost_all += lg_act_l2 * getattr(self.par, 'lg_act_l2_scale', 0.0)
        if 'lg_weight_l1' in self.par.which_costs:
            cost_all += lg_weight_l1 * getattr(self.par, 'lg_weight_l1_scale', 0.0) * ramp
        if capacity_scale != 0:
            cost_all += l_capacity * cap_scale_eff
        if iso_scale != 0:
            cost_all += l_grid_iso * iso_scale
        if 'l_act_pred' in self.par.which_costs and act_pred_scale != 0:
            if not act_pred_available:
                raise ValueError("Action prediction loss is enabled, but act_pred is unavailable.")
            cost_all += l_act_pred * act_pred_scale


        losses.train_loss = cost_all
        return losses




    def seq_train(self, num_episode=200, last_epi=0):
        self.dynamic.to(self.device)
        self.dynamic.train()

        e = last_epi - 1
        for e in range(last_epi, last_epi+num_episode):
            # late ramp on the representation weights (see compute_loss); recomputed
            # every epoch so a resumed run and a fresh run behave the same at the
            # same *absolute* epoch
            self._ramp = self._ramp_at(e)
            self._task_ramp = self._task_ramp_at(e)
            self._task_warmup = self._task_warmup_at(e)
            # alpha of the error-correction loop: 0 -> 1 over the first ec_alpha_ramp
            # steps (paper A.2: "the scalar alpha controls the integration ratio, which
            # slowly increases in the first 2000 gradient steps").
            _ec_n = float(getattr(self.par, "ec_alpha_ramp", 2000) or 0)
            self._ec_alpha = min(1.0, e / _ec_n) if _ec_n > 0 else 1.0
            # the loop's alpha lives on the model, which is a different object
            if hasattr(self, "dynamic"):
                self.dynamic._ec_alpha = self._ec_alpha
            env = init_env(self.par, one_instance=USE_SINGLE_ENV)
            data = gen_data(env, self.par, one_instance=USE_SINGLE_ENV)

            act = to_tensor(data["dirs"], dtype=torch.float32)
            sen_seq = to_tensor(data["sense"], dtype=torch.float32) # (B, T, D)
            s_visited = to_tensor(data["visit"], dtype=torch.float32) # (B,T)
            # flat state indices (B, T) -- required by the grid-code capacity loss
            pos_seq = to_tensor(np.asarray(data["pos"]), dtype=torch.float32) if "pos" in data else None
            # arena side of every sample in this batch (the width table is mixed:
            # 11/10/10/11/8/9/...), so flat pos -> (x, y) needs it per sample
            self.train_widths = (to_tensor(np.asarray(data["width"]), dtype=torch.float32)
                                 if "width" in data else None)

            if self.grad_debug and hasattr(self.dynamic.memory, "reset_debug"):
                self.dynamic.memory.reset_debug()
            res, _ = self.dynamic(act, sen_seq, last_g=None, cache=None)
            self._prepare_grad_debug(res)
            loss = self.compute_loss(
                res,
                sen_seq,
                s_visited,
                act,
                pos_seq=pos_seq,
                lx_gen_scale=getattr(self.par, "lx_gen_scale", 1.0),
                lx_gen_rel_scale=getattr(self.par, "lx_gen_rel_scale", 1.0),
                widths=self.train_widths,
            )

            self.opt_train.zero_grad()
            loss.train_loss.backward()
            total_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.dynamic.parameters(), max_norm=5.0)
            self._log_grad_debug(res, e, total_norm_before_clip)
            self.opt_train.step()
            self.schedule_train.step()
            
            self.train_writer.add_scalar("x_gen", loss.lx_gen.item(), e)
            self.train_writer.add_scalar("lx_gen_rel", loss.lx_gen_rel.item(), e)
            # these two only exist for backends that actually produce them; writing
            # a constant 0 -- or a constant mean(g^2) -- to TensorBoard hid that
            if loss.act_pred_available:
                self.train_writer.add_scalar("action_pred", loss.l_act_pred.item(), e)
            if loss.l_mem_available:
                self.train_writer.add_scalar("gs2g", loss.l_mem.item(), e)
            self.train_writer.add_scalar("train_acc_pred", loss.acc_gen.item(), e)
            self.train_writer.add_scalar("train_acc_rel", loss.acc_rel.item(), e)
            self.train_writer.add_scalar("loss_scale/lx_gen", loss.lx_gen_scale, e)
            self.train_writer.add_scalar("loss_scale/lx_gen_rel", loss.lx_gen_rel_scale, e)
            self.train_writer.add_scalar("loss_scale/action_pred", loss.act_pred_scale, e)
            self.train_writer.add_scalar("capacity/l_capacity", loss.l_capacity.item(), e)
            self.train_writer.add_scalar("capacity/weighted", loss.l_capacity.item() * loss.capacity_scale, e)
            self.train_writer.add_scalar("capacity/scale", loss.capacity_scale, e)
            if loss.grid_iso_scale != 0:
                self.train_writer.add_scalar("grid_iso/l_iso", loss.l_grid_iso.item(), e)
                self.train_writer.add_scalar("grid_iso/weighted",
                                             loss.l_grid_iso.item() * loss.grid_iso_scale, e)
                self.train_writer.add_scalar("grid_iso/scale", loss.grid_iso_scale, e)
                if loss.grid_iso_detail.get("n_pairs"):
                    det = loss.grid_iso_detail
                    self.train_writer.add_scalar("grid_iso/step_mean", det.get("mean", float("nan")), e)
                    if "std" in det:                       # v1 only
                        self.train_writer.add_scalar("grid_iso/step_std", det["std"], e)
                    if "cv" in det:                        # v1: pooled CV
                        self.train_writer.add_scalar("grid_iso/step_cv", det["cv"], e)
                    for key in ("cv_direction", "cv_pooled", "cv_direction_p90",
                                "n_units_live", "n_cells_used", "displacement"):
                        if key in det:                     # v2
                            self.train_writer.add_scalar("grid_iso/" + key, det[key], e)
                    for name, value in (det.get("per_direction") or {}).items():   # v1 only
                        self.train_writer.add_scalar("grid_iso/dir_" + name, value, e)
            
            if e % 20 == 0:
                # unavailability is printed as "n/a" instead of a number that is
                # constant by construction (see compute_loss)
                act_pred_txt = ("%.4f" % loss.l_act_pred.item()) if loss.act_pred_available else "n/a"
                gs2g_txt = ("%.4f" % loss.l_mem.item()) if loss.l_mem_available else "n/a"
                iso_txt = ("%.4f(x%g)" % (loss.l_grid_iso.item(), loss.grid_iso_scale)
                           if loss.grid_iso_scale != 0 else "n/a")
                print(f"epoch={e}: x_gen={loss.lx_gen.item():.2f}, lx_gen_rel={loss.lx_gen_rel.item():.2f}, "
                      f"act_pred={act_pred_txt}, gs2g={gs2g_txt}, "
                      f"scale_gen={loss.lx_gen_scale:.3f}, scale_rel={loss.lx_gen_rel_scale:.3f}, "
                      f"scale_act={loss.act_pred_scale:.3f}, "
                      f"acc_pred={loss.acc_gen.item():.3f}, acc_rel={loss.acc_rel.item():.3f}, "
                      f"capac={loss.l_capacity.item():.4f}(x{loss.capacity_scale:g}), "
                      f"iso={iso_txt}, "
                      f"total={loss.train_loss.item():.2f}, lr={current_lr(self.opt_train):.2e}"
                      + ("" if getattr(loss, "x2g_scale", 0.0) == 0
                         and getattr(loss, "con_scale", 0.0) == 0
                         and getattr(loss, "inf_scale", 0.0) == 0 else
                         ", x2g=%.4f(x%g), con=%.4f(x%g), inf=%.2f(x%g), acc_inf=%.3f, wr=%.3f"
                         % (loss.l_x2g.item(), loss.x2g_scale,
                            loss.l_con.item(), loss.con_scale,
                            loss.lx_inf.item(), loss.inf_scale,
                            loss.acc_inf.item(), loss.l_write.item()))
                      + ("" if abs(loss.ramp - 1.0) < 1e-9 else f", ramp={loss.ramp:.2f}")
                      + ("" if abs(loss.task_ramp - 1.0) < 1e-9 else f", task={loss.task_ramp:.2f}"))
                if CFG.repro_check:
                    report_fingerprint(e, loss)
                if loss.grid_iso_scale != 0 and loss.grid_iso_detail.get("n_pairs"):
                    print("    " + format_iso_detail(loss.grid_iso_detail), flush=True)
            
            save_interval = max(1, int(num_episode / 5))
            if e % save_interval == 0:
                torch.save(self.dynamic.state_dict(), self.get_train_last_ckpt_path())

            # SNAP_EVERY=k：每 k 个 epoch 存一份完整权重快照（放大实验用；默认 0 = 关闭）
            _snap = CFG.snap_every
            if _snap > 0 and e % _snap == 0:
                _sdir = os.path.join(self.train_dir, "snap")
                os.makedirs(_sdir, exist_ok=True)
                torch.save(self.dynamic.state_dict(), os.path.join(_sdir, "e%04d.pth" % e))

            # DIAG_EVERY: cadence of the (expensive) 200-epoch grid diagnostic.
            # A long run must widen it or the per-diagnostic files (~5.9 MB each,
            # including a PDF) fill the disk: at 2e5 epochs the default 200 means
            # 1000 dumps ~ 5.9 GB per run.
            _diag_every = CFG.diag_every
            if e % _diag_every == 0:
                if int(getattr(self.par, "probe_every", 0)) > 0 \
                        and e % int(self.par.probe_every) == 0:
                    try:
                        from probe_readout import probe_readout
                        probe_readout(res, sen_seq, "e%d" % e, act_seq=act)
                    except Exception as exc:      # a diagnostic must never kill a run
                        print("[warn] probe failed: %s" % exc, flush=True)
                gen_acc, acc_1 = self.eval(4, context=int(self.par.seq_len/2), load=False)
                self.train_writer.add_scalar("train/gen_acc", gen_acc, e)
                self.train_writer.add_scalar("train/acc_1", acc_1, e)
                draw_path = os.path.join(self.train_dir, f"e{e}S{self.par.seq_len}W{self.par.env.widths[0]}") 
                self.draw_position_code(draw_path=draw_path, epoch=e)    
                
        return e


    @torch.no_grad()
    def draw_position_code(self, draw_path, epoch=None):
        self.dynamic.eval()
        with torch.no_grad():
            batch_size_tmp = self.eval_par.batch_size
            seq_len_tmp = self.eval_par.seq_len 
            
            self.eval_par.batch_size = 16
            self.eval_par.seq_len = 256 # longer seq for eval
            sync_env_config(self.eval_par, width=max(self.par.env.widths), height=max(self.par.env.heights))
            
            env = init_env(self.eval_par, one_instance=False)
            data = gen_data(env, self.eval_par, eval=True)

            act_seq = to_tensor(data["dirs"], dtype=torch.float32)
            sen_seq = to_tensor(data["sense"], dtype=torch.float32)
            pos_seq = to_tensor(data["pos"], dtype=torch.float32)
            mask = to_tensor(data["visit"], dtype=torch.float32)
            
            res_dict, _ = self.dynamic(act_seq, sen_seq)
            pred = res_dict["x_gen"]
            
            pred_idx = torch.argmax(pred, dim=-1)
            label_idx = torch.argmax(sen_seq, dim=-1)
            acc = ((pred_idx == label_idx).float() * mask).sum().item() / (mask.sum().item() + 1e-8)
            print(f"Evaluation Accuracy (seq 256): {acc:.3f}")
        
            if hasattr(self, 'eval_par'):
                # 200-epoch diagnostic: GridScore60 + GridScore90 with a +/-N step time-shift
                # search (predictive grid cells).  Legacy behaviour (grid_cell.draw_grid) is
                # still available through the same draw_path files (grid.pdf, gridness.npy).
                self.run_predictive_diag(self.eval_par, pos_seq, res_dict["g_gen"], draw_path, epoch,
                                         widths=data.get("width"))

            self.eval_par.batch_size = batch_size_tmp
            self.eval_par.seq_len = seq_len_tmp
      
        self.dynamic.train()
        
        
    def run_predictive_diag(self, par, pos_seq, grid_seq, draw_path, epoch=None, widths=None):
        """Rate maps + GS60/GS90 + best-shift statistics, never fatal for training.

        ``widths`` (optional, ``(B,)``) is the per-sample arena side, so flat state
        indices are decoded with the width they were built from.  The evaluation
        batch uses one shared arena, so this is normally a constant array.
        """
        try:
            from predictive_grid_diag import draw_grid_shift, format_summary
        except Exception as exc:  # matplotlib / scipy missing -> fall back to the legacy plot
            print("[warn] predictive_grid_diag unavailable (%s); using legacy draw_grid" % exc, flush=True)
            return draw_grid(par, pos_seq, grid_seq, draw_path=draw_path)

        shift_range = int(getattr(self.par, "diag_shift_range", 2))
        top_k = int(getattr(self.par, "diag_top_k", 10))
        diag_kwargs = {}
        if getattr(self.par, "diag_sub", None) is not None:
            diag_kwargs["sub"] = int(self.par.diag_sub)
        if getattr(self.par, "diag_sigma", None) is not None:
            diag_kwargs["sigma"] = float(self.par.diag_sigma)
        if getattr(self.par, "diag_legacy_ring", False):
            diag_kwargs["adapt_ring"] = False
        if getattr(self.par, "diag_save_sacs", None) is not None:
            diag_kwargs["save_sacs"] = str(self.par.diag_save_sacs)
        if widths is not None:
            diag_kwargs["widths"] = widths
        try:
            summary = draw_grid_shift(par, pos_seq, grid_seq, draw_path,
                                      shift_range=shift_range, top_k=top_k, epoch=epoch,
                                      **diag_kwargs)
        except Exception:
            import traceback
            print("[warn] predictive grid diagnostic failed at epoch %s, training continues:" % epoch, flush=True)
            traceback.print_exc()
            return None

        print(format_summary(summary), flush=True)
        if getattr(self, "train_writer", None) is not None and epoch is not None:
            for key, value in summary.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.train_writer.add_scalar("grid/" + key, float(value), int(epoch))
            for delta, count in summary["counts_gs60"].items():
                self.train_writer.add_scalar("grid/count_gs60_shift%+d" % int(delta), float(count), int(epoch))
            for delta, count in summary["counts_gs90"].items():
                self.train_writer.add_scalar("grid/count_gs90_shift%+d" % int(delta), float(count), int(epoch))
        return summary

    @torch.no_grad()
    def eval(self, num, context=64, load=False, write=False, draw=None):
        if load:
            ckpt_path = self.get_train_last_ckpt_path()
            ckpt = load_ckpt(ckpt_path)
            self.dynamic.load_state_dict(ckpt)
            print("load", ckpt_path)

        self.dynamic.to(self.device)
        self.dynamic.eval()

        self.eval_par.seq_len = self.par.seq_len

        if self.eval_use_legacy_batch:
            eval_batch_size = self.eval_par.batch_size
            self.eval_par.batch_size = num
            sync_env_config(self.eval_par, width=max(self.par.env.widths), height=max(self.par.env.heights))
            env = init_env(self.eval_par, one_instance=False)
            data = gen_data(env, self.eval_par, eval=False)
            self.eval_par.batch_size = eval_batch_size
            sync_env_config(self.eval_par, width=max(self.par.env.widths), height=max(self.par.env.heights))

            act = to_tensor(data["dirs"], dtype=torch.float32)
            sen_seq = to_tensor(data["sense"], dtype=torch.float32)  # (B, T, D)
            s_visited = to_tensor(data["visit"], dtype=torch.float32)  # (B,T)
            pos_seq = to_tensor(data["pos"], dtype=torch.float32)
        else:
            all_act = []
            all_sen = []
            all_pos = []
            all_visit = []
            for i in range(num):
                env = init_env(self.par, one_instance=True)
                data = gen_data(env, self.par, one_instance=True)
                all_act.append(data["dirs"])
                all_sen.append(data["sense"])
                all_pos.append(data["pos"])
                all_visit.append(data["visit"])
            act = to_tensor(np.concatenate(all_act, axis=0), dtype=torch.float32)
            sen_seq = to_tensor(np.concatenate(all_sen, axis=0), dtype=torch.float32)  # (B, T, D)
            s_visited = to_tensor(np.concatenate(all_visit, axis=0), dtype=torch.float32)  # (B,T)
            pos_seq = to_tensor(np.concatenate(all_pos, axis=0), dtype=torch.float32)
        
        # 64-context multi-step pred
        res_one_step, one_step_cache = self.dynamic(
            act,
            sen_seq,
            last_g=None,
            cache=None,
        )

        res_with_context, context_cache = self.dynamic(
            act,
            sen_seq,
            last_g=None,
            cache=None,
            context=context,
        )

        if len(pos_seq.shape) > 2:
            context_after_mask = (pos_seq[:, context:].unsqueeze(2) == pos_seq[:, :context].unsqueeze(1)).all(dim=-1).float().amax(dim=-1)  # amax -> tensor (max(dim=) would return a (values, indices) tuple)
        else:
            context_after_mask = (pos_seq[:, context:].unsqueeze(2) == pos_seq[:, :context].unsqueeze(1)).float().amax(dim=-1)  # same: amax keeps the tensor semantics

        _, acc_multi = self.dynamic.visual_encdec.compute_loss(
                sen_seq[:, context:],
                res_with_context['x_gen'][:, context:],
                mask=context_after_mask,
                dim=1,
            )
        
        _, acc_1step = self.dynamic.visual_encdec.compute_loss(sen_seq, res_one_step['x_gen'], mask=s_visited, dim=1)


        print(f"seq={self.par.seq_len}, context={context}, acc_multi={acc_multi.mean():.3f}, acc_1step={acc_1step.mean():.3f}")
            
        if write:
            compare_path = os.path.join(self.run_dir, "result_compare_withname.csv")
            with open(compare_path, 'a', encoding='utf-8') as f:
                for i in range(acc_1step.shape[0]):
                    f.write(f'\n{self.prefix}, {self.prefix}, {context}, {self.par.seq_len}, {i}, {acc_1step[i]}, {acc_multi[i]}')

        if draw is not None:
            # same high-resolution pipeline as the 200-epoch diagnostic, so that
            # _s*.pdf / _s*.npz written by `--mode eval` match the training figures
            if getattr(self.par, "diag_on_eval", True) and hasattr(self, "eval_par"):
                self.run_predictive_diag(self.eval_par, pos_seq, res_one_step["g_gen"], draw)
            else:
                draw_grid(self.eval_par, pos_seq, res_one_step["g_gen"], draw_path=draw)

        self.dynamic.train()
        return acc_multi.mean(), acc_1step.mean()


def seq_train(model_name, pretrain_ckpt="", args=None, par=None, run_layout=None):
    pretrain_ckpt = pretrain_ckpt or ""

    if os.path.exists(pretrain_ckpt):
        ckpt = load_ckpt(pretrain_ckpt)
        print("Load from", pretrain_ckpt)
    else:
        assert pretrain_ckpt == "", "pretrain_ckpt not exist!"
        ckpt = None

    num_stages = len(args.train_seq_bucket)
    gs2g_scale_list = expand_schedule_arg(args.gs2g_scale_list, num_stages, "--gs2g_scale_list")
    train_epochs_list = expand_schedule_arg(args.train_epochs_list, num_stages, "--train_epochs_list")
    train_batch_list = expand_schedule_arg(args.train_batch_list, num_stages, "--train_batch_list")
    lx_gen_scale_list = expand_schedule_arg(args.lx_gen_scale_list, num_stages, "--lx_gen_scale_list")
    lx_gen_rel_scale_list = expand_schedule_arg(args.lx_gen_rel_scale_list, num_stages, "--lx_gen_rel_scale_list")
    act_pred_scale_list = expand_schedule_arg(args.act_pred_scale_list, num_stages, "--act_pred_scale_list")

    last_epi = 1
    for seq, mem_batch, gs2g_scale, train_epoch, train_batch, lx_gen_scale, lx_gen_rel_scale, act_pred_scale in zip(
        args.train_seq_bucket,
        args.train_mem_batch,
        gs2g_scale_list,
        train_epochs_list,
        train_batch_list,
        lx_gen_scale_list,
        lx_gen_rel_scale_list,
        act_pred_scale_list,
    ):
        par.seq_len = seq
        par.mem_batch_size = mem_batch
        par.chunk_size = mem_batch
        par.gs2g_scale = gs2g_scale
        par.train_epochs = train_epoch
        par.batch_size = train_batch
        par.lx_gen_scale = lx_gen_scale
        par.lx_gen_rel_scale = lx_gen_rel_scale
        par.act_pred_scale = act_pred_scale

        sync_env_config(par, width=WIDTH, height=HEIGHT)

        print(f"seq={seq}, batch={par.batch_size}")
        trainer = Trainer(model_name, False, par, legacy_eval=args.legacy_eval if args else False, run_layout=run_layout)
        if getattr(args, "data_seed", None) is not None:
            _ds = int(args.data_seed)
            random.seed(_ds); np.random.seed(_ds)
            torch.manual_seed(_ds)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_ds)
            print("[data_seed] 数据流重设为 %d，权重仍来自 seed=%d" % (_ds, args.seed))
        _pert = CFG.init_perturb
        if _pert > 0:
            _g = torch.Generator(device="cpu").manual_seed(20251108)
            with torch.no_grad():
                for _name, _p in trainer.dynamic.named_parameters():
                    _noise = torch.empty(_p.shape, dtype=torch.float32, device="cpu").normal_(0.0, 1.0, generator=_g)
                    _p.add_(_noise.to(_p.device, _p.dtype) * (_pert * _p.detach().float().abs().mean().clamp_min(1e-12)))
            print("[perturb] 全部参数加相对扰动 %.3g（独立 CPU generator，不消耗全局 RNG）" % _pert, flush=True)
        try:
            _sd = trainer.dynamic.state_dict()
            _k = [k for k in _sd if k.endswith("a2trans_layer.0.weight")][0]
            _w = _sd[_k].detach().double()
            print("[init] %s  sum=%.10e  absum=%.10e" % (_k, float(_w.sum()), float(_w.abs().sum())))
        except Exception as _e:
            print("[init] checksum unavailable:", _e)
        print("[dyn] M(a)=I+W(a)  a2trans_init=%s  iso=%g"
              % (getattr(par, "a2trans_init", "?"),
                 float(getattr(par, "grid_iso_scale", 0.0))),
              flush=True)
        if ckpt is not None:
            trainer.dynamic.load_state_dict(ckpt, strict=False)
        last_epi = trainer.seq_train(num_episode=par.train_epochs, last_epi=last_epi)
        ckpt = load_ckpt(trainer.get_train_last_ckpt_path(), map_location="cpu")


def eval(load_dir, ckpt_name="", args=None, par=None):
    seed = 101
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    ckpt_path = os.path.join(load_dir, ckpt_name)
    run_root = infer_run_root(ckpt_path)
    run_layout = ensure_run_layout(run_root)
    model_name = os.path.basename(run_root).split("+")[0]
    trainer = Trainer(model_name, True, legacy_eval=args.legacy_eval if args else False, par=par, run_layout=run_layout)
    ckpt = load_ckpt(ckpt_path)
    print(model_name, trainer.dynamic.load_state_dict(ckpt, strict=False))

    trainer.par.batch_size = 1
    seq_len_list = [65,128,256,384,512,1024,2048,4096]

    for s in seq_len_list:
        trainer.par.seq_len = s
        trainer.eval_par.batch_size = 16
        test_env_num = 32
        trainer.eval(test_env_num, context=64, load=False, write=True, draw=os.path.join(load_dir, f'_s{s}'))


def configure_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_run_prefix(seed, prefix):
    if prefix:
        return f"s{seed}_{prefix}"
    return f"s{seed}"


def parse_optional_float(value):
    value = str(value).strip().lower()
    if value in {"none", "null"}:
        return None
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or draw_position_code mmTEM.")
    parser.add_argument(
        "--mode",
        choices=["pretrain", "seq_train", "eval"],
        default="pretrain",
        help="Which entrypoint to run.",
    )
    parser.add_argument("--seed", type=int, default=4533, 
                        help="Random seed. please use [4533, 7791, 101]")
    parser.add_argument("--data_seed", type=int, default=None,
                        help="只重设数据流（walk/batch 顺序）的随机种子；模型初始化仍由 --seed 决定。"
                             "用于'固定初始化、只换数据顺序'的对照实验。")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint path. Used as pretrain ckpt for seq_train, or eval checkpoint for eval mode.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional suffix for the run prefix. Final prefix is always s{seed}_{prefix}.",
    )
    parser.add_argument(
        "--legacy-eval",
        action="store_true",
        help="Use the legacy eval batch construction path.",
    )
    parser.add_argument(
        "--no-legacy-eval",
        dest="legacy_eval",
        action="store_false",
        help="Disable the legacy eval batch construction path.",
    )

    parser.add_argument('--train_epochs_list', '--train_epochs', nargs='+', dest='train_epochs_list', type=int, default=[6000], help='Number of epochs for sequential training')
    parser.add_argument('--mem_batch', type=int, default=1, help='Chunk size for training')
    parser.add_argument('--use_last_gint', action='store_true', help='Whether to use last g_int as input for next step')
    parser.add_argument('--train_batch_list', '--train_batch', nargs='+', dest='train_batch_list', type=int, default=[64], help='Batch size list for training')
    parser.add_argument('--lr_train', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--device', type=str, default=None,
                        help='torch device: cuda:0 / cuda / cpu (default: cuda:0 when available)')
    parser.add_argument('--width', nargs='+', type=int, default=[11, 10, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9], help='Width of the environment')
    parser.add_argument('--lr_pretrain', type=float, default=1e-3, help='Learning rate for pretraining')
    parser.add_argument('--lg_act_l2_scale', type=float, default=1e-5, help='lg_act l2 regularization scale')
    parser.add_argument('--lg_weight_l1_scale', type=float, default=1e-5, help='lg weight l1 regularization scale')
    parser.add_argument('--gs2g_scale_list', nargs='+', type=float, default=[0.1], help='gs2g regularization scale list')
    parser.add_argument('--lx_gen_scale_list', nargs='+', type=float, default=[0.1], help='Starting scale for lx_gen; linearly increases during training')
    parser.add_argument('--lx_gen_rel_scale_list', nargs='+', type=float, default=[0.1], help='Starting scale for lx_gen_rel; linearly increases during training')
    parser.add_argument('--act_pred_scale_list', nargs='+', type=float, default=[0.0], help='Scale list for action prediction auxiliary loss')
    parser.add_argument('--heads', type=int, default=None, help='Number of heads in Titans memory')
    parser.add_argument('--mem_dim_head', type=int, default=None, help='Per-head dimension in Titans memory')
    parser.add_argument('--neural_mem_expand', type=int, default=None, help='Expansion factor for the Titans memory MLP')
    parser.add_argument('--mem_grad_norm', type=parse_optional_float, default=None, help='Soft clamp norm for memory write gradients; use none to disable')
    parser.add_argument('--hebbian_hidden_dim', type=int, default=None, help='Hidden/key dimension for Hebbian relational memory')
    parser.add_argument('--hebbian_eta', type=float, default=None, help='Hebbian memory write rate')
    parser.add_argument('--hebbian_decay', type=float, default=None, help='Hebbian memory decay factor')
    parser.add_argument('--hebbian_use_oja', type=int, default=None, help='Use Oja/delta correction in Hebbian memory: 1 or 0')
    parser.add_argument('--hebbian_detach_fast_weight', type=int, default=None, help='Detach Hebbian fast weights between recurrent steps: 1 or 0')
    parser.add_argument('--hebbian_normalize_activity', type=int, default=None, help='Normalize Hebbian RNN activities for fast-weight branch: 1 or 0')
    parser.add_argument('--grad_debug', action='store_true', help='Print and TensorBoard-log key tensor and parameter gradient statistics')
    parser.add_argument('--grad_debug_interval', type=int, default=20, help='Epoch interval for gradient debug print/TensorBoard logging')
    parser.add_argument('--grad_debug_param_topk', type=int, default=8, help='Number of largest parameter gradient norms to print')
    parser.add_argument('--capacity_scale', type=float, default=1e-3, help='Weight of the grid-code capacity (anti-sparsity) loss; 0 disables it')
    parser.add_argument('--g_size', type=int, default=None,
                        help='Number of grid cells (default 64). Lowering it is the architectural '
                             'BOTTLENECK experiment: fewer units must cover the same arena under the '
                             'same per-unit capacity budget, which is the regime where the densest '
                             'packing (hexagonal) is the cheapest solution. See '
                             'cloud_phase9_2026-10-27/REPORT_TASK.md section 4')
    parser.add_argument('--capacity_n_steps', type=int, default=10, help='Number of leading steps averaged by the capacity loss')
    # ---- mm-TEM error-correction loop + the paper's remaining loss terms --------
    parser.add_argument('--use_error_correction', type=int, default=0,
                        help='1 = enable mm-TEM\'s EC feedback loop (needs --rel_memory_impl fastweights)')
    parser.add_argument('--ec_block', type=int, default=0,
                        help='m_b of the error-correction loop: memory update interval (0 = chunk_size)')
    parser.add_argument('--probe_every', type=int, default=0,
                        help='Print decodability probes (ridge/MLP on g) every N epochs; 0 = off')
    parser.add_argument('--ec_alpha_ramp', type=int, default=2000,
                        help='Steps over which the EC integration ratio ramps 0 -> 1')
    parser.add_argument('--f_delta_hidden', type=int, default=128, help='Width of the f_delta MLP')
    parser.add_argument('--x2g_scale', type=float, default=0.0, help='gamma of L_x2g (structure from content)')
    parser.add_argument('--con_scale', type=float, default=0.0, help='gamma of L_con (consistency)')
    parser.add_argument('--inf_scale', type=float, default=0.0, help='gamma of L_inf (inference prediction)')
    parser.add_argument('--write_scale', type=float, default=0.0,
                        help='Weight of the memory write loss (paper Eq. 4); the only path that trains W_k/W_v')
    parser.add_argument('--fw_heads', type=int, default=2, help='Heads of the fast-weight meta-MLP')
    parser.add_argument('--fw_head_dim', type=int, default=0, help='Per-head width (0 = m_dim / heads)')
    parser.add_argument('--grid_ramp_start', type=int, default=None,
                        help='Epoch where the late ramp on the REPRESENTATION weights starts '
                             '(lg_weight_l1 only; the task loss is untouched). '
                             'Default 0 = no ramp unless --grid_ramp_mult is also set')
    parser.add_argument('--grid_ramp_end', type=int, default=None,
                        help='Epoch where the ramp reaches its full multiplier (default 0 = no ramp)')
    parser.add_argument('--grid_ramp_mult', type=float, default=None,
                        help='Final multiplier for the representation weights (default 1.0 = off). '
                             'Motivation: nucleation happens at e~1200-1800, once the task gradient '
                             'has decayed; see cloud_phase6_2026-10-26/DIAGNOSIS.md')
    parser.add_argument('--task_ramp_start', type=int, default=None,
                        help='Epoch where the late TASK anneal starts (scales lx_gen and lx_gen_rel '
                             'only). Default 0 = off unless --task_ramp_mult is set')
    parser.add_argument('--task_ramp_end', type=int, default=None,
                        help='Epoch where the task anneal reaches its multiplier (default 0 = off)')
    parser.add_argument('--task_ramp_mult', type=float, default=None,
                        help='Final multiplier for the reconstruction weights (default 1.0 = off). '
                             'Test of "the task gradient is what destroys the lattice"; see '
                             'cloud_phase8_2026-10-27/REPORT_DECAY.md')
    parser.add_argument('--task_warmup_epochs', type=int, default=0,
                        help='TASK-FREE OPENING: task weight is 0 at epoch 0 and ramps linearly to 1.0 '
                             'at this epoch (0 = off, the historical behaviour). Targets the e400-1200 '
                             'window where the grid/place basin is already decided; see '
                             'cloud_phase11_2026-10-28/REPORT_GZ2.md section 5')
    parser.add_argument('--a2trans_init', choices=['xavier', 'orthogonal', 'orthogonal_scaled'], default='xavier',
                        help="Init of the g_size x g_size action-to-transition weights: 'xavier' "
                             "(historical default) or 'orthogonal' (same entry scale ~1/sqrt(g_size) "
                             "but no preferred direction -- a random rotation instead of a small nudge). "
                             "Removes the direction bias of the initial transition matrices")
    parser.add_argument('--grid_iso_scale', type=float, default=None,
                        help='Weight of the grid-code ISOTROPY loss (variance of the distance between '
                             'neighbouring g over one-cell moves); 0 disables it. Default: the calibrated '
                             'value in config/titan_params_large.py (see GRID_ISO_LOSS.md)')
    parser.add_argument('--grid_iso_mode', choices=['var', 'direction', 'cv'], default=None,
                        help="'var' (default) = variance of the step length over all one-cell pairs; "
                             "'direction' = variance of the four per-direction means; "
                             "'cv' = variance / mean^2 (scale-free, cannot be gamed by shrinking all steps)")
    parser.add_argument('--grid_iso_dist', choices=['l2', 'sq'], default=None,
                        help="Distance used for the step length: 'l2' (default) = ||dg||, 'sq' = ||dg||^2")
    parser.add_argument('--grid_iso_target', type=float, default=None,
                        help='Typical step length to keep; only active together with --grid_iso_anchor > 0')
    parser.add_argument('--grid_iso_anchor', type=float, default=None,
                        help='Weight of the one-sided anti-collapse guard anchor*relu(target-mean)^2 (0 = off)')
    parser.add_argument('--grid_iso_version', type=int, choices=[1, 2], default=None,
                        help='1 = original pooled Var(||dg||) over one-cell steps; '
                             '2 = per-unit, scale-free, metric-density CV (grid_losses.py v2)')
    parser.add_argument('--grid_iso_k', type=str, default=None,
                        help="v2 only: lag(s) in steps, e.g. '1' (default, measures the metric), "
                             "'2' (also samples the diagonal), '2,3' (average)")
    parser.add_argument('--grid_iso_nbins', type=int, default=None,
                        help='v2 only: orientation bins over [0, pi) (default 8)')
    parser.add_argument('--grid_iso_min_disp', type=float, default=None,
                        help='v2 only: drop k-step pairs whose net displacement is below this many '
                             'cells (default 1.0; guards the fold / divide-by-zero)')
    parser.add_argument('--grid_iso_agg', choices=['cv', 'var'], default=None,
                        help='v2 only: aggregate the per-unit dispersions as CV (default, scale-free) '
                             'or raw variance')
    parser.add_argument('--grid_iso_perunit', type=int, default=None,
                        help='v2 only: 1 (default) = per code dimension, then average; '
                             '0 = v1-style pooled over the batch (ablation: it homogenises lambda)')
    parser.add_argument('--grid_iso_metric', choices=['within_radius', 'raw'], default=None,
                        help='v2 only: subtract the per-displacement-radius mean before taking the '
                             'spread (default), or use the raw spread over orientation bins')
    parser.add_argument('--diag_shift_range', type=int, default=2, help='The 200-epoch grid diagnostic scans temporal shifts -N..+N with this N')
    parser.add_argument('--diag_top_k', type=int, default=10, help='Top-k units averaged in the grid-score summary')
    parser.add_argument('--diag_sub', type=str, default="4",
                        help="Fine bins per cell in the rate map / SAC: an integer, or 'auto' "
                             "(~44 bins per side). Default 4 = 0.25 cell per bin, i.e. 44x44 at "
                             "W=11 and 84x84 at W=21; use 'auto' only for wide sweeps")
    parser.add_argument('--diag_sigma', type=float, default=0.375,
                        help='Gaussian smoothing of the fine rate map, in cells (default 0.375 = 1.5 bins at sub=4)')
    parser.add_argument('--diag_legacy_ring', action='store_true',
                        help='Score on the fixed 0.2..0.9 annulus instead of the period-adapted ring')
    parser.add_argument('--diag_save_sacs', choices=['all', 'dt0'], default='all',
                        help="Store the SAC of every temporal shift ('all', default) or only "
                             "delta=0 ('dt0', opt-in: smaller npz, no best-shift SAC montage)")
    parser.add_argument('--no_diag_on_eval', dest='diag_on_eval', action='store_false',
                        help='In --mode eval, keep the legacy draw_grid instead of the high-resolution diagnostic')
    parser.set_defaults(diag_on_eval=True)
    parser.add_argument('--ssm_state_dim', type=int, default=None, help='State dimension for SSM relational memory')
    parser.add_argument('--rel_transformer_mode', choices=['flat', 'hierarchical'], default=None, help='Transformer memory mode')
    parser.add_argument('--rel_transformer_d_model', type=int, default=None, help='Transformer memory model dimension')
    parser.add_argument('--rel_transformer_nhead', type=int, default=None, help='Transformer memory attention heads')
    parser.add_argument('--rel_transformer_layers', type=int, default=None, help='Transformer memory local encoder layers')
    parser.add_argument('--rel_transformer_summary_layers', type=int, default=None, help='Transformer memory summary encoder layers')
    parser.add_argument('--rel_transformer_window_size', type=int, default=None, help='Recent-token window size for Transformer memory')
    parser.add_argument('--rel_transformer_segment_size', type=int, default=None, help='Segment size for hierarchical Transformer summaries')
    parser.add_argument('--rel_transformer_max_segments', type=int, default=None, help='Maximum retained hierarchical Transformer summaries')
    parser.add_argument('--rel_transformer_ff_mult', type=int, default=None, help='Transformer memory feed-forward expansion factor')
    parser.add_argument('--rel_transformer_dropout', type=float, default=None, help='Transformer memory dropout')
    parser.add_argument(
        '--rel-memory-impl',
        '--rel_memory_impl',
        dest='rel_memory_impl',
        choices=['legacy', 'v1', 'factorized', 'factorized_shared_qk', 'hebbian_rnn', 'hebbian_fast_rnn', 'block_hebbian_fast_rnn', 'hebbian_assoc', 'transformer', 'ssm', 'fastweights', 'meta_mlp'],
        default='legacy',
        help='Relational memory implementation used by mm_TEM.',
    )
    parser.add_argument('--train_seq_bucket', nargs='+', type=int, 
                        default=[64,],
                        help='Training sequence bucket sizes')
    parser.add_argument('--train_mem_batch', nargs='+', type=int,
                        default=[1,],
                        help='Training memory batch sizes') 

    parser.add_argument('--pretrain_seq_bucket', nargs='+', type=int, 
                        default=[64, 128, 128, 128, 64, 128],
                        help='Pretraining sequence bucket sizes')
    parser.add_argument('--pretrain_mem_batch', nargs='+', type=int,
                        default=[64, 128, 32, 8, 1, 1],
                        help='Training memory batch sizes')
    
    parser.add_argument('--world_type', default='rectangle',
                        choices=['rectangle', 'hexagonal'],
                        help='World / action set.  rectangle (default) = 4-connected '
                             'square world, n_actions=4 -- the setting of every run so '
                             'far.  hexagonal = six-neighbour hexagonal world, '
                             'n_actions=6 at 60 deg, so the motor repertoire itself is '
                             '6-fold symmetric.  Applies to the trainer and to the '
                             'diagnostic eval_par alike; default changes nothing.')
    parser.add_argument('--hex_boundary', action='store_true',
                        help='hexagonal only: clip the world to the inscribed hexagon '
                             'via square2hex().  OFF by default because it renumbers '
                             'states and would invalidate the divmod(width)-based '
                             'rate-map diagnostics; turning it on is untested here.')

    parser.add_argument('--preset', choices=['none', 'r17'], default='none',
                        help='把一整份配方收成一个名字：r17 = r17both6k_s1007 的 29 个 '
                             'flag（见 r17_preset.py）。显式给出的 flag 仍然优先，'
                             '所以可以用它当起点再改一两个参数。')
    parser.add_argument('--allow-env-drift', action='store_true',
                        help='preset r17 下允许 WALK_VEL / WALK_FAST / GRID_NORM 等环境'
                             '变量与 r17 不一致。默认是直接中止，因为 WALK_VEL=cont 会让 '
                             'act_dim 变成 2、权重形状对不上，而日志不会报错。')
    parser.add_argument('--repro-check', action='store_true',
                        help='在 e20/100/200/400 打印与归档 train.log 的逐项对账。')

    parser.set_defaults(legacy_eval=True)
    _argv = apply_preset(sys.argv[1:])
    args = parser.parse_args(_argv)
    if args.preset == 'r17':
        r17_banner()
        if not r17_check_env(allow=args.allow_env_drift):
            raise SystemExit(2)
    if getattr(args, 'repro_check', False):
        os.environ['REPRO_CHECK'] = '1'
    return args

# ===================== torch device / checkpoint helpers =====================
def _set_torch_device(device):
    """torch 风格设备字符串（'cuda:7' / 'gpu:7' / 'cpu'）-> torch.device。

    同时更新模块级 _DEFAULT_DEVICE，使 to_tensor() 与模型落在同一张卡上。
    """
    global _DEFAULT_DEVICE
    name = str(device).lower()
    if name.startswith("cuda") or name.startswith("gpu"):
        index = 0
        if ":" in name:
            try:
                index = int(name.split(":")[-1])
            except ValueError:
                index = 0
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            _DEFAULT_DEVICE = torch.device("cuda:{}".format(index))
            return _DEFAULT_DEVICE
        print("[warn] cuda:{} unavailable, falling back to cpu".format(index))
    _DEFAULT_DEVICE = torch.device("cpu")
    return _DEFAULT_DEVICE


def load_ckpt(path, map_location="cpu"):
    """读取 checkpoint（torch .pth/.pt）。返回 state_dict。"""
    try:
        return torch.load(path, map_location=map_location)
    except TypeError:
        # torch < 2.0 has no weights_only kwarg
        return torch.load(path, map_location=map_location)


if __name__ == '__main__':
    args = parse_args()
    # World / action set.  Set before any default_params() call so the trainer AND the
    # diagnostics' eval_par are built for the same world.  Default 'rectangle' keeps
    # every existing run bit-identical.
    _world = str(getattr(args, 'world_type', None) or 'rectangle').strip().lower()
    os.environ['MM_WORLD_TYPE'] = _world
    validate_args(args)
    configure_runtime(args.seed)
    prefix = args.prefix
    run_layout = ensure_run_layout(prefix)
    
    random.shuffle(args.width)

    # import pdb
    # pdb.set_trace()
    
    par = default_params(width=args.width, height=args.width)
    # --width 必须写回模块级 WIDTH/HEIGHT，否则 sync_env_config 会用
    # width=None 重新调 get_env_params，把宽度表换回默认的 [11,10,10,...]。
    globals()["WIDTH"] = int(max(args.width))
    globals()["HEIGHT"] = int(max(args.width))
    par.hex_boundary = bool(getattr(args, 'hex_boundary', False))
    if getattr(args, "device", None):
        par.device = args.device
    par.use_last_gint = args.use_last_gint
    par.mem_batch_size = args.mem_batch
    par.chunk_size = args.mem_batch
    par.train_epochs_list = args.train_epochs_list
    par.train_mem_batch = args.train_mem_batch
    par.train_batch_list = args.train_batch_list
    par.lr_train = args.lr_train
    par.lr_pretrain = args.lr_pretrain
    par.lg_act_l2_scale = args.lg_act_l2_scale
    par.lg_weight_l1_scale = args.lg_weight_l1_scale
    par.gs2g_scale_list = args.gs2g_scale_list
    par.lx_gen_scale_list = args.lx_gen_scale_list
    par.lx_gen_rel_scale_list = args.lx_gen_rel_scale_list
    par.act_pred_scale_list = args.act_pred_scale_list
    par.rel_memory_impl = args.rel_memory_impl
    par.grad_debug = args.grad_debug
    par.grad_debug_interval = args.grad_debug_interval
    par.grad_debug_param_topk = args.grad_debug_param_topk
    par.capacity_scale = args.capacity_scale
    par.use_error_correction = bool(int(args.use_error_correction))
    par.ec_alpha_ramp = int(args.ec_alpha_ramp)
    par.ec_block = int(args.ec_block)
    par.probe_every = int(args.probe_every)
    par.f_delta_hidden = int(args.f_delta_hidden)
    par.x2g_scale = float(args.x2g_scale)
    par.con_scale = float(args.con_scale)
    par.inf_scale = float(args.inf_scale)
    par.write_scale = float(args.write_scale)
    par.fw_heads = int(args.fw_heads)
    par.fw_head_dim = int(args.fw_head_dim) or None
    par.capacity_n_steps = args.capacity_n_steps
    # grid-code dimension: the only architectural knob.  g_size drives the grid module,
    # projection_g (g_size -> g_size_project) and every diagnostic that iterates over
    # units, so nothing else has to change; mem_dim does not depend on it.
    if args.g_size is not None:
        if int(args.g_size) < 4:
            raise SystemExit("--g_size must be >= 4")
        par.g_size = int(args.g_size)
    # late ramp on the representation weights (defaults: off)
    for _k in ("grid_ramp_start", "grid_ramp_end", "grid_ramp_mult"):
        _v = getattr(args, _k)
        if _v is not None:
            setattr(par, _k, _v)
    par.grid_ramp_start = int(getattr(par, "grid_ramp_start", 0))
    par.grid_ramp_end = int(getattr(par, "grid_ramp_end", 0))
    par.grid_ramp_mult = float(getattr(par, "grid_ramp_mult", 1.0))
    if par.grid_ramp_mult != 1.0 and par.grid_ramp_end <= par.grid_ramp_start:
        print("[warn] grid_ramp_mult=%g but grid_ramp_end(%d) <= grid_ramp_start(%d): ramp disabled"
              % (par.grid_ramp_mult, par.grid_ramp_end, par.grid_ramp_start), flush=True)
        par.grid_ramp_mult = 1.0
    # late task anneal (scales the reconstruction weights only; default off)
    for _k in ("task_ramp_start", "task_ramp_end", "task_ramp_mult"):
        _v = getattr(args, _k)
        if _v is not None:
            setattr(par, _k, _v)
    par.task_ramp_start = int(getattr(par, "task_ramp_start", 0))
    par.task_ramp_end = int(getattr(par, "task_ramp_end", 0))
    par.task_ramp_mult = float(getattr(par, "task_ramp_mult", 1.0))
    if par.task_ramp_mult != 1.0 and par.task_ramp_end <= par.task_ramp_start:
        print("[warn] task_ramp_mult=%g but task_ramp_end(%d) <= task_ramp_start(%d): disabled"
              % (par.task_ramp_mult, par.task_ramp_end, par.task_ramp_start), flush=True)
        par.task_ramp_mult = 1.0
    # task-free opening + transition-matrix init (phase 12)
    par.task_warmup_epochs = int(getattr(args, "task_warmup_epochs", 0) or 0)
    par.a2trans_init = str(getattr(args, "a2trans_init", "xavier"))
    # grid-code isotropy loss: CLI wins, otherwise the calibrated config default
    if args.grid_iso_scale is not None:
        par.grid_iso_scale = args.grid_iso_scale
    if args.grid_iso_mode is not None:
        par.grid_iso_mode = args.grid_iso_mode
    if args.grid_iso_dist is not None:
        par.grid_iso_dist = args.grid_iso_dist
    if args.grid_iso_target is not None:
        par.grid_iso_target = args.grid_iso_target
    if args.grid_iso_anchor is not None:
        par.grid_iso_anchor = args.grid_iso_anchor
    if args.grid_iso_version is not None:
        par.grid_iso_version = args.grid_iso_version
    if args.grid_iso_k is not None:
        par.grid_iso_k = args.grid_iso_k
    if args.grid_iso_nbins is not None:
        par.grid_iso_nbins = args.grid_iso_nbins
    if args.grid_iso_min_disp is not None:
        par.grid_iso_min_disp = args.grid_iso_min_disp
    if args.grid_iso_agg is not None:
        par.grid_iso_agg = args.grid_iso_agg
    if args.grid_iso_perunit is not None:
        par.grid_iso_perunit = args.grid_iso_perunit
    if args.grid_iso_metric is not None:
        par.grid_iso_metric = args.grid_iso_metric
    par.diag_shift_range = args.diag_shift_range
    par.diag_top_k = args.diag_top_k
    par.diag_sub = None if args.diag_sub in (None, "auto", "") else int(args.diag_sub)
    par.diag_sigma = args.diag_sigma
    par.diag_legacy_ring = bool(args.diag_legacy_ring)
    par.diag_save_sacs = args.diag_save_sacs
    par.diag_on_eval = bool(args.diag_on_eval)
    if args.diag_sub in (None, "auto", ""):
        # ``--diag_sub auto`` must reach predictive_grid_diag, which reads the env
        # var at import time; without this the module default (4) would win.
        os.environ["DIAG_HIRES_SUB"] = "auto"
    from predictive_grid_diag import diag_settings
    _diag = diag_settings()
    if par.diag_sub is not None:
        _diag["sub"] = int(par.diag_sub)
        _diag["sub_rule"] = "forced"
        _diag["pipeline"] = "legacy" if int(par.diag_sub) == 1 else "hires"
    if par.diag_sigma is not None:
        _diag["sigma"] = float(par.diag_sigma)
    if par.diag_legacy_ring:
        _diag["adapt_ring"] = False
    if par.diag_save_sacs is not None:
        _diag["save_sacs"] = str(par.diag_save_sacs)
    print(f"rel_memory_impl={par.rel_memory_impl}")
    print(f"capacity_scale={par.capacity_scale:g} (n_steps={par.capacity_n_steps})")
    print("grid_iso_scale=%g (version=%d, mode=%s, dist=%s, target=%s, anchor=%g)"
          % (getattr(par, "grid_iso_scale", 0.0), int(getattr(par, "grid_iso_version", 1)),
             getattr(par, "grid_iso_mode", "var"),
             getattr(par, "grid_iso_dist", "l2"), getattr(par, "grid_iso_target", None),
             getattr(par, "grid_iso_anchor", 0.0)))
    if int(getattr(par, "grid_iso_version", 1)) >= 2:
        print("grid_iso_v2: k=%s nbins=%d min_disp=%g agg=%s per_unit=%d metric=%s"
              % (getattr(par, "grid_iso_k", 1), int(getattr(par, "grid_iso_nbins", 8)),
                 float(getattr(par, "grid_iso_min_disp", 1.0)),
                 getattr(par, "grid_iso_agg", "cv"),
                 int(getattr(par, "grid_iso_perunit", 1)),
                 getattr(par, "grid_iso_metric", "within_radius")))
    print(f"diag_shift_range=+/-{par.diag_shift_range}")
    print("diag=%s" % _diag)
    if par.grid_ramp_mult != 1.0:
        print("grid_ramp: x%.3g over epochs [%d, %d] on lg_weight_l1"
              % (par.grid_ramp_mult, par.grid_ramp_start, par.grid_ramp_end))
    if par.task_ramp_mult != 1.0:
        print("task_ramp: x%.3g over epochs [%d, %d] on lx_gen + lx_gen_rel"
              % (par.task_ramp_mult, par.task_ramp_start, par.task_ramp_end))
    if args.heads is not None:
        par.heads = args.heads
    if args.mem_dim_head is not None:
        par.mem_dim_head = args.mem_dim_head
    if args.neural_mem_expand is not None:
        par.neural_mem_expand = args.neural_mem_expand
    if args.mem_grad_norm is not None:
        par.mem_grad_norm = args.mem_grad_norm
    memory_override_names = [
        'hebbian_hidden_dim',
        'hebbian_eta',
        'hebbian_decay',
        'hebbian_detach_fast_weight',
        'hebbian_normalize_activity',
        'ssm_state_dim',
        'rel_transformer_mode',
        'rel_transformer_d_model',
        'rel_transformer_nhead',
        'rel_transformer_layers',
        'rel_transformer_summary_layers',
        'rel_transformer_window_size',
        'rel_transformer_segment_size',
        'rel_transformer_max_segments',
        'rel_transformer_ff_mult',
        'rel_transformer_dropout',
    ]
    for name in memory_override_names:
        value = getattr(args, name)
        if value is not None:
            setattr(par, name, value)
    if args.hebbian_use_oja is not None:
        par.hebbian_use_oja = bool(args.hebbian_use_oja)
    if args.hebbian_detach_fast_weight is not None:
        par.hebbian_detach_fast_weight = bool(args.hebbian_detach_fast_weight)
    if args.hebbian_normalize_activity is not None:
        par.hebbian_normalize_activity = bool(args.hebbian_normalize_activity)


    # 环境变量是第二套 CLI，不进 argparse 也就不进 flag 列表；这里至少把
    # 被覆盖的那些打出来，避免"配置只存在于上一个人的 shell 里"。
    _ov = CFG.overrides()
    print("[runtime] 环境变量开关 %d 个，被覆盖 %d 个%s"
          % (len(CFG.names()), len(_ov),
             (": " + ", ".join("%s=%s" % kv for kv in sorted(_ov.items()))) if _ov else ""),
          flush=True)
    if _ov:
        print(CFG.describe(), flush=True)

    if args.mode == "pretrain":
        raise NotImplementedError("pretrain mode is currently disabled in train_mm_TEM.py; use --mode seq_train.")
    
    elif args.mode == "seq_train":
        snapshot_run_sources(__file__, run_layout["code"])
        seq_train(model_name=prefix, pretrain_ckpt=args.ckpt, args=args, par=par, run_layout=run_layout)

    elif args.mode == "eval":
        if not args.ckpt:
            raise ValueError("eval mode requires --ckpt")
        ckpt_path = os.path.abspath(args.ckpt)
        load_dir = os.path.dirname(ckpt_path)
        ckpt_name = os.path.basename(ckpt_path)
        eval(load_dir=load_dir, ckpt_name=ckpt_name, args=args, par=par)

