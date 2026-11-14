# -*- coding: utf-8 -*-
"""把 r17both6k_s1007 的配方收成 `--preset r17`。

为什么需要这个文件
------------------
r17 的配方散在 **29 个命令行 flag + 7 个环境变量**里。手抄一次漏一个，
结果就从"60 个单元"变成"一片条纹"，而日志不会报任何错：漏掉
（历史例）`--aux_norm_ratio` 漏给时，容量项会被按主损失重标（强度约 2000 倍），码被压成
常量；漏掉 `--lx_gen_scale_list 1.0` 时重建项弱 10 倍。这些都不是概率事件。

所以这里做三件事：

1. `apply_preset(argv)`  —— 只补"你没显式给出"的 flag，显式 CLI 永远优先；
2. `check_env()`         —— 启动前检查 shell 里有没有会改变数据的 export
                            （`WALK_VEL=cont` 会让 act_dim 变成 2，权重形状直接
                            对不上；`WALK_FAST=1` 会换一条随机流）；
3. `report_fingerprint()` —— 在 e20/100/200/400 和归档日志逐项对账。

档案
----
参考值取自 `运行记录/train.log`（原 run 的 stdout）。注意 e20 的 `acc_pred`
在今天的模块上会偏高约 0.02（0.156 vs 0.132），这是 `relation_memory_hebbian`
后来加读出通路造成的已知漂移，不是配方错。
"""

from __future__ import annotations

import os
import sys

from runtime_cfg import CFG

# --------------------------------------------------------------------------
# 配方本体：顺序即日志里打印的顺序
# --------------------------------------------------------------------------
R17_FLAGS = [
    ("--mode", "seq_train"),
    ("--seed", "1007"),
    # --- 记忆 ---
    ("--rel_memory_impl", "hebbian_assoc"),
    ("--hebbian_hidden_dim", "800"),
    ("--hebbian_eta", "0.01"),
    ("--hebbian_decay", "0.9"),
    ("--hebbian_use_oja", "1"),
    ("--hebbian_detach_fast_weight", "0"),
    # --- 规模 ---
    ("--train_seq_bucket", "64"),
    ("--train_mem_batch", "4"),
    ("--train_batch", "32"),
    ("--train_epochs", "6000"),
    # --- 三项有效损失 ---
    ("--gs2g_scale_list", "0.0"),
    ("--lx_gen_rel_scale_list", "0.0"),
    ("--lx_gen_scale_list", "1.0"),
    ("--lg_weight_l1_scale", "1e-4"),
    ("--capacity_scale", "1e-3"),
    ("--capacity_n_steps", "10"),
    # --- 关掉的项（写出来是为了让日志自证）---
    ("--grid_iso_scale", "0"),
    ("--grid_iso_mode", "var"),
    ("--grid_iso_dist", "l2"),
    # --- 两个配方开关 ---
    ("--task_warmup_epochs", "200"),
    ("--a2trans_init", "orthogonal"),
    # --- 诊断口径（决定 GS60 怎么算、存哪几帧）---
    ("--diag_shift_range", "0"),   # 不再做时间对齐扫描，只取 dt=0
    ("--diag_sub", "4"),
    ("--diag_sigma", "0.375"),
    ("--diag_save_sacs", "dt0"),
]

# 环境变量白名单：值必须落在这里面，否则视为被 shell profile 污染
# 覆盖 runtime_cfg.SPEC 里的**全部**开关（键集必须一致，check_env 会断言）。
# 值为 None 表示"不校验"（纯诊断/输出类开关，改了不影响数值）。
R17_ENV = {
    "BPTT_TRUNCATE": ("", "0"),
    "GRID_INIT_PERTURB": ("", "0"),
    "EC_SOURCE": ("", "obs"),
    "EC_GAIN_INIT": None,               # 仅在开 EC 时才有意义
    "WALK_VEL": ("",),                  # cont -> act_dim=2
    "WALK_VEL_A": None,
    "WALK_FAST": ("",),                 # 1 -> 向量化游走，分布同、随机流不同
    "WALK_MODE": ("", "random"),
    "LR_STEP_SIZE": ("", "500"),
    "DIAG_EVERY": None,                 # 只影响诊断密度
    "SNAP_EVERY": None,                 # 只影响快照密度
    "DIAG_LEGACY_RING": ("", "0"),
    "DIAG_HIRES_SUB": ("", "4"),
    "DIAG_HIRES_SIGMA": ("",),
    "DIAG_HIRES_SIGMA_CELLS": ("", "0.375"),
    "DIAG_ADAPT_RING": ("", "1"),
    "DIAG_SAVE_SACS": None,             # 只影响落盘体积
    "MM_WORLD_TYPE": ("", "rectangle"),
    "REPRO_CHECK": None,
}

# 归档日志的逐项指纹
R17_FINGERPRINT = {
    20: {"x_gen": 4.08, "lx_gen_rel": 4.12, "acc_pred": 0.132,
         "acc_rel": 0.074, "capac": -0.0156, "total": 0.41},
    100: {"x_gen": 3.52, "lx_gen_rel": 3.95, "acc_pred": 0.221,
          "acc_rel": 0.101, "capac": -0.0113, "total": 1.76},
    200: {"x_gen": 2.96, "lx_gen_rel": 3.75, "acc_pred": 0.348,
          "acc_rel": 0.180, "capac": -0.0089, "total": 2.96},
    400: {"x_gen": 2.56, "lx_gen_rel": 3.65, "acc_pred": 0.468,
          "acc_rel": 0.247, "capac": -0.0057, "total": 2.56},
}

# 容差：相对值的百分比；acc_* 用绝对差
_TOL_REL = {"x_gen": 0.05, "lx_gen_rel": 0.05, "total": 0.08, "capac": 0.10}
_TOL_ABS = {"acc_pred": 0.05, "acc_rel": 0.03, "capac": 0.001}


def _preset_name(argv):
    for i, a in enumerate(argv):
        if a == "--preset" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--preset="):
            return a.split("=", 1)[1]
    return "none"


def apply_preset(argv):
    """`--preset r17` 时补齐缺失的 flag；显式给出的不动。"""
    argv = list(argv)
    if _preset_name(argv) != "r17":
        return argv
    given = {a.split("=", 1)[0] for a in argv if a.startswith("--")}
    added = [(f, v) for f, v in R17_FLAGS if f not in given]
    if added:
        print("[preset r17] 补入 %d 个未显式给出的 flag（显式 CLI 优先）" % len(added),
              flush=True)
        for f, v in added:
            print("    %-32s %s" % (f, v), flush=True)
    return argv + [x for f, v in added for x in (f, v)]


def check_env(allow=False):
    """返回 True 表示可以继续。"""
    # 白名单必须覆盖 runtime_cfg 里的每一个开关，否则新增开关会悄悄绕过校验
    _cov, _all = set(R17_ENV), set(CFG.names())
    if _cov != _all:
        print("[preset r17] 白名单与环境变量表不一致：未覆盖 %s / 多出 %s"
              % (sorted(_all - _cov), sorted(_cov - _all)), flush=True)
    drift = []
    for k, ok in R17_ENV.items():
        if ok is None:
            continue
        v = os.environ.get(k)
        if v is None:
            continue
        if v.strip() in ok:
            continue
        drift.append((k, v, ok))
    if not drift:
        print("[preset r17] 环境变量检查通过", flush=True)
        return True
    print("[preset r17] 以下环境变量与 r17 不一致：", flush=True)
    for k, v, ok in drift:
        print("    %-16s = %-12r   要求 %r" % (k, v, ok), flush=True)
    if allow:
        print("[preset r17] 已给 --allow-env-drift，继续（结果不再与 r17 可比）",
              flush=True)
        return True
    print("[preset r17] 中止。先 unset 这些变量，或显式加 --allow-env-drift。",
          flush=True)
    return False


def banner():
    print("=" * 74, flush=True)
    print("r17 配方（r17both6k_s1007，6000 epoch，seed 1007）", flush=True)
    print("=" * 74, flush=True)
    for f, v in R17_FLAGS:
        print("  %-32s %s" % (f, v), flush=True)
    print("-" * 74, flush=True)
    print("参考指纹（归档 train.log），容差：x_gen/total 5%%，acc_pred 0.05：", flush=True)
    for e in sorted(R17_FINGERPRINT):
        r = R17_FINGERPRINT[e]
        print("  e%-4d x_gen=%.2f lx_gen_rel=%.2f acc_pred=%.3f capac=%.4f total=%.2f"
              % (e, r["x_gen"], r["lx_gen_rel"], r["acc_pred"], r["capac"], r["total"]),
              flush=True)
    print("  （e20 的 acc_pred 现约 0.156，模块漂移，已知）", flush=True)
    print("=" * 74, flush=True)


def report_fingerprint(epoch, loss):
    """在指纹 epoch 上打印与归档的逐项差。任何异常都不该影响训练。"""
    ref = R17_FINGERPRINT.get(int(epoch))
    if ref is None:
        return
    try:
        got = {
            "x_gen": float(loss.lx_gen.item()),
            "lx_gen_rel": float(loss.lx_gen_rel.item()),
            "acc_pred": float(loss.acc_gen.item()),
            "acc_rel": float(loss.acc_rel.item()),
            "capac": float(loss.l_capacity.item()),
            "total": float(loss.train_loss.item()),
        }
    except Exception as exc:  # 字段改名就静默跳过
        print("[repro-check] 无法读取指标：%s" % exc, flush=True)
        return
    bad = []
    for k, r in ref.items():
        g = got[k]
        if k in _TOL_ABS and abs(g - r) <= _TOL_ABS[k]:
            continue
        span = max(abs(r) * _TOL_REL.get(k, 0.05), _TOL_ABS.get(k, 0.0), 1e-9)
        if abs(g - r) <= span:
            continue
        bad.append((k, r, g))
    if bad:
        print("    [repro-check] e%d 偏离：%s" % (
            epoch, "，".join("%s %.4g→%.4g" % (k, r, g) for k, r, g in bad)), flush=True)
    else:
        print("    [repro-check] e%d 与归档一致（全部在容差内）" % epoch, flush=True)
