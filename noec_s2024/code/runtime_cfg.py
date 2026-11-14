# -*- coding: utf-8 -*-
"""运行时开关的唯一来源（环境变量的集中定义）。

为什么需要它
------------
本项目有约 20 个 `os.environ.get(...)` 开关，它们**不进 argparse**，因此
不进日志的 flag 列表；想查一次运行的实际配置只能去 stdout 里捞，而
`r17_preset.check_env` 的白名单只覆盖了其中 7 个。

这里把它们收成一张表：定义、默认值、类型转换、白名单、启动打印。

读取时机（重要）
----------------
采用**逐变量、首次访问时缓存**，而不是 import 时一次性快照。原因：
`MM_WORLD_TYPE` / `DIAG_HIRES_SUB` / `REPRO_CHECK` 这三个变量是程序自己
在 parse_args 之后写入的（见 train_mm_TEM.py），一次性快照会读到旧值。
逐变量缓存则完全保持原有的读取时机：
  * `grid_cell` 里的 BPTT_TRUNCATE 在构造模型时求值 —— 时机不变；
  * `predictive_grid_diag` 的 DIAG_* 在首次 import 该模块时求值 ——
    而该模块是在 parse_args 之后才被惰性导入的。

约束
----
取值与默认值与原实现逐字一致；未设置环境变量时行为不变。
"""
from __future__ import annotations

import os

__all__ = ["CFG", "SPEC"]


#: 变量名 -> (默认值字符串, 说明)。顺序即启动打印顺序。
SPEC = [
    ("BPTT_TRUNCATE",      "0",    "每 k 步 detach 一次状态（0 = 不截断）"),
    ("GRID_INIT_PERTURB",  "0",    "初始化相对扰动幅度（小误差放大实验用）"),
    ("EC_SOURCE",          "obs",  "EC 锚：obs = 观测码，ret = 记忆检索码"),
    ("EC_GAIN_INIT",       "0.105","EC 逐维增益的初值"),
    ("WALK_VEL",           "",     "cont = 连续速度臂（act_dim 变成 2）"),
    ("WALK_VEL_A",         "1.2247448713915895", "连续臂速度采样上界"),
    ("WALK_FAST",          "",     "1 = 向量化游走（分布同、随机流不同）"),
    ("WALK_MODE",          "random","游走策略：random / deficit / aligned"),
    ("LR_STEP_SIZE",       "500",  "StepLR 的 step（长跑需放大）"),
    ("DIAG_EVERY",         "200",  "诊断间隔 epoch（磁盘开销的主要来源）"),
    ("SNAP_EVERY",         "0",    "权重快照间隔 epoch（0 = 关闭）"),
    ("DIAG_LEGACY_RING",   "0",    "1 = 旧诊断管线（sub=1 + 固定环带）"),
    ("DIAG_HIRES_SUB",     "4",    "诊断每格细分数，auto = 目标 44 bin"),
    ("DIAG_HIRES_SIGMA",   "",     "诊断平滑（原始细 bin），空 = 用 CELLS 版本"),
    ("DIAG_HIRES_SIGMA_CELLS", "0.375", "诊断平滑（以 cell 为单位）"),
    ("DIAG_ADAPT_RING",    "1",    "1 = 环带随实测周期自适应"),
    ("DIAG_SAVE_SACS",     "all",  "all / dt0（只存 delta=0 的 SAC）"),
    ("MM_WORLD_TYPE",      "rectangle", "世界类型（parse_args 会用 --world_type 覆盖）"),
    ("REPRO_CHECK",        "",     "1 = 打印与归档日志的指纹对账"),
]

_BOOL_TRUE = ("1", "true", "yes", "on")


class _Cfg:
    """逐变量懒读取 + 缓存。所有类型错误都退回默认值，绝不因环境变量炸训练。"""

    def __init__(self):
        self._cache = {}

    # -- 原语 ----------------------------------------------------------
    def raw(self, name, default=""):
        if name not in self._cache:
            val = os.environ.get(name)
            self._cache[name] = default if val is None else val
        return self._cache[name]

    def _str(self, name, default):
        return str(self.raw(name, default)).strip()

    def _int(self, name, default):
        try:
            return int(float(self._str(name, default) or default))
        except (TypeError, ValueError):
            return int(float(default))

    def _float(self, name, default):
        try:
            return float(self._str(name, default) or default)
        except (TypeError, ValueError):
            return float(default)

    def _bool(self, name):
        return self._str(name, "").lower() in _BOOL_TRUE

    # -- 具体开关 ------------------------------------------------------
    @property
    def bptt_truncate(self):
        return max(0, self._int("BPTT_TRUNCATE", 0))

    @property
    def init_perturb(self):
        return max(0.0, self._float("GRID_INIT_PERTURB", 0.0))

    @property
    def ec_source(self):
        return self._str("EC_SOURCE", "obs") or "obs"

    @property
    def ec_gain_init(self):
        return self._float("EC_GAIN_INIT", 0.105)

    @property
    def walk_vel_cont(self):
        return self._str("WALK_VEL", "").lower() == "cont"

    @property
    def walk_vel_a(self):
        return self._float("WALK_VEL_A", 1.2247448713915895)

    @property
    def walk_fast(self):
        return self._str("WALK_FAST", "") == "1"

    @property
    def walk_mode(self):
        return self._str("WALK_MODE", "random") or "random"

    @property
    def lr_step_size(self):
        return max(1, self._int("LR_STEP_SIZE", 500))

    @property
    def diag_every(self):
        return max(1, self._int("DIAG_EVERY", 200))

    @property
    def snap_every(self):
        return max(0, self._int("SNAP_EVERY", 0))

    @property
    def diag_legacy_ring(self):
        return self._bool("DIAG_LEGACY_RING")

    @property
    def diag_hires_sub(self):
        return (self._str("DIAG_HIRES_SUB", "4") or "4").lower()

    @property
    def diag_hires_sigma(self):
        return self._str("DIAG_HIRES_SIGMA", "")

    @property
    def diag_sigma_cells(self):
        return self._float("DIAG_HIRES_SIGMA_CELLS", 0.375)

    @property
    def diag_adapt_ring(self):
        return self._bool("DIAG_ADAPT_RING")

    @property
    def diag_save_sacs(self):
        return self._str("DIAG_SAVE_SACS", "all") or "all"

    @property
    def world_type(self):
        return (self._str("MM_WORLD_TYPE", "rectangle") or "rectangle").lower()

    @property
    def repro_check(self):
        return self._bool("REPRO_CHECK")

    # -- 打印 ----------------------------------------------------------
    def describe(self):
        """一行一个变量：实际取值 + 是否被环境覆盖。"""
        out = ["runtime switches (环境变量，未列入 argparse):"]
        for name, default, desc in SPEC:
            raw = os.environ.get(name)
            shown = "默认 %-10s" % default if raw is None else "环境 %-10s" % raw.strip()
            out.append("  %-24s %s  %s" % (name, shown, desc))
        return "\n".join(out)

    def names(self):
        return [n for n, _d, _s in SPEC]

    def overrides(self):
        """被环境显式设置成非默认值的变量（启动摘要用；默认值不刷屏）。"""
        out = {}
        for name, default, _desc in SPEC:
            raw = os.environ.get(name)
            if raw is not None and raw.strip() != default:
                out[name] = raw.strip()
        return out


CFG = _Cfg()
