import torch
from torch import nn
import torch.nn.functional as F

import os
import shutil
import numpy as np 


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        # We trust the dict to init itself better than we can.
        dict.__init__(self, *args, **kwargs)
        # Because of that, we do duplicate work, but it's worth it.
        for k, v in self.items():
            self.__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            # Maintain consistent syntactical behaviour.
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, DotDict.__convert(v))

    __setattr__ = __setitem__

    def __delattr__(self, k):
        try:
            dict.__delitem__(self, k)
        except KeyError:
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    @staticmethod
    def __convert(o):
        """
        Recursively convert `dict` objects in `dict`, `list`, `set`, and
        `tuple` objects to `attrdict` objects.
        """
        if isinstance(o, dict):
            o = DotDict(o)
        elif isinstance(o, list):
            o = list(DotDict.__convert(v) for v in o)
        elif isinstance(o, set):
            o = set(DotDict.__convert(v) for v in o)
        elif isinstance(o, tuple):
            o = tuple(DotDict.__convert(v) for v in o)
        return o

    @staticmethod
    def to_dict(data):
        """
        Recursively transforms a dotted dictionary into a dict
        """
        if isinstance(data, dict):
            data_new = {}
            for k, v in data.items():
                data_new[k] = DotDict.to_dict(v)
            return data_new
        elif isinstance(data, list):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, set):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, tuple):
            return [DotDict.to_dict(i) for i in data]
        else:
            return data



def snapshot_run_sources(entry_file, code_dir):
    """把整份源码快照到 run/code/，不只是入口脚本。

    以前这里只复制 train_mm_TEM.py（和同名 .sh），于是每次运行留下的
    code/ 目录都缺 grid_cell.py、mm_TEM.py、envs/*、config/*。等到几个月
    后想复现，手上只有入口脚本，其余模块早被改过——这就是 r17 复现不了
    的根因。现在整包复制，run 目录自己就能自证。
    """
    entry_path = os.path.abspath(entry_file)
    target_dir = os.path.abspath(code_dir)
    os.makedirs(target_dir, exist_ok=True)

    shutil.copy2(entry_path, os.path.join(target_dir, os.path.basename(entry_path)))

    shell_script_path = os.path.splitext(entry_path)[0] + ".sh"
    if os.path.isfile(shell_script_path):
        shutil.copy2(shell_script_path, os.path.join(target_dir, os.path.basename(shell_script_path)))

    src_root = os.path.dirname(entry_path)
    skip_dirs = {"__pycache__", ".git", ".idea", ".vscode", "runs", "figures", "node_modules"}
    n_copied = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames
                       if d not in skip_dirs and not d.startswith(".")
                       and os.path.abspath(os.path.join(dirpath, d)) != target_dir]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            src = os.path.join(dirpath, name)
            if os.path.abspath(src) == entry_path:
                continue
            rel = os.path.relpath(src, src_root)
            dst = os.path.join(target_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n_copied += 1
    if n_copied:
        print("[snapshot] 源码快照：入口 1 + 模块 %d 个 -> %s"
              % (n_copied, target_dir), flush=True)



def build_run_layout(run_root):
    run_root = os.path.abspath(run_root)
    return {
        "root": run_root,
        "pretrain": os.path.join(run_root, "pretrain"),
        "train": os.path.join(run_root, "train"),
        "logs_train": os.path.join(run_root, "logs/train"),
        "logs_pretrain": os.path.join(run_root, "logs/pretrain"),
        "code": os.path.join(run_root, "code"),
    }


def ensure_run_layout(run_root):
    layout = build_run_layout(run_root)
    for path in layout.values():
        os.makedirs(path, exist_ok=True)
    return layout


def infer_run_root(path):
    abs_path = os.path.abspath(path)
    load_dir = abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path)
    if os.path.basename(load_dir) in {"pretrain", "train", "logs", "code"}:
        return os.path.dirname(load_dir)
    return load_dir


def validate_args(args):
    if len(args.train_seq_bucket) != len(args.train_mem_batch):
        raise ValueError("--train_seq_bucket and --train_mem_batch must have the same length")
    if len(args.pretrain_seq_bucket) != len(args.pretrain_mem_batch):
        raise ValueError("--pretrain_seq_bucket and --pretrain_mem_batch must have the same length")
    for arg_name in ("train_epochs_list", "train_batch_list", "gs2g_scale_list", "act_pred_scale_list"):
        if not hasattr(args, arg_name):
            continue
        values = getattr(args, arg_name)
        if values is None:
            continue
        if len(values) not in (1, len(args.train_seq_bucket)):
            raise ValueError(f"--{arg_name} must have length 1 or match --train_seq_bucket")
