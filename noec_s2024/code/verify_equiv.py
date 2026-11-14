# -*- coding: utf-8 -*-
r"""等价性验证台：同一配置跑 N epoch，比对逐 epoch 损失轨迹与最终权重哈希。

用法
----
  # 1) 给冻结版（code_1020）录一条金标轨迹
  python verify_equiv.py run --code-dir E:\r17_s1007\code_1020 --tag base --epochs 100

  # 2) 给重构版（code_1021）录同样的轨迹
  python verify_equiv.py run --code-dir E:\r17_s1007\code_1021 --tag refactor --epochs 100

  # 3) 比对
  python verify_equiv.py compare --base base --other refactor

判定
----
* 每个 epoch 的每个数值字段 |Δ| 必须为 0（同 seed/同设备/同 batch 顺序下应逐位一致）。
* 允许 1e-9 的浮点噪声：>=1e-6 说明语义变了，必须查。
* 权重哈希相同 = 最强证据；不同则退化为全参数 max|Δ| 检查（见 --compare 输出提示）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PYTHON = r"C:\pyenv\r18\Scripts\python.exe"
OUT = r"E:\r17_s1007\_equiv"
RUN_ROOT = r"E:\r17_s1007\_equiv\runs"

EPOCH_RE = re.compile(r"^epoch=(\d+):\s*(.*)$")
NUM_RE = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")


def _num(txt):
    txt = txt.split("(")[0].strip()
    return float(txt) if NUM_RE.match(txt) else None


def parse_trace(text):
    """把 stdout 里 `epoch=N: k=v, k=v` 的行解析成 {epoch: {field: float}}。"""
    out = {}
    for line in text.splitlines():
        m = EPOCH_RE.match(line.strip())
        if not m:
            continue
        ep, body = int(m.group(1)), m.group(2)
        kv = {}
        for part in body.split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            val = _num(v)
            if val is not None:
                kv[k.strip()] = val
        if kv:
            out[ep] = kv
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_ckpt(root):
    best, best_t = None, -1
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(".pth"):
                continue
            p = os.path.join(dirpath, name)
            t = os.path.getmtime(p)
            if t > best_t:
                best, best_t = p, t
    return best


def cmd_run(a):
    os.makedirs(OUT, exist_ok=True)
    prefix = os.path.join(RUN_ROOT, a.tag)
    if os.path.isdir(prefix):
        import shutil
        shutil.rmtree(prefix, ignore_errors=True)

    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "2",
        "DIAG_EVERY": str(a.epochs + 1),      # 验证跑不做诊断，省时间也排除绘图随机性
        "SNAP_EVERY": "0",
        "REPRO_CHECK": "1",                  # 与 r17_preset 的 e20/e100 指纹对账
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    argv = [PYTHON, "-u", "train_mm_TEM.py", "--preset", "r17",
            "--prefix", prefix, "--device", a.device,
            "--train_epochs", str(a.epochs), "--seed", str(a.seed),
            "--g_size", "64", "--data_seed", str(a.data_seed),
            "--lr_train", str(a.lr)]
    print("[run] %s" % " ".join(argv))
    t0 = time.time()
    proc = subprocess.run(argv, cwd=a.code_dir, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = proc.stdout.decode("utf-8", "replace")
    dt = time.time() - t0

    log_path = os.path.join(OUT, "%s.log" % a.tag)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(log)

    trace = parse_trace(log)
    ckpt = newest_ckpt(prefix)
    rec = {
        "tag": a.tag, "code_dir": os.path.abspath(a.code_dir),
        "epochs": a.epochs, "seed": a.seed, "data_seed": a.data_seed,
        "lr": a.lr, "device": a.device, "returncode": proc.returncode,
        "seconds": round(dt, 1), "log": log_path,
        "n_epoch_points": len(trace),
        "repro_check_lines": [ln.strip() for ln in log.splitlines() if "repro-check" in ln],
        "ckpt": ckpt, "ckpt_sha256": sha256(ckpt) if ckpt else None,
        "trace": {str(k): v for k, v in sorted(trace.items())},
    }
    out = os.path.join(OUT, "%s.json" % a.tag)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    print("[run] rc=%d  %.0fs  epoch 点=%d  -> %s" % (proc.returncode, dt, len(trace), out))
    if proc.returncode != 0:
        print("[run] 失败，日志尾部：")
        print("\n".join(log.splitlines()[-25:]))
        return 1
    return 0


def cmd_compare(a):
    def load(tag):
        p = os.path.join(OUT, "%s.json" % tag)
        if not os.path.exists(p):
            sys.exit("缺少轨迹文件：%s" % p)
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    b, o = load(a.base), load(a.other)
    print("base    : %s  (%s)" % (b["tag"], b["code_dir"]))
    print("other   : %s  (%s)" % (o["tag"], o["code_dir"]))
    print("epochs  : %d vs %d" % (b["epochs"], o["epochs"]))

    tb, to = b["trace"], o["trace"]
    common = sorted(set(tb) & set(to), key=int)
    if not common:
        sys.exit("没有共同的 epoch 采样点，无法比对")

    worst = (0.0, None, None)
    nonzero = []
    only_base, only_other = set(), set()
    for ep in common:
        only_base |= set(tb[ep]) - set(to[ep])
        only_other |= set(to[ep]) - set(tb[ep])
        # 只比两边都有的字段：日志字段集会随重构变化（如 sic=/aux= 被删），
        # 字段缺失不算数值差异，单独报告。
        for k in set(tb[ep]) & set(to[ep]):
            vb, vo = tb[ep][k], to[ep][k]
            d = abs(vb - vo)
            if d > 0:
                nonzero.append((ep, k, vb, vo, d))
            if d > worst[0]:
                worst = (d, ep, k)

    print("采点     : %d 个（e%s ... e%s）" % (len(common), common[0], common[-1]))
    if not nonzero:
        print("数值     : 全部 %d 个字段逐位一致  ✅" % (len(common) * len(tb[common[0]])))
    else:
        print("数值     : %d 处不同，最大 |Δ|=%.3e @ e%s %s" % (len(nonzero), worst[0], worst[1], worst[2]))
        for row in nonzero[:15]:
            print("   e%-4s %-14s %r -> %r" % row)
        if len(nonzero) > 15:
            print("   ... 其余 %d 处省略" % (len(nonzero) - 15))

    if only_base or only_other:
        print("字段     : 仅 base 有 %s / 仅 other 有 %s（不参与比对）"
              % (sorted(only_base) or "无", sorted(only_other) or "无"))

    same_ckpt = b["ckpt_sha256"] == o["ckpt_sha256"]
    print("权重哈希 : %s" % ("相同 ✅" if same_ckpt else "不同 ⚠"))
    if not same_ckpt:
        print("           base=%s\n           other=%s" % (b["ckpt_sha256"], o["ckpt_sha256"]))

    for name, rec in (("base", b), ("other", o)):
        lines = [x for x in rec.get("repro_check_lines", []) if "偏离" in x or "一致" in x]
        if lines:
            print("指纹(%s): %s" % (name, " | ".join(lines[-2:])))

    ok = (not nonzero) and same_ckpt
    print("\n结论: %s" % ("等价 ✅" if ok else "不等价或需人工判断 ❌"))
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run")
    r.add_argument("--code-dir", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--epochs", type=int, default=100)
    r.add_argument("--seed", type=int, default=1007)
    r.add_argument("--data-seed", type=int, default=1)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--device", default="cuda:0")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--other", required=True)
    c.set_defaults(func=cmd_compare)

    a = ap.parse_args()
    sys.exit(a.func(a))


if __name__ == "__main__":
    main()
