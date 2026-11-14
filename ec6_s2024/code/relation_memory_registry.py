from __future__ import annotations

import importlib

from torch import nn


TITANS_IMPLS = {"legacy", "v1", "factorized", "factorized_shared_qk"}


# ---------------------------------------------------------------------------
# 本包是「单个 run 的复现包」，该 run 用的是 rel_memory_impl=hebbian_assoc，
# 因此只保留了 hebbian_assoc 实际用到的后端实现文件：
#     relation_memory_interface.py, relation_memory_hebbian.py
# 其余后端（v1 / factorized / factorized_shared_qk / legacy / hebbian_rnn /
# block_hebbian_fast_rnn / transformer / ssm）改为「按需惰性导入」，
# 只有真的请求该后端时才会去 import，所以删掉那些文件不影响 hebbian_assoc。
#
# 与原版的唯一差别：原来的 6 条模块顶层 `from ... import ...` 被移进了各自的分支。
# 这些模块在 import 时不做任何随机初始化（已静态核查：模块顶层无 RNG 调用），
# 因此对 hebbian_assoc 的训练结果是逐比特等价的（已用 400 epoch 复跑验证）。
# 详见 PRUNED.md。
# ---------------------------------------------------------------------------


def _load_backend(impl: str, module: str):
    """按需导入后端模块；文件不在本包里时给出可操作的提示。"""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"rel_memory_impl={impl!r} requires module {module!r}, which is not "
            "included in this slimmed package (only 'hebbian_assoc' is kept). "
            "Either take it from the original project mm_tem_diff_rnn/, or "
            "restore ../xl_repro_hebbian_assoc_0810a_preprune_backup.tar.gz "
            "— see PRUNED.md.\n"
            f"underlying error: {exc}"
        ) from exc


def build_relational_memory(
    par,
    p2g_mu: nn.Module | None = None,
    action_pred_head: nn.Module | None = None,
):
    memory_impl = getattr(par, "rel_memory_impl", "legacy")

    if memory_impl in TITANS_IMPLS:
        build_titans_memory_adapter = _load_backend(
            memory_impl, "relation_memory_titans_adapter"
        ).build_titans_memory_adapter
        return build_titans_memory_adapter(
            memory_impl,
            par,
            p2g_mu=p2g_mu,
            action_pred_head=action_pred_head,
        )

    if memory_impl in {"hebbian_rnn", "hebbian_fast_rnn"}:
        HebbianRNNRelationalMemory = _load_backend(
            memory_impl, "relation_memory_hebbian_rnn"
        ).HebbianRNNRelationalMemory
        hebbian_update_interval = getattr(par, "hebbian_update_interval", None)
        if hebbian_update_interval is None:
            hebbian_update_interval = getattr(par, "chunk_size", 1)
        return HebbianRNNRelationalMemory(
            g_dim=par.g_size_project,
            x_dim=par.s_size_project,
            hidden_dim=getattr(par, "hebbian_hidden_dim", par.mem_dim_head),
            eta=getattr(par, "hebbian_eta", 0.05),
            decay=getattr(par, "hebbian_decay", 0.99),
            use_oja=getattr(par, "hebbian_use_oja", True),
            detach_fast_weight=getattr(par, "hebbian_detach_fast_weight", True),
            normalize_activity=getattr(par, "hebbian_normalize_activity", True),
            update_interval=hebbian_update_interval,
            p2g_mu=p2g_mu,
        )

    if memory_impl == "block_hebbian_fast_rnn":
        BlockHebbianRNNRelationalMemory = _load_backend(
            memory_impl, "relation_memory_block_hebbian_rnn"
        ).BlockHebbianRNNRelationalMemory
        return BlockHebbianRNNRelationalMemory(
            g_dim=par.g_size_project,
            x_dim=par.s_size_project,
            hidden_dim=getattr(par, "hebbian_hidden_dim", par.mem_dim_head),
            eta=getattr(par, "hebbian_eta", 0.05),
            decay=getattr(par, "hebbian_decay", 0.99),
            use_oja=getattr(par, "hebbian_use_oja", True),
            detach_fast_weight=getattr(par, "hebbian_detach_fast_weight", True),
            normalize_activity=getattr(par, "hebbian_normalize_activity", True),
            p2g_mu=p2g_mu,
        )

    if memory_impl in {"fastweights", "meta_mlp"}:
        # The paper's relational memory: a meta-MLP with Titans-style fast weights and a
        # *joint* m -> m_hat reconstruction.  This is the only backend in the package
        # that can serve L_x2g / L_g2g / the error-correction query, because those need
        # the memory to hand structure back and the Hebbian store is g -> x only.
        FastWeightRelationalMemory = _load_backend(
            memory_impl, "relation_memory_fastweights"
        ).FastWeightRelationalMemory
        return FastWeightRelationalMemory(
            g_dim=par.g_size,
            x_dim=par.s_size_project,
            hidden_dim=getattr(par, "fw_head_dim", 128),
            n_heads=getattr(par, "fw_heads", 2),
            n_layers=3,
        )

    if memory_impl == "hebbian_assoc":
        # ★ 本包复现的 run 走的就是这条路；构造参数与原版逐字一致。
        HebbianRelationalMemory = _load_backend(
            memory_impl, "relation_memory_hebbian"
        ).HebbianRelationalMemory
        return HebbianRelationalMemory(
            g_dim=par.g_size_project,
            x_dim=par.s_size_project,
            hidden_dim=getattr(par, "hebbian_hidden_dim", par.mem_dim_head),
            eta=getattr(par, "hebbian_eta", 0.1),
            decay=getattr(par, "hebbian_decay", 0.99),
            use_oja=getattr(par, "hebbian_use_oja", True),
        )

    if memory_impl == "transformer":
        transformer_module = _load_backend(memory_impl, "relation_memory_transformer")
        TransformerMemoryConfig = transformer_module.TransformerMemoryConfig
        TransformerRelationalMemory = transformer_module.TransformerRelationalMemory
        config = TransformerMemoryConfig(
            mode=getattr(par, "rel_transformer_mode", "hierarchical"),
            d_model=getattr(par, "rel_transformer_d_model", par.mem_dim_head),
            nhead=getattr(par, "rel_transformer_nhead", par.heads),
            num_layers=getattr(par, "rel_transformer_layers", 2),
            summary_layers=getattr(par, "rel_transformer_summary_layers", 1),
            window_size=getattr(par, "rel_transformer_window_size", 32),
            segment_size=getattr(par, "rel_transformer_segment_size", 8),
            max_segments=getattr(par, "rel_transformer_max_segments", 16),
            ff_mult=getattr(par, "rel_transformer_ff_mult", 4),
            dropout=getattr(par, "rel_transformer_dropout", 0.0),
        )
        return TransformerRelationalMemory(
            par.g_size_project,
            par.s_size_project,
            config,
        )

    if memory_impl == "ssm":
        SSMRelationalMemory = _load_backend(
            memory_impl, "relation_memory_ssm"
        ).SSMRelationalMemory
        return SSMRelationalMemory(
            g_dim=par.g_size_project,
            x_dim=par.s_size_project,
            state_dim=getattr(par, "ssm_state_dim", par.mem_dim_head),
        )

    raise ValueError(f"unknown relational memory implementation: {memory_impl}")
