import numpy as np

from config.default_config import*
import os
try:                       # 以子目录为 cwd 直接运行时也能导入
    from runtime_cfg import CFG
except ImportError:        # pragma: no cover
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runtime_cfg import CFG 
from itertools import cycle, islice

import gzip

def default_params(width=None, height=None):
    params = DotDict()
    params.batch_size = 32
    params.seq_len=64
    params.s_size= 64  # origin 45

    # grid model
    project_size = 64
    params.g_size = 64  # 16*12
    params.g_size_project = project_size # g->P

    # sense model
    params.s_size_project = project_size # s->p

    # neural memory
    params.chunk_size = 1
    params.mem_batch_size = params.chunk_size
    params.heads = 4
    params.mem_dim_head = 128
    params.rel_memory_impl = "v1"
    params.hebbian_hidden_dim = params.mem_dim_head
    params.hebbian_eta = 0.1
    params.hebbian_decay = 0.99
    params.hebbian_use_oja = True
    params.hebbian_detach_fast_weight = True
    params.hebbian_normalize_activity = True
    params.hebbian_update_interval = None
    params.ssm_state_dim = params.mem_dim_head
    
    params.rel_transformer_mode = "hierarchical"
    params.rel_transformer_d_model = params.mem_dim_head
    params.rel_transformer_nhead = params.heads
    params.rel_transformer_layers = 2
    params.rel_transformer_summary_layers = 1
    params.rel_transformer_window_size = 32
    params.rel_transformer_segment_size = 8
    params.rel_transformer_max_segments = 16
    params.rel_transformer_ff_mult = 4
    params.rel_transformer_dropout = 0.0

    # loss parameters
    params.which_costs = ['lx_gen', 'lx_gen_rel', 'l_mem', 'lg_act_l2', 'lg_weight_l1', 'l_act_pred']
    params.lg_act_l2_scale = 1e-4
    params.lg_weight_l1_scale = 1e-5
    params.act_pred_scale = 0.0
    # capacity (anti-sparsity) loss of the grid code; 0 disables it (original behaviour),
    # this copy of the project uses 1e-3 (see train_mm_TEM.py --capacity_scale)
    params.capacity_scale = 1e-3
    params.capacity_n_steps = 10
    # grid-code ISOTROPY loss (see grid_losses.py and GRID_ISO_LOSS.md).
    #   v1 = Var(||g_t - g_{t-1}||) pooled over one-cell moves  (2026-10-25)
    #   v2 = per-unit, scale-free metric-density CV              (2026-10-26)
    # The 36-run sweep of 2026-10-26 showed v1 does not help and *does* homogenise:
    # it cut the coarse module from 33 units to 1 and dropped GS60 top-10 from 1.38
    # to 0.40 (it pools across units, and units with different lambda are what a
    # module count means).  v2 fixes that by construction; the default is still
    # **off**, so a baseline run stays bit-identical to the capacity-only model and
    # every experiment has to opt in with --grid_iso_scale.  Re-run
    # `calibrate_grid_iso.py --version 2` on the training box for the weight.
    params.grid_iso_scale = 0.0
    params.grid_iso_version = 2
    # v1 knobs (only read when version = 1)
    params.grid_iso_mode = 'var'
    params.grid_iso_dist = 'l2'
    # v2 knobs
    params.grid_iso_k = '1'        # '1' = metric; '2' also samples the diagonal (confounded)
    params.grid_iso_nbins = 8
    params.grid_iso_min_disp = 1.0
    params.grid_iso_agg = 'cv'
    params.grid_iso_perunit = 1
    params.grid_iso_metric = 'within_radius'
    params.grid_iso_target = None
    params.grid_iso_anchor = 0.0

    # --- late ramp on the representation weights --------------------------------
    # Nucleation only happens at e~1200-1800, once the task gradient has decayed
    # enough for these terms to matter (cloud_phase6_2026-10-26/DIAGNOSIS.md).  The
    # ramp multiplies ONLY lg_weight_l1_scale -- never the
    # task losses, and it never introduces a lattice target.  mult=1.0 disables it,
    # which keeps every earlier run reproducible.
    params.grid_ramp_start = 0
    params.grid_ramp_end = 0
    params.grid_ramp_mult = 1.0
    # --- late anneal of the TASK weights ---------------------------------------
    # The task loss is the one force that keeps pulling the code away from a lattice
    # after e~2200 (the representation weights are non-monotone in strength, so the
    # decay is not a dose problem -- cloud_phase8_2026-10-27/REPORT_DECAY.md).  This
    # scales lx_gen / lx_gen_rel only; it adds no lattice target.
    params.task_ramp_start = 0
    params.task_ramp_end = 0
    params.task_ramp_mult = 1.0
    # 200-epoch grid diagnostic: temporal shifts -N..+N and top-k units in the summary
    params.diag_shift_range = 2
    params.diag_top_k = 10

    # system 
    params.device = 'cuda:0'

    # World / action set.  'rectangle' = 4-connected square world (n_actions=4, the
    # setting of every round so far); 'hexagonal' = six-neighbour hexagonal world
    # (n_actions=6 at 60 deg).  The choice is read from MM_WORLD_TYPE so that *every*
    # default_params() call in the process -- the trainer AND the separate eval_par
    # built for diagnostics -- agrees; --world_type sets that variable before any
    # params are built.  Default stays 'rectangle', so old runs are bit-identical.
    params.world_type = CFG.world_type

    env_par = get_env_params(params, width, height)
    params.n_actions = env_par.n_actions
    params.env = env_par

    return params

def get_env_params(pars, width=None, height=None):
    
    world = str(getattr(pars, 'world_type', 'rectangle')).strip().lower()
    if world == 'hexagonal':
        # Hexagonal.relation() returns exactly these names; n_actions = len-1 = 6,
        # i.e. six moves at 60 degrees (left/right + 4 diagonals of the offset grid).
        rels = ['down left', 'down right', 'up left', 'up right', 'left', 'right', 'stay still']
    elif world == 'rectangle':
        rels = ['down', 'up', 'left', 'right', 'stay still']
    else:
        rels = ['down', 'up', 'left', 'right', 'stay still']

    if isinstance(width, int):
        width = [width] * pars.batch_size
        height = [height] * pars.batch_size
    elif isinstance(width, list):
        width = width * int(pars.batch_size / len(width))
        height = height * int(pars.batch_size / len(height))
    else:
        width = width
        height = height

    par_env = DotDict({'stay_still': True,
                  'bias_type': 'angle',
                  'direc_bias': 0.25,
                  'angle_bias_change': 0.4,
                  'restart_max': 40,
                  'restart_min': 5,
                  'seq_jitter': 30,
                  'save_walk': 30,
                  'sum_inf_walk': 30,
                  'widths': [11, 10, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9] if not width else width,
                  'heights': [11, 10, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9] if not height else height,
                  'rels': rels,
                  # Hexagonal only: False keeps the full w*h offset grid (state s ->
                  # (s//w, s%w), exactly like Rectangle, so every diagnostic keeps
                  # working unchanged); True shrinks it to the inscribed hexagon via
                  # square2hex(), which changes the state indexing and breaks divmod.
                  'hex_boundary': bool(getattr(pars, 'hex_boundary', False)),
                  })
    size_increase = 0
    par_env.widths = [width + size_increase for width in par_env.widths]
    par_env.heights = [height + size_increase for height in par_env.heights]
    if world == 'hexagonal':
        if par_env.hex_boundary:
            from envs.environments import Hexagonal
            n_states = [Hexagonal.get_n_states(width)[0] for width in par_env.widths]
        else:
            # full offset grid: state count identical to Rectangle, so max_states,
            # the walk length rule and the divmod-based diagnostics are untouched.
            n_states = [w * h for w, h in zip(par_env.widths, par_env.heights)]
    else:
        n_states = [Rectangle.get_n_states(width, height) for width, height in zip(par_env.widths, par_env.heights)]

    # repeat widths and height
    par_env.widths = list(islice(cycle(par_env.widths), pars.batch_size))
    par_env.heights = list(islice(cycle(par_env.heights), pars.batch_size))
    # print(par_env.widths)
    pars.max_states = np.max(n_states)
    par_env.n_actions = len(par_env.rels) if 'stay still' not in par_env.rels else len(par_env.rels) - 1

    return par_env
