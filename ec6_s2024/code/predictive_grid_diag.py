"""Predictive-grid diagnostics: GridScore60 / GridScore90 with a time-shift search.

This module replaces ``grid_cell.draw_grid`` for the periodic (every 200 epochs)
diagnostic.  Everything that computes a rate map or a grid score is taken from
``grid_cell.GridScorer`` -- nothing is re-implemented here, the module only adds
the shift bookkeeping and the plotting.

Two changes with respect to the original analysis
-------------------------------------------------
1. **Both scores are reported for every unit**

   * ``gs60`` -- six-fold symmetry, ``(corr60 + corr120) / 2 - (corr30 + corr90 + corr150) / 3``,
     i.e. ``GridScorer.grid_score_60``; this is the standard / paper measure.
   * ``gs90`` -- four-fold symmetry, ``corr90 - (corr45 + corr135) / 2``,
     i.e. ``GridScorer.grid_score_90`` (what the old code saved as ``gridness.npy``).

   For both scores the ring mask (``mask_min`` = 0.2, ``mask_max`` in
   ``linspace(0.4, 0.9, 10)``) maximising the score is used, as in ``get_scores``.
   **Since 2026-10-20 the default is the high-resolution, period-adapted ring**
   (see below); the fixed 0.2..0.9 annulus is kept behind ``DIAG_LEGACY_RING=1``
   only as a comparison baseline.

High-resolution rate maps (default since 2026-10-20)
----------------------------------------------------
The old pipeline binned ``cells`` x ``cells`` cells into ``cells`` bins (one bin
per cell) and only then looked at the SAC.  Two artefacts came out of that:

* the ring radii are fractions of ``nbins`` (``mask_min * nbins``), so the *same*
  code measured a 2.2-cell annulus at W=11 and a 4.2-cell one at W=21 -- for a
  lattice whose first ring is at 4.24 cells the W=21 band sat on the ring and the
  score collapsed (the "W=21 falls apart" result was a tool artefact);
* a 11 x 11 map cannot resolve a 1.8-cell period at all.

Now the map is binned **directly on a fine grid** with ``DIAG_HIRES_SUB`` bins per
cell (default 4 -> 44 x 44 for W=11) over the *same* ``coords_range``
``[[0, cells], [0, cells]]``, smoothed with a Gaussian of ``DIAG_HIRES_SIGMA``
fine bins (default 1.5 = 0.375 cell), and the Skaggs SAC is taken on that fine
map (87 x 87).  No interpolation is involved: the raw visit positions are
histogrammed at the fine resolution, which is strictly better than upsampling a
coarse map.  The scoring ring is then *adapted to the measured lattice period*
(``DIAG_ADAPT_RING=1``, default on): the period is taken from the nearest 2-D
local maximum of the SAC (``grid_spacing.first_ring_spacing``) and the band is
``[0.7 s, 1.3 s]``, identical at every W.

Measured cost: ~70 s per (cap, width) group on 4 CPU threads for 5 shifts x 64
units, and ~7 MB per ``*gridshift_ratemaps.npz`` (float32 maps + SACs).

2. **Temporal shift (predictive grid cells)**

   Following Sinha, Fasehun & Redman, *Predictive Grid Cell Coding Does Not
   Require Explicit Planning*: a *predictive* grid cell fires ahead of the animal,
   so its spatial periodicity is weak when activity is aligned with the current
   position (``delta = 0``) and strong when the activity of step ``t - delta`` is
   binned at the position of step ``t``.  For ``delta in {-N, ..., +N}`` we rebuild

       R_i(j, k; delta) = sum_{t in T(j,k)} s_i(t - delta) / O(j,k)

   ``delta > 0`` is prospective (activity leads position), ``delta < 0`` is
   retrospective.  Shifts stay inside a single trajectory, so the first/last
   ``|delta|`` steps of each trajectory are dropped.

   Reported per epoch: the score at ``delta = 0``, the best score over ``delta``
   together with its shift, and **the number of units whose best shift is each
   ``delta``** (``counts_gs60`` / ``counts_gs90``).  Ties are broken towards
   ``|delta|`` smallest, hence towards ``delta = 0`` (a non-predictive unit is not
   turned into a predictive one by the search).

All figure text is ASCII; the matplotlib font is pinned to DejaVu Sans so that no
CJK font is needed and no missing-glyph boxes can appear.  Every file name is
ASCII as well.

Outputs (``draw_path`` is a prefix such as ``.../train/e200S64W11``)
------------------------------------------------------------------
``{draw_path}grid.pdf``              rate map + SAC per unit at delta = 0
``{draw_path}gridness.npy``          GS90 at delta = 0  (legacy name, GS90)
``{draw_path}gridscore60.npy``       GS60 at delta = 0
``{draw_path}gridscore90.npy``       GS90 at delta = 0
``{draw_path}gridshift_scores.npz``  deltas, all GS60/GS90, best shift per unit
``{draw_path}gridshift_ratemaps.npz`` rate maps + SACs + grid spacing, per shift
``{draw_path}gridshift_summary.json`` counts per shift + summary numbers
``{draw_path}gridshift_summary.png`` unit counts / score-vs-shift / gain
``{draw_path}gridshift.pdf``         one page per shift (all units)
``{draw_path}gridshift_best.pdf``    every unit at its own best shift
``{draw_path}gridshift_autocorr.pdf`` SAC of the shifted rate maps, one page per shift
``{draw_path}gridshift_autocorr_dt0.png`` the delta = 0 SAC page as PNG
"""

import json
import os
from runtime_cfg import CFG

import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from grid_cell import GridScorer  # noqa: E402
from grid_spacing import estimate_grid_spacing, first_ring_spacing  # noqa: E402

# 图中只允许出现 ASCII，字体固定为 DejaVu Sans（Linux/AI Studio 上必然存在）
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

RING_MINS = [0.2] * 10
RING_MAXES = np.linspace(0.4, 0.9, num=10).tolist()

# ---- high-resolution diagnostics ----------------------------------------- #
# DIAG_HIRES_SUB: fine bins per environment cell.  sub=1 is the old
#                  one-bin-per-cell map; sub=4 cuts every environment cell into
#                  4x4 bins, i.e. 0.25 cell per pixel (W=11 -> 44x44 rate map and
#                  87x87 SAC; W=21 -> 84x84 and 167x167).  Default 4: the same
#                  *physical* bin size at every width, which is what makes the
#                  W=11 and W=21 tables comparable.  "auto" keeps the map side at
#                  ~44 bins instead (W=21 -> sub 2); it is ~6x cheaper but the bin
#                  size then changes with W, so use it for wide sweeps only.
# DIAG_HIRES_SIGMA_CELLS: Gaussian smoothing in CELLS (0.375 = 1.5 fine bins at
#                  sub=4), so the kernel stays physically identical at any sub.
# DIAG_HIRES_SIGMA: legacy override in *raw fine bins* (0 disables smoothing),
#                  only needed to replay an archived run bit for bit.
# DIAG_ADAPT_RING: score on a ring adapted to the measured lattice period
#                  instead of the fixed 2.2..9.9 cells annulus.
# DIAG_LEGACY_RING: force the pre-2026-10-20 pipeline (sub=1, fixed ring,
#                  radial spacing estimator); kept for side-by-side comparison.
LEGACY = CFG.diag_legacy_ring
DEFAULT_SUB = "4"        # high-precision default: 0.25 cell per bin at any width
DIAG_SUB_ENV = CFG.diag_hires_sub
# Smoothing is specified in CELLS so that it stays the same physical kernel at
# every sub (0.375 cell = 1.5 fine bins at sub=4, the value the archived
# 2026-10-20 tables were produced with).  DIAG_HIRES_SIGMA still overrides it in
# raw fine bins, for bit-exact replay of an old run.
SIGMA_CELLS = CFG.diag_sigma_cells
_SIGMA_BINS = CFG.diag_hires_sigma
HIRES_SIGMA = float(_SIGMA_BINS) if _SIGMA_BINS else None
ADAPT_RING = (not LEGACY) and CFG.diag_adapt_ring
# DIAG_SAVE_SACS: "all" (default) keeps the SAC of every temporal shift, so the
#                 best-shift SAC montage and the per-shift SAC pages exist.
#                 "dt0" (delta = 0 only, ~60% smaller npz) is opt-in only.
SAVE_SACS = CFG.diag_save_sacs
SUB_TARGET = 44          # target rate-map side for the "auto" rule


def sub_rule_text(sub=None):
    """Human-readable provenance of the sub used for one run (goes into the json)."""
    if LEGACY:
        return "legacy(sub=1)"
    if sub is not None:
        return "forced(cli %s)" % sub
    if DIAG_SUB_ENV in ("", "auto"):
        return "auto(44 bins)"
    return "forced(env %s)" % DIAG_SUB_ENV


def resolve_sub(cells):
    """Fine bins per cell: forced value (default 4), or the constant-pixel rule."""
    if LEGACY:
        return 1
    if DIAG_SUB_ENV in ("", "auto"):
        return max(1, min(6, int(round(float(SUB_TARGET) / float(max(1, cells))))))
    return max(1, int(DIAG_SUB_ENV))


def resolve_sigma(sub, sigma_cells=None):
    """Gaussian sigma in *fine bins* from a sigma given in cells."""
    sub = max(1, int(sub))
    if sigma_cells is None:
        if HIRES_SIGMA is not None:          # DIAG_HIRES_SIGMA: raw fine bins
            return float(HIRES_SIGMA)
        sigma_cells = SIGMA_CELLS
    return max(0.0, float(sigma_cells) * sub)

CORR_ANGLES = [30, 45, 60, 90, 120, 135, 150]


def diag_settings(cells=None):
    """The knobs that produced a given npz -- written into every summary json."""
    sub = resolve_sub(cells) if cells is not None else DIAG_SUB_ENV
    return {
        "pipeline": "legacy" if LEGACY else "hires",
        "sub": sub,
        "sub_rule": sub_rule_text(),
        "sigma": float(resolve_sigma(sub if isinstance(sub, int) else 4)),
        "sigma_cells": float(SIGMA_CELLS if HIRES_SIGMA is None else
                             HIRES_SIGMA / max(1, int(sub if isinstance(sub, int) else 4))),
        "adapt_ring": bool(ADAPT_RING),
        "save_sacs": str(SAVE_SACS),
        "ring_rule": "[0.7 s, 1.3 s] on nearest 2-D SAC peak" if ADAPT_RING
                     else "fixed mask_min*nbins .. mask_max*nbins",
    }


def _adaptive_ring(spacing_bins, nbins, adapt=None):
    """Ring that brackets the first lattice ring, as fractions of ``nbins``.

    ``spacing_bins`` is the estimated lattice period in *fine* bins.  When the
    estimate fails (noise-like map) fall back to the legacy innermost ring so
    that the number stays comparable with the old table.
    """
    adapt = ADAPT_RING if adapt is None else bool(adapt)
    if not adapt or not np.isfinite(spacing_bins) or spacing_bins <= 0:
        return RING_MINS[0], RING_MAXES[0]
    lo = 0.7 * float(spacing_bins) / float(nbins)
    hi = 1.3 * float(spacing_bins) / float(nbins)
    lo = max(0.12, lo)
    hi = min(0.95, hi)
    if hi - lo < 0.05:
        return RING_MINS[0], RING_MAXES[0]
    return lo, hi


_MASK_CACHE = {}


def _ring_mask(scorer, lo_frac, hi_frac):
    """``scorer._get_ring_mask`` with a cache.

    The adapted band hardly varies between units, so without the cache the same
    circular mask is rebuilt for all 64 units of all 5 shifts; the mask only
    depends on ``(nbins, lo_frac, hi_frac)``, so caching is safe.
    """
    key = (int(scorer._nbins), round(float(lo_frac), 4), round(float(hi_frac), 4))
    mask = _MASK_CACHE.get(key)
    if mask is None:
        if len(_MASK_CACHE) > 512:
            _MASK_CACHE.clear()
        mask = scorer._get_ring_mask(float(lo_frac), float(hi_frac))
        _MASK_CACHE[key] = mask
    return mask


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_numpy(x):
    """torch.Tensor / np.ndarray / list -> np.ndarray (detached, on host)."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def shift_order(shift_range):
    """Shifts ordered by |delta| then by sign preference, so ``argmax`` breaks
    ties towards ``delta = 0`` (then ``+1``) instead of ``-N``."""
    return sorted(range(-int(shift_range), int(shift_range) + 1),
                  key=lambda d: (abs(d), -d))


def make_scorer(nbins, coords_range=None):
    """``GridScorer`` with the ring masks used by ``draw_grid_score``."""
    if coords_range is None:
        coords_range = [[0.0, float(nbins)], [0.0, float(nbins)]]
    return GridScorer(int(nbins), coords_range, list(zip(RING_MINS, RING_MAXES)))


def grid_extent(par):
    """Side length (in cells) of the square rate-map grid.

    Taken from the evaluation environment (``par.env.widths`` / ``heights``), so
    the binning does not drift with the part of the maze that happened to be
    visited.  Falls back to ``par.env_width``.
    """
    env_spec = getattr(par, "env", None)
    candidates = []
    for name in ("widths", "heights"):
        value = getattr(env_spec, name, None)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            candidates.append(max(int(v) for v in value))
        elif isinstance(value, (int, float)):
            candidates.append(int(value))
    if not candidates:
        value = getattr(par, "env_width", None)
        if value is None:
            raise AttributeError("grid diagnostic needs par.env.widths or par.env_width")
        candidates.append(int(value))
    return max(2, max(candidates))


def positions_xy(par, pos_seq, max_width=None, widths=None):
    """Flat state indices (B, S) -> (B, S, 2) integer grid coordinates.

    ``widths`` gives the arena side **per sample** (``(B,)``, as recorded by
    ``gen_data`` in ``data["width"]``).  It matters whenever the batch mixes arena
    sizes -- the training batch does (``[11, 10, 10, 11, 8, 9, ...]``) -- because
    a flat index only means a cell together with the width it was built from:
    ``107`` is ``(9, 8)`` at width 11 but ``(10, 7)`` -- off the board -- at width
    10, and with a single scalar width those samples silently get the wrong cell.

    Diagnostics that evaluate on one shared arena (all ``eval_par.env.widths``
    equal) pass ``max_width`` and keep the old, bit-identical behaviour.
    """
    pos_np = _as_numpy(pos_seq).astype(np.float64)
    if pos_np.ndim == 3 and pos_np.shape[-1] == 2:   # already absolute coordinates
        return pos_np
    if widths is not None:
        w = np.asarray(widths, dtype=np.float64).reshape(-1)
        if w.size not in (1, pos_np.shape[0]):
            raise ValueError("widths must have length 1 or %d, got %d"
                             % (pos_np.shape[0], w.size))
        if w.size == 1:
            w = np.repeat(w, pos_np.shape[0])
        shape = (-1,) + (1,) * (pos_np.ndim - 1)
        w = w.reshape(shape)
        pos_x = np.floor(pos_np / w)
        pos_y = np.mod(pos_np, w)
        return np.stack([pos_x, pos_y], axis=-1)
    if max_width is None:
        max_width = grid_extent(par)
    pos_flat = pos_np.reshape(-1)
    pos_x = np.floor(pos_flat / float(max_width))
    pos_y = pos_flat % float(max_width)
    xy = np.stack([pos_x, pos_y], axis=-1)
    if pos_np.ndim == 2:
        return xy.reshape(pos_np.shape + (2,))
    return xy


def shift_pairs(pos_xy, act, delta):
    """``(positions, activities)`` that survive a shift of ``delta`` steps.

    Position index ``t`` is paired with activity index ``t - delta``, so a
    positive ``delta`` aligns the *past* activity with the *current* position.
    """
    S = act.shape[1]
    lo = max(0, int(delta))
    hi = S + min(0, int(delta))
    if hi <= lo:
        raise ValueError("shift %s is incompatible with sequence length %s" % (delta, S))
    return pos_xy[:, lo:hi], act[:, lo - int(delta):hi - int(delta)]


def _first_ring_spacing(sac, nbins, sub):
    """Nearest non-central SAC peak, **in fine bins**.

    Thin wrapper around ``grid_spacing.first_ring_spacing`` (single source of
    truth, shared with the clustering tools) which returns cells; this keeps the
    fine-bin convention used inside the diagnostics (ring fractions are taken
    over ``nbins`` fine bins).
    """
    return float(first_ring_spacing(sac, nbins, sub)) * float(max(1, sub))


def score_shift(scorer, nbins, pos_xy, act, delta, cells=None, sigma=None,
                adapt_ring=None, sub=None):
    """Rate maps, SACs and both grid scores of every unit at one temporal shift.

    ``cells`` is the side of the environment in grid cells.  When it is given
    and differs from ``nbins``, the map is computed on a ``nbins x nbins`` fine
    grid spanning the same ``[0, cells]`` extent, smoothed with a Gaussian of
    ``sigma`` *fine bins* (= ``sigma * cells`` cells), and scored with a ring
    adapted to the measured lattice period.  ``sub`` overrides the inferred
    ``nbins / cells``; ``adapt_ring=False`` forces the legacy fixed annulus.
    """
    if sub is None:
        sub = 1 if cells is None else max(1, int(round(float(nbins) / float(cells))))
    sub = max(1, int(sub))
    sigma_bins = resolve_sigma(sub, sigma)
    adapt = ADAPT_RING if adapt_ring is None else bool(adapt_ring)
    pos_sel, act_sel = shift_pairs(pos_xy, act, delta)
    pos_sel = pos_sel.reshape(-1, pos_sel.shape[-1])
    act_sel = act_sel.reshape(-1, act_sel.shape[-1])
    n_units = int(act_sel.shape[1])

    maps = np.zeros((n_units, nbins, nbins), dtype=np.float64)
    sacs = np.zeros((n_units, 2 * nbins - 1, 2 * nbins - 1), dtype=np.float64)
    score_60 = np.zeros(n_units, dtype=np.float64)
    score_90 = np.zeros(n_units, dtype=np.float64)
    mask_60 = [None] * n_units
    mask_90 = [None] * n_units

    spacing = np.full(n_units, np.nan, dtype=np.float64)

    # Occupancy of the fine grid: how many samples a non-empty bin averages.
    # This is the number that decides whether a finer ``sub`` buys resolution or
    # only noise: with few samples per bin the extra pixels are interpolation.
    occ_hist, _, _ = np.histogram2d(pos_sel[:, 0], pos_sel[:, 1],
                                    bins=int(scorer._nbins), range=scorer._coords_range)
    occ_nonzero = occ_hist[occ_hist > 0]
    occupancy = {
        "bins": int(occ_hist.size),
        "occupied": int(occ_nonzero.size),
        "frac_occupied": float(occ_nonzero.size / max(1, occ_hist.size)),
        "samples_per_bin_mean": float(occ_hist.mean()),
        "samples_per_occupied_mean": float(occ_nonzero.mean()) if occ_nonzero.size else 0.0,
        "samples_per_occupied_median": float(np.median(occ_nonzero)) if occ_nonzero.size else 0.0,
    }

    for i in range(n_units):
        rate_map = scorer.calculate_ratemap(pos_sel[:, 0], pos_sel[:, 1], act_sel[:, i])
        if sub > 1:
            if sigma_bins > 0:
                rate_map = ndimage.gaussian_filter(np.nan_to_num(rate_map), sigma_bins)
            else:
                rate_map = np.nan_to_num(rate_map)
            sac = scorer.calculate_sac(rate_map)
            rot = scorer.rotated_sacs(sac, CORR_ANGLES)
            sp_bins = _first_ring_spacing(sac, nbins, sub)
            spacing[i] = float(sp_bins) / sub if np.isfinite(sp_bins) else np.nan
            lo_frac, hi_frac = _adaptive_ring(sp_bins, nbins, adapt)
            mask = _ring_mask(scorer, lo_frac, hi_frac)
            s60, s90, _ = scorer.get_grid_scores_for_mask(sac, rot, mask)
            m60 = m90 = (lo_frac, hi_frac)
            sac = np.asarray(sac, dtype=np.float64)
        else:
            s60, s90, m60, m90, sac = scorer.get_scores(rate_map)
            # grid spacing (lattice period, in cells) of this shift's rate map
            spacing[i] = estimate_grid_spacing(sac, nbins)["spacing"]
        maps[i] = np.nan_to_num(rate_map)
        sacs[i] = np.nan_to_num(sac)
        score_60[i] = float(s60) if np.isfinite(s60) else 0.0
        score_90[i] = float(s90) if np.isfinite(s90) else 0.0
        mask_60[i] = m60
        mask_90[i] = m90

    return {
        "delta": int(delta),
        "maps": maps,
        "sacs": sacs,
        "score_60": score_60,
        "score_90": score_90,
        "mask_60": mask_60,
        "mask_90": mask_90,
        "spacing": spacing,
        "occupancy": occupancy,
        "n_pairs": int(pos_sel.shape[0]),
    }


# --------------------------------------------------------------------------- #
# figures (English only)
# --------------------------------------------------------------------------- #
def _plot_ratemap(ratemap, ax, title, cmap="jet"):
    ax.imshow(ratemap, interpolation="none", cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7)


def scorers_nbins(sac):
    """Recover ``nbins`` from a SAC array (shape ``2 * nbins - 1``)."""
    return (sac.shape[0] + 1) // 2


def _layout(n_units, cols=16):
    return cols, int(np.ceil(n_units / cols))


def _page_all_units(maps, sacs, mask_60, score_60, score_90, order, path, suptitle):
    """One PDF page: rate map + SAC for every unit, scores in the panel title."""
    n_units = maps.shape[0]
    cols, rows = _layout(n_units)
    fig = plt.figure(figsize=(24, max(2.0, rows * 4.0)))
    for slot, idx in enumerate(order):
        ax_map = fig.add_subplot(rows * 2, cols, slot + 1)
        ax_sac = fig.add_subplot(rows * 2, cols, rows * cols + slot + 1)
        title = "%d (GS60=%.2f, GS90=%.2f)" % (idx, score_60[idx], score_90[idx])
        _plot_ratemap(maps[idx], ax_map, title)
        _plot_sac_simple(sacs[idx], mask_60[idx], ax_sac, title)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def _plot_sac_simple(sac, mask_params, ax, title, cmap="jet"):
    """SAC panel; the unused corners are hidden with NaN, like ``plot_sac``."""
    nbins = scorers_nbins(sac)
    size = sac.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.sqrt((xx - (nbins - 1)) ** 2 + (yy - (nbins - 1)) ** 2)
    visible = np.where(radius <= nbins, 1.0, np.nan)
    ax.imshow(sac * visible, interpolation="none", cmap=cmap)
    if mask_params is not None:
        point = (nbins - 1, nbins - 1)
        for r in mask_params:
            ax.add_artist(plt.Circle(point, r * nbins, fill=False, edgecolor="k", lw=0.6))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7)


def _page_shift_all_units(result, order, pdf, epoch):
    n_units = result["maps"].shape[0]
    cols, rows = _layout(n_units)
    fig = plt.figure(figsize=(24, max(2.0, rows * 2.4)))
    for slot, idx in enumerate(order):
        ax = fig.add_subplot(rows, cols, slot + 1)
        _plot_ratemap(result["maps"][idx], ax,
                      "%d (GS60=%.2f)" % (idx, result["score_60"][idx]))
    fig.suptitle("%sdt = %+d : activity of step t%+d binned at the position of step t"
                 % ("" if epoch is None else "epoch %d, " % int(epoch), result["delta"], -result["delta"]),
                 fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _sac_page(result, order, suptitle, cols=8):
    """One page of SACs (spatial autocorrelograms) for every unit, with spacing."""
    n_units = result["maps"].shape[0]
    cols = int(min(cols, n_units))
    rows = int(np.ceil(n_units / float(cols)))
    fig = plt.figure(figsize=(3.0 * cols, 3.0 * rows + 0.4))
    for slot, idx in enumerate(order):
        ax = fig.add_subplot(rows, cols, slot + 1)
        spacing = result["spacing"][idx]
        title = "%d (GS60=%.2f" % (idx, result["score_60"][idx])
        if np.isfinite(spacing):
            title += ", sp=%.2f" % spacing
        title += ")"
        _plot_sac_simple(result["sacs"][idx], result["mask_60"][idx], ax, title)
    fig.suptitle(suptitle, fontsize=min(30.0, 2.4 * cols), y=0.992)
    # tight_layout does not cope with 8 x 8 panel pages: reserve the title band by hand
    fig.subplots_adjust(left=0.012, right=0.992, bottom=0.012,
                        top=0.945 if rows >= 4 else 0.962,
                        wspace=0.06, hspace=0.24)
    return fig


def _sac_suptitle(result, epoch=None):
    head = "SAC" if epoch is None else "epoch %d, SAC" % int(epoch)
    return ("%s of the rate map at time shift %+d (activity of step t%+d binned at "
            "the position of step t)" % (head, result["delta"], -result["delta"]))


def _summary_figure(path, deltas, score_60, score_90, best_60_idx, best_90_idx,
                    counts60, counts90, valid, epoch=None):
    """Three panels: units per best shift, score vs shift, gain from shifting.

    ``deltas`` arrives in *search* order (``shift_order``: 0, +1, -1, +2, -2 ...) so
    that ``argmax`` breaks ties towards small ``|delta|``.  That order is wrong for
    an x axis -- plotting the bars in it made the "grid score vs time shift" panel
    zig-zag and hid the monotone falloff that the predictive/retrospective reading
    relies on -- so every panel that draws a shift axis is re-indexed into
    increasing numeric order here.  Nothing else depends on this order: the score
    arrays stay indexed by search position, and only the ticks are re-labelled.
    """
    n_valid = int(valid.sum())
    order = [int(k) for k in np.argsort(np.asarray(deltas, dtype=float), kind="stable")]
    ticks = [deltas[k] for k in order]
    xs = np.arange(len(order))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.bar(xs - width / 2, [counts60[deltas[k]] for k in order], width,
           label="GS60 (six-fold)")
    ax.bar(xs + width / 2, [counts90[deltas[k]] for k in order], width,
           label="GS90 (four-fold)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%+d" % d for d in ticks])
    ax.set_xlabel("best time shift of the unit (steps)")
    ax.set_ylabel("number of units")
    ax.set_title("Units per best shift (n_valid = %d)" % n_valid)
    ax.legend(fontsize=8)

    ax = axes[1]
    mean60 = np.array([np.nanmean(score_60[k]) for k in order])
    mean90 = np.array([np.nanmean(score_90[k]) for k in order])
    sem60 = np.array([np.nanstd(score_60[k]) / max(1.0, np.sqrt(n_valid)) for k in order])
    sem90 = np.array([np.nanstd(score_90[k]) / max(1.0, np.sqrt(n_valid)) for k in order])
    ax.errorbar(xs, mean60, yerr=sem60, marker="o", label="GS60")
    ax.errorbar(xs, mean90, yerr=sem90, marker="s", ls="--", label="GS90")
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.axvline(xs[int(np.argmin(np.abs(np.asarray(ticks, dtype=float))))], color="k", lw=0.6, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%+d" % d for d in ticks])
    ax.set_xlabel("time shift of the activity (steps)")
    ax.set_ylabel("grid score (mean +/- SEM)")
    ax.set_title("Grid score vs time shift")
    ax.legend(fontsize=8)

    ax = axes[2]
    base = deltas.index(0)
    best60 = np.array([score_60[best_60_idx[i], i] for i in range(len(best_60_idx))])
    ax.scatter(score_60[base], best60, s=14, label="GS60", alpha=0.85)
    lo = min(-0.5, float(np.min(score_60[base])) - 0.1)
    hi = max(1.5, float(np.max(np.asarray([score_60[base], best60]))) + 0.1)
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("GS60 at shift 0")
    ax.set_ylabel("GS60 at the best shift")
    ax.set_title("Gain from the best shift")
    ax.legend(fontsize=8)

    if epoch is not None:
        fig.suptitle("epoch %d" % int(epoch), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def draw_grid_shift(par, pos_seq, grid_seq, draw_path, shift_range=2, top_k=10,
                    epoch=None, save_plots=True, min_std=1e-9, sub=None,
                    sigma=None, adapt_ring=None, save_sacs=None, widths=None):
    """Rate maps + GS60/GS90 with a +/-``shift_range`` step time-shift search.

    ``par``      params carrying ``par.env.widths`` (evaluation env)
    ``pos_seq``  (B, S) flat state indices, as returned by ``gen_data``
    ``grid_seq`` (B, S, g_size) grid code (``res["g_gen"]``)
    ``draw_path`` file-name prefix, e.g. ``<train_dir>/e200S64W11``
    ``sub``      fine bins per cell (default: ``DIAG_HIRES_SUB`` / par field)
    ``sigma``    Gaussian smoothing in fine bins (default ``DIAG_HIRES_SIGMA``)
    ``adapt_ring`` score on the period-adapted ring (default ``DIAG_ADAPT_RING``)
    ``save_sacs`` ``"all"`` or ``"dt0"`` (default ``DIAG_SAVE_SACS``)
    ``widths``   per-sample arena side ``(B,)``, or ``None`` for the single-arena
                 assumption (identical to the old behaviour)

    Returns the summary dict (also serialised to ``gridshift_summary.json``).
    """
    act = _as_numpy(grid_seq).astype(np.float64)
    if act.ndim == 2:
        act = act[None, ...]
    if act.ndim != 3:
        raise ValueError("grid_seq must be (B, S, U), got shape %s" % (act.shape,))

    cells = grid_extent(par)
    sub_arg = sub
    sub = int(resolve_sub(cells) if sub is None else max(1, int(sub)))
    sigma = None if sigma is None else float(sigma)      # cells
    sigma_bins = resolve_sigma(sub, sigma)
    adapt = ADAPT_RING if adapt_ring is None else bool(adapt_ring)
    save_sacs = SAVE_SACS if save_sacs is None else str(save_sacs)
    nbins = cells * sub
    pos_xy = positions_xy(par, pos_seq, max_width=cells, widths=widths)
    if pos_xy.shape[:2] != act.shape[:2]:
        pos_xy = pos_xy.reshape(act.shape[0], act.shape[1], pos_xy.shape[-1])

    scorer = make_scorer(nbins, coords_range=[[0.0, float(cells)], [0.0, float(cells)]])
    deltas = shift_order(shift_range)
    results = [score_shift(scorer, nbins, pos_xy, act, d, cells=cells, sigma=sigma,
                           adapt_ring=adapt, sub=sub)
               for d in deltas]

    score_60 = np.stack([r["score_60"] for r in results], axis=0)   # (D, U)
    score_90 = np.stack([r["score_90"] for r in results], axis=0)
    spacing = np.stack([r["spacing"] for r in results], axis=0)      # (D, U), in cells
    base = deltas.index(0)
    n_units = int(act.shape[-1])

    # units whose rate map at shift 0 is not constant (dead units are excluded from
    # the shift statistics, they would otherwise "prefer" a random shift)
    std_base = np.nanstd(results[base]["maps"], axis=(1, 2))
    valid = std_base > min_std
    n_valid = int(valid.sum())

    best_60_idx = np.argmax(score_60, axis=0)
    best_90_idx = np.argmax(score_90, axis=0)
    best_60 = np.array([deltas[k] for k in best_60_idx], dtype=int)
    best_90 = np.array([deltas[k] for k in best_90_idx], dtype=int)
    best_score_60 = np.array([score_60[best_60_idx[i], i] for i in range(n_units)])
    best_score_90 = np.array([score_90[best_90_idx[i], i] for i in range(n_units)])

    counts60 = {int(d): int(((best_60 == d) & valid).sum()) for d in deltas}
    counts90 = {int(d): int(((best_90 == d) & valid).sum()) for d in deltas}

    def _mean_topk(values, k, mask=None):
        values = np.asarray(values, dtype=np.float64)
        if mask is not None:
            values = values[mask]
        if values.size == 0:
            return float("nan")
        return float(np.sort(values)[::-1][:min(int(k), values.size)].mean())

    summary = {
        "epoch": None if epoch is None else int(epoch),
        "n_units": n_units,
        "n_valid": n_valid,        "nbins": int(nbins),
        "cells": int(cells),
        "sub": int(sub),
        "diag": {"pipeline": "legacy" if sub == 1 else "hires",
                 "sub": int(sub), "sigma_bins": float(sigma_bins),
                 "sigma_cells": float(sigma_bins) / max(1, sub),
                 "sub_rule": sub_rule_text(None if sub_arg is None else int(sub)),
                 "adapt_ring": bool(adapt), "save_sacs": str(save_sacs),
                 "ring_rule": ("[0.7 s, 1.3 s] on nearest 2-D SAC peak" if adapt
                               else "fixed mask_min*nbins .. mask_max*nbins")},
        "shift_range": int(shift_range),
        "deltas": [int(d) for d in deltas],
        "n_pairs": {int(r["delta"]): r["n_pairs"] for r in results},
        "occupancy_dt0": results[base]["occupancy"],
        "gs60_per_shift_mean": {int(r["delta"]): float(np.nanmean(r["score_60"])) for r in results},
        "gs90_per_shift_mean": {int(r["delta"]): float(np.nanmean(r["score_90"])) for r in results},
        "gs60_mean_dt0": float(np.nanmean(score_60[base])),
        "gs90_mean_dt0": float(np.nanmean(score_90[base])),
        "gs60_mean_best": float(np.nanmean(best_score_60[valid])) if n_valid else float("nan"),
        "gs90_mean_best": float(np.nanmean(best_score_90[valid])) if n_valid else float("nan"),
    }
    for k in (1, top_k):
        summary["gs60_top%d_dt0" % k] = _mean_topk(score_60[base], k, valid)
        summary["gs90_top%d_dt0" % k] = _mean_topk(score_90[base], k, valid)
        summary["gs60_top%d_best" % k] = _mean_topk(best_score_60, k, valid)
        summary["gs90_top%d_best" % k] = _mean_topk(best_score_90, k, valid)
    summary["n_units_gs60_above_0.5_dt0"] = int(((score_60[base] > 0.5) & valid).sum())
    summary["n_units_gs90_above_0.5_dt0"] = int(((score_90[base] > 0.5) & valid).sum())
    summary["counts_gs60"] = counts60
    summary["counts_gs90"] = counts90
    summary["frac_prospective_gs60"] = float(np.mean(best_60[valid] > 0)) if n_valid else float("nan")
    summary["frac_retrospective_gs60"] = float(np.mean(best_60[valid] < 0)) if n_valid else float("nan")
    summary["frac_zero_shift_gs60"] = float(np.mean(best_60[valid] == 0)) if n_valid else float("nan")
    summary["frac_prospective_gs90"] = float(np.mean(best_90[valid] > 0)) if n_valid else float("nan")
    summary["frac_retrospective_gs90"] = float(np.mean(best_90[valid] < 0)) if n_valid else float("nan")
    summary["frac_zero_shift_gs90"] = float(np.mean(best_90[valid] == 0)) if n_valid else float("nan")
    # grid spacing (lattice period, cells) of the shifted rate maps
    summary["spacing_dt0_median"] = float(np.nanmedian(spacing[base][valid])) if n_valid else float("nan")
    summary["spacing_dt0_iqr"] = float(np.nanpercentile(spacing[base][valid], 75)
                                        - np.nanpercentile(spacing[base][valid], 25)) if n_valid else float("nan")
    summary["spacing_dt0_mean"] = float(np.nanmean(spacing[base][valid])) if n_valid else float("nan")
    summary["spacing_per_shift_median"] = {int(r["delta"]): float(np.nanmedian(r["spacing"])) for r in results}
    summary["spacing_best_median"] = float(np.nanmedian(
        [spacing[best_60_idx[i], i] for i in range(n_units) if valid[i]])) if n_valid else float("nan")

    # ---- arrays ------------------------------------------------------------ #
    np.save(draw_path + "gridscore60.npy", score_60[base])
    np.save(draw_path + "gridscore90.npy", score_90[base])
    np.save(draw_path + "gridness.npy", score_90[base])  # legacy name, keep GS90 at dt=0
    with open(draw_path + "gridshift_scores.npz", "wb") as fh:
        np.savez(fh,
                 deltas=np.array(deltas, dtype=int),
                 score_60=score_60,
                 score_90=score_90,
                 best_shift_60=best_60,
                 best_shift_90=best_90,
                 best_score_60=best_score_60,
                 best_score_90=best_score_90,
                 valid=valid,
                 nbins=np.array(int(nbins)))
    # rate maps + SACs + spacings of every shift (input of cluster_grid_spacing.py)
    # ``cells``/``sub`` make the file self-describing: bins are NOT cells once
    # sub > 1, and the spacing array is always in cells.
    _sacs = np.stack([r["sacs"] for r in results], axis=0).astype(np.float32)
    if str(save_sacs) == "dt0":
        _sacs = _sacs[base:base + 1]
    np.savez_compressed(draw_path + "gridshift_ratemaps.npz",
                        deltas=np.array(deltas, dtype=int),
                        maps=np.stack([r["maps"] for r in results], axis=0).astype(np.float32),
                        sacs=_sacs,
                        spacing=spacing.astype(np.float32),
                        score_60=score_60.astype(np.float32),
                        score_90=score_90.astype(np.float32),
                        best_shift_60=best_60,
                        best_shift_90=best_90,
                        best_index_60=best_60_idx.astype(int),
                        best_index_90=best_90_idx.astype(int),
                        valid=valid,
                        nbins=np.array(int(nbins)),
                        cells=np.array(int(cells)),
                        sub=np.array(int(sub)))
    with open(draw_path + "gridshift_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    # ---- figures ----------------------------------------------------------- #
    if save_plots:
        ordering = [int(i) for i in np.argsort(-score_60[base])]
        _multi = int(shift_range) > 0
        _page_all_units(results[base]["maps"], results[base]["sacs"], results[base]["mask_60"],
                        score_60[base], score_90[base], ordering, draw_path + "grid.pdf",
                        "rate map + SAC, epoch shift 0"
                        + ("" if epoch is None else " (epoch %d)" % int(epoch)))
        if _multi:
            _summary_figure(draw_path + "gridshift_summary.png", deltas, score_60, score_90,
                            best_60_idx, best_90_idx, counts60, counts90, valid, epoch=epoch)
            with PdfPages(draw_path + "gridshift.pdf") as pdf:
                for res in results:
                    _page_shift_all_units(res, ordering, pdf, epoch)
            best_maps = np.stack([results[best_60_idx[i]]["maps"][i] for i in range(n_units)], axis=0)
            cols, rows = _layout(n_units)
            fig = plt.figure(figsize=(24, max(2.0, rows * 2.4)))
            for slot, idx in enumerate(ordering):
                ax = fig.add_subplot(rows, cols, slot + 1)
                _plot_ratemap(best_maps[idx], ax,
                              "%d (dt=%+d, GS60=%.2f)" % (idx, best_60[idx], best_score_60[idx]))
            fig.suptitle("each unit at its own best shift (+/-%d steps)%s"
                         % (int(shift_range), "" if epoch is None else ", epoch %d" % int(epoch)),
                         fontsize=10)
            fig.tight_layout()
            with PdfPages(draw_path + "gridshift_best.pdf") as pdf:
                pdf.savefig(fig)
            plt.close(fig)

        # --- spatial autocorrelograms (SAC) of the *shifted* rate maps -------- #
        with PdfPages(draw_path + "gridshift_autocorr.pdf") as pdf:
            for res in results:
                page = _sac_page(res, ordering, _sac_suptitle(res, epoch))
                pdf.savefig(page)
                plt.close(page)
            if _multi:
                best_res = {
                    "delta": 0,
                    "maps": best_maps,
                    "sacs": np.stack([results[best_60_idx[i]]["sacs"][i] for i in range(n_units)], axis=0),
                    "mask_60": [results[best_60_idx[i]]["mask_60"][i] for i in range(n_units)],
                    "score_60": best_score_60,
                    "spacing": np.array([spacing[best_60_idx[i], i] for i in range(n_units)]),
                }
                page = _sac_page(best_res, ordering,
                                 "SAC of every unit at its own best shift (+/-%d steps)%s"
                                 % (int(shift_range), "" if epoch is None else ", epoch %d" % int(epoch)))
                pdf.savefig(page)
                plt.close(page)
        page = _sac_page(results[base], ordering, _sac_suptitle(results[base], epoch))
        page.savefig(draw_path + "gridshift_autocorr_dt0.png", dpi=120)
        plt.close(page)

    return summary


def format_summary(summary):
    """One-line ASCII log string for the training console."""
    counts60 = " ".join("%+d:%d" % (d, summary["counts_gs60"][d]) for d in summary["deltas"])
    counts90 = " ".join("%+d:%d" % (d, summary["counts_gs90"][d]) for d in summary["deltas"])
    return ("[grid] dt=0: GS60=%.3f GS90=%.3f | best-shift GS60=%.3f | "
            "units per best shift (GS60) %s | (GS90) %s | "
            "prospective=%.2f retrospective=%.2f zero=%.2f (n_valid=%d/%d) | spacing=%.2f"
            " | %s nbins=%s cells=%s sub=%s sigma=%s ring=%s samp/bin=%.1f"
            % (summary["gs60_mean_dt0"], summary["gs90_mean_dt0"], summary["gs60_mean_best"],
               counts60, counts90,
               summary["frac_prospective_gs60"], summary["frac_retrospective_gs60"],
               summary["frac_zero_shift_gs60"], summary["n_valid"], summary["n_units"],
               summary["spacing_dt0_median"],
               summary.get("diag", {}).get("pipeline", "?"), summary["nbins"],
               summary.get("cells"), summary.get("sub"),
               summary.get("diag", {}).get("sigma_cells"),
               "adapt" if summary.get("diag", {}).get("adapt_ring") else "fixed",
               summary.get("occupancy_dt0", {}).get("samples_per_occupied_mean", float("nan"))))
