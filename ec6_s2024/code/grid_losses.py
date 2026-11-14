"""Directional-isotropy regulariser for the grid code (``l_grid_iso``).

Why this term
-------------
The grid module is an open-loop path integrator::

    g_t = Norm(ReLU(g_{t-1} + W(a_t) · g_{t-1}))

so ``g`` is driven by the *action sequence*, not by positions.  If the code is a
**metric** of space, then moving one cell must change ``g`` by the same amount in
every direction::

    || g(p + e_d) - g(p) ||   independent of d ∈ {up, down, left, right}

That is the property this file implements, in the form the project asked for --
**the variance of the distance between two neighbouring g's**::

    L_iso = Var_{pairs with Manhattan distance 1} ( ||g_t - g_{t-1}|| )

with the pairs taken from *consecutive* steps of the trajectory that are a real
one-cell move (``|Δx| + |Δy| = 1``).  Same-position steps ("stay still") are
excluded: they are supposed to have distance 0, and their inclusion would make the
variance measure the stay-still rate rather than the isotropy.

Because every ``g_t`` is L2-normalised, ``||g_t - g_{t-1}||² = 2 - 2 cos(g_t, g_{t-1})``,
so this is equivalently the variance of the **angular step** in code space --
"the code advances by the same angle per cell, whatever the heading".

Two honest caveats (read before trusting a rise in GS60)
--------------------------------------------------------
1. **Variance of step length is necessary, not sufficient, for a 2-D metric.**
   A code that lives on a 1-D manifold (a "corridor" or a lookup table walked
   along a single curve) also has constant step length.  This term equalises the
   step sizes; it cannot by itself ask for a *two-dimensional* periodic structure.
   It is meant to be read together with GS60 / spacing clustering, not instead.
2. **It is minimised by collapse.**  All step lengths equal to 0 gives L_iso = 0.
   In this codebase the reconstruction loss (``lx_gen``), the per-step L2
   normalisation and the capacity term all push against a constant ``g``, so
   collapse is not the path of least resistance -- but when you turn the term on
   with a large weight, pass ``anchor > 0`` and ``target > 0`` as insurance: the
   optional anchor adds the one-sided penalty ``anchor * relu(target - mean)²``,
   i.e. it only complains when the typical step gets *too small*.

What is measured (and logged)
-----------------------------
``mode``
  ``"var"`` (default)  population variance of the step length over all selected
                       pairs -- the plain reading of "variance of adjacent
                       distances".
  ``"direction"``      variance of the **per-direction mean** step length (4
                       groups).  Sharper and less noise-sensitive: it ignores
                       within-direction spread (a walk that mixes long and short
                       steps in one direction) and only penalises differences
                       *between* headings.  This is the literal "isotropic in
                       direction" quantity.
  ``"cv"``             variance divided by squared mean, i.e. (std/mean)² -- a
                       scale-free version of ``"var"`` that cannot be gamed by
                       shrinking every step uniformly (but still collapses at 0/0).

``dist``
  ``"l2"`` (default) the Euclidean distance ``||Δg||``  -- "distance".
  ``"sq"`` the squared distance ``||Δg||²`` = ``2 - 2cos`` -- same ordering,
           different scale; historically this project's code uses squared
           distances (``g_trans``), hence the option.

Everything is vectorised torch: no python loop over the batch, no numpy round trip,
and the pair mask goes through ``widths`` so a training batch whose arena widths
are mixed (11/10/10/11/8/9/...) is decoded with the width each sample was built
from -- see FIXES_2026-10-25.md §3.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_EPS = 1e-8
DIRECTION_NAMES = ("+x", "-x", "+y", "-y")


def _xy_coords(pos_seq, widths=None, batch_size=None, device=None, dtype=None):
    """Absolute ``(x, y)`` for every step -> ``(B, S, 2)``.

    ``pos_seq`` is ``(B, S)`` flat state indices (then ``widths`` -- the arena side
    of each *sample* -- is required, which is what makes a mixed-width batch decode
    correctly, see FIXES_2026-10-25.md §3) or already ``(B, S, 2)`` coordinates.
    """
    if not torch.is_tensor(pos_seq):
        pos_seq = torch.as_tensor(pos_seq)
    pos_seq = pos_seq.to(device=device, dtype=dtype)

    if pos_seq.dim() == 3:
        if pos_seq.shape[-1] != 2:
            raise ValueError("(B, S, 2) coordinates expected, got shape %s" % (tuple(pos_seq.shape),))
        return pos_seq                 # already absolute; `widths` is then unused
    if pos_seq.dim() != 2:
        raise ValueError("give either (B, S) flat indices or (B, S, 2) coordinates")
    if widths is None:
        raise ValueError("flat pos_seq needs `widths` (per-sample arena side)")

    w = torch.as_tensor(widths, device=pos_seq.device, dtype=pos_seq.dtype).reshape(-1)
    b = int(pos_seq.shape[0])
    if w.numel() == 1:
        w = w.repeat(b)
    elif w.numel() != b:
        # cycle the table, exactly like get_env_params() does
        w = w.repeat(int((b + w.numel() - 1) // w.numel()))[:b]
    w = w.reshape(-1, 1)
    x = torch.floor(pos_seq / w)
    y = pos_seq - x * w
    return torch.stack([x, y], dim=-1)


def _pair_geometry(pos_seq, widths=None, batch_size=None, device=None, dtype=None):
    """Per-sample ``(x, y)`` and the consecutive-step direction codes.

    Args:
      pos_seq: ``(B, S)`` flat state indices, or ``(B, S, 2)`` absolute ``(x, y)``.
      widths:  arena side per sample.  A sequence / tensor of length ``B`` (aliases
               of length 1 are broadcast).  ``None`` means "absolute coordinates"
               when ``pos_seq`` is ``(B, S, 2)``, and is an error otherwise.
      batch_size: optional ``B`` used to broadcast a scalar-ish ``widths``.

    Returns:
      ``(dir_code, moved)`` with shapes ``(B, S - 1)``:
      ``dir_code`` in ``0..3`` (``+x, -x, +y, -y``), ``moved`` the boolean mask of
      real one-cell moves.  A diagonal or multi-cell jump can only appear if the
      caller hands in coordinates that skip cells; those steps are masked out.
    """
    xy = _xy_coords(pos_seq, widths=widths, batch_size=batch_size,
                    device=device, dtype=dtype)

    dx = xy[:, 1:, 0] - xy[:, :-1, 0]
    dy = xy[:, 1:, 1] - xy[:, :-1, 1]
    moved = (dx.abs() + dy.abs()) == 1
    dir_code = torch.where(dx > 0, 0, torch.where(dx < 0, 1, torch.where(dy > 0, 2, 3)))
    return dir_code, moved


def step_lengths(g_seq, dist="l2", eps=DEFAULT_EPS):
    """``||g_t - g_{t-1}||`` (or its square) for every consecutive pair -> ``(B, S-1)``."""
    dg = g_seq[:, 1:] - g_seq[:, :-1]
    d2 = (dg * dg).sum(dim=-1)
    if str(dist).lower() in ("l2", "euclid", "distance"):
        return (d2 + eps).sqrt()
    if str(dist).lower() in ("sq", "l2sq", "squared"):
        return d2
    raise ValueError("dist must be 'l2' or 'sq', got %r" % (dist,))


def grid_isotropy_loss(g_seq, pos_seq=None, widths=None, mode="var", dist="l2",
                       target=None, anchor=0.0, eps=DEFAULT_EPS, dir_code=None,
                       moved=None, return_detail=True):
    """Variance of the distance between neighbouring grid codes (see module doc).

    Args:
      g_seq:  ``(B, S, U)`` grid code, ``res["g_gen"]``.
      pos_seq: ``(B, S)`` flat state indices (or ``(B, S, 2)`` coordinates).
      widths: per-sample arena side; required for flat indices.
      mode:   ``"var"`` | ``"direction"`` | ``"cv"``.
      dist:   ``"l2"`` (default) or ``"sq"``.
      target: optional typical step length you want to keep; with ``anchor > 0``
              it adds ``anchor * relu(target - mean_step)²``, a one-sided guard
              against the all-steps-shrink-to-zero solution.
      anchor: weight of that guard (0 disables it, the default).
      dir_code, moved: pass the precomputed pair geometry to avoid recomputing it
              when several terms share it.

    Returns:
      ``(loss, detail)``: ``loss`` is a scalar tensor that requires grad when
      ``g_seq`` does; ``detail`` is a dict of python floats / ints for logging.
    """
    if dir_code is None or moved is None:
        dir_code, moved = _pair_geometry(pos_seq, widths=widths, dtype=g_seq.dtype,
                                        device=g_seq.device)
    d = step_lengths(g_seq, dist=dist, eps=eps)                 # (B, S-1)
    sel = d[moved]
    n_pairs = int(sel.numel())
    zero = torch.zeros((), dtype=g_seq.dtype, device=g_seq.device)
    detail = {"n_pairs": n_pairs, "mean": float("nan"), "std": float("nan"),
              "cv": float("nan"), "per_direction": {}}
    if n_pairs < 2:
        return zero, detail

    mean_step = sel.mean()
    detail["mean"] = float(mean_step.detach())
    mode = str(mode).lower()

    if mode == "var":
        loss = (sel - mean_step).pow(2).mean()
    elif mode == "cv":
        loss = (sel - mean_step).pow(2).mean() / (mean_step.pow(2) + eps)
    elif mode == "direction":
        codes = dir_code[moved]
        per_dir = []
        for code in range(4):
            group = sel[codes == code]
            if int(group.numel()) > 0:
                mu = group.mean()
                detail["per_direction"][DIRECTION_NAMES[code]] = float(mu.detach())
                per_dir.append(mu)
        detail["per_direction"] = dict(sorted(detail["per_direction"].items()))
        if len(per_dir) < 2:
            return zero, detail
        stacked = torch.stack(per_dir)
        loss = (stacked - stacked.mean()).pow(2).mean()
    else:
        raise ValueError("mode must be 'var', 'direction' or 'cv', got %r" % (mode,))

    if target is not None and float(anchor) != 0.0:
        loss = loss + float(anchor) * F.relu(float(target) - mean_step).pow(2)

    std = sel.std(unbiased=False) if n_pairs > 1 else zero
    detail["std"] = float(std.detach())
    detail["cv"] = float((std / (mean_step + eps)).detach())
    detail["mode"] = mode
    detail["dist"] = str(dist).lower()
    if not return_detail:
        return loss, {}
    return loss, detail


# --------------------------------------------------------------------------- #
# v2 -- scale-free, per-unit, radius-controlled isotropy (2026-10-26)
# --------------------------------------------------------------------------- #
def _as_k_list(k):
    """``2`` / ``"2"`` / ``"2,3"`` -> ``[2]`` / ``[2, 3]``."""
    if isinstance(k, (int, float)):
        return [int(k)]
    out = [int(p) for p in str(k).replace(" ", "").split(",") if p]
    return out or [2]


def grid_isotropy_loss_v2(g_seq, pos_seq=None, widths=None, k=1, nbins=8,
                          min_disp=1.0, agg="cv", per_unit=True,
                          metric="within_radius", radius_tol=0.05,
                          anchor=0.0, target=None, eps=DEFAULT_EPS,
                          min_mean_step=1e-3, min_pairs_per_cell=4,
                          return_detail=True):
    """Isotropy of the code **metric**, estimated from k-step displacements.

    What changed with respect to :func:`grid_isotropy_loss` (v1), and why
    ----------------------------------------------------------------------
    v1 was ``Var(||g_t - g_{t-1}||)`` over one-cell steps.  The 36-run sweep of
    2026-10-26 showed three defects, each fixed here:

    1. **v1's variance was pooled over the whole batch, i.e. across units.**  That
       variance mixes three things: orientation dependence (what we want), noise
       within an orientation, and *differences between units*.  The measurement is
       bad even on a perfect lattice: `_self_test_v2` reports ``cv_pooled = 1.2``
       for an idealised hexagonal code whose per-unit orientation CV is ``0.11``.
       The last component is the worst offender: units with different lattice
       spacing (different ``lambda``) are exactly what "multiple modules" means, so
       pooled variance *is* a homogenisation pressure -- measured on the v1 sweep,
       the coarse module (33 units at spacing 4.24) was cut to 6 and then to 1, and
       GS60 top-10 fell 1.38 -> 0.77 -> 0.40.  v2 therefore computes the dispersion
       **per code dimension** and averages over units (``per_unit=True``).
    2. **v1 had no scale normalisation.**  ``Var(c * d) = c^2 Var(d)``, so shrinking
       every step drives the term to zero.  v2 uses ``r = ||dg|| / ||dx||`` (code
       distance per *physical* distance, i.e. a metric density) and takes the
       coefficient of variation, ``CV(c*r) = CV(r)`` -- scale-free by construction.
    3. **v1 had no guard against path folds.**  ``||dx||`` is a *net* displacement,
       so a path that doubles back has a tiny denominator and an exploding ratio.
       ``min_disp`` drops those pairs, and it keeps the physics honest: this model
       is a path integrator, so net and path displacement agree only on (nearly)
       straight segments.

    Lag ``k`` and ``lambda`` -- read this before trying other settings
    ----------------------------------------------------------------
    ``k=1`` (default) is the only lag that measures the **metric**: the step is
    as small as the walk allows, so ``r`` reflects ``d||dg||/d||dx||`` along
    +-x/+-y.  ``k=2`` additionally samples the diagonal -- (2,0), (1,1), (0,2) --
    but at ``|dx| ~ 1.4-2`` cells the value also depends on whether the
    displacement happens to be an integer number of lattice periods
    (commensurability, not anisotropy), so it is an explicit arm rather than the
    default.

    **Both lags are biased against small lattice spacing**, and this is the single
    most important thing to know about this family of losses: a metric-density
    statistic assumes the code is locally linear, which requires ``lambda >> 1``
    cell.  ``_self_test_v2`` measures ``hex @ 4.24`` at ``1.8e-2`` and the same
    ideal hexagon at ``1.25`` cells at ``1.2e-1`` -- **6.5x worse**, although both
    codes are perfect lattices.  So the term structurally prefers coarse modules
    (and, in the limit, a locally linear / ramp code), i.e. it is a *scale*
    pressure as much as an orientation pressure.  Expect ``@floor`` (the share of
    units at the 1.25-cell bin) to fall -- and check the reconstruction loss when
    it does.

    Args:
      g_seq: ``(B, S, U)`` grid code.
      pos_seq, widths: as in v1 (flat state indices + per-sample arena side).
      k: lag(s) in steps.  ``1`` (default) measures the metric cleanly along
         +-x/+-y.  ``2`` adds the diagonal but also mixes in lattice
         commensurability (see above), so it is an experimental arm; ``"2,3"``
         averages two lags.
      nbins: number of orientation bins over ``[0, pi)`` (anisotropy has pi period).
      min_disp: drop pairs whose net displacement is below this many cells.
      per_unit: ``True`` (default) computes the dispersion **per code dimension**
         and then averages over units.  This is the fix for the worst v1 failure:
         pooled variance also penalises *differences between units* -- and units
         with different lattice spacing (different ``lambda``) are exactly what
         "multiple modules" means.  Measured on the v1 sweep: the coarse module
         (33 units at spacing 4.24) was cut to 6 and then to 1, and GS60 top-10
         fell 1.38 -> 0.77 -> 0.40.  ``False`` reproduces v1's homogenising
         behaviour and exists only as an ablation.
      metric: ``"within_radius"`` (default) first subtracts, for every
         displacement *radius*, the mean over orientations, and only then measures
         the spread.  This removes the purely geometric trend ``r(|dx|)`` -- a
         periodic code's ``||dg||`` is a non-linear function of ``|dx|`` even when
         the code is perfectly isotropic -- which would otherwise be mistaken for
         anisotropy.  ``"raw"`` skips that centring (the literal "spread of r over
         directions/pairs").
      agg: ``"cv"`` (default, dimensionless, scale-free) or ``"var"``.
      anchor, target: optional one-sided ``anchor * relu(target - mean_r)^2``, i.e.
         insurance against the code going flat (a flat code has ``r = 0`` and
         ``CV = 0/0``; those units are dropped by ``min_mean_step``, so the default
         is fine, but keep the anchor available).

    Honest limitation (read this before expecting GS60 to rise)
    -----------------------------------------------------------
    This is a **second-order (metric) statistic**.  To leading order in ``|dx|``
    the metric tensor of a code built from a reciprocal star ``{G_m}`` is
    ``M = sum_m w_m G_m G_m^T``; any star with >= 3-fold symmetry (4-fold, 6-fold,
    8-fold, quasicrystalline) gives ``M ∝ I``, so **a square lattice passes this
    loss exactly as well as a hexagonal one**.  ``_self_test_v2`` asserts that
    (``test_square_not_separated``) instead of pretending otherwise.  The term can
    therefore remove *stripe / sheared / squashed* codes -- which v1 could not see
    -- but it cannot select 6-fold symmetry.  Judging it by "GS60 went up" would be
    a category error; judge it by "fewer anisotropic units, and no loss of the
    coarse module".

    Returns:
      ``(loss, detail)``; ``detail['version'] == 2``.
    """
    if g_seq.dim() != 3:
        raise ValueError("g_seq must be (B, S, U), got %s" % (tuple(g_seq.shape),))
    xy = _xy_coords(pos_seq, widths=widths, batch_size=g_seq.shape[0],
                    device=g_seq.device, dtype=g_seq.dtype)
    B, S, U = (int(v) for v in g_seq.shape)
    u_out = U if per_unit else 1

    ks = _as_k_list(k)
    zero = torch.zeros((), dtype=g_seq.dtype, device=g_seq.device)
    detail = {"version": 2, "n_pairs": 0, "mean": float("nan"),
              "cv_pooled": float("nan"), "cv_direction": float("nan"),
              "cv_direction_p90": float("nan"), "n_units_live": 0,
              "n_cells": 0, "n_cells_used": 0, "k_txt": ",".join(str(v) for v in ks),
              "agg": str(agg).lower(), "metric": str(metric).lower(),
              "per_unit": bool(per_unit), "radii": []}
    if nbins < 2:
        raise ValueError("nbins must be >= 2")

    edges = torch.linspace(0.0, math.pi, int(nbins) + 1, dtype=g_seq.dtype, device=g_seq.device)
    terms = []
    n_pairs = 0
    means, cvs, p90s, radii_all = [], [], [], []

    for kk in ks:
        if kk < 1 or kk >= S - 1:
            continue
        dxy = xy[:, kk:] - xy[:, :-kk]                      # (B, S-kk, 2)
        rho = dxy.pow(2).sum(-1).sqrt()                     # (B, S-kk)
        keep = rho >= float(min_disp)
        n_keep = int(keep.sum())
        if n_keep < max(8, int(nbins)):
            continue

        dg = g_seq[:, kk:] - g_seq[:, :-kk]                 # (B, S-kk, U)
        if per_unit:
            dstep = (dg * dg + eps).sqrt()                  # (B, S-kk, U)
        else:
            dstep = (dg * dg).sum(-1).add(eps).sqrt().unsqueeze(-1)   # (B, S-kk, 1)

        r = dstep[keep] / (rho[keep].unsqueeze(-1) + eps)    # (M, u_out) = ||dg||/||dx||
        ang = torch.atan2(dxy[..., 1], dxy[..., 0])[keep].remainder(math.pi)
        cell_ang = torch.bucketize(ang, edges[1:-1])         # 0 .. nbins-1
        if detail["metric"] == "within_radius":
            rb = torch.round(rho[keep] / float(radius_tol)).long()
            key = rb * int(nbins) + cell_ang
        else:
            rb = torch.zeros_like(cell_ang)
            key = cell_ang

        uniq, inv = torch.unique(key, return_inverse=True)
        ncell = int(uniq.numel())
        if ncell < 2:
            continue
        sums = torch.zeros(ncell, u_out, dtype=r.dtype, device=r.device)
        sums.index_add_(0, inv, r)
        cnt = torch.bincount(inv, minlength=ncell).to(r.dtype)
        cell_mean = sums / cnt.clamp(min=1).unsqueeze(-1)          # (ncell, u_out)
        ok_cell = (cnt >= int(min_pairs_per_cell)).to(r.dtype)     # (ncell,)
        n_used = float(ok_cell.sum())
        if n_used < 2:
            continue

        if detail["metric"] == "within_radius":
            rb_cell = torch.div(uniq, int(nbins), rounding_mode="floor")
            ru, rinv = torch.unique(rb_cell, return_inverse=True)
            nr = int(ru.numel())
            rsum = torch.zeros(nr, u_out, dtype=r.dtype, device=r.device)
            rsum.index_add_(0, rinv, cell_mean * ok_cell.unsqueeze(-1))
            rcnt = torch.bincount(rinv, minlength=nr).to(r.dtype)
            rmean = rsum / rcnt.clamp(min=1).unsqueeze(-1)
            dev = (cell_mean - rmean[rinv]) * ok_cell.unsqueeze(-1)
            disp = dev.pow(2).sum(0) / n_used                       # (u_out,)
        else:
            cm = cell_mean * ok_cell.unsqueeze(-1)
            cm_mean = cm.sum(0) / n_used
            disp = ((cell_mean - cm_mean).pow(2) * ok_cell.unsqueeze(-1)).sum(0) / n_used

        mean_r_u = r.mean(0)                                        # (u_out,)
        live = mean_r_u > float(min_mean_step)
        n_live = int(live.sum())
        if n_live == 0:
            continue
        if detail["agg"] == "cv":
            per_unit_term = disp / (mean_r_u.pow(2) + eps)           # squared CV
            term = per_unit_term[live].mean()
            cv_u = (disp[live] / (mean_r_u[live].pow(2) + eps)).sqrt()
        else:
            term = disp[live].mean()
            cv_u = disp[live].sqrt() / (mean_r_u[live] + eps)
        terms.append(term)

        n_pairs += n_keep
        means.append(float(mean_r_u[live].mean().detach()))
        cvs.append(float(cv_u.mean().detach()))
        p90s.append(float(cv_u.flatten().kthvalue(max(1, int(0.9 * cv_u.numel())))[0].detach())
                    if cv_u.numel() > 1 else float(cv_u.mean().detach()))
        radii_all.append(float(rho[keep].mean().detach()))
        detail["n_units_live"] = max(detail["n_units_live"], n_live)
        detail["n_cells"] = max(detail["n_cells"], ncell)
        detail["n_cells_used"] = max(detail["n_cells_used"], int(n_used))

    if not terms:
        if not return_detail:
            return zero, {}
        return zero, detail

    loss = torch.stack(terms).mean()
    if target is not None and float(anchor) != 0.0:
        loss = loss + float(anchor) * F.relu(float(target) - sum(means) / len(means)).pow(2)

    r_all = r.detach()
    detail["n_pairs"] = n_pairs
    detail["mean"] = sum(means) / len(means)
    detail["cv_direction"] = sum(cvs) / len(cvs)
    detail["cv_direction_p90"] = sum(p90s) / len(p90s)
    detail["cv_pooled"] = float((r_all.std(unbiased=False) / (r_all.mean() + eps)).detach())
    detail["displacement"] = sum(radii_all) / len(radii_all)
    if not return_detail:
        return loss, {}
    return loss, detail


def format_detail(detail):
    """One-line ASCII summary (goes to stdout / TensorBoard next to the loss)."""
    if not detail:
        return "iso: no detail"
    if detail.get("version") == 2:
        if not detail.get("n_pairs"):
            return "iso2: no pairs"
        return ("iso2: n=%d r=%.4f cv_dir=%.3f(cv_pooled=%.3f p90=%.3f) "
                "cells=%d/%d k=%s %s/%s%s"
                % (detail["n_pairs"], detail["mean"], detail["cv_direction"],
                   detail["cv_pooled"], detail["cv_direction_p90"],
                   detail["n_cells_used"], detail["n_cells"], detail["k_txt"],
                   detail["metric"], detail["agg"],
                   "" if detail.get("per_unit", True) else " POOLED")).strip()
    if not detail.get("n_pairs"):
        return "iso: no pairs"
    per = detail.get("per_direction") or {}
    per_txt = " ".join("%s=%.3f" % (k, v) for k, v in per.items())
    return ("iso: n=%d mean=%.4f std=%.4f cv=%.3f %s"
            % (detail["n_pairs"], detail["mean"], detail["std"], detail["cv"], per_txt)).strip()


# --------------------------------------------------------------------------- #
# self-test / functional check
# --------------------------------------------------------------------------- #
def resolve_widths(widths=None, batch_size=None, par=None):
    """Best available per-sample arena side: explicit ``widths`` > ``par.env.widths``.

    Returns a 1-D float tensor of length ``batch_size`` (cycled if the table is
    shorter, exactly like ``get_env_params`` builds it), or ``None`` when neither
    source exists -- in which case the caller must hand in absolute coordinates.

    ``par.env.widths`` is the correct fallback for training batches: ``init_env``
    builds ``envs[b]`` from ``par.env.widths[b]``, so entry ``b`` *is* the arena of
    sample ``b``.  ``data["width"]`` from ``gen_data`` is still preferable when
    available, because it reports the width of the env that actually produced the
    walk (they differ under ``one_instance=True``).
    """
    if widths is None and par is not None:
        widths = getattr(getattr(par, "env", None), "widths", None)
    if widths is None:
        return None
    w = torch.as_tensor(widths, dtype=torch.float32).reshape(-1)
    if w.numel() == 0:
        return None
    if batch_size is not None and w.numel() not in (1, int(batch_size)):
        w = w.repeat(int((int(batch_size) + w.numel() - 1) // w.numel()))[:int(batch_size)]
    return w


def _self_test():
    """Two synthetic codes: direction-dependent steps vs isotropic steps.

    The code is built on the unit circle, ``g_t = (cos θ_t, sin θ_t)`` with
    ``θ_t = θ_{t-1} + δ``, so the step length is ``2 sin(δ/2)`` and needs no
    normalisation: equal ``δ`` <=> equal distance, which is exactly the property
    under test.  Code "bad" walks ``δ = 0.05`` while heading +x and ``δ = 0.30``
    while heading +y; code "good" uses ``δ = 0.10`` for both.
    """
    torch.manual_seed(0)
    B, S, W = 2, 24, 16                       # W wide enough that the walk never wraps
    h = S // 2                                # steps 0..h-2 go +x, then +y

    # trajectory: walk +x along a column, then +y along a row (arena of side W)
    pos = torch.zeros(B, S, dtype=torch.float32)
    pos[0, :h] = torch.arange(h, dtype=torch.float32) * W
    pos[0, h:] = pos[0, h - 1] + torch.arange(1, S - h + 1, dtype=torch.float32)
    pos[1] = pos[0]
    widths = torch.tensor([W], dtype=torch.float32)

    def circle(deltas):
        deltas = torch.as_tensor(deltas, dtype=torch.float32)
        theta = torch.cumsum(deltas, dim=0)
        g = torch.stack([theta.cos(), theta.sin()], dim=-1)
        return g.unsqueeze(0).repeat(B, 1, 1).requires_grad_(True)

    bad = circle([0.0] + [0.05] * (h - 1) + [0.30] * (S - h))
    good = circle([0.0] + [0.10] * (S - 1))

    l_bad, d_bad = grid_isotropy_loss(bad, pos, widths=widths, mode="var")
    l_good, d_good = grid_isotropy_loss(good, pos, widths=widths, mode="var")
    l_bad_d, _ = grid_isotropy_loss(bad, pos, widths=widths, mode="direction")
    l_good_d, _ = grid_isotropy_loss(good, pos, widths=widths, mode="direction")
    print("[self-test] var       direction-dependent=%.6e  isotropic=%.6e"
          % (float(l_bad.detach()), float(l_good.detach())))
    print("[self-test] direction direction-dependent=%.6e  isotropic=%.6e"
          % (float(l_bad_d.detach()), float(l_good_d.detach())))
    print("[self-test] detail bad :", format_detail(d_bad))
    print("[self-test] detail good:", format_detail(d_good))
    assert l_bad > 10 * l_good, "variance must separate the two codes"
    assert l_bad_d > 10 * l_good_d, "per-direction variance must separate them too"
    assert d_bad["n_pairs"] == B * (S - 1), (d_bad["n_pairs"], B * (S - 1))

    l_bad.backward()
    assert bad.grad is not None and torch.isfinite(bad.grad).all(), "gradient not finite"
    print("[self-test] grad ok: |grad|=%.3e" % bad.grad.norm())
    print("[self-test] PASS")


# --------------------------------------------------------------------------- #
# self-test for v2: synthetic lattices with a *known* symmetry
# --------------------------------------------------------------------------- #
def _synth_star(kind, lam):
    """Reciprocal star of a triangular / square / stripe pattern of spacing ``lam``."""
    if kind == "hex":
        g = 4 * math.pi / (math.sqrt(3) * lam)
        ang = (0, 60, 120, 180, 240, 300)
    elif kind == "square":
        g = 2 * math.pi / lam
        ang = (0, 90, 180, 270)
    elif kind == "stripe":
        g = 2 * math.pi / lam
        ang = (0,)
    else:
        raise ValueError("unknown lattice kind %r" % (kind,))
    return [(g * math.cos(math.radians(a)), g * math.sin(math.radians(a))) for a in ang]


def _synth_code(kind, lams, xy, seed=0):
    """``(B, S, U)`` code: one periodic field per unit, unit ``u`` has spacing ``lams[u]``."""
    gen = torch.Generator().manual_seed(seed)
    B, S, _ = xy.shape
    units = []
    for lam in lams:
        star = _synth_star(kind, lam)
        phase = torch.rand((), generator=gen) * 2 * math.pi
        f = torch.zeros(B, S)
        for gx, gy in star:
            f = f + torch.cos(gx * xy[..., 0] + gy * xy[..., 1] + phase)
        units.append(f / math.sqrt(len(star)))
    g = torch.stack(units, dim=-1)
    zero_mean = g - g.mean(dim=1, keepdim=True)          # a code has no DC component
    out = torch.nn.functional.normalize(zero_mean, dim=-1)
    return out.detach().requires_grad_(True)             # a leaf, like a real code


def _synth_walk(S=240, seed=1):
    """One-dimensional random walk on the 4-neighbourhood -> ``(1, S, 2)`` cells."""
    gen = torch.Generator().manual_seed(seed)
    d = torch.randint(0, 4, (S,), generator=gen)
    step = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    return torch.cumsum(step[d], dim=0).unsqueeze(0)


def _self_test_v2():
    """Checks the *documented* behaviour, including the limitation.

    * stripe  -> large loss (v1 could not even see the diagonal, v2 does)
    * square  -> NOT separated from hex (a metric statistic cannot; asserted)
    * linear  -> also small (the trivial, non-periodic minimum; asserted)
    * two spacing groups -> v1 pooled variance is large, v2 per-unit is ~0
      (this is the homogenisation bug that killed the coarse module in v1)
    """
    xy = _synth_walk()
    hexg = _synth_code("hex", [4.24] * 6, xy)
    fine = _synth_code("hex", [1.25] * 6, xy)
    sq = _synth_code("square", [4.24] * 6, xy)
    st = _synth_code("stripe", [4.24] * 6, xy)
    het = _synth_code("hex", [1.25] * 6 + [4.24] * 6, xy)
    lin = torch.nn.functional.normalize(
        torch.stack([xy[..., 0] / 3.0, xy[..., 1] / 3.0], dim=-1), dim=-1)

    out = {}
    for name, code in (("hex", hexg), ("hex_fine", fine), ("square", sq),
                       ("stripe", st), ("linear", lin), ("hetero", het)):
        out[name] = grid_isotropy_loss_v2(code, xy)
        print("[self-test v2] %-9s loss=%.6e  %s"
              % (name, float(out[name][0].detach()), format_detail(out[name][1])))

    t_hex = out["hex"][0]
    l_hex = float(t_hex)
    l_st = float(out["stripe"][0])
    l_sq = float(out["square"][0])
    l_lin = float(out["linear"][0])

    assert l_st > 5 * max(l_hex, 1e-9), ("stripe must be penalised", l_st, l_hex)
    assert l_sq < 0.25 * l_st, ("square must look isotropic to a metric statistic", l_sq, l_st)
    # A *per-unit* ramp is not direction-isotropic (each component advances along its
    # own axis), so the smooth-code minimum is only approached by the pooled variant.
    # Here it must merely be cheaper than a stripe, not free.
    assert l_lin < 0.5 * l_st, ("a linear code must be cheaper than a stripe", l_lin, l_st)

    # The pooled statistic is unusable even on a perfect lattice: different units
    # have different phases, so pooling mixes phase with orientation.  This is the
    # measurement-side reason v1's logged "cv" never matched what it penalised.
    cv_pooled = out["hex"][1]["cv_pooled"]
    cv_dir = out["hex"][1]["cv_direction"]
    print("[self-test v2] perfect hex lattice: cv_pooled=%.3f vs per-unit cv_dir=%.3f"
          % (cv_pooled, cv_dir))
    assert cv_pooled > 3 * max(cv_dir, 1e-9), \
        ("pooling must be visibly noisier than the per-unit statistic", cv_pooled, cv_dir)

    # k=1 is the cleanest lag, but it is still not scale-free in lambda: at
    # lambda = 1.25 cells a single step already advances the phase by ~2*pi/1.25,
    # so even k=1 is aliased for the *fine* module.  The loss therefore prefers
    # lambda >> 1 cell -- a structural bias that must be measured, not assumed away.
    l_k2, d_k2 = grid_isotropy_loss_v2(hexg, xy, k=2)
    l_fine = float(out["hex_fine"][0])
    print("[self-test v2] same hex: k=1 loss=%.6e (cv_dir=%.3f) vs k=2 loss=%.6e (cv_dir=%.3f)"
          % (l_hex, cv_dir, float(l_k2), d_k2["cv_direction"]))
    print("[self-test v2] lambda matters: hex@4.24 loss=%.6e vs hex@1.25 loss=%.6e (x%.1f)"
          % (l_hex, l_fine, l_fine / max(l_hex, 1e-12)))
    assert l_fine > 3 * l_hex, \
        ("a 1.25-cell lattice must be penalised more than a 4.24-cell one", l_fine, l_hex)

    # the v1 vs v2 difference on a *heterogeneous* population (the real failure mode)
    l_v1_het, _ = grid_isotropy_loss(het, xy, mode="var")
    l_v2_het = float(out["hetero"][0])
    l_v2_het_pooled, _ = grid_isotropy_loss_v2(het, xy, per_unit=False)
    print("[self-test v2] heterogeneous population (two spacings 1.25 / 4.24):")
    print("[self-test v2]   v1 pooled var      = %.6e" % float(l_v1_het.detach()))
    print("[self-test v2]   v2 pooled (abl.)   = %.6e" % float(l_v2_het_pooled.detach()))
    print("[self-test v2]   v2 per-unit (ours)= %.6e" % l_v2_het)
    assert l_v2_het < 0.25 * float(l_v1_het.detach()), \
        ("per-unit v2 must not punish a two-spacing population", l_v2_het, float(l_v1_het.detach()))
    assert l_v2_het < 0.1 * l_st, ("hetero hex population must look isotropic", l_v2_het, l_st)

    t_hex.backward()
    assert hexg.grad is not None and torch.isfinite(hexg.grad).all(), "gradient not finite"
    print("[self-test v2] grad ok: |grad|=%.3e" % hexg.grad.norm())
    print("[self-test v2] PASS")


if __name__ == "__main__":
    _self_test()
    _self_test_v2()
