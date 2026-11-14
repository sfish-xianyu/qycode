"""Grid spacing (lattice period) estimation and grid-module clustering.

The spatial autocorrelogram (SAC) of a grid cell's rate map is a hexagonal
lattice of peaks centred on the origin.  The distance of the six first-ring
peaks from the centre equals the grid spacing (``lambda``), so a module
(spacing-homogeneous group of cells) can be recovered by

1. estimating ``lambda`` per unit from its SAC, and
2. clustering the 1-D distribution of ``lambda``.

The estimation uses two complementary steps:

* a *radial profile* of the SAC (mean over annuli of width ``step`` bins); the
  first local maximum outside ``min_r`` is an initial spacing guess, and
* a *2-D peak picking* step (``maximum_filter`` + relative threshold) whose
  peaks within 35 % of the radial guess are averaged to refine ``lambda``.

Because the rate map has ``nbins`` bins over an environment that is exactly
``nbins`` cells wide, one bin == one cell, so the spacing is reported in cells.

High-resolution rate maps (``cells * sub`` fine bins over the same ``cells``
wide arena, see ``predictive_grid_diag``) break that identity: there one bin is
``1 / sub`` cells.  Two entry points cover the two cases:

* ``estimate_grid_spacing`` -- radial + 2-D refinement for the coarse legacy
  maps (``sub = 1``); its radial first guess is diluted by a single wavevector.
* ``first_ring_spacing`` -- nearest 2-D local maximum, ``sub`` aware, returns
  cells directly.  Use this one for fine SACs; it is what the diagnostics store
  in ``*gridshift_ratemaps.npz`` (``spacing`` is always in cells).

All figure text is ASCII only (DejaVu Sans), same convention as
``predictive_grid_diag``.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def radius_map(size):
    """Distance (in bins) of every pixel from the centre of a ``size`` x ``size`` grid."""
    centre = (float(size) - 1.0) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    return np.hypot(xx - centre, yy - centre)


def sac_disc(sac, nbins):
    """SAC restricted to the disc that actually holds data (rest -> NaN)."""
    sac = np.asarray(sac, dtype=np.float64)
    radius = radius_map(sac.shape[0])
    return np.where(radius <= (float(nbins) - 1.0) + 1e-9, sac, np.nan)


def first_ring_spacing(sac, nbins, sub=1, zero_lag_r=1.2, rel_thr=0.3, span_frac=0.6,
                       edge_guard_steps=1.0):
    """Radius of the nearest non-central SAC peak, **in cells**.

    This is the estimator that survives a dominant single wavevector.  Radial
    (annulus) averaging is the wrong tool for a stripe-like map or for a
    lattice whose first ring sits close to the centre: the peak then lives on
    one or two narrow angular sectors and the annulus mean dilutes it below the
    noise, which is exactly how the old radial estimator turned a 1.80-cell
    stripe into a 3.12-cell "spacing".  Taking the nearest 2-D local maximum
    instead recovers both the hexagonal ring and the stripe period.

    Args:
      sac: spatial autocorrelogram, shape ``(2 * nbins - 1,) * 2``.
      nbins: number of rate-map bins per side (fine bins, i.e. ``cells * sub``).
      sub: fine bins per environment cell; the result is divided by it.
      zero_lag_r: radius (cells) of the zero-lag lobe that is masked out.
      rel_thr: keep local maxima with at least this fraction of the best peak.
      span_frac: search radius as a fraction of the arena side.
      edge_guard_steps: reject peaks sitting within this many radial grid steps
        *outside* the mask boundary. Masking with ``-inf`` creates a cliff, so
        ``maximum_filter`` always sees the first bin beyond ``zero_lag_r`` as a
        local maximum whenever the central lobe is still decaying. That was not
        hypothetical: with ``zero_lag_r=1.2`` and ``sub=4`` the first usable
        radius is 1.25 cells, and 10 of 12 runs reported exactly 1.25 as their
        spacing while their real lattice sat at 3.61 (visible only after raising
        the mask). Such a peak is the boundary, not a ring, so it is dropped;
        if nothing interior passes ``rel_thr`` the result is ``nan`` ("no ring"),
        which is the honest answer rather than a number that happens to equal
        the mask radius. Set to 0 to restore the pre-2026-10-31 behaviour.

    Returns ``nan`` when no peak passes the threshold (noise-like map).  The
    default ``zero_lag_r = 1.2`` cells matters: with 0.9 cells the flank of the
    zero-lag lobe was picked and cap1e-3 reported a 0.90-cell spacing.
    """
    sac = np.asarray(sac, dtype=np.float64)
    sub = float(max(1, sub))
    n = sac.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rad = np.hypot(xx - c, yy - c) / sub          # in cells
    span = max(2.0, span_frac * float(nbins) / sub)

    work = sac.copy()
    work[rad < float(zero_lag_r)] = -np.inf
    work[rad > span] = -np.inf
    if not np.isfinite(work).any():
        return float("nan")

    size = max(3, int(round(0.5 * sub)))
    mx = ndimage.maximum_filter(work, size=size, mode="constant", cval=-np.inf)
    cand = np.isfinite(work) & (work >= mx - 1e-12)
    if not cand.any():
        return float("nan")

    vals = work[cand]
    rr = rad[cand]
    top = float(np.max(vals))
    if not np.isfinite(top) or top <= 0:
        return float("nan")
    keep = vals >= float(rel_thr) * top
    if not keep.any():
        return float("nan")
    # 边界峰不是环：屏蔽用 -inf 造了一个断崖，断崖外侧第一个 bin 必然是局部极大。
    # 它们恰好等于 zero_lag_r + 一个网格步长（sub=4、zero_lag_r=1.2 时就是 1.25），
    # 曾被 10/12 个 run 当成"间距"报出来，而真实点阵在 3.61 cell。见 docstring。
    edge = float(zero_lag_r) + float(edge_guard_steps) / float(sub)
    interior = rr >= edge
    if not interior.any():
        return float("nan")
    keep = keep & interior
    if not keep.any():
        return float("nan")
    return float(np.min(rr[keep]))


def radial_profile(sac, nbins, step=0.5):
    """Mean SAC per annulus of width ``step`` bins.

    Returns ``(centres, means, counts)``; empty annuli are skipped.
    """
    disc = sac_disc(sac, nbins)
    radius = radius_map(np.asarray(sac).shape[0])
    finite = np.isfinite(disc)
    r_max = float(nbins) - 1.0
    edges = np.arange(0.0, r_max + step, step)
    centres, means, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = finite & (radius > lo) & (radius <= hi)
        vals = disc[sel]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        centres.append(0.5 * (lo + hi))
        means.append(float(vals.mean()))
        counts.append(int(vals.size))
    return np.array(centres), np.array(means), np.array(counts)


def _masked_gaussian(values, mask, sigma):
    """Gaussian smoothing that ignores the invalid part instead of bleeding zeros in."""
    v = np.where(mask, np.nan_to_num(values, nan=0.0), 0.0)
    w = ndimage.gaussian_filter(mask.astype(np.float64), sigma)
    v = ndimage.gaussian_filter(v, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(w > 1e-6, v / np.maximum(w, 1e-6), np.nan)
    return out


def sac_peaks(sac, nbins, min_r=2.0, max_r_frac=0.9, rel_thr=0.35, smooth_sigma=0.8,
              footprint_size=3):
    """2-D local maxima of the SAC outside the central peak.

    Returns ``(radii, angles, values)`` of the kept peaks (arrays, possibly empty).
    """
    sac = np.asarray(sac, dtype=np.float64)
    disc = sac_disc(sac, nbins)
    mask = np.isfinite(disc)
    radius = radius_map(sac.shape[0])
    r_max = max_r_frac * (float(nbins) - 1.0)
    annulus = mask & (radius >= min_r) & (radius <= r_max)
    if not annulus.any():
        return np.array([]), np.array([]), np.array([])

    smooth = _masked_gaussian(disc, mask, smooth_sigma)
    smooth = np.where(np.isfinite(smooth), smooth, -np.inf)
    maxima = ndimage.maximum_filter(smooth, size=int(footprint_size))
    top = float(np.nanmax(smooth[annulus]))
    keep = annulus & np.isfinite(smooth) & (smooth >= maxima - 1e-12) & (smooth > rel_thr * top)
    if not keep.any():
        return np.array([]), np.array([]), np.array([])
    yy, xx = np.nonzero(keep)
    centre = (float(sac.shape[0]) - 1.0) / 2.0
    radii = radius[keep]
    angles = np.degrees(np.arctan2(yy - centre, xx - centre))
    values = smooth[keep]
    order = np.argsort(radii)
    return radii[order], angles[order], values[order]


def estimate_grid_spacing(sac, nbins, min_r=2.0, max_r_frac=0.9, rel_thr=0.35,
                          step=0.5, ring_tol=0.35, prominence=0.02, upsample=4):
    """Grid spacing (in cells) of one unit from its SAC.

    The rate map of an ``nbins x nbins`` arena has at most ~``nbins/2`` bins per
    lattice period, so the raw SAC peak positions are quantised to
    ``sqrt(i^2 + j^2)`` and a spacing histogram degenerates into a handful of
    spikes (2.25, 2.75, 4.24, 4.47, ...).  The SAC is therefore *upsampled*
    (cubic interpolation, factor ``upsample``) before peak picking, which gives
    sub-bin radii.

    Returns a dict with

    ``spacing``        refined spacing (NaN when no peak was found)
    ``spacing_radial`` initial guess from the radial profile
    ``n_peaks``        number of 2-D peaks used for the refinement
    ``peak_value``     SAC value of the strongest first-ring peak
    ``sac_max``        largest SAC value outside the central peak
    """
    nan = float("nan")
    empty = {"spacing": nan, "spacing_radial": nan, "n_peaks": 0,
             "peak_value": nan, "sac_max": nan}
    sac = np.asarray(sac, dtype=np.float64)
    if not np.isfinite(sac_disc(sac, nbins)).any():
        return empty

    scale = float(max(1, int(upsample)))
    if scale > 1.0:
        sac = ndimage.zoom(sac, scale, order=3, mode="nearest")
        nb = float(nbins) * scale
        smooth_sigma = 1.2 * max(1.0, scale / 2.0)
        footprint = int(2 * round(smooth_sigma) + 1)
    else:
        nb = float(nbins)
        smooth_sigma = 0.8
        footprint = 3

    centres, means, _ = radial_profile(sac, nb, step=step)
    if centres.size < 3:
        return empty
    sel = centres >= min_r
    if not sel.any():
        return empty
    profile = means.copy()
    idx_peaks, _ = find_peaks(profile, prominence=prominence * max(1e-9, np.nanmax(profile)))
    idx_peaks = [int(i) for i in idx_peaks if centres[i] >= min_r]
    if idx_peaks:
        spacing_radial = float(centres[idx_peaks[0]])
    else:
        spacing_radial = float(centres[sel][int(np.nanargmax(means[sel]))])

    radii, _angles, values = sac_peaks(sac, nb, min_r=min_r, max_r_frac=max_r_frac,
                                       rel_thr=rel_thr, smooth_sigma=smooth_sigma,
                                       footprint_size=footprint)
    sac_max = nan
    radius = radius_map(sac.shape[0])
    ann = np.isfinite(sac_disc(sac, nb)) & (radius >= min_r) & (radius <= max_r_frac * (nb - 1.0))
    if ann.any():
        sac_max = float(np.nanmax(sac_disc(sac, nb)[ann]))

    if radii.size:
        near = np.abs(radii - spacing_radial) <= max(ring_tol * spacing_radial, 0.75 * scale)
        if near.sum() >= 2:
            spacing = float(np.median(radii[near]))
            peak_value = float(np.max(values[near]))
            n_peaks = int(near.sum())
        else:
            spacing = spacing_radial
            peak_value = float(np.max(values))
            n_peaks = int(radii.size)
    else:
        spacing = spacing_radial
        peak_value = nan
        n_peaks = 0

    return {"spacing": spacing / scale, "spacing_radial": spacing_radial / scale,
            "n_peaks": n_peaks, "peak_value": peak_value, "sac_max": sac_max}


# --------------------------------------------------------------------------- #
# 1-D clustering of the spacings (= grid modules)
# --------------------------------------------------------------------------- #
def cluster_1d(values, k_max=6, min_size=2, sil_thr=0.5, seed=0):
    """Cluster 1-D values with KMeans, ``k`` chosen by silhouette.

    ``k = 1`` (a single module) is returned when

    * fewer than ``2 * min_size`` values are given, or
    * the best silhouette among ``k >= 2`` is below ``sil_thr``, or
    * a candidate partition has a cluster smaller than ``min_size``.

    Returns a dict with ``labels`` (0-based, ordered by increasing centre),
    ``centres``, ``k``, ``silhouette``, ``history`` (all tried ``k``) and
    ``method`` (string).
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = v.size
    nan_out = {"labels": np.zeros(n, dtype=int),
               "centres": [float(np.nanmean(v))] if n else [],
               "k": 1, "silhouette": float("nan"), "history": [], "method": "single"}
    if n == 0:
        return nan_out
    if n < 2 * max(1, min_size):
        return nan_out

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    x = v.reshape(-1, 1)
    best = None
    history = []
    max_k = int(min(k_max, n // max(1, min_size)))
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x)
        labels = km.labels_
        counts = np.bincount(labels, minlength=k)
        if int(counts.min()) < min_size:
            continue
        sil = float(silhouette_score(x, labels))
        centres_sorted = sorted(float(c) for c in km.cluster_centers_.ravel())
        history.append({"k": int(k), "silhouette": sil, "centres": centres_sorted,
                        "sizes": sorted(int(c) for c in counts)})
        if best is None or sil > best["silhouette"]:
            best = {"labels": labels.copy(), "silhouette": sil, "k": int(k),
                    "centres_raw": km.cluster_centers_.ravel().copy()}

    if best is None or best["silhouette"] < sil_thr:
        out = dict(nan_out)
        out["method"] = "single" if best is None else "single (best silhouette %.2f < %.2f)" % (
            best["silhouette"], sil_thr)
        out["history"] = history
        if best is not None:
            out["silhouette"] = float(best["silhouette"])
        return out

    # relabel so that module 0 is the smallest spacing (labels come back as KMeans ids)
    raw = np.asarray(best["centres_raw"], dtype=np.float64).reshape(-1)
    order = np.argsort(raw)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    labels = rank[best["labels"]]
    centres = [float(raw[i]) for i in order]
    return {"labels": labels, "centres": [float(c) for c in centres],
            "k": int(best["k"]), "silhouette": float(best["silhouette"]),
            "history": history, "method": "kmeans+silhouette"}
