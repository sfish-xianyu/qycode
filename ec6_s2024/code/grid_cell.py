# 文件包含了grid定义和grid分析
import os
from runtime_cfg import CFG
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.signal
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
import math
import scipy



def XAVIER(tensor):
    nn.init.xavier_uniform_(tensor)


def ZERO(tensor):
    nn.init.constant_(tensor, 0.0)


def capacity_loss(g_gen_seq, n_steps=10, eps=1e-8):
    """Capacity (anti-sparsity) regulariser for the grid code.

    This is the expression that used to sit *inside* ``GridModuleFast.compute_loss``
    (line 286 of the original file) but was never reachable from training:

        capacity_loss = -((g[:, :10].reshape(B * 10, -1).sum(0) / B / 10) ** 2).mean()

    It takes the mean activity of every unit over the first ``n_steps`` steps of
    the batch, squares that mean, averages over units and flips the sign, so
    *minimising* the loss *maximises* the mean firing rate of every unit.  No
    spatial structure is imposed, it only keeps all ``g_size`` units in use
    (no dead / collapsed code).

    Args:
      g_gen_seq: (B, S, d) grid code sequence, already L2-normalised per step.
      n_steps: number of steps averaged over (original code used 10).

    Returns:
      scalar tensor; the value is ``-(mean_activity ** 2).mean()`` over units,
      i.e. ``-1 / d`` when every unit fires at its maximum.
    """
    B, S, d = g_gen_seq.shape
    n = int(min(max(1, int(n_steps)), S))
    g_ca = g_gen_seq[:, :n].reshape([B * n, d])
    mean_act = g_ca.sum(0) / float(B * n)
    return -(mean_act ** 2).mean()


class GridModuleFast(nn.Module):
    def __init__(self, action_dim, g_size, act_hidden=192):
        super(GridModuleFast, self).__init__()
        self.a2trans_layer = nn.Sequential(
            nn.Linear(action_dim, act_hidden, bias=False),
            nn.Tanh(),
            nn.Linear(act_hidden, g_size**2, bias=False),
        )
        self._init_weights()

        # 与最初实现一致的初始化（相同的 RNG 消耗顺序与归一化方式），
        # same normalisation), so both implementations start from the same g0.
        g0 = torch.rand([1, g_size]) + 0.1
        g0 = F.normalize(g0, p=2, dim=-1)
        self.g0 = nn.Parameter(g0.detach().clone(), requires_grad=True)
        # [refactor] 原实现里还有 scale / normalize_g 两个从未被调用的成员，
        # 已删除。它们不参与任何计算、也不消耗 RNG，删除不改变数值；
        # 旧 checkpoint 用 strict=False 加载，多出来的键会被忽略。
        self.action_dim = action_dim
        self.g_size = g_size
        # 截断 BPTT：每 bptt_truncate 步 detach 一次状态，把最长的
        # 雅可比乘积从 S-1 重降到 k-1 重。默认 0 = 完全不改变原行为。
        self.bptt_truncate = CFG.bptt_truncate
        if self.bptt_truncate > 0:
            print("[bptt] 截断 BPTT：每 %d 步 detach 一次" % self.bptt_truncate, flush=True)
        self.use_compile = False
        self._compiled_step_fn = None
        self._compiled_gen_seq_fn = None
        self._compiled_gen_seq_addlast_fn = None
        self._maybe_init_compile()

    def _maybe_init_compile(self):
        # torch.compile on bound methods with optional kwargs is fragile (and the
        # 64-step recurrence is dominated by tiny kernels anyway), so it stays off.
        self._compiled_step_fn = None
        self._compiled_gen_seq_fn = None
        self._compiled_gen_seq_addlast_fn = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                XAVIER(m.weight)
                if m.bias is not None:
                    ZERO(m.bias)

    def norm_g(self,g):
        g = F.relu(g)
        clip_g = F.normalize(g, p=2, dim=-1, eps=1e-5)
        return clip_g

    def action_to_features(self, act):
        return self.a2trans_layer(act).reshape(list(act.shape[:-1]) + [self.g_size, self.g_size])

    def _step_from_action_features(self, prev_g, action_features):
        g_delta = torch.matmul(action_features, prev_g.unsqueeze(-1)).squeeze(-1)
        # M(a) 固定为 I + W(a)。
        new_g = g_delta + prev_g
        return self.norm_g(new_g)

    def _gen_seq_from_action_features(self, last_g, action_features_tm):
        S, B = action_features_tm.shape[:2]
        out = torch.empty([B, S, self.g_size], dtype=last_g.dtype, device=last_g.device)
        cur_g = last_g
        for t in range(S):
            cur_g = self._step_from_action_features(cur_g, action_features_tm[t])
            if self.bptt_truncate and (t + 1) % self.bptt_truncate == 0:
                cur_g = cur_g.detach()
            out[:, t] = cur_g
        return out

    def _gen_seq_from_action_features_addlast(self, last_g, action_features_tm):
        S, B = action_features_tm.shape[:2]
        out = torch.empty([B, S + 1, self.g_size], dtype=last_g.dtype, device=last_g.device)
        out[:, 0] = last_g
        cur_g = last_g
        for t in range(S):
            cur_g = self._step_from_action_features(cur_g, action_features_tm[t])
            if self.bptt_truncate and (t + 1) % self.bptt_truncate == 0:
                cur_g = cur_g.detach()
            out[:, t + 1] = cur_g
        return out
    
    def forward(self, prev_g, prev_a):
        action_features = self.action_to_features(prev_a)
        step_fn = self._compiled_step_fn if self._compiled_step_fn is not None else self._step_from_action_features
        return step_fn(prev_g, action_features)

    def gen_seq_g(self, last_g, act_seq, add_last=False):
        # last_g: B, g_size
        # act_seq: B, S, action_dim
        action_features = self.action_to_features(act_seq).transpose(1, 0).contiguous()
        if add_last:
            gen_fn = self._compiled_gen_seq_addlast_fn if self._compiled_gen_seq_addlast_fn is not None else self._gen_seq_from_action_features_addlast
        else:
            gen_fn = self._compiled_gen_seq_fn if self._compiled_gen_seq_fn is not None else self._gen_seq_from_action_features
        g_gen_seq = gen_fn(last_g, action_features)
        if bool(torch.isnan(g_gen_seq).any()):
            # this used to be pdb.set_trace(), which blocks unattended training on stdin
            print("[warn] GridModuleFast.gen_seq_g produced NaN, check lr / memory write scale", flush=True)
        return g_gen_seq
    
def circle_mask(size, radius, in_val=1.0, out_val=0.0):
  """Calculating the grid scores with different radius."""
  sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
  x = np.linspace(-sz[0], sz[1], size[1])
  x = np.expand_dims(x, 0)
  x = x.repeat(size[0], 0)
  y = np.linspace(-sz[0], sz[1], size[1])
  y = np.expand_dims(y, 1)
  y = y.repeat(size[1], 1)
  z = np.sqrt(x**2 + y**2)
  z = np.less_equal(z, radius)
  vfunc = np.vectorize(lambda b: b and in_val or out_val)
  return vfunc(z)


class GridScorer(object):
  """Class for scoring ratemaps given trajectories."""

  def __init__(self, nbins, coords_range, mask_parameters, min_max=False):
    """Scoring ratemaps given trajectories.

    Args:
      nbins: Number of bins per dimension in the ratemap.
      coords_range: Environment coordinates range.
      mask_parameters: parameters for the masks that analyze the angular
        autocorrelation of the 2D autocorrelation.
      min_max: Correction.
    """
    self._nbins = nbins
    self._min_max = min_max
    self._coords_range = coords_range
    self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
    # Create all masks
    self._masks = [(self._get_ring_mask(mask_min, mask_max), (mask_min,
                                                              mask_max))
                   for mask_min, mask_max in mask_parameters]
    # Mask for hiding the parts of the SAC that are never used
    self._plotting_sac_mask = circle_mask(
        [self._nbins * 2 - 1, self._nbins * 2 - 1],
        self._nbins,
        in_val=1.0,
        out_val=np.nan)

  def calculate_ratemap(self, xs, ys, activations, statistic='mean'):
    return scipy.stats.binned_statistic_2d(
        xs,
        ys,
        activations,
        bins=self._nbins,
        statistic=statistic,
        range=self._coords_range)[0]

  def _get_ring_mask(self, mask_min, mask_max):
    n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
    return (circle_mask(n_points, mask_max * self._nbins) *
            (1 - circle_mask(n_points, mask_min * self._nbins)))

  def grid_score_60(self, corr):
    if self._min_max:
      return np.minimum(corr[60], corr[120]) - np.maximum(
          corr[30], np.maximum(corr[90], corr[150]))
    else:
      return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

  def grid_score_90(self, corr):
    return corr[90] - (corr[45] + corr[135]) / 2

  def calculate_sac(self, seq1):
    """Calculating spatial autocorrelogram."""
    seq2 = seq1

    def filter2(b, x):
      stencil = np.rot90(b, 2)
      return scipy.signal.convolve2d(x, stencil, mode='full')

    seq1 = np.nan_to_num(seq1)
    seq2 = np.nan_to_num(seq2)

    ones_seq1 = np.ones(seq1.shape)
    ones_seq1[np.isnan(seq1)] = 0
    ones_seq2 = np.ones(seq2.shape)
    ones_seq2[np.isnan(seq2)] = 0

    seq1[np.isnan(seq1)] = 0
    seq2[np.isnan(seq2)] = 0

    seq1_sq = np.square(seq1)
    seq2_sq = np.square(seq2)

    seq1_x_seq2 = filter2(seq1, seq2)
    sum_seq1 = filter2(seq1, ones_seq2)
    sum_seq2 = filter2(ones_seq1, seq2)
    sum_seq1_sq = filter2(seq1_sq, ones_seq2)
    sum_seq2_sq = filter2(ones_seq1, seq2_sq)
    n_bins = filter2(ones_seq1, ones_seq2)
    n_bins_sq = np.square(n_bins)

    std_seq1 = np.power(
        np.subtract(
            np.divide(sum_seq1_sq, n_bins),
            (np.divide(np.square(sum_seq1), n_bins_sq))), 0.5)
    std_seq2 = np.power(
        np.subtract(
            np.divide(sum_seq2_sq, n_bins),
            (np.divide(np.square(sum_seq2), n_bins_sq))), 0.5)
    covar = np.subtract(
        np.divide(seq1_x_seq2, n_bins),
        np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
    x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
    x_coef = np.real(x_coef)
    x_coef = np.nan_to_num(x_coef)
    return x_coef

  def rotated_sacs(self, sac, angles):
    return [
        scipy.ndimage.rotate(sac, angle, reshape=False)
        for angle in angles
    ]

  def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
    """Calculate Pearson correlations of area inside mask at corr_angles."""
    masked_sac = sac * mask
    ring_area = np.sum(mask)
    # Calculate dc on the ring area
    masked_sac_mean = np.sum(masked_sac) / ring_area
    # Center the sac values inside the ring
    masked_sac_centered = (masked_sac - masked_sac_mean) * mask
    variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
    corrs = dict()
    for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
      masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
      cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
      corrs[angle] = cross_prod / variance
    return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

  def get_scores(self, rate_map):
    """Get summary of scrores for grid cells."""
    sac = self.calculate_sac(rate_map)
    rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

    scores = [
        self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
        for mask, mask_params in self._masks  # pylint: disable=unused-variable
    ]
    scores_60, scores_90, variances = map(np.asarray, zip(*scores))  # pylint: disable=unused-variable
    max_60_ind = np.argmax(scores_60)
    max_90_ind = np.argmax(scores_90)

    return (scores_60[max_60_ind], scores_90[max_90_ind],
            self._masks[max_60_ind][1], self._masks[max_90_ind][1], sac)

  def plot_ratemap(self, ratemap, ax=None, title=None, smooth=True, sigma=0.8, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Plot ratemaps."""
    if ax is None:
      ax = plt.gca()
    # Apply Gaussian smoothing if requested
    if smooth:
      ratemap = scipy.ndimage.gaussian_filter(ratemap, sigma=sigma)
    # Plot the ratemap
    ax.imshow(ratemap, interpolation='none', *args, **kwargs)
    # ax.pcolormesh(ratemap, *args, **kwargs)
    ax.axis('off')
    if title is not None:
      ax.set_title(title)

  def plot_sac(self,
               sac,
               mask_params=None,
               ax=None,
               title=None,
               *args,
               **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Plot spatial autocorrelogram."""
    if ax is None:
      ax = plt.gca()
    # Plot the sac
    useful_sac = sac * self._plotting_sac_mask
    ax.imshow(useful_sac, interpolation='none', *args, **kwargs)
    # ax.pcolormesh(useful_sac, *args, **kwargs)
    # Plot a ring for the adequate mask
    if mask_params is not None:
      center = self._nbins - 1
      ax.add_artist(
          plt.Circle(
              (center, center),
              mask_params[0] * self._nbins,
              # lw=bump_size,
              fill=False,
              edgecolor='k'))
      ax.add_artist(
          plt.Circle(
              (center, center),
              mask_params[1] * self._nbins,
              # lw=bump_size,
              fill=False,
              edgecolor='k'))
    ax.axis('off')
    if title is not None:
      ax.set_title(title)

def get_scores_and_plot(scorer,
                        data_abs_xy,
                        activations,
                        filename,
                        plot_graphs=True,
                        nbins=20,  
                        cm="jet",  # 'viridis' 
                        sort_by_score_60=True):
  """Plotting function."""

  # Concatenate all trajectories
  xy = data_abs_xy.reshape(-1, data_abs_xy.shape[-1])
  act = activations.reshape(-1, activations.shape[-1])
  n_units = act.shape[1]
  # Get the rate-map for each unit
  s = [
      scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i])
      for i in range(n_units)
  ]
  # Get the scores
  score_60, score_90, max_60_mask, max_90_mask, sac = zip(
      *[scorer.get_scores(rate_map) for rate_map in s])
  # Separations
  # separations = map(np.mean, max_60_mask)
  # Sort by score if desired
  if sort_by_score_60:
    ordering = np.argsort(-np.array(score_60))
    using_score = score_60
    using_mask = max_60_mask
  else:
    ordering = np.argsort(-np.array(score_90))
    using_score = score_90
    using_mask = max_90_mask
  # else:
  #   ordering = range(n_units)
  cols = 16
  rows = int(np.ceil(n_units / cols))
  fig = plt.figure(figsize=(24, rows * 4))
  for i in range(n_units):
    rf = plt.subplot(rows * 2, cols, i + 1)
    acr = plt.subplot(rows * 2, cols, n_units + i + 1)
    if i < n_units:
      index = ordering[i]
      title = "%d (%.2f)" % (index, using_score[index])
      # Plot the activation maps
      scorer.plot_ratemap(s[index], ax=rf, title=title, cmap=cm)
      # Plot the autocorrelation of the activation maps
      scorer.plot_sac(
          sac[index],
          mask_params=using_mask[index],
          ax=acr,
          title=title,
          cmap=cm)
  # Save
  with PdfPages(filename, "w") as f:
    plt.savefig(f, format="pdf")
  plt.close(fig)
  return (np.asarray(score_60), np.asarray(score_90),
          np.asarray(map(np.mean, max_60_mask)),
          np.asarray(map(np.mean, max_90_mask)))


def draw_grid_score(pos, fr, file_name):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 0.9, num=10)
    masks_parameters = zip(starts, ends.tolist())
    env_len = int(np.max(pos))
    scorer = GridScorer(env_len, None, masks_parameters)
    score_60, score_90, _, _ = get_scores_and_plot(scorer, pos, fr, file_name, sort_by_score_60=False)
    return score_90

def draw_grid(par, pos_seq, grid_seq, draw_path, top_k=10):
    B, S = pos_seq.shape
    pos_ = pos_seq.reshape([B * S]).detach().cpu().numpy()
    env_spec = getattr(par, "env", None)
    if hasattr(env_spec, "widths") and isinstance(env_spec.widths, list):
      max_width = max(env_spec.widths)
    elif hasattr(env_spec, "widths"):
      max_width = env_spec.widths
    elif hasattr(par, "env_width"):
      max_width = par.env_width
    else:
      raise AttributeError("draw_grid requires par.env.widths or par.env_width")
    pos_x = pos_ / max_width
    pos_y = pos_ % max_width
    pos_xy = np.stack([pos_x, pos_y], axis=-1).astype(np.int32)
    # print(pos_xy[0:20])
    score_90 = draw_grid_score(pos_xy, grid_seq.reshape([B * S, -1]).detach().cpu().numpy(),
                    f"{draw_path}grid.pdf")
    np.save(f"{draw_path}gridness.npy", score_90)
    mean_90_topk = np.sort(-score_90)[:top_k].mean() * -1
    return mean_90_topk
    # loss, detail = self.dynamic.grid.compute_loss(grid_seq, pos_seq)
    # print(detail)
