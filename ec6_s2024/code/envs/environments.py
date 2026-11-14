#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""
import pdb

import numpy as np
import copy as cp
import itertools
import os
try:                       # 以子目录为 cwd 直接运行时也能导入
    from runtime_cfg import CFG
except ImportError:        # pragma: no cover
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runtime_cfg import CFG

# ---------------------------------------------------------------------------
# 均匀访问模式（WALK_MODE=deficit）
#
# 默认的随机游走在矩形里不是均匀的：角落只能从两个方向进来，驻留概率低
# （实测 0.60x），内部 1.07x，最大/最小相差约 2 倍。训练损失按时间步采样，
# 于是梯度偏向高频格。deficit 模式把"已访问次数"记成全局账本（跨 env、跨
# epoch，模块级），每一步在可去的邻居里按 exp(-(c-c_min)/tau) 加权 —— 越少
# 去过越优先。全局账本很关键：每个 epoch 只有 64 步，若按本轮计数做缺口，
# 121 个格子里绝大多数计数都是 0，权重全相等，等于没改。
# ---------------------------------------------------------------------------
_OCC = {}


def reset_occupancy(width=None, height=None):
    """清空全局访问账本（测基线时必须调用）。"""
    if width is None:
        _OCC.clear()
    else:
        _OCC.pop((int(width), int(height)), None)


def _occ_get(width, height):
    key = (int(width), int(height))
    if key not in _OCC:
        _OCC[key] = np.zeros(int(width) * int(height), dtype=np.float64)
    return _OCC[key]


def _deficit_choice(occ, available, tau=1.0):
    """在 available 里选一个访问最少的（差值为 0 的按权重均分）。"""
    c = occ[available]
    w = np.exp(-(c - c.min()) / tau)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return int(np.random.choice(available))
    return int(np.random.choice(available, p=w / s))


# ---------------------------------------------------------------------------
# aligned：共享起点 + 共享终点，中间的顺序随机（RyLan2023 的 permute 构造）
#
# 每个 epoch 抽一次 (起点 s, 净位移 (dx, dy))，整批共用；每个 env 各自把
# 一步数多重集随机重排，所以 32 条轨迹起点相同、终点也相同、路径不同。
# 与论文的差别只在环境：连续速度 -> 4 邻格移动，所以边界必须显式处理。
#
# 边界处理不用「排完再检查」（净位移 4 格时约 1/3 的排列必然出界），而是
# 按轴拆开：每个轴单独做 1 维桥上走，各自先保证不越界，再随机交织两轴。
# 实测 W=17/25/31、T=64 时 1 维重抽平均 1.2-1.8 次、失败率 0。
# ---------------------------------------------------------------------------
_ALIGN = {}
_ALIGN_FAIL = [0]


def _align_note_fail():
    """生成失败退回普通游走时计数；每 1000 次打一行 stderr，避免静默混入。"""
    _ALIGN_FAIL[0] += 1
    if _ALIGN_FAIL[0] == 1 or _ALIGN_FAIL[0] % 1000 == 0:
        import sys
        print("[aligned] fallback to random walk, n=%d" % _ALIGN_FAIL[0], file=sys.stderr)


def reset_alignment(width=None, height=None, seq_len=None):
    """清空共享的 (起点, 净位移) 配置；gen_data 每个 epoch 调一次。"""
    if width is None:
        _ALIGN.clear()
        return
    for k in list(_ALIGN):
        if k[0] == int(width) and k[1] == int(height) and (seq_len is None or k[2] == int(seq_len)):
            _ALIGN.pop(k, None)


def _align_cfg(width, height, seq_len, max_disp=4):
    """本 epoch 的共享起点与净位移（同 (width,height,seq_len) 复用同一份）。"""
    key = (int(width), int(height), int(seq_len))
    cfg = _ALIGN.get(key)
    if cfg is not None:
        return cfg

    # T 个位置只有 T-1 次移动；两轴步数之和必须正好是 T-1，否则最后一个
    # 移动用不上，终点会随排列随机差一步（32 条轨迹就凑不出共同终点了）。
    n_total = max(0, int(seq_len) - 1)
    # 63 步分给两轴必然是 31/32；哪条轴多一步要随机，否则该轴净位移的奇偶
    # 被永久钉死（x 恒奇、y 恒偶），2000 个 epoch 下来会印出一个方向偏置。
    if np.random.rand() < 0.5:
        n_x = n_total // 2
    else:
        n_x = n_total - n_total // 2
    n_y = n_total - n_x

    # 抽之前先验证这个 (s, dx, dy) 到底走不走得通：两轴各试一次 1 维桥。
    # 不然整批 32 个 env 会各自重试 200 次才失败，慢 300 倍还会混进随机游走。
    for _draw in range(40):
        dx = int(np.random.randint(-max_disp, max_disp + 1))
        dy = int(np.random.randint(-max_disp, max_disp + 1))
        if (n_x + dx) % 2:
            dx += 1 if dx < max_disp else -1
        if (n_y + dy) % 2:
            dy += 1 if dy < max_disp else -1
        dx = int(np.clip(dx, -n_x, n_x))
        dy = int(np.clip(dy, -n_y, n_y))
        sx = int(np.random.randint(0, int(width)))
        sy = int(np.random.randint(0, int(height)))
        if (_axis_seq(sx, n_x, dx, int(width), max_try=12) is not None and
                _axis_seq(sy, n_y, dy, int(height), max_try=12) is not None):
            cfg = {"sx": sx, "sy": sy, "dx": dx, "dy": dy,
                   "n_x": n_x, "n_y": n_y, "s": sx * int(width) + sy}
            _ALIGN[key] = cfg
            return cfg

    # 场地太小、怎么也放不下（如 W=11）：退化成零净位移，起点居中
    sx = int(width) // 2
    sy = int(height) // 2
    cfg = {"sx": sx, "sy": sy, "dx": 0, "dy": 0,
           "n_x": n_x, "n_y": n_y, "s": sx * int(width) + sy}
    _ALIGN[key] = cfg
    return cfg


def _axis_seq(start, n_move, delta, side, max_try=200):
    """1 维：n_move 步、净位移 delta、起点 start，返回 +/-1 序列或 None。

    先随机排列，再检查是否越出 [0, side-1]；1 维上重抽很便宜（实测 1-2 次）。
    """
    up = (int(n_move) + int(delta)) // 2
    dn = int(n_move) - up
    if up < 0 or dn < 0:
        return None
    base = np.concatenate([np.ones(up, dtype=np.int8), -np.ones(dn, dtype=np.int8)])
    for _ in range(max_try):
        seq = np.random.permutation(base)
        cum = int(start) + np.concatenate([[0], np.cumsum(seq.astype(np.int64))])
        if cum.min() >= 0 and cum.max() <= side - 1:
            return seq
    return None


def _aligned_positions(cfg, width, height, seq_len, max_try=200):
    """按 cfg 生成一条轨迹（位置序列）或 None。

    两轴各自成桥（保证各自不越界），再随机交织 —— 交织不改变任一轴的
    取值范围，所以整体一定不越界，且终点恒为 (sx+dx, sy+dy)。
    """
    for _ in range(max_try):
        sx = _axis_seq(cfg["sx"], cfg["n_x"], cfg["dx"], int(width))
        sy = _axis_seq(cfg["sy"], cfg["n_y"], cfg["dy"], int(height))
        if sx is not None and sy is not None:
            break
    else:
        return None

    # 0 = 走 x 轴，1 = 走 y 轴；随机交织
    who = np.concatenate([np.zeros(cfg["n_x"], dtype=np.int8),
                          np.ones(cfg["n_y"], dtype=np.int8)])
    who = np.random.permutation(who)
    ix = iy = 0
    x, y = cfg["sx"], cfg["sy"]
    pos = np.zeros(int(seq_len), dtype=np.int16)
    pos[0] = x * int(width) + y
    for t in range(1, int(seq_len)):
        if who[t - 1] == 0:
            x += int(sx[ix]); ix += 1
        else:
            y += int(sy[iy]); iy += 1
        pos[t] = x * int(width) + y
    return pos



class Environment:
    def __init__(self, params, width, height):
        super(Environment, self).__init__()
        self.par = params
        self.width = width
        self.height = height
        self.n_actions = self.par.env.n_actions
        self.rels = self.par.env.rels
        self.walk_len = None
        self.start_state, self.adj, self.tran, self.states_mat = None, None, None, None


class Rectangle(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width, height)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states(width, height):
        return width * height

    def world(self, torus=False):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        states = int(self.width * self.height)
        adj = np.zeros((states, states))

        for i in range(states):
            # stay still
            if self.par.env.stay_still:
                adj[i, i] = 1
            # up - down
            if i + self.width < states:
                adj[i, i + self.width] = 1
                adj[i + self.width, i] = 1
                # left - right
            if np.mod(i, self.width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1

            if torus and np.mod(i, self.width) == 0:
                adj[i, i + self.width - 1] = 1
                adj[i + self.width - 1, i] = 1

            if torus and int(i / self.width) == 0:
                adj[i, i + states - self.width] = 1
                adj[i + states - self.width, i] = 1

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran
        allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
        self.start_state = np.random.choice(allowed_states)

    def relation(self, s1, s2):  # 根据state计算真实应该行动的方向
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == self.width or diff == -self.width * (self.height - 1):  # down
            rel_type = 'down'
        elif diff == -self.width or diff == self.width * (self.height - 1):  # up
            rel_type = 'up'
        elif diff == -1 or diff == (self.width - 1):  # left
            rel_type = 'left'
        elif diff == 1 or diff == -(self.width - 1):  # right
            rel_type = 'right'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):  # 确定每个位置对应的sense
        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)

        for i in range(self.n_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        self.states_mat = states_vec.astype(int)

    def walk(self, rand_begin=False, seq_len=0, return_visited=False, walk_mode=None):  # 获得了随机的轨迹和对应的方向
        """  pos-t 代表t时刻的位置，act-t代表上一时刻采取的动作，逻辑是先a后p
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        mode = walk_mode if walk_mode is not None else CFG.walk_mode
        occ = None
        if mode == "deficit":
            occ = _occ_get(self.width, self.height)
            allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
            self.start_state = _deficit_choice(occ, allowed_states)
            occ[int(self.start_state)] += 1
        elif rand_begin:
            allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
            self.start_state = np.random.choice(allowed_states)
        time_steps = self.walk_len if seq_len==0 else seq_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))
        visited = np.zeros(self.n_states)
        pos_visited = np.zeros(time_steps)

        current_angle = np.random.uniform(-np.pi, np.pi)

        # consider rectangular environment. if move +1=up, -1=down, +length=right, -length=left.
        if self.height * self.width != len(self.adj):
            raise ValueError('incorrect height/width : height * width not equal to number of states')

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1
        visited[position[0]] = 1
        if mode == "aligned":
            aligned = self._walk_aligned(time_steps)
            if aligned is not None:
                pos_a, dir_a, vis_a = aligned
                if return_visited:
                    return pos_a, dir_a, vis_a
                return pos_a, dir_a
            # 生成失败就退回普通随机游走（不应发生，留个安全网）
            _align_note_fail()
        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)

            # head towards objects, or in straight lines
            if occ is not None:
                new_poss_pos = _deficit_choice(occ, available)
            elif self.par.env.bias_type == 'angle':
                new_poss_pos, current_angle = self.move_straight_bias(current_angle, position[i], available)
            else:
                new_poss_pos = np.random.choice(available)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))
            if occ is not None:
                occ[int(position[i + 1])] += 1
            pos_visited[i+1] = visited[position[i + 1]]
            visited[position[i + 1]] = 1
            relation_taken, _ = self.relation(position[i], position[i + 1])
            if relation_taken < self.n_actions:
                direc[relation_taken, i + 1] = 1
            # stay still is just a set of zeros
        if return_visited:
            return position, direc, pos_visited
        return position, direc

    def walk_cont(self, seq_len=0, step_a=None, rng=None):
        """连续速度游走（RyLan2023 的设定）：v ~ U(-a, a)^2，位置累积，越界原地不动。

        与 walk() 的唯一差别是速度的分布：walk() 从 4 个正交单位向量里等概率选，
        这里从连续均匀分布里抽。位置因此落在连续平面上，不再限制在格点。

        步长半宽 a 默认 sqrt(1.5)，使每轴方差 = a^2/3 = 0.5，与 4 邻域游走
        （每轴方差 0.5）完全一致 —— 两种走法的扩散速度相同，只有"是否连续"不同。

        返回 (pos (T,2) float, vel (2,T) float, visited (T,))。
        vel 与 walk() 的 direc 口径一致：第 t 列是走到 pos[t] 所走的那一步，
        被边界挡住时记零（等价于 walk() 里的"stay"）。
        """
        a = CFG.walk_vel_a if step_a is None else float(step_a)
        T = int(self.walk_len if seq_len == 0 else seq_len)
        W, H = int(self.width), int(self.height)
        rng = np.random.default_rng() if rng is None else rng
        pos = np.empty((T, 2), dtype=np.float32)
        vel = np.zeros((2, T), dtype=np.float32)
        xy = np.array([W / 2.0, H / 2.0], dtype=np.float32)
        pos[0] = xy
        for t in range(1, T):
            d = rng.uniform(-a, a, size=2).astype(np.float32)
            new = xy + d
            if (new[0] >= 0) and (new[0] < W) and (new[1] >= 0) and (new[1] < H):
                xy = new
                vel[:, t] = d
            pos[t] = xy
        visited = np.zeros(T, dtype=np.float32)
        return pos, vel, visited

    def _walk_aligned(self, time_steps):
        """共享起终点的重排轨迹（RyLan2023 的 permute 构造，格点版）。

        返回 (position, direc, pos_visited)；生成失败返回 None。
        direc 由相邻位置反推，与普通游走口径一致，loss 侧不需要任何改动。
        """
        cfg = _align_cfg(self.width, self.height, time_steps)
        pos = _aligned_positions(cfg, self.width, self.height, time_steps)
        if pos is None:
            return None
        self.start_state = int(cfg["s"])
        direc = np.zeros((self.n_actions, time_steps))
        visited = np.zeros(self.n_states)
        pos_visited = np.zeros(time_steps)
        direc[0, 0] = 1
        visited[pos[0]] = 1
        for i in range(time_steps - 1):
            pos_visited[i + 1] = visited[pos[i + 1]]
            visited[pos[i + 1]] = 1
            rel, _ = self.relation(pos[i], pos[i + 1])
            if rel < self.n_actions:
                direc[rel, i + 1] = 1
        return pos, direc, pos_visited

    def move_straight_bias(self, current_angle, position, available):
        # angle is allo-centric
        # from available position - find distance and angle from current pos
        diff_angle_min = np.pi / 4
        angles = [self.angle_between_states_square(position, x) if x != position else 10000 for x in available]
        # find angle closest to current angle
        a_diffs = [np.abs(a - current_angle) for a in angles]
        a_diffs = [a if a < np.pi else np.abs(2 * np.pi - a) for a in a_diffs]

        angle_diff = np.min(a_diffs)

        if angle_diff < diff_angle_min:
            a_min_index = np.where(a_diffs == angle_diff)[0][0]
            angle = current_angle
        else:  # hit a wall - then do random non stationary choice
            p_angles = [1 if a < 100 else 0.000001 for a in angles]
            a_min_index = np.random.choice(np.arange(len(available)), p=np.asarray(p_angles) / sum(p_angles))
            angle = angles[a_min_index]

        new_poss_pos = int(available[a_min_index])

        angle += np.random.uniform(-self.par.env.angle_bias_change, self.par.env.angle_bias_change)
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # keep between +- pi

        if np.random.rand() > self.par.env.direc_bias:
            p = self.tran[int(position), available]
            new_poss_pos = np.random.choice(available, p=p)

        return new_poss_pos, angle

    def angle_between_states_square(self, s1, s2):
        x1 = s1 % self.width
        x2 = s2 % self.width

        y1 = np.floor(s1 / self.width)
        y2 = np.floor(s2 / self.width)

        angle = np.arctan2(y1 - y2, x2 - x1)

        return angle

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xs = xs.flatten() - (self.width - 1) / 2
        ys = - (ys.flatten() - (self.height - 1) / 2)

        if cells is not None:
            cell_prepared = cp.deepcopy(cells).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys

    def around_walk(self, seq_len=0, return_visited=False):
        allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
        assert 0 in allowed_states, "0 not in allowed states"
        self.start_state = 0

        time_steps = self.walk_len if seq_len==0 else seq_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))
        visited = np.zeros(self.n_states)
        pos_visited = np.zeros(time_steps)

        position[0] = int(self.start_state)
        direc[0, 0] = 1
        visited[position[0]] = 1
        
        # Helper function to get next boundary state in clockwise order
        def get_next_boundary_state(current_state):
            row = current_state // self.width
            col = current_state % self.width
            
            # Top edge (row=0), moving right
            if row == 0 and col < self.width - 1:
                return current_state + 1
            # Right edge (col=width-1), moving down
            elif col == self.width - 1 and row < self.height - 1:
                return current_state + self.width
            # Bottom edge (row=height-1), moving left
            elif row == self.height - 1 and col > 0:
                return current_state - 1
            # Left edge (col=0), moving up
            elif col == 0 and row > 0:
                return current_state - self.width
            # Corner cases
            elif row == 0 and col == self.width - 1:  # Top-right corner
                return current_state + self.width
            elif row == self.height - 1 and col == self.width - 1:  # Bottom-right corner
                return current_state - 1
            elif row == self.height - 1 and col == 0:  # Bottom-left corner
                return current_state - self.width
            elif row == 0 and col == 0 and self.height > 1:  # Top-left corner (start)
                return current_state + 1
            else:
                # Single row or column case, cycle back to start
                return 0
        
        # Generate trajectory
        for i in range(time_steps - 1):
            current_pos = position[i]
            next_pos = get_next_boundary_state(current_pos)
            
            # Ensure the next position is valid and adjacent
            if self.adj[current_pos, next_pos] == 1:
                position[i + 1] = next_pos
            else:
                # If not adjacent (shouldn't happen on boundary), stay in place
                position[i + 1] = current_pos
            
            # Update visited tracking
            pos_visited[i + 1] = visited[position[i + 1]]
            visited[position[i + 1]] = 1
            
            # Determine the action/relation taken
            relation_taken, _ = self.relation(position[i], position[i + 1])
            if relation_taken < self.n_actions:
                direc[relation_taken, i + 1] = 1
        
        if return_visited:
            return position, direc, pos_visited
        return position, direc
     

class Hexagonal(Environment):
    def __init__(self, params, width):
        self.n_states, self.graph_states = self.get_n_states(width)
        # Without the square2hex() boundary the world keeps the full w*w offset grid
        # (state s -> row s//w, col s%w).  get_n_states() would still report the
        # inscribed-hexagon count (91 for w=11), which then disagrees with adj's 121
        # rows; anything sizing a one-hot from env.n_states would be silently wrong.
        if not bool(getattr(params.env, 'hex_boundary', False)):
            self.n_states = self.graph_states
        super().__init__(params, width, width)

    @staticmethod
    def get_n_states(width):
        return int((3 * (width ** 2) + 1) / 4), width ** 2

    def world(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        width = self.width
        states = int(width ** 2)
        adj = np.zeros((states, states))

        for n1 in range(width):  # row
            for n2 in range(width):  # col
                if n2 < width - 1:
                    # go right
                    adj[n1 * width + n2, n1 * width + n2 + 1] = 1
                if n1 < width - 1:
                    # go down
                    adj[n1 * width + n2, (n1 + 1) * width + n2] = 1
                    if np.mod(n1, 2) == 0 and n2 < width - 1:
                        # go down and right
                        adj[n1 * width + n2, (n1 + 1) * width + n2 + 1] = 1
                    elif np.mod(n1, 2) == 1 and n2 > 0:
                        # go down and left
                        adj[n1 * width + n2, (n1 + 1) * width + n2 - 1] = 1

                if self.par.env.stay_still:
                    adj[n1 * width + n2, n1 * width + n2] = 1
        adj = adj + adj.T
        adj = np.minimum(adj, 1)

        if self.par.env.hex_boundary:
            adj = square2hex(adj)

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran
        allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
        self.start_state = np.random.choice(allowed_states)

    def relation(self, s1, s2):
        width = self.width
        level = np.mod(int(s1 / width), 2)
        diff = s2 - s1
        if diff == width - 1 or (diff == width and level == 0):
            rel_type = 'down left'
        elif diff == width + 1 or (diff == width and level == 1):
            rel_type = 'down right'
        elif diff == -(width + 1) or (diff == -width and level == 0):
            rel_type = 'up left'
        elif diff == -(width - 1) or (diff == -width and level == 1):
            rel_type = 'up right'
        elif diff == -1:
            rel_type = 'left'
        elif diff == 1:
            rel_type = 'right'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.graph_states)
        choices = np.arange(self.par.s_size)

        for i in range(self.graph_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        self.states_mat = states_vec.astype(int)

    def walk(self, rand_begin=False, seq_len=0, return_visited=False):
        """Same contract as Rectangle.walk: (pos, direc[, pos_visited]).

        The original Hexagonal.walk() took no arguments and always used
        ``self.walk_len``, but envs/data_utils.gen_data calls
        ``env.walk(rand_begin=False, seq_len=seq_len, return_visited=True)`` for every
        world -- so the hex world could not be trained at all before this.
        """
        if rand_begin:
            allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
            self.start_state = np.random.choice(allowed_states)
        time_steps = self.walk_len if seq_len == 0 else seq_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))
        visited = np.zeros(self.n_states)
        pos_visited = np.zeros(time_steps)

        current_angle = np.random.uniform(-np.pi, np.pi)
        # breakpoint()
        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1
        visited[position[0]] = 1

        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)

            if self.par.env.bias_type == 'angle':
                new_poss_pos, current_angle = self.move_straight_bias(current_angle, position[i], available)
            else:
                new_poss_pos = np.random.choice(available)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            pos_visited[i + 1] = visited[position[i + 1]]
            visited[position[i + 1]] = 1

            relation_taken, _ = self.relation(position[i], position[i + 1])
            if relation_taken < self.n_actions:
                direc[relation_taken, i + 1] = 1
            # stay still is just a set of zeros

        if return_visited:
            return position, direc, pos_visited
        return position, direc

    def move_straight_bias(self, current_angle, position, available):
        # angle is allo-centric
        # from available position - find distance and angle from current pos
        diff_angle_min = np.pi / 6
        angles = [self.angle_between_states_hex(position, x) if x != position else 10000 for x in available]
        # find angle closest to current angle
        a_diffs = [np.abs(a - current_angle) for a in angles]
        a_diffs = [a if a < np.pi else np.abs(2 * np.pi - a) for a in a_diffs]

        angle_diff = np.min(a_diffs)

        if angle_diff < diff_angle_min:
            a_min_index = np.where(a_diffs == angle_diff)[0][0]
            angle = current_angle
        else:  # hit a wall - then do random non stationary choice
            p_angles = [1 if a < 100 else 0.000001 for a in angles]
            a_min_index = np.random.choice(np.arange(len(available)), p=np.asarray(p_angles) / sum(p_angles))
            angle = angles[a_min_index]

        new_poss_pos = int(available[a_min_index])

        angle += np.random.uniform(-self.par.env.angle_bias_change, self.par.env.angle_bias_change)
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # keep between +- pi

        if np.random.rand() > self.par.env.direc_bias:
            p = self.tran[int(position), available]
            new_poss_pos = np.random.choice(available, p=p)

        return new_poss_pos, angle

    def angle_between_states_hex(self, s1, s2):
        width = self.width
        level_1 = np.mod(int(s1 / width), 2)
        level_2 = np.mod(int(s2 / width), 2)

        x1 = s1 % width
        x2 = s2 % width

        y1 = np.floor(s1 / width)
        y2 = np.floor(s2 / width)

        if level_1 == level_2:
            angle = np.arctan2(y1 - y2, x2 - x1)
        elif level_1 == 0 and level_2 == 1:
            angle = np.arctan2((np.sqrt(3) / 2) * (y1 - y2), x2 - 0.5 - x1)
        elif level_1 == 1 and level_2 == 0:
            angle = np.arctan2((np.sqrt(3) / 2) * (y1 - y2), x2 - x1 + 0.5)
        else:
            raise ValueError('something wrong!!')

        return angle

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        width = self.width

        xs, ys = np.meshgrid(np.arange(width), np.arange(width))
        xs = (xs.flatten() - (width - 1) / 2) - np.mod(ys.flatten(), 2) * 0.5
        ys = - (ys.flatten() - (width - 1) / 2) * np.sqrt(3) / 2

        if cells is not None:
            cell_prepared = np.asarray(cells).flatten()
            cell_prepared[~in_hexagon(xs, ys, width)] = np.nan

            return xs, ys, cell_prepared
        else:
            return xs, ys


def sample_data(position, states_mat, s_size):
    # makes one-hot encoding of sensory at each time-step
    time_steps = np.shape(position)[0]
    sense_data = np.zeros((s_size, time_steps))
    for i, pos in enumerate(position):
        ind = int(pos)
        sense_data[states_mat[ind], i] = 1
    return sense_data


def square2hex(a):
    # length must be odd
    n_states = len(a)
    length = int(np.sqrt(len(a)))
    hex_length = (length + 1) / 2

    middle = int((n_states - 1) / 2)
    init = np.zeros(n_states)
    init[middle] = 1

    n_hops = int(hex_length - 1)
    jumps = [init]
    for i in range(n_hops):
        jumps.append(np.dot(a, jumps[i]))

    jumps_add = np.sum(jumps, 0)

    a_new = cp.deepcopy(a)
    for i, val in enumerate(list(jumps_add)):
        if val == 0:
            a_new[i, :] = 0
            a_new[:, i] = 0

    return a_new


def in_hexagon(x, y, width):
    # x, y, are centered about middle
    return np.abs(y) / np.sqrt(3) <= np.minimum(width / 4, width / 2 - np.abs(x))
