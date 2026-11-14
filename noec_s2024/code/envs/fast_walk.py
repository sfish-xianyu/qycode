"""向量化的随机游走：语义与 Rectangle.walk / walk_cont 完全一致，只是把 32 个 env 一起推进。

原实现每个 env 单独跑 64 步 Python 循环，每步要做
  np.where(tran[cur] > 0)  →  找可用邻居
  move_straight_bias(...)  →  角度偏置地选一个
合计约 80 µs，32 env × 64 步 = 160 ms，占一个 epoch 的 36%。

离散游走**不是均匀随机**：config 里 bias_type='angle'、direc_bias=0.25、
angle_bias_change=0.4。每步先按「朝向最接近」挑一个邻居（撞墙则均匀重选），
朝向再加 U(-0.4,0.4) 抖动；然后以 75% 的概率整体改回均匀重选。

本模块按同样的规则做数组化实现，候选顺序也照原样（状态下标升序）：
    up (cur-W) < left (cur-1) < self (cur) < right (cur+1) < down (cur+W)
因为 np.where(tran[cur] > 0) 返回的就是升序下标，顺序决定平局时取哪个。
"""
import numpy as np

PI = np.pi
SLOT_ANGLE = np.array([PI / 2, PI, 10000.0, 0.0, -PI / 2])   # up, left, self, right, down


def _tables(env):
    """邻居表（缓存）：nb (n_states, max_deg) 用 -1 填充，deg (n_states,)。"""
    t = getattr(env, "_fw_tables", None)
    if t is not None:
        return t
    adj = env.adj
    n_states = adj.shape[0]
    max_deg = int((adj > 0).sum(1).max())
    nb = np.full((n_states, max_deg), -1, dtype=np.int64)
    deg = np.zeros(n_states, dtype=np.int64)
    for s in range(n_states):
        w = np.where(adj[s] > 0)[0]
        nb[s, :len(w)] = w
        deg[s] = len(w)
    env._fw_tables = (nb, deg)
    return nb, deg


def _categorical(p, valid, rng, rows):
    """按概率 p 采样下标；浮点误差导致落在所有累积和之外时，退回最后一个有效槽。"""
    cum = np.cumsum(p, axis=1)
    cum = cum / np.maximum(cum[:, -1:], 1e-12)
    u = rng.random((p.shape[0], 1))
    idx = np.argmax(u < cum, axis=1)
    bad = ~valid[rows, idx]
    if bad.any():
        last_valid = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
        idx = np.where(bad, last_valid, idx)
    return idx


def _slot_candidates(cur, W, n_states, stay_still):
    """按原实现的升序顺序构造 5 个候选槽位及其有效性。"""
    cand = np.stack([cur - W, cur - 1, cur, cur + 1, cur + W], axis=1)
    x = cur % W
    valid = np.stack([
        cur - W >= 0,
        x != 0,
        np.full(cur.shape, bool(stay_still)),
        x != (W - 1),
        cur + W < n_states,
    ], axis=1)
    return cand, valid


def walk_batch(envs, seq_len, rng=None):
    """离散偏置随机游走，语义同 Rectangle.walk(rand_begin=False, walk_mode='random')。

    返回 (pos (n,T) int64, direc (n, n_actions, T) float64, visited (n,T) float64)。
    动作顺序沿用 config 的 rels: down=0 up=1 left=2 right=3，停留 = 全零。
    """
    rng = np.random.default_rng() if rng is None else rng
    envs = list(envs)
    n = len(envs)
    T = int(seq_len)
    e0 = envs[0]
    W = int(e0.width)
    n_states = int(e0.n_states)
    n_act = int(e0.n_actions)
    stay = bool(getattr(e0.par.env, "stay_still", False))
    dbias = float(getattr(e0.par.env, "direc_bias", 0.0))
    achg = float(getattr(e0.par.env, "angle_bias_change", 0.0))
    use_angle = str(getattr(e0.par.env, "bias_type", "")) == "angle"

    rows = np.arange(n)
    cur = np.array([int(e.start_state) for e in envs], dtype=np.int64)
    theta = rng.uniform(-PI, PI, n)

    pos = np.zeros((n, T), dtype=np.int64)
    visited = np.zeros((n, T), dtype=np.float64)
    seen = np.zeros((n, n_states), dtype=bool)
    pos[:, 0] = cur
    seen[rows, cur] = True

    for t in range(1, T):
        cand, valid = _slot_candidates(cur, W, n_states, stay)
        if use_angle:
            # 每个候选相对当前朝向的角差（折到 [0, pi]）
            adiff = np.abs(SLOT_ANGLE[None, :] - theta[:, None])
            adiff = np.where(adiff < PI, adiff, np.abs(2 * PI - adiff))
            # 无效槽位（越界邻居）不能参与 argmin，否则会选到场地外
            adiff = np.where(valid, adiff, np.inf)
            angmin = adiff.min(axis=1)
            idx_follow = np.argmin(adiff, axis=1)          # 首个最小值，与原实现一致
            follow = angmin < (PI / 4)

            # 撞墙：在 4 个邻居里均匀重选（self 的角是 10000，权重 1e-6，可忽略）
            p_wall = (valid & (SLOT_ANGLE[None, :] < 100)).astype(np.float64)
            p_wall /= np.maximum(p_wall.sum(1, keepdims=True), 1e-12)
            idx_wall = _categorical(p_wall, valid, rng, rows)

            idx = np.where(follow, idx_follow, idx_wall)
            theta = np.where(follow, theta, SLOT_ANGLE[idx])
            newpos = cand[rows, idx]

            # 朝向抖动
            theta = theta + (rng.random(n) * 2 - 1) * achg
            theta = np.mod(theta + PI, 2 * PI) - PI
        else:
            newpos = None

        # 以 1-direc_bias 的概率整体改回均匀重选（tran 在可用邻居上是均匀的）
        if newpos is None or dbias < 1.0:
            p_u = valid.astype(np.float64)
            p_u /= np.maximum(p_u.sum(1, keepdims=True), 1e-12)
            idx_u = _categorical(p_u, valid, rng, rows)
            pos_u = cand[rows, idx_u]
            if newpos is None:
                newpos = pos_u
            else:
                redraw = rng.random(n) > dbias
                newpos = np.where(redraw, pos_u, newpos)

        cur = newpos
        pos[:, t] = cur
        visited[:, t] = seen[rows, cur]
        seen[rows, cur] = True

    # 位移 → 动作下标
    d = pos[:, 1:] - pos[:, :-1]
    idx = np.full(d.shape, 4, dtype=np.int64)          # 4 = 停留
    idx[d == 1] = 3                                    # right
    idx[d == -1] = 2                                   # left
    idx[d == W] = 0                                    # down
    idx[d == -W] = 1                                   # up

    direc = np.zeros((n, n_act, T), dtype=np.float64)
    tt = np.arange(T - 1)[None, :]
    moved = idx < n_act
    direc[rows[:, None], np.minimum(idx, n_act - 1), tt + 1] = moved.astype(np.float64)
    direc[:, 0, 0] = 1.0
    return pos, direc, visited


def walk_cont_batch(envs, seq_len, a, rng=None):
    """连续速度游走，语义同 Rectangle.walk_cont。

    返回 (pos (n,T,2) float32, vel (n,2,T) float32, visited (n,T) float32)。
    """
    rng = np.random.default_rng() if rng is None else rng
    envs = list(envs)
    n = len(envs)
    T = int(seq_len)
    lim = np.array([[e.width, e.height] for e in envs], dtype=np.float32)

    pos = np.empty((n, T, 2), dtype=np.float32)
    vel = np.zeros((n, 2, T), dtype=np.float32)
    xy = lim / 2.0
    pos[:, 0] = xy
    for t in range(1, T):
        d = (rng.random((n, 2)) * 2 - 1).astype(np.float32) * a
        new = xy + d
        ok = np.all((new >= 0) & (new < lim), axis=1)
        xy = np.where(ok[:, None], new, xy)
        vel[:, :, t] = np.where(ok[:, None], d, 0.0)
        pos[:, t] = xy
    visited = np.zeros((n, T), dtype=np.float32)
    return pos, vel, visited
