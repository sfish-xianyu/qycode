import os
try:                       # 以子目录为 cwd 直接运行时也能导入
    from runtime_cfg import CFG
except ImportError:        # pragma: no cover
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runtime_cfg import CFG
import pdb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.environments import *
from envs.environments import *


### 下面是batch下的env初始化和数据生成
def init_env(pars, fix_width=None, fix_height=None, one_instance=False):
    batch_envs = []
    for b in range(pars.batch_size):
        if fix_width is None:
            wid, hei = pars.env.widths[b], pars.env.heights[b]
        else:
            wid, hei = fix_width, fix_height
        if pars.world_type == 'rectangle':
            env = Rectangle(pars, wid, hei)
        if pars.world_type == 'hexagonal':
            env = Hexagonal(pars, wid)
        env.world()
        env.state_data()
        env.walk_len = pars.seq_len  # 建议设置为>4*W*H的
        batch_envs.append(env)

        if one_instance:
            break
    return batch_envs

def gen_data(batch_envs, pars, seq_len=0, one_hot=True, shift=False, eval=False, toy_task=False, discrete_pos=True, one_instance=False):
    # WALK_VEL=cont：连续速度臂。速度 v ~ U(-a,a)^2，位置连续，
    # 与离散臂的唯一差别就是速度的分布（见 Rectangle.walk_cont）。
    cont_vel = CFG.walk_vel_cont
    # 每个 epoch 重新抽一次「共享起点 + 净位移」；WALK_MODE=aligned 时才生效。
    if not eval:
        reset_alignment()
    # 准备数据
    batch_pos = []
    batch_sense = []
    batch_dirs = []
    batch_visit = []
    batch_width = []

    # WALK_FAST=1：32 个 env 的游走一次性向量化推进（见 envs/fast_walk.py）。
    # 分布不变，只是随机数流不同；原路径完全保留，默认关闭。
    # deficit/aligned 两种模式共享全局状态（occ、start_state），不能走批量路径。
    fast = (CFG.walk_fast and not toy_task
            and CFG.walk_mode in ("", "random"))
    poss = dirs_all = viss = None
    if fast:
        from envs.fast_walk import walk_batch, walk_cont_batch
        envs_list = [batch_envs[0] if (eval or one_instance) else batch_envs[b]
                     for b in range(pars.batch_size)]
        T_len = int(seq_len) if seq_len else int(envs_list[0].walk_len)
        if cont_vel:
            _a = CFG.walk_vel_a
            poss, dirs_all, viss = walk_cont_batch(envs_list, T_len, _a)
        else:
            poss, dirs_all, viss = walk_batch(envs_list, T_len)

    for b in range(pars.batch_size):
        if eval or one_instance:
            env = batch_envs[0]
        else:
            env = batch_envs[b]
        if toy_task:
            pos, direc, visited = env.around_walk(seq_len=seq_len, return_visited=True)
        elif fast:
            pos, direc, visited = poss[b], dirs_all[b], viss[b]
        elif cont_vel:
            pos, direc, visited = env.walk_cont(seq_len=seq_len)
        else:
            # WALK_MODE=deficit 只用于训练：诊断/评估一律走原来的随机游走，
            # 这样新旧运行的 rate map 口径一致，才能和之前 10 个 run 对比。
            pos, direc, visited = env.walk(rand_begin=False, seq_len=seq_len,
                                           return_visited=True,
                                           walk_mode='random' if eval else None)

        if cont_vel:
            # sense 只为兼容下游形状（预测关了时它不参与编码）：取最近的格点做 one-hot。
            # 位置是连续的 (x, y)，要转回扁平下标 y*W + x 才能喂 sample_data。
            _ix = np.clip(np.floor(pos[:, 0]).astype(np.int64), 0, int(env.width) - 1)
            _iy = np.clip(np.floor(pos[:, 1]).astype(np.int64), 0, int(env.height) - 1)
            _flat = _iy * int(env.width) + _ix
            sense = sample_data(_flat, env.states_mat, s_size=pars.s_size, one_hot=one_hot)
            batch_pos.append(pos)
            batch_sense.append(sense)
            batch_dirs.append(direc)
            batch_visit.append(visited)
            batch_width.append(int(env.width))
            continue

        sense = sample_data(pos, env.states_mat, s_size=pars.s_size, one_hot=one_hot)
        if not one_hot:
            direc = np.where(direc.sum(0)==0, 0, direc.argmax(axis=0)+1)
        # batch_pos.append([pos//env.width, pos%env.width])
        if discrete_pos:
            batch_pos.append(pos)
        else:
            batch_pos.append([pos//env.width, pos%env.width])
        batch_sense.append(sense)
        batch_dirs.append(direc)
        batch_visit.append(visited)
        # 本样本所属环境的宽度：`pos` 是每个 env 自己的扁平状态下标，
        # 只有用它自己的宽度才能还原 (x, y)。训练 batch 的宽度表是混合的
        # （[11,10,10,11,8,9,...]），用单一宽度解码会把坐标算错。
        batch_width.append(int(env.width))
    if one_hot:
        data = {
            "pos": np.asarray(batch_pos),  # .transpose(0,2,1),  # B,S
            "sense": np.asarray(batch_sense).transpose(0,2,1),  # B,S,_
            "dirs": np.asarray(batch_dirs).transpose(0,2,1),
            "visit": np.asarray(batch_visit),
            "width": np.asarray(batch_width),  # B, 每个样本的环境宽度
        }
    else:
        data = {
            "pos": np.asarray(batch_pos),  # .transpose(0,2,1),  # B,S
            "sense": np.asarray(batch_sense),
            "dirs": np.asarray(batch_dirs),
            "visit": np.asarray(batch_visit),
            "width": np.asarray(batch_width),  # B, 每个样本的环境宽度
        }
    if shift:
        data["dirs"][:, :-1] = data["dirs"][:, 1:]  # act需要提前一个t以确保按照p,s,a->p,s,a->...的顺序进行
    if not discrete_pos:
        data["pos"] = data["pos"].transpose(0,2,1)
    
    return data

def sample_data(position, states_mat, s_size, one_hot=True):
    # makes one-hot encoding of sensory at each time-step
    time_steps = np.shape(position)[0]
    if one_hot:
        sense_data = np.zeros((s_size, time_steps))
    else:
        sense_data = np.zeros(time_steps)
    for i, pos in enumerate(position):
        ind = int(pos)
        if one_hot:
            sense_data[states_mat[ind], i] = 1
        else:
            sense_data[i] = states_mat[ind]
    return sense_data
