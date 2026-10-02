"""下压段姿态指令生成(对应论文图2/图3的攻角指令剖面)。"""
from __future__ import annotations

import numpy as np


def attack_angle_command(t: float) -> float:
    """攻角指令 alpha_r: 平衡滑翔 -> 大攻角减速 -> 下压。"""
    if t < 200.0:
        return 0.30
    if t < 310.0:
        return 0.60
    return 0.05


def roll_command(t: float) -> float:
    """倾侧角指令(下压段置零, 转弯段给出常值倾侧)。"""
    if 120.0 <= t < 200.0:
        return 0.20
    return 0.0


def sideslip_command(t: float) -> float:
    """侧滑角指令恒为零(BTT 协调转弯)。"""
    return 0.0


def commands(t: float) -> np.ndarray:
    """返回 [gamma_v_r, alpha_r, beta_r], 顺序与控制器通道一致(滚转/俯仰/偏航)。"""
    return np.array([roll_command(t), attack_angle_command(t), sideslip_command(t)])
