"""多源干扰与不确定性表征(论文第1节)。

包括:
  * 器件失效/退化  -> 舵面失效因子 lambda_i 与加性故障
  * 烧蚀/燃料消耗  -> 质量、转动惯量时变, 变质心附加力矩
  * 气动外部干扰   -> Dryden 型一阶马尔可夫紊流等效力矩
  * 弹性振动       -> 已知频率的谐波干扰 d (部分信息已知)
"""
from __future__ import annotations

import math

import numpy as np

from .params import DisturbanceConfig


class GustProcess:
    """Dryden 紊流的一阶近似: tau*xdot = -x + sigma*sqrt(2*tau/dt)*w。"""

    def __init__(self, cfg: DisturbanceConfig, dt: float) -> None:
        self.sigma = cfg.gust_sigma
        self.tau = max(cfg.gust_tau, 1e-6)
        self.dt = dt
        self.rng = np.random.default_rng(cfg.gust_seed)
        self.state = np.zeros(3)

    def step(self) -> np.ndarray:
        if self.sigma <= 0.0:
            return self.state
        beta = math.exp(-self.dt / self.tau)
        noise = self.rng.standard_normal(3) * self.sigma * math.sqrt(1.0 - beta**2)
        self.state = beta * self.state + noise
        return self.state


def mass_properties(cfg: DisturbanceConfig, t: float, m0: float,
                    inertia0: tuple[float, float, float]):
    """烧蚀与燃料消耗引起的质量/转动惯量时变(每 200 s 均匀减少初值的给定比例)。"""
    frac = max(0.0, 1.0 - cfg.mass_loss_rate_per_200s * t / 200.0)
    frac = max(frac, 0.3)
    return m0 * frac, tuple(i * frac for i in inertia0)


def fault_factors(cfg: DisturbanceConfig, t: float) -> tuple[float, float, float]:
    """三轴舵面失效因子 lambda_l, lambda_m, lambda_n (式(5))。"""
    if t < cfg.fault_time:
        return 1.0, 1.0, 1.0
    return cfg.fault_factor_l, cfg.fault_factor_m, cfg.fault_factor_n


def harmonic_disturbance(cfg: DisturbanceConfig, t: float) -> np.ndarray:
    """与弹性振动相关的谐波干扰 d_i(t) = A_i*sin(Omega_i*t + phi_i)。

    其频率 Omega_i 由地面测试获得, 因此属于"部分信息已知的干扰",
    由干扰观测器(式(6))进行精细估计与前馈补偿。
    """
    return np.array([
        amp * math.sin(freq * t + phase)
        for amp, freq, phase in zip(cfg.harmonic_amp, cfg.harmonic_freq, cfg.harmonic_phase)
    ])
