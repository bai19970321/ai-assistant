"""干扰在线预示的核心环节: 干扰观测器与扩张状态观测器(论文第2节, 式(6)(7))。

论文为俯仰通道给出两类观测器:
  * 式(6) 谐波干扰观测器(DO): 针对"部分信息已知"的干扰(频率 Omega 由地面测试
    获得), 以谐波内模对其精细估计, 得到 d_hat;
  * 式(7) 扩张状态观测器(ESO): 把"变化率有界"的集总干扰作为扩张状态 z2 估计,
    并在 p*(delta + d_hat) 项中前馈补偿 d_hat。

二者在闭环中相互耦合, 本质上构成一个联合观测器。若把谐波内模直接并入扩张状态
观测器, 即得到数值稳定、可极点配置的"广义扩张状态观测器(GESO)", 其子块分别对应
式(6)的 d_hat 与式(7)的 z2。本模块同时提供:
  * HarmonicDisturbanceObserver —— 式(6) 的直接实现(含阻尼泄漏, 便于单独使用/教学);
  * ExtendedStateObserver       —— 式(7) 的二阶实现(ADRC 使用, 集总单一同质干扰);
  * GeneralizedESO              —— 式(6)+(7) 联合实现(EADC 使用, 分离谐波与集总干扰)。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import place_poles


class HarmonicDisturbanceObserver:
    """式(6) 谐波干扰观测器(带阻尼泄漏的内模观测器)。

        mu1' = -lambda * mu1 + Omega * mu2 + g1 * e
        mu2' = -Omega * mu1 - lambda * mu2 + g2 * e
        d_hat = mu1

    e 为量测残差(y - z1)。阻尼 lambda 使齐次矩阵特征值 -lambda +- j*Omega 位于
    左半平面, lambda -> 0 时退化为式(6)的纯内模。
    """

    def __init__(self, omega: float, gain1: float, gain2: float,
                 damping: float = 0.0) -> None:
        self.omega = float(omega)
        self.gain1 = float(gain1)
        self.gain2 = float(gain2)
        self.damping = float(damping)
        self.mu = np.zeros(2)

    @property
    def d_hat(self) -> float:
        return float(self.mu[0])

    def derivative(self, mu: np.ndarray, residual: float) -> np.ndarray:
        lam = self.damping
        return np.array([
            -lam * mu[0] + self.omega * mu[1] + self.gain1 * residual,
            -self.omega * mu[0] - lam * mu[1] + self.gain2 * residual,
        ])

    def update(self, residual: float, dt: float) -> float:
        k1 = self.derivative(self.mu, residual)
        k2 = self.derivative(self.mu + dt / 2 * k1, residual)
        k3 = self.derivative(self.mu + dt / 2 * k2, residual)
        k4 = self.derivative(self.mu + dt * k3, residual)
        self.mu = self.mu + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.d_hat

    def reset(self) -> None:
        self.mu = np.zeros(2)


class ExtendedStateObserver:
    """式(7) 二阶扩张状态观测器(ADRC 核心)。

        z1' = z2 + 2*sigma*(y - z1) + p*(delta + d_hat)
        z2' = sigma^2*(y - z1)

    z1 估计量测 y, z2 估计"变化率有界"的集总扩张状态。d_hat 缺省为 0, 即 ADRC
    把所有干扰(含谐波)视作单一同质集总干扰 —— 带宽受限, 难以跟踪快变谐波。
    """

    def __init__(self, bandwidth: float, p_nominal: float) -> None:
        self.sigma = float(bandwidth)
        self.p = float(p_nominal)
        self.z = np.zeros(2)

    @property
    def z1(self) -> float:
        return float(self.z[0])

    @property
    def z2(self) -> float:
        return float(self.z[1])

    def derivative(self, z: np.ndarray, y: float, u_eff: float) -> np.ndarray:
        e = y - z[0]
        return np.array([
            z[1] + 2.0 * self.sigma * e + self.p * u_eff,
            self.sigma ** 2 * e,
        ])

    def update(self, y: float, delta: float, d_hat: float, dt: float,
               p_nominal: float | None = None) -> None:
        if p_nominal is not None:
            self.p = float(p_nominal)
        u_eff = delta + d_hat
        k1 = self.derivative(self.z, y, u_eff)
        k2 = self.derivative(self.z + dt / 2 * k1, y, u_eff)
        k3 = self.derivative(self.z + dt / 2 * k2, y, u_eff)
        k4 = self.derivative(self.z + dt * k3, y, u_eff)
        self.z = self.z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def initialize(self, y: float) -> None:
        self.z = np.array([y, 0.0])

    def reset(self) -> None:
        self.z = np.zeros(2)


class GeneralizedESO:
    """广义扩张状态观测器: 式(6)谐波内模 + 式(7)集总扩张状态的联合稳定实现。

    被控通道(角加速度回路)为  y' = p*(delta) + w + d,
      * w  —— "变化率有界"的集总慢干扰(扩张状态, 对应式(7)的 z2);
      * d  —— 频率 Omega 已知的谐波干扰(对应式(6)的 d_hat), 内模
              s1' = Omega*s2, s2' = -Omega*s1, d = s1。

    观测器状态  xhat = [z1, w_hat, s1_hat, s2_hat], 量测 y = z1:

        z1'   = w_hat + s1_hat + p*delta + L1*(y - z1)
        w_hat'= L2*(y - z1)
        s1_hat'= Omega*s2_hat + L3*(y - z1)
        s2_hat'= -Omega*s1_hat + L4*(y - z1)

    增益 L 由极点配置保证误差动态 (A - L C) 稳定, 极点半径取 ~bandwidth。
    当 n_harm=0 时退化为式(7)的二阶 ESO(即 ADRC)。
    """

    def __init__(self, bandwidth: float, omega: float = 0.0,
                 n_harm: int = 1) -> None:
        self.sigma = float(bandwidth)
        self.omega = float(omega)
        self.n_harm = int(n_harm)
        self.n = 2 + 2 * self.n_harm
        A = np.zeros((self.n, self.n))
        A[0, 1] = 1.0                       # z1' <- w
        if self.n_harm:
            A[0, 2] = 1.0                   # z1' <- s1 (= d)
            A[2, 3] = self.omega            # s1' <- Omega*s2
            A[3, 2] = -self.omega           # s2' <- -Omega*s1
        C = np.zeros((1, self.n))
        C[0, 0] = 1.0
        self.A, self.C = A, C
        self.L = self._design_gains()
        self.x = np.zeros(self.n)

    def _design_gains(self) -> np.ndarray:
        # 期望极点: 以 sigma 为半径的一组相异实/复极点, 保证可配置且稳定
        wo = self.sigma
        if self.n_harm:
            poles = np.array([
                -1.0 * wo, -1.6 * wo,
                complex(-1.2 * wo, 0.4 * wo), complex(-1.2 * wo, -0.4 * wo),
            ])
        else:
            poles = np.array([-1.0 * wo, -1.8 * wo])
        res = place_poles(self.A.T, self.C.T, poles)
        return res.gain_matrix.flatten()

    @property
    def z1(self) -> float:
        return float(self.x[0])

    @property
    def z2(self) -> float:
        return float(self.x[1])

    @property
    def d_hat(self) -> float:
        return float(self.x[2]) if self.n_harm else 0.0

    def derivative(self, x: np.ndarray, y: float, p_delta: float) -> np.ndarray:
        e = y - x[0]
        dx = self.A @ x + self.L * e
        dx[0] += p_delta
        return dx

    def update(self, y: float, delta: float, p_nominal: float, dt: float) -> None:
        pd = p_nominal * delta
        k1 = self.derivative(self.x, y, pd)
        k2 = self.derivative(self.x + dt / 2 * k1, y, pd)
        k3 = self.derivative(self.x + dt / 2 * k2, y, pd)
        k4 = self.derivative(self.x + dt * k3, y, pd)
        self.x = self.x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def initialize(self, y: float) -> None:
        self.x = np.zeros(self.n)
        self.x[0] = y

    def reset(self) -> None:
        self.x = np.zeros(self.n)
