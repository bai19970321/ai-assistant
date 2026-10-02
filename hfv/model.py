"""高超声速飞行器姿态/轨迹/干扰深耦合系统模型(式(1)~(5))。

状态向量 x = [alpha, beta, gamma_v, omega_l, omega_m, omega_n,
              eta1, eta2, eta3, deta1, deta2, deta3, V, theta, h]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .disturbances import (
    GustProcess,
    fault_factors,
    harmonic_disturbance,
    mass_properties,
)
from .params import DisturbanceConfig, FlexParams, VehicleParams

G0 = 9.80665
RHO0 = 1.225
H_SCALE = 7200.0

IDX = {
    "alpha": 0, "beta": 1, "gamma_v": 2,
    "omega_l": 3, "omega_m": 4, "omega_n": 5,
    "eta": slice(6, 9), "deta": slice(9, 12),
    "V": 12, "theta": 13, "h": 14,
}
N_STATE = 15


def air_density(h: float) -> float:
    return RHO0 * math.exp(-max(h, 0.0) / H_SCALE)


@dataclass
class HFVState:
    """便于读取的状态封装。"""

    x: np.ndarray

    @property
    def alpha(self) -> float:
        return float(self.x[IDX["alpha"]])

    @property
    def beta(self) -> float:
        return float(self.x[IDX["beta"]])

    @property
    def gamma_v(self) -> float:
        return float(self.x[IDX["gamma_v"]])

    @property
    def omega(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def eta(self) -> np.ndarray:
        return self.x[IDX["eta"]]

    @property
    def V(self) -> float:
        return float(self.x[IDX["V"]])

    @property
    def theta(self) -> float:
        return float(self.x[IDX["theta"]])

    @property
    def h(self) -> float:
        return float(self.x[IDX["h"]])


class HFVModel:
    """深耦合系统模型: 姿态(式(1))、角运动(式(3))、力矩(式(4)(5))与弹性模态。

    干扰以加性、乘性和与状态深耦合的隐性三种形式注入。
    """

    def __init__(self, vp: VehicleParams, dist: DisturbanceConfig,
                 flex: FlexParams, dt: float) -> None:
        self.vp = vp
        self.dist = dist
        self.flex = flex
        self.dt = dt
        self.gust = GustProcess(dist, dt)
        self._gust_moment = np.zeros(3)

    # ---------- 辅助量 ----------
    def initial_state(self, cfg) -> np.ndarray:
        x = np.zeros(N_STATE)
        x[IDX["alpha"]] = cfg.alpha0
        x[IDX["beta"]] = cfg.beta0
        x[IDX["gamma_v"]] = cfg.gamma_v0
        x[3:6] = np.asarray(cfg.omega0, dtype=float)
        x[IDX["eta"]] = np.asarray(self.flex.eta0, dtype=float)
        x[IDX["deta"]] = np.asarray(self.flex.deta0, dtype=float)
        x[IDX["V"]] = cfg.V0
        x[IDX["theta"]] = cfg.theta0
        x[IDX["h"]] = cfg.h0
        return x

    def mass_inertia(self, t: float):
        return mass_properties(self.dist, t, self.vp.m0,
                               (self.vp.Il0, self.vp.Im0, self.vp.In0))

    def dyn_pressure(self, x: np.ndarray) -> float:
        return 0.5 * air_density(x[IDX["h"]]) * x[IDX["V"]] ** 2

    def control_effectiveness(self, t: float, x: np.ndarray) -> np.ndarray:
        """标称舵效 p_i = q S len c^{delta} / I_i (式(8)中的 p_m)。

        仅使用标称系数与标称质量特性, 不含失效因子与拉偏 —— 这是控制器可用的信息。
        """
        q = self.dyn_pressure(x)
        vp = self.vp
        return np.array([
            q * vp.S * vp.b * vp.cl_delta / vp.Il0,
            q * vp.S * vp.l * vp.cm_delta / vp.Im0,
            q * vp.S * vp.b * vp.cn_delta / vp.In0,
        ])

    def measurements(self, x: np.ndarray) -> np.ndarray:
        """式(6)~(8)中的量测 y_i, 即三个姿态角的变化率的运动学部分。

        y_m = omega_m - tan(beta)(omega_l cos(alpha) - omega_n sin(alpha))
        y_n = omega_l sin(alpha) + omega_n cos(alpha)
        y_l = sec(beta)(omega_l cos(alpha) - omega_n sin(alpha))
        """
        a, b = x[IDX["alpha"]], x[IDX["beta"]]
        wl, wm, wn = x[3], x[4], x[5]
        proj = wl * math.cos(a) - wn * math.sin(a)
        y_l = proj / math.cos(b)
        y_m = wm - math.tan(b) * proj
        y_n = wl * math.sin(a) + wn * math.cos(a)
        return np.array([y_l, y_m, y_n])

    # ---------- 气动力/力矩 ----------
    def forces(self, t: float, x: np.ndarray):
        vp, d = self.vp, self.dist
        q = self.dyn_pressure(x)
        kf = 1.0 + d.coef_bias_force
        a, b = x[IDX["alpha"]], x[IDX["beta"]]
        L = q * vp.S * kf * (vp.CL0 + vp.CLa * a)
        D = q * vp.S * kf * (vp.CD0 + vp.CDa2 * a * a)
        Z = q * vp.S * kf * vp.Czb * b
        return L, D, Z

    def moments(self, t: float, x: np.ndarray, delta: np.ndarray):
        """式(4)(5): 含舵面失效因子(乘性)、系数拉偏、弹性耦合与紊流/变质心加性干扰。"""
        vp, d = self.vp, self.dist
        q = self.dyn_pressure(x)
        a, b = x[IDX["alpha"]], x[IDX["beta"]]
        wl, wm, wn = x[3], x[4], x[5]
        eta = x[IDX["eta"]]
        lam = fault_factors(d, t)
        km = 1.0 + d.coef_bias_moment
        kd = 1.0 + d.coef_bias_delta
        V = max(x[IDX["V"]], 1.0)

        cl = (vp.cl0 + km * vp.cl_beta * b + kd * vp.cl_delta * lam[0] * delta[0]
              + float(np.dot(vp.cl_eta, eta)) + vp.cl_p * wl * vp.b / (2 * V))
        cm = (vp.cm0 + km * vp.cm_alpha * a + kd * vp.cm_delta * lam[1] * delta[1]
              + float(np.dot(vp.cm_eta, eta)) + vp.cm_q * wm * vp.l / (2 * V))
        cn = (vp.cn0 + km * vp.cn_beta * b + kd * vp.cn_delta * lam[2] * delta[2]
              + float(np.dot(vp.cn_eta, eta)) + vp.cn_r * wn * vp.b / (2 * V))

        Ml = 0.5 * cl * air_density(x[IDX["h"]]) * x[IDX["V"]] ** 2 * vp.S * vp.b
        Mm = 0.5 * cm * air_density(x[IDX["h"]]) * x[IDX["V"]] ** 2 * vp.S * vp.l
        Mn = 0.5 * cn * air_density(x[IDX["h"]]) * x[IDX["V"]] ** 2 * vp.S * vp.b

        # 变质心偏置 + 紊流等效力矩 + 谐波干扰(弹性振动)
        _, (Il, Im, In) = self.mass_inertia(t)
        cg = np.asarray(d.cg_shift, dtype=float) * q * vp.S
        gust = self._gust_moment * q * vp.S * vp.l
        # d_i 按式(7)(8)定义为"等效输入干扰", 因而实际力矩为
        # q*S*length*c_delta*lambda*d_i，使角加速度通道呈 p_i*(delta_i+d_i)。
        d_eq = harmonic_disturbance(d, t)
        harm = q * vp.S * np.array([
            vp.b * vp.cl_delta * lam[0] * kd * d_eq[0],
            vp.l * vp.cm_delta * lam[1] * kd * d_eq[1],
            vp.b * vp.cn_delta * lam[2] * kd * d_eq[2],
        ])
        return (np.array([Ml, Mm, Mn]) + cg + gust + harm), (Il, Im, In)

    # ---------- 微分方程 ----------
    def derivative(self, t: float, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        m, _ = self.mass_inertia(t)
        a, b, gv = x[IDX["alpha"]], x[IDX["beta"]], x[IDX["gamma_v"]]
        wl, wm, wn = x[3], x[4], x[5]
        V = max(x[IDX["V"]], 1.0)
        th = x[IDX["theta"]]
        g = G0

        L, D, Z = self.forces(t, x)
        M, (Il, Im, In) = self.moments(t, x, delta)

        # 式(2): 姿态运动学中的气动/重力耦合项 r1, r2, r3
        cb = max(math.cos(b), 1e-3)
        r1 = (m * g * math.cos(th) * math.cos(gv) - L) / (m * V * cb)
        r2 = (Z + m * g * math.cos(th) * math.sin(gv)) / (m * V)
        r3 = (L * (math.tan(b) + math.tan(th) * math.sin(gv))
              + Z * math.tan(th) * math.cos(gv)) / (m * V) \
            - g * math.cos(th) * math.tan(b) * math.cos(gv) / V

        proj = wl * math.cos(a) - wn * math.sin(a)
        dx = np.zeros(N_STATE)
        # 式(1)
        dx[IDX["alpha"]] = wm - math.tan(b) * proj + r1
        dx[IDX["beta"]] = wl * math.sin(a) + wn * math.cos(a) + r2
        dx[IDX["gamma_v"]] = proj / cb + r3
        # 式(3)
        dx[3] = (M[0] - (Im - In) * wm * wn) / Il
        dx[4] = (M[1] - (In - Il) * wn * wl) / Im
        dx[5] = (M[2] - (Il - Im) * wl * wm) / In
        # 弹性模态
        eta, deta = x[IDX["eta"]], x[IDX["deta"]]
        q = self.dyn_pressure(x)
        for i in range(3):
            w_i, z_i, k_i = self.flex.omega[i], self.flex.zeta[i], self.flex.gain[i]
            excite = k_i * q * (delta[1] + a)
            dx[9 + i] = -2 * z_i * w_i * deta[i] - w_i ** 2 * eta[i] + excite
        dx[IDX["eta"]] = deta
        # 质心运动(下压段)
        dx[IDX["V"]] = (-D - m * g * math.sin(th)) / m
        dx[IDX["theta"]] = (L * math.cos(gv) - Z * math.sin(gv) - m * g * math.cos(th)) / (m * V)
        dx[IDX["h"]] = V * math.sin(th)
        return dx

    def step(self, t: float, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """四阶龙格库塔积分一步(紊流在步内保持不变)。"""
        self._gust_moment = self.gust.step()
        dt = self.dt
        k1 = self.derivative(t, x, delta)
        k2 = self.derivative(t + dt / 2, x + dt / 2 * k1, delta)
        k3 = self.derivative(t + dt / 2, x + dt / 2 * k2, delta)
        k4 = self.derivative(t + dt, x + dt * k3, delta)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
