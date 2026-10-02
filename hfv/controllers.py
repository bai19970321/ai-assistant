"""强抗扰控制(EADC)与对比方法(ADRC/PID)(论文第3节, 式(6)~(8))。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observers import GeneralizedESO


@dataclass
class ChannelGains:
    """单通道增益: PD 增益、ESO 带宽与已知谐波频率。"""

    kp: float
    kd: float
    eso_bandwidth: float
    do_freq: float = 0.0
    do_gain1: float = 0.0       # 保留以对应论文式(6)参数 xi_1
    do_gain2: float = 0.0       # 保留以对应论文式(6)参数 xi_2
    do_damping: float = 0.0


class EADCController:
    """三通道强抗扰控制器(式(6)+(7)+(8))。

    数值实现采用广义 ESO 联合估计:
      * obs.z2    —— 未知慢变集总扩张状态;
      * obs.d_hat —— 已知频率谐波的角加速度估计。

    b0_scale: 控制器所用舵效 b0 = b0_scale * p_nominal。舵面失效 50% 叠加舵效系数
    [-40%,40%] 拉偏时真实舵效位于 [0.3, 0.7]*p_nominal, 取区间中值 0.6 作为 b0。

    对应式(8):
        delta = [-kp*(angle-angle_r) - kd*z1 - z2 - d_acc_hat] / p
              = [-kp*e - kd*z1 - z2]/p - d_input_hat
    """

    name = "EADC"

    def __init__(self, gains: list[ChannelGains], delta_max: float,
                 use_disturbance_observer: bool = True,
                 use_eso: bool = True,
                 b0_scale: float = 0.6) -> None:
        if len(gains) != 3:
            raise ValueError("需要给出滚转/俯仰/偏航三个通道的增益")
        self.gains = gains
        self.delta_max = delta_max
        self.use_do = use_disturbance_observer
        self.use_eso = use_eso
        self.b0_scale = b0_scale
        self.eso = [GeneralizedESO(g.eso_bandwidth, g.do_freq,
                                  n_harm=1 if use_disturbance_observer else 0)
                    for g in gains]
        self.delta = np.zeros(3)
        self.d_hat = np.zeros(3)  # 等效输入干扰估计, 与论文 d_hat 同量纲
        self._initialized = False

    def reset(self) -> None:
        for obs in self.eso:
            obs.reset()
        self.delta = np.zeros(3)
        self.d_hat = np.zeros(3)
        self._initialized = False

    def update(self, y: np.ndarray, angles: np.ndarray, refs: np.ndarray,
               p_nominal: np.ndarray, dt: float) -> np.ndarray:
        if not self._initialized:
            for i in range(3):
                self.eso[i].initialize(float(y[i]))
            self._initialized = True

        out = np.zeros(3)
        for i, g in enumerate(self.gains):
            p = self.b0_scale * float(p_nominal[i])
            if abs(p) < 1e-9:
                p = 1e-9 * np.sign(p) if p != 0 else 1e-9
            obs = self.eso[i]
            obs.update(float(y[i]), float(self.delta[i]), p, dt)
            z1 = obs.z1 if self.use_eso else float(y[i])
            z2 = obs.z2 if self.use_eso else 0.0
            d_acc = obs.d_hat if self.use_do else 0.0
            self.d_hat[i] = d_acc / p
            err = float(angles[i]) - float(refs[i])
            out[i] = (-g.kp * err - g.kd * z1 - z2 - d_acc) / p
        self.delta = np.clip(out, -self.delta_max, self.delta_max)
        return self.delta.copy()


class ADRCController(EADCController):
    """传统 ADRC: 关闭谐波内模, 所有干扰被视作单一同质集总扩张状态。"""

    name = "ADRC"

    def __init__(self, gains: list[ChannelGains], delta_max: float,
                 b0_scale: float = 0.6) -> None:
        super().__init__(gains, delta_max, use_disturbance_observer=False, use_eso=True,
                         b0_scale=b0_scale)


class PIDController:
    """PD/PID 基线控制器。"""

    name = "PID"

    def __init__(self, kp: np.ndarray, kd: np.ndarray, ki: np.ndarray,
                 delta_max: float) -> None:
        self.kp, self.kd, self.ki = np.asarray(kp), np.asarray(kd), np.asarray(ki)
        self.delta_max = delta_max
        self.integral = np.zeros(3)
        self.delta = np.zeros(3)
        self.d_hat = np.zeros(3)

    def reset(self) -> None:
        self.integral = np.zeros(3)
        self.delta = np.zeros(3)

    def update(self, y: np.ndarray, angles: np.ndarray, refs: np.ndarray,
               p_nominal: np.ndarray, dt: float) -> np.ndarray:
        err = np.asarray(angles, dtype=float) - np.asarray(refs, dtype=float)
        self.integral = np.clip(self.integral + err * dt, -0.5, 0.5)
        raw = -(self.kp * err + self.kd * np.asarray(y) + self.ki * self.integral)
        self.delta = np.clip(raw / np.sign(p_nominal + 1e-12) / np.maximum(np.abs(p_nominal), 1e-6),
                             -self.delta_max, self.delta_max)
        return self.delta.copy()
