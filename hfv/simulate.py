"""闭环仿真: 深耦合模型 + 极端环境干扰注入 + 控制方法(论文第4节)。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controllers import ChannelGains
from .disturbances import harmonic_disturbance
from .model import IDX, HFVModel
from .params import DisturbanceConfig, SimConfig, VehicleParams
from .reference import commands

HARMONIC_FREQ = 6.0  # 已知频率谐波干扰 / (rad/s), 频率信息可由地面测试获得


def default_gains(vp: VehicleParams | None = None) -> list[ChannelGains]:
    """三通道默认增益(滚转/俯仰/偏航), 谐波内模频率取 HARMONIC_FREQ。"""
    return [
        ChannelGains(kp=9.0, kd=6.0, eso_bandwidth=12.0,
                     do_freq=HARMONIC_FREQ, do_gain1=14.0, do_gain2=-10.0, do_damping=3.0),
        ChannelGains(kp=12.0, kd=7.0, eso_bandwidth=12.0,
                     do_freq=HARMONIC_FREQ, do_gain1=16.0, do_gain2=-12.0, do_damping=3.0),
        ChannelGains(kp=10.0, kd=6.0, eso_bandwidth=12.0,
                     do_freq=HARMONIC_FREQ, do_gain1=14.0, do_gain2=-10.0, do_damping=3.0),
    ]


@dataclass
class SimResult:
    """单次仿真结果。"""

    t: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma_v: np.ndarray
    omega: np.ndarray
    height: np.ndarray
    velocity: np.ndarray
    delta: np.ndarray
    d_hat: np.ndarray
    d_true: np.ndarray
    z2: np.ndarray
    alpha_cmd: np.ndarray
    diverged: bool
    completed: bool
    controller: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def tracking_rmse(self) -> float:
        n = len(self.alpha)
        if n == 0:
            return float("nan")
        return float(np.sqrt(np.mean((self.alpha - self.alpha_cmd) ** 2)))

    def steady_rmse(self, t0: float = 100.0, t1: float = 190.0) -> float:
        """指令无跳变时段内的攻角跟踪 RMSE, 反映对持续干扰的抑制精度。"""
        mask = (self.t >= t0) & (self.t < t1)
        if not mask.any():
            return float("nan")
        return float(np.sqrt(np.mean((self.alpha[mask] - self.alpha_cmd[mask]) ** 2)))


def simulate(controller, cfg: SimConfig | None = None,
             vp: VehicleParams | None = None,
             dist: DisturbanceConfig | None = None) -> SimResult:
    """运行一次下压段闭环仿真。

    判稳准则: 攻角或角速率超限、或状态出现非有限值视为失稳;
    高度降至 cfg.h_stop 以下视为完成下压段飞行。
    """
    cfg = cfg or SimConfig()
    vp = vp or VehicleParams()
    dist = dist or DisturbanceConfig()
    model = HFVModel(vp, dist, cfg.flex, cfg.dt)
    controller.reset()

    x = model.initial_state(cfg)
    n_steps = int(cfg.t_end / cfg.dt)
    rec = max(1, cfg.record_every)

    log: dict[str, list] = {k: [] for k in
                            ("t", "alpha", "beta", "gamma_v", "omega", "h", "V",
                             "delta", "d_hat", "d_true", "z2", "alpha_cmd")}
    diverged = False
    completed = False
    delta = np.zeros(3)

    for k in range(n_steps):
        t = k * cfg.dt
        y = model.measurements(x)
        angles = np.array([x[IDX["gamma_v"]], x[IDX["alpha"]], x[IDX["beta"]]])
        refs = commands(t)
        p_nom = model.control_effectiveness(t, x)
        delta = controller.update(y, angles, refs, p_nom, cfg.dt)

        if k % rec == 0:
            log["t"].append(t)
            log["alpha"].append(x[IDX["alpha"]])
            log["beta"].append(x[IDX["beta"]])
            log["gamma_v"].append(x[IDX["gamma_v"]])
            log["omega"].append(x[3:6].copy())
            log["h"].append(x[IDX["h"]])
            log["V"].append(x[IDX["V"]])
            log["delta"].append(delta.copy())
            log["d_hat"].append(np.asarray(getattr(controller, "d_hat", np.zeros(3))).copy())
            log["d_true"].append(harmonic_disturbance(dist, t))
            log["z2"].append(np.array([o.z2 for o in getattr(controller, "eso", [])])
                             if hasattr(controller, "eso") else np.zeros(3))
            log["alpha_cmd"].append(refs[1])

        x = model.step(t, x, delta)

        if not np.all(np.isfinite(x)):
            diverged = True
            break
        if abs(x[IDX["alpha"]]) > cfg.alpha_limit or np.max(np.abs(x[3:6])) > cfg.rate_limit:
            diverged = True
            break
        if abs(x[IDX["beta"]]) > cfg.alpha_limit:
            diverged = True
            break
        if x[IDX["h"]] <= cfg.h_stop:
            completed = True
            break
        if x[IDX["V"]] <= 100.0:
            diverged = True
            break
    else:
        # 仿真时间窗内保持稳定且未提前退出, 视为完成设定飞行段。
        completed = True

    arr = {k: np.array(v) for k, v in log.items()}
    return SimResult(
        t=arr["t"], alpha=arr["alpha"], beta=arr["beta"], gamma_v=arr["gamma_v"],
        omega=arr["omega"], height=arr["h"], velocity=arr["V"], delta=arr["delta"],
        d_hat=arr["d_hat"], d_true=arr["d_true"], z2=arr["z2"],
        alpha_cmd=arr["alpha_cmd"], diverged=diverged, completed=completed,
        controller=getattr(controller, "name", type(controller).__name__),
    )
