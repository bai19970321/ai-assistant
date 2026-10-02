"""高超声速飞行器(HFV)标称模型参数与多源干扰配置。

对应论文: 郭雷等, 多源干扰下高超声速飞行器自主精细控制, 宇航学报 2023, 44(4).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VehicleParams:
    """滑翔类高超声速飞行器标称参数(式(1)~(5))。"""

    m0: float = 900.0          # 初始质量 / kg
    Il0: float = 150.0          # 滚转转动惯量 / kg m^2
    Im0: float = 800.0         # 俯仰转动惯量 / kg m^2
    In0: float = 900.0         # 偏航转动惯量 / kg m^2
    S: float = 1.50             # 参考面积 / m^2
    b: float = 1.00             # 翼展 / m
    l: float = 1.00             # 平均气动弦长 / m

    # 力系数: L = q S (CL0 + CLa * alpha), D = q S (CD0 + CDa2 * alpha^2),
    #          Z = q S Czb * beta
    CL0: float = 0.02
    CLa: float = 2.20
    CD0: float = 0.05
    CDa2: float = 1.20
    Czb: float = -1.20

    # 力矩系数 (式(5)): c = c0 + c_state * state + c_delta * lambda * delta + sum c_eta_i * eta_i
    cl0: float = 0.0
    cl_beta: float = -0.02
    cl_delta: float = 0.18
    cm0: float = 0.0
    cm_alpha: float = 0.015     # >0 -> 静不稳定
    cm_delta: float = -0.35
    cn0: float = 0.0
    cn_beta: float = 0.05
    cn_delta: float = -0.16
    # 弹性模态对三通道力矩的耦合系数 c^{eta_i}
    cl_eta: tuple[float, float, float] = (2.0e-3, 1.0e-3, 5.0e-4)
    cm_eta: tuple[float, float, float] = (6.0e-3, 3.0e-3, 1.5e-3)
    cn_eta: tuple[float, float, float] = (2.5e-3, 1.2e-3, 6.0e-4)

    # 气动阻尼(未建入式(5)的小量, 用于保证标称模型合理)
    cl_p: float = -0.05
    cm_q: float = -0.08
    cn_r: float = -0.06

    # 舵偏限幅 / rad
    delta_max: float = 0.35


@dataclass
class FlexParams:
    """弹性模态(广义坐标 eta_i)参数, 其振动构成与状态深耦合的谐波型干扰。"""

    omega: tuple[float, float, float] = (20.0, 48.0, 95.0)   # 模态频率 / (rad/s)
    zeta: tuple[float, float, float] = (0.01, 0.015, 0.02)   # 阻尼比
    gain: tuple[float, float, float] = (4.0e-4, 1.5e-4, 6.0e-5)  # 舵面/气动激励增益
    eta0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    deta0: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DisturbanceConfig:
    """极端环境多源干扰与不确定性配置。

    - fault_factor_*: 三轴舵面失效因子 lambda (式(5)), 1 表示健康, 0.5 表示失效 50%
    - mass_loss_rate_per_200s: 烧蚀/燃料消耗导致质量与转动惯量每 200 s 减少初值的比例
    - coef_bias_*: 力/力矩系数拉偏(乘性不确定), 取值范围 [-0.4, 0.4]
    - harmonic_*: 与弹性振动相关的谐波干扰 d (频率可由地面测试获得)
    - gust_*: Dryden 紊流/风切变等效外部干扰
    - cg_shift: 变质心引起的附加常值力矩偏置
    """

    fault_factor_l: float = 1.0
    fault_factor_m: float = 1.0
    fault_factor_n: float = 1.0
    fault_time: float = 0.0

    mass_loss_rate_per_200s: float = 0.0

    coef_bias_force: float = 0.0
    coef_bias_moment: float = 0.0
    coef_bias_delta: float = 0.0

    harmonic_freq: tuple[float, float, float] = (20.0, 20.0, 20.0)
    harmonic_amp: tuple[float, float, float] = (0.0, 0.0, 0.0)
    harmonic_phase: tuple[float, float, float] = (0.0, 0.0, 0.0)

    gust_sigma: float = 0.0
    gust_tau: float = 1.0
    gust_seed: int | None = None

    cg_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def with_updates(self, **kwargs) -> DisturbanceConfig:
        data = dict(self.__dict__)
        data.update(kwargs)
        return DisturbanceConfig(**data)


@dataclass
class SimConfig:
    """仿真与初始条件配置(下压段飞行)。"""

    dt: float = 0.005
    t_end: float = 500.0
    h0: float = 7.5e4
    V0: float = 6000.0
    theta0: float = -0.02
    alpha0: float = 0.25
    beta0: float = 0.0
    gamma_v0: float = 0.0
    omega0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    h_stop: float = 1.0e3        # 到达该高度视为完成下压段
    alpha_limit: float = 1.5     # 攻角超限视为失稳 / rad
    rate_limit: float = 3.0      # 角速率超限视为失稳 / (rad/s)
    record_every: int = 50
    flex: FlexParams = field(default_factory=FlexParams)
