"""蒙特卡洛测试评估(论文第4节): 三轴舵面失效 50% + 烧蚀变质量 + 力/力矩系数 [-40%,40%] 拉偏。"""
from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np

from .controllers import ADRCController, EADCController
from .params import DisturbanceConfig, SimConfig, VehicleParams
from .simulate import HARMONIC_FREQ, SimResult, default_gains, simulate


def extreme_environment(rng: np.random.Generator, seed: int) -> DisturbanceConfig:
    """构造一次极端环境干扰样本。"""
    return DisturbanceConfig(
        fault_factor_l=0.5, fault_factor_m=0.5, fault_factor_n=0.5, fault_time=0.0,
        mass_loss_rate_per_200s=0.01,
        coef_bias_force=float(rng.uniform(-0.4, 0.4)),
        coef_bias_moment=float(rng.uniform(-0.4, 0.4)),
        coef_bias_delta=float(rng.uniform(-0.4, 0.4)),
        harmonic_freq=(HARMONIC_FREQ,) * 3,
        harmonic_amp=(0.12, 0.30, 0.15),
        harmonic_phase=tuple(rng.uniform(0, 2 * np.pi, 3)),
        gust_sigma=2.0e-3, gust_tau=1.0, gust_seed=seed,
        cg_shift=(2.0e-3, 6.0e-3, 2.0e-3),
    )


def make_controller(kind: str, vp: VehicleParams) -> EADCController:
    gains = default_gains(vp)
    if kind.upper() == "EADC":
        return EADCController(gains, vp.delta_max)
    if kind.upper() == "ADRC":
        return ADRCController(gains, vp.delta_max)
    raise ValueError(f"未知控制方法: {kind}")


@dataclass
class MonteCarloSummary:
    """单一控制方法的蒙特卡洛测试统计。"""

    controller: str
    runs: list[SimResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.runs if r.completed)

    @property
    def n_diverged(self) -> int:
        return sum(1 for r in self.runs if not r.completed)

    @property
    def pass_rate(self) -> float:
        return self.n_completed / self.n if self.n else float("nan")

    @property
    def rmse_completed(self) -> float:
        vals = [r.tracking_rmse for r in self.runs if r.completed]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def steady_rmse_completed(self) -> float:
        vals = [r.steady_rmse() for r in self.runs if r.completed]
        return float(np.mean(vals)) if vals else float("nan")

    def __str__(self) -> str:
        return (f"{self.controller}: {self.n_completed}/{self.n} 次完成下压段飞行, "
                f"{self.n_diverged} 次失稳, 通过率 {self.pass_rate*100:.0f}%, "
                f"攻角跟踪 RMSE {self.rmse_completed:.4f} rad, "
                f"稳态(100-190 s) RMSE {self.steady_rmse_completed:.4f} rad")


def _run_one(args: tuple[str, int, SimConfig, VehicleParams]) -> SimResult:
    kind, run_seed, cfg, vp = args
    dist = extreme_environment(np.random.default_rng(run_seed), run_seed)
    res = simulate(make_controller(kind, vp), cfg=cfg, vp=vp, dist=dist)
    res.meta = {"seed": run_seed, "coef_bias": (dist.coef_bias_force,
                                                dist.coef_bias_moment,
                                                dist.coef_bias_delta)}
    return res


def monte_carlo(kind: str, n_runs: int = 50, seed: int = 2023,
                cfg: SimConfig | None = None,
                vp: VehicleParams | None = None,
                verbose: bool = False, workers: int = 1) -> MonteCarloSummary:
    """对指定控制方法执行 n_runs 次极端环境蒙特卡洛仿真(第 i 次样本种子为 seed+i)。"""
    cfg = cfg or SimConfig()
    vp = vp or VehicleParams()
    summary = MonteCarloSummary(controller=kind.upper())
    jobs = [(kind, seed + i, cfg, vp) for i in range(n_runs)]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(_run_one, jobs)
    else:
        results = [_run_one(j) for j in jobs]
    for i, res in enumerate(results):
        summary.runs.append(res)
        if verbose:
            tag = "完成" if res.completed else "失稳"
            print(f"  [{kind.upper()}] run {i+1:02d}/{n_runs} {tag}", flush=True)
    return summary
