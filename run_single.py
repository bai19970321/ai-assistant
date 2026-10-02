"""单次极端环境仿真: 对比 EADC 与 ADRC 的干扰估计与跟踪性能。"""
from __future__ import annotations

import numpy as np

from hfv import SimConfig, VehicleParams, simulate
from hfv.montecarlo import extreme_environment, make_controller
from hfv.plotting import plot_disturbance_estimation

if __name__ == "__main__":
    cfg = SimConfig()
    vp = VehicleParams()
    dist = extreme_environment(np.random.default_rng(7), 7)
    for kind in ("EADC", "ADRC"):
        res = simulate(make_controller(kind, vp), cfg=cfg, vp=vp, dist=dist)
        print(f"{kind}: completed={res.completed} diverged={res.diverged} "
              f"t_end={res.t[-1]:.1f}s h_end={res.height[-1]:.0f}m "
              f"alpha RMSE={res.tracking_rmse:.4f} rad "
              f"steady RMSE(100-190s)={res.steady_rmse():.4f} rad")
        p = plot_disturbance_estimation(res, f"figures/{kind.lower()}_single.png")
        print(f"  figure -> {p}")
