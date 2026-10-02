"""蒙特卡洛测试评估: 复现论文图2(EADC)与图3(ADRC)。"""
from __future__ import annotations

import argparse

from hfv import SimConfig, VehicleParams
from hfv.montecarlo import monte_carlo
from hfv.plotting import plot_monte_carlo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cfg, vp = SimConfig(), VehicleParams()
    for kind, fig in (("EADC", "figures/fig2_eadc.png"), ("ADRC", "figures/fig3_adrc.png")):
        s = monte_carlo(kind, n_runs=args.runs, seed=args.seed, cfg=cfg, vp=vp,
                        verbose=args.verbose, workers=args.workers)
        print(s)
        print("  figure ->", plot_monte_carlo(s, fig))
