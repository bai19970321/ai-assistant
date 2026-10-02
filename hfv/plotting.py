"""测试结果可视化(复现论文图2/图3以及干扰估计效果)。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .montecarlo import MonteCarloSummary
from .simulate import SimResult


def plot_monte_carlo(summary: MonteCarloSummary, path: str | Path,
                     title: str | None = None) -> Path:
    """绘制蒙特卡洛结果: (a) 飞行器高度 (b) 飞行器攻角, 对应论文图2/图3。"""
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    for r in summary.runs:
        axes[0].plot(r.t, r.height, lw=0.9)
        axes[1].plot(r.t, r.alpha, lw=0.9)
    axes[0].set_ylabel("height / m")
    axes[0].set_ylim(0, 8.5e4)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("angle of attack / rad")
    axes[1].set_xlabel("time / s")
    axes[1].set_ylim(-0.2, 1.1)
    axes[1].grid(alpha=0.3)
    head = title or f"Test results of {summary.controller}"
    fig.suptitle(f"{head}  (pass rate {summary.pass_rate*100:.0f}%, "
                 f"{summary.n_completed}/{summary.n})")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_disturbance_estimation(res: SimResult, path: str | Path,
                                channel: int = 1) -> Path:
    """绘制谐波干扰估计(式(6))与扩张状态估计(式(7))效果。"""
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.0), sharex=True)
    axes[0].plot(res.t, res.d_true[:, channel], label="true harmonic disturbance")
    axes[0].plot(res.t, res.d_hat[:, channel], "--", label="DO estimate $\\hat d$")
    axes[0].set_ylabel("$d_m$")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(res.t, res.z2[:, channel], label="ESO lumped state $z_{m2}$")
    axes[1].set_ylabel("$z_{m2}$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[2].plot(res.t, res.alpha, label=r"$\alpha$")
    axes[2].plot(res.t, res.alpha_cmd, "--", label=r"$\alpha_r$")
    axes[2].set_ylabel("rad")
    axes[2].set_xlabel("time / s")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"{res.controller}: disturbance indication and tracking")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
