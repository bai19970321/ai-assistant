import math

import numpy as np
import pytest

from hfv import (
    ADRCController,
    DisturbanceConfig,
    EADCController,
    ExtendedStateObserver,
    HarmonicDisturbanceObserver,
    HFVModel,
    SimConfig,
    VehicleParams,
    simulate,
)
from hfv.disturbances import harmonic_disturbance, mass_properties
from hfv.model import IDX
from hfv.montecarlo import extreme_environment, make_controller
from hfv.simulate import HARMONIC_FREQ, default_gains


def test_harmonic_observer_converges():
    omega, dt = 20.0, 1e-3
    do = HarmonicDisturbanceObserver(omega, 14.0, -10.0, damping=3.0)
    # 用 d - d_hat 作为残差模拟闭环中的量测残差注入
    amp, phase = 0.3, 0.7
    for k in range(int(20.0 / dt)):
        t = k * dt
        d = amp * math.sin(omega * t + phase)
        do.update(d - do.d_hat, dt)
    t = 20.0
    assert abs(do.d_hat - amp * math.sin(omega * t + phase)) < 0.12


def test_eso_tracks_output_and_lumped_disturbance():
    dt, sigma, p = 1e-3, 30.0, 2.0
    eso = ExtendedStateObserver(sigma, p)
    y, d = 0.0, 0.4
    delta = 0.0
    for _ in range(int(5.0 / dt)):
        eso.update(y, delta, 0.0, dt)
        y += dt * (p * delta + d)
    assert abs(eso.z1 - y) < 1e-2
    assert abs(eso.z2 - d) < 5e-2


def test_mass_and_inertia_decrease_with_ablation():
    cfg = DisturbanceConfig(mass_loss_rate_per_200s=0.01)
    m0, I0 = 1000.0, (200.0, 1200.0, 1300.0)
    m1, I1 = mass_properties(cfg, 200.0, m0, I0)
    assert m1 == pytest.approx(990.0)
    assert I1[1] == pytest.approx(1188.0)


def test_fault_factor_halves_control_moment():
    vp, flex = VehicleParams(), SimConfig().flex
    x = HFVModel(vp, DisturbanceConfig(), flex, 1e-3).initial_state(SimConfig())
    healthy = HFVModel(vp, DisturbanceConfig(), flex, 1e-3)
    faulty = HFVModel(vp, DisturbanceConfig(fault_factor_m=0.5), flex, 1e-3)
    delta = np.array([0.0, 0.1, 0.0])
    m_h, _ = healthy.moments(0.0, x, delta)
    m_f, _ = faulty.moments(0.0, x, delta)
    m_0, _ = healthy.moments(0.0, x, np.zeros(3))
    assert (m_f[1] - m_0[1]) == pytest.approx(0.5 * (m_h[1] - m_0[1]), rel=1e-9)


def test_measurements_match_kinematics():
    vp, flex = VehicleParams(), SimConfig().flex
    model = HFVModel(vp, DisturbanceConfig(), flex, 1e-3)
    x = model.initial_state(SimConfig())
    x[IDX["beta"]] = 0.05
    x[3:6] = np.array([0.02, -0.03, 0.01])
    y = model.measurements(x)
    dx = model.derivative(0.0, x, np.zeros(3))
    # y_i 为式(1)中去掉 r_i 的运动学部分
    b = x[IDX["beta"]]
    cb, g = math.cos(b), 9.80665
    L, _, _ = model.forces(0.0, x)
    m, _ = model.mass_inertia(0.0)
    V, th, gv = x[IDX["V"]], x[IDX["theta"]], x[IDX["gamma_v"]]
    r1 = (m * g * math.cos(th) * math.cos(gv) - L) / (m * V * cb)
    assert y[1] == pytest.approx(dx[IDX["alpha"]] - r1, rel=1e-9, abs=1e-12)


def test_eadc_degenerates_to_adrc_and_pd():
    vp = VehicleParams()
    adrc = ADRCController(default_gains(vp), vp.delta_max)
    assert adrc.use_do is False and adrc.use_eso is True
    pd = EADCController(default_gains(vp), vp.delta_max,
                        use_disturbance_observer=False, use_eso=False)
    out = pd.update(np.zeros(3), np.array([0.0, 0.5, 0.0]), np.zeros(3),
                    np.array([1.0, -1.0, -1.0]), 1e-3)
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) <= vp.delta_max + 1e-12)


def test_nominal_closed_loop_tracks_command():
    cfg = SimConfig(t_end=60.0, record_every=100)
    vp = VehicleParams()
    res = simulate(make_controller("EADC", vp), cfg=cfg, vp=vp,
                   dist=DisturbanceConfig())
    assert not res.diverged
    tail = res.alpha[int(0.8 * len(res.alpha)):]
    assert np.all(np.abs(tail - 0.30) < 0.05)


def test_eadc_estimates_harmonic_disturbance_in_closed_loop():
    cfg = SimConfig(t_end=40.0, record_every=50)
    vp = VehicleParams()
    dist = DisturbanceConfig(harmonic_amp=(0.0, 0.3, 0.0), harmonic_freq=(HARMONIC_FREQ,) * 3)
    res = simulate(make_controller("EADC", vp), cfg=cfg, vp=vp, dist=dist)
    mask = res.t > 20.0
    err = np.sqrt(np.mean((res.d_hat[mask, 1] - res.d_true[mask, 1]) ** 2))
    assert err < 0.5 * 0.3  # 估计误差显著小于干扰幅值


def test_extreme_environment_sample_ranges():
    dist = extreme_environment(np.random.default_rng(0), 0)
    assert dist.fault_factor_m == 0.5
    for bias in (dist.coef_bias_force, dist.coef_bias_moment, dist.coef_bias_delta):
        assert -0.4 <= bias <= 0.4
    assert np.any(harmonic_disturbance(dist, 0.3) != 0.0)


def test_eadc_rejects_known_harmonic_better_than_adrc():
    cfg = SimConfig(t_end=190.0, record_every=20)
    vp = VehicleParams()
    dist = extreme_environment(np.random.default_rng(2023), 2023)
    eadc = simulate(make_controller("EADC", vp), cfg=cfg, vp=vp, dist=dist)
    adrc = simulate(make_controller("ADRC", vp), cfg=cfg, vp=vp, dist=dist)
    assert eadc.completed and adrc.completed
    assert eadc.steady_rmse() < 0.5 * adrc.steady_rmse()
