# 多源干扰下高超声速飞行器自主精细控制 —— 核心技术代码实现

对应论文: 郭雷 等, 《多源干扰下高超声速飞行器自主精细控制》, 宇航学报, 2023, 44(4).

本项目用 Python 实现论文提出的增强抗扰控制 (EADC) 框架的核心环节, 并复现论文第 4 节的
极端环境蒙特卡洛对比(EADC vs. ADRC)。模型为面向控制设计的简化深耦合模型, 参数为示意值,
不对应任何真实飞行器。

## 论文内容与代码对应

| 论文内容 | 代码 |
| --- | --- |
| 式(1) 姿态运动学 (攻角/侧滑角/倾侧角) | `hfv/model.py` `HFVModel.derivative` |
| 式(2) 重力/升力/侧力耦合项 r1, r2, r3 | `hfv/model.py` `HFVModel.derivative` |
| 式(3) 三轴角速度动力学 | `hfv/model.py` `HFVModel.derivative` |
| 式(4)(5) 气动力矩、舵面失效因子 lambda、弹性模态耦合 c^eta | `hfv/model.py` `HFVModel.moments` |
| 弹性模态广义坐标 eta_i (二阶振动) | `hfv/model.py`, `hfv/params.py` `FlexParams` |
| 多源干扰: 舵面失效、系数拉偏、烧蚀变质量/惯量、变质心、紊流、谐波干扰 | `hfv/disturbances.py`, `hfv/params.py` `DisturbanceConfig` |
| 式(6) 谐波干扰观测器 (部分信息已知干扰) | `hfv/observers.py` `HarmonicDisturbanceObserver` |
| 式(7) 扩张状态观测器 (变化率有界的集总干扰) | `hfv/observers.py` `ExtendedStateObserver` |
| 式(6)+(7) 联合实现 (谐波内模 + 扩张状态, 极点配置) | `hfv/observers.py` `GeneralizedESO` |
| 式(8) EADC 复合控制律 (可退化为 ADRC / PD) | `hfv/controllers.py` `EADCController` |
| 传统 ADRC 对比方法 | `hfv/controllers.py` `ADRCController` |
| 第 4 节 蒙特卡洛测试评估 (图2/图3) | `hfv/montecarlo.py`, `run_monte_carlo.py` |

## 算法要点

被控通道(以俯仰通道为例)写成

    y' = p * (delta + d) + w

- `d`: 频率 Omega 已知(地面测试获得)的谐波干扰, 属于"部分信息已知"的干扰;
- `w`: 其余变化率有界的集总干扰(模型不确定、失效、变质量、紊流、通道耦合等);
- `p`: 舵效。

EADC 的联合观测器状态为 `[z1, w_hat, s1, s2]`, 其中 `s1' = Omega*s2, s2' = -Omega*s1`
为谐波内模, 增益由极点配置保证观测误差稳定。控制律(式(8)):

    delta = [-kp*(angle - angle_r) - kd*z1 - z2 - d_hat_acc] / b0

ADRC 关闭谐波内模, 所有干扰都由单一扩张状态 `z2` 估计, 对持续谐波只能做有限带宽的近似跟踪。

`b0 = b0_scale * p_nominal`: 舵面失效 50% 叠加舵效系数 [-40%, 40%] 拉偏时, 真实舵效位于
`[0.3, 0.7] * p_nominal`, 两种控制器都取区间中值 `b0_scale = 0.6`。

直接按式(6)实现的无阻尼谐波观测器与 ESO 分开运行时, 在模型失配下会出现闭环谐振发散,
因此 EADC 实际使用 `GeneralizedESO` 联合实现; `HarmonicDisturbanceObserver` 保留为式(6)
的独立实现(带阻尼泄漏项)。

## 极端环境 (每次蒙特卡洛样本)

- 三轴舵面失效 50%;
- 质量与转动惯量每 200 s 减少初值的 1%;
- 力、力矩、舵效系数在 [-40%, 40%] 内随机拉偏;
- 已知频率 (6 rad/s) 谐波干扰, 随机相位;
- 一阶 Dryden 紊流等效力矩、变质心附加力矩。

## 运行

```bash
pip install -r requirements.txt
python3 run_single.py                         # 单次极端环境 EADC/ADRC 对比 -> figures/*_single.png
python3 run_monte_carlo.py --runs 50          # 蒙特卡洛 -> figures/fig2_eadc.png, figures/fig3_adrc.png
python3 -m pytest tests -q
```

## 评价指标

- 通过率: 仿真时间内攻角/侧滑角/角速率未超限、状态有限即视为通过;
- 攻角跟踪 RMSE: 全程, 含指令跳变的过渡过程;
- 稳态 RMSE (100–190 s): 指令无跳变时段, 反映对持续干扰的抑制精度。

## 结果 (50 次蒙特卡洛, seed 2023-2072)

| 方法 | 通过率 | 攻角跟踪 RMSE (全程) | 稳态 RMSE (100–190 s) |
| --- | --- | --- | --- |
| EADC | 50/50 | 0.0352 rad | 0.0011 rad |
| ADRC | 50/50 | 0.0567 rad | 0.0126 rad |

单次极端环境 (seed 7): EADC 稳态 RMSE 0.0008 rad, ADRC 0.0109 rad; EADC 对谐波干扰的估计
与真值基本重合(`figures/eadc_single.png`), ADRC 的集总状态和攻角中残留明显振荡(`figures/adrc_single.png`)。

## 局限

- 模型与参数是示意性的, 结果只说明两种控制结构的相对差异, 不能作为工程指标;
- 在本模型参数下, 两种方法在 50% 失效 + 40% 拉偏下都能保持稳定, EADC 的优势体现在
  已知频率谐波干扰下的稳态精度; 论文图2/图3中 ADRC 的失稳案例依赖原文未公开的模型参数, 未做数值复现。
