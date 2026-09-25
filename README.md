# CFD × Foundation Model validation workbench

先独立验证 **FM 能否在足够准确、可校准的前提下替代 CFD**。当前不搭建外层 MDO。

## 正式实验前：先按 v1 协议准备

[CFD × Foundation Model 实验协议 v1](docs/EXPERIMENT_PROTOCOL_V1.md) 已约定 Large/微调对照、数据来源、训练参数、置信区间、计时及推进条件。
配套 [固定清单](configs/protocol_v1_split.json) 按几何划分为 144 Train / 48 Dev / 48 Calibration / 48 Test，训练只用其中 450 个真实 CFD 工况。
已看过的 77 个机翼全部放入训练范围；旧 pilot 的测试结果作为探索记录。

**这是已冻结的实验计划，完整四分组运行器尚待接入。下方 `run_validation.sh` 仍运行旧 72 样本 pilot，不执行 v1。**

## 已可运行的 72 样本 pilot：真实适配 + 置信度

```bash
git clone https://github.com/hongyi-lab/MDO.git
cd MDO
bash scripts/bootstrap_server.sh cuda
bash scripts/run_validation.sh cuda:0
```

已有仓库先 `git pull --ff-only`。脚本下载固定版本的 72 个真实 CRMpert 样本（约 170 MB），
按几何分为 24 训练 / 24 校准 / 24 测试，冻结主干、训练原模型输出层，再比较原模型与适配模型。
配置在 `configs/cfd_validation.json`；固定训练轮数，测试集不用于调参或选择 checkpoint。

输出 `results/validation_<时间>/report.html` 可直接双击：
三维机翼压力同步旋转、CFD/原模型/适配模型对照、翼剖面曲线、独立测试误差、校准区间与回退决定。
JSON 和完整数组保存在同目录，可追溯每一项数字。

**首轮 CPU 实测：阻力平均误差 8.71 → 6.54 drag counts，降低约 24.9%。**
但校准区间半宽仍有 14.10 counts，超过事先设置的 5-count 容差，故 24 个测试样本全部要求 CFD。
这说明轻量适配改善了平均精度，**尚未达到保精度放行条件，也没有证明整体加速**。
名义覆盖率 90%，本次 24 个测试几何实测为 87.5%；不是可靠性认证。

这一步使用公开 CFD 标签，没有新跑 CFD。真实服务器求解入口与新定义的同几何开发案例见
[独立验证运行说明](docs/CFD_VALIDATION.md) 和 [匹配 CFD 案例](docs/MATCHED_CFD.md)。
不要将新案例的置信度直接沿用 CRMpert 的校准结果。
可复核数字见 [首轮 CPU 结果记录](docs/cfd_validation_cpu.json)。

第一条主线复刻 [AeroTransformer](https://arxiv.org/abs/2604.18062)：使用作者公开的 ATsurf 权重、[CRMpert](https://huggingface.co/datasets/thuerey-group/CRMpert) 表面数据，以及 floGen / cfdpost 的原始模型和力积分代码。源码、权重和数据均固定版本；不是另写一个同名网络。

## 现在能运行什么

| 命令 / 模块 | 做什么 | 结果含义 |
|---|---|---|
| `mdo-demo smoke` | 合成数据检查报告与接口 | 软件测试，无 AI / CFD 加速结论 |
| `scripts/fetch_assets.py` | 官方模型 + 8 个真实机翼样本 | 只取所需数据，默认约 25 MB，而非下载整套预训练体积场 |
| `mdo-demo evaluate` | 真权重预测 Cp、Cf、CL/CD/CM | 与公开 CFD 标签比较，生成 HTML / JSON / NPZ |
| `mdo-demo screen` | 固定 Mach 下筛选离散候选机翼 | 检查代理选出的候选在 CFD 标签下是否满足升力要求 |
| `scripts/run_cfd.py` | ADflow 真实 CPU 求解 | 记录收敛、系数、网格、MPI 数量、耗时与资源 |

已实现输出层 probing 和全参数微调入口、几何隔离校准、拒绝不可靠预测的服务接口。
完整 FEM 耦合、STW Case 4、节点力映射、优化梯度验证仍未实现；`screen` 只是较早的离散候选演示。

## 较早的 8 样本模型安装检查（可选）

Linux + Git + Python 3.12（推荐）；模型环境与 CFD Docker 环境分开。已有驱动不需要重装。

```bash
git clone https://github.com/hongyi-lab/MDO.git
cd MDO
bash scripts/bootstrap_server.sh cuda
bash scripts/run_demo.sh cuda:0 ATsurf_S
```

输出在 `results/<时间>_ATsurf_S/`：

- `report.html`：直接双击可查看的压力分布、预测/真值曲线、系数误差。
- `evaluation.json`：每个样本的误差、计时范围、模型和数据版本。
- `case_*.npz`：用于复核与重新绘图的数组。
- `environment.json`：服务器实际环境；安装版本另存 `results/environment.freeze.txt`。

这一步不需要 Docker。S 跑通后，运行 `bash scripts/run_demo.sh cuda:0 ATsurf_L` 检查大模型。**快多少要实测；报告不会拿读取 CFD 标签的速度当真实 CFD 时间。**

想在浏览器看服务器报告，可先把整个结果目录复制回本地。若使用 HTTP 服务，只绑定本机并通过 SSH 转发；不要把服务器项目根目录公开到互联网。

## 本地只搭流程

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m mdo_demo smoke --output results/local_smoke
```

输出目录必须为空，失败重跑请使用新目录，避免新旧数组与报告混合。

## 接着测真实 CFD

```bash
bash scripts/bootstrap_cfd.sh
```

然后按 [CFD 运行说明](docs/CFD.md) 启动公开 MACH-Aero L3 机翼算例。它用于验证 CPU 求解环境和资源开销；**这个机翼不是 CRMpert 中同一个机翼，不能用它的耗时直接宣称模型加速倍率。**

同几何、同工况的 CFD 复核需要匹配体积网格、参考量与系数定义。[两天测试顺序](docs/SERVER_RUNBOOK.md) 标出这一关和应收集的结果。

## 数据和方法约定

- 输入为表面几何与 `[AoA（度）, Mach]`；该权重不接受任意 Reynolds 数输入，也不接受 `target_CL` 替代 AoA。
- 模型输出为 `Cp, 150×Cf_stream, 300×Cf_span`，还原后调用上游表面积分。几何坐标为 x 流向、y 厚度、z 展向。
- 同时保留 CFD 求解器系数和重采样表面积分系数；默认模型误差对后者计算，二者不混用。
- 默认使用预训练权重和不同几何的便利子集；不是论文官方测试划分，不声称复现论文中的微调分数。
- 数据、权重和依赖下载到被 Git 忽略的目录，不上传到本仓库。CRMpert 数据保留原 CC-BY-SA-4.0 许可和来源；模型 MIT、AeroTransformer Apache-2.0、floGen/cfdpost MIT。

## 代码入口

```text
configs/upstream.lock.json  固定源码 / 模型 / 数据版本
scripts/                   下载、安装、服务器演示、真实 CFD
src/mdo_demo/
  dataset.py               数据映射、单位和原始样本 ID
  aerotransformer.py       原版模型加载、推理、力积分
  cfd.py                   ADflow、收敛判据和运行记录
  evaluation.py            精度与计时
  screening.py             离散候选筛选与标签复核
  report.py                本地 HTML demo
tests/                     数据、指标、失败状态与流程验证
docs/                      论文对应关系、服务器步骤和证据边界
```

研究依据与开源许可见 [复刻地图](docs/REPRODUCTION_MAP.md)、[模型资产说明](docs/MODEL_ASSETS.md) 和 [BibTeX](docs/references.bib)。验证状态见 [本地验证记录](docs/VALIDATION.md)。
