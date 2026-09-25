# MDO · AeroTransformer reproduction workbench

先把论文中的**真实模型 + 真实 CFD 数据**跑通，再判断是否值得进入 AI 辅助优化。

第一条主线复刻 [AeroTransformer](https://arxiv.org/abs/2604.18062)：使用作者公开的 ATsurf 权重、[CRMpert](https://huggingface.co/datasets/thuerey-group/CRMpert) 表面数据，以及 floGen / cfdpost 的原始模型和力积分代码。源码、权重和数据均固定版本；不是另写一个同名网络。

## 现在能运行什么

| 命令 / 模块 | 做什么 | 结果含义 |
|---|---|---|
| `mdo-demo smoke` | 合成数据检查报告与接口 | 软件测试，无 AI / CFD 加速结论 |
| `scripts/fetch_assets.py` | 官方模型 + 8 个真实机翼样本 | 只取所需数据，默认约 25 MB，而非下载整套预训练体积场 |
| `mdo-demo evaluate` | 真权重预测 Cp、Cf、CL/CD/CM | 与公开 CFD 标签比较，生成 HTML / JSON / NPZ |
| `mdo-demo screen` | 固定 Mach 下筛选离散候选机翼 | 检查代理选出的候选在 CFD 标签下是否满足升力要求 |
| `scripts/run_cfd.py` | ADflow 真实 CPU 求解 | 记录收敛、系数、网格、MPI 数量、耗时与资源 |

当前还没有实现完整 FEM 耦合、STW Case 4、经过验证的连续形状优化或新的 probing 算法。`screen` 是离散候选演示，不包装成完整 MDO。

## A6000 服务器：先跑这一组

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
