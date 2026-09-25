# 开源复刻路线与证据边界

核查日期：2026-09-25。本文区分「上游有公开实现」「本仓库接好了接口」和「服务器上已经验证」；前两项不代表第三项。本文是源码与公开发布文件审查，不是已经完成的 CFD / GPU 实验结果。

## 1. 第一条主线怎么选

**先复刻 AeroTransformer 的表面预测，在 CRMpert 上检查误差，再用同源 ADflow 对少量完全匹配的案例复核。** 选择它是因为论文、代码、权重、几何和 CFD 标签同时有公开入口，不代表已经证明它是所有 MDO 方法中的 SOTA。

最小闭环：

```text
公开 CRMpert 几何 + AoA/Mach
             │
     ┌───────┴────────┐
     ▼                ▼
AeroTransformer   公开 CFD 标签 / 新跑 ADflow
     │                │
     └──── Cp、Cf、CL、CD 对照 ────┘
                      │
             明确计时范围与误差
                      │
       小范围参数搜索 → 真实 CFD 复核
```

论文来源：[AeroTransformer](https://arxiv.org/abs/2604.18062)、[SuperWing](https://arxiv.org/abs/2512.14397)。预训练模型在新数据上的直接推理只是复刻起点；论文报告的微调结果不能当作当前默认权重的预期实测值。

## 2. 候选代码实际提供什么

| 候选 | 核实到的内容 | 本阶段用法 |
|---|---|---|
| [AeroTransformer](https://github.com/tum-pbs/AeroTransformer) | 训练、评估、CRMpert/SuperWing 网格与 ADflow 调用脚本；模型主体在 floGen | 主要论文复刻入口 |
| [floGen](https://github.com/YangYunjia/floGen) | 模型定义、配置加载、表面积分及 Wing API | 按固定提交调用实现，不另写一个假同名模型 |
| [cfdpost](https://github.com/YangYunjia/cfdpost) | 表面网格、摩擦分量处理、力和力矩积分 | 保留原坐标、归一化、参考量约定 |
| [WebWing](https://github.com/YangYunjia/webwing) | Flask/Celery/Redis 的交互式预测工具，调用 floGen Wing API | UI 与参数化参考；不是现成完整 MDO 优化器 |
| [DAFoam CRM](https://github.com/DAFoam/tutorials/tree/8f3b5ff706bed8f6ec4d6a636540644554c24650/CRM_Wing) | OpenFOAM 网格处理、MPhys/OpenMDAO、FFD、伴随和受约束减阻脚本 | 后续替代 CFD 路线；首次验证不混入另一求解器的分布偏移 |
| [MACH-Aero](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/) | ADflow / pyGeo / IDWarp / pyOptSparse 等气动优化教程 | 安装与真实气动优化参考；该教程本身不是完整 TACS 气动结构案例 |
| [STW](https://mdobenchmarks.github.io/MDOAeroelasticBenchmark/) | 几何、气动/结构网格、规范、性能计算及参考结果 | 后续多学科基准；不是下载后立即运行的完整 UM 优化程序 |

DAFoam 的 CRM `runScript.py` 确有 `run_model`、`compute_totals`、`check_totals` 和 `run_driver` 入口；它优化阻力并约束升力、厚度、体积。当前默认优化器是 `Uno`，不要套用老教程里默认 SNOPT 的说法。预处理还要下载 CRM 表面网格并加载 OpenFOAM 环境。[固定源码](https://github.com/DAFoam/tutorials/blob/8f3b5ff706bed8f6ec4d6a636540644554c24650/CRM_Wing/runScript.py)

## 3. 可复现的源码快照

以下是审查时读取的提交，不表示已经通过 Linux CFD 的组合兼容性测试。安装应以本仓库 lock 文件为准；二者不一致时先解释并更新记录，不能默默用最新分支。

| 仓库 | Git commit | 许可核查 |
|---|---|---|
| tum-pbs/AeroTransformer | `3dc350ff69e354d1451eae468c686a368bd6151f` | Apache-2.0 |
| YangYunjia/floGen | `ff3abda23e10e1073c07ffd78dad96979e940c77` | MIT |
| YangYunjia/cfdpost | `c9fb313a4f3f3912a2e1a1aaf56e9e7a338bfa5a` | MIT |
| YangYunjia/cst-modeling3d | `9500b19a732463c26b32f6860ac6c517c7d212a5` | MIT |
| YangYunjia/webwing | `d8491437b60f23fe6f6be2de49b063a684c886ef` | MIT |
| DAFoam/tutorials | `8f3b5ff706bed8f6ec4d6a636540644554c24650` | 仓库未检出顶层 LICENSE；不当作 MIT 源码复制 |
| mdolab/MACH-Aero | `47545f536d41bd8075ee7e7dd750fb078edffe63` | 仓库未检出顶层 LICENSE；各底层软件另有许可 |
| mdobenchmarks/MDOAeroelasticBenchmark | `48e54b4dcc1c6c696ffc6625c01714f3c3c1244e` | 仓库未检出顶层 LICENSE；公开可读不等于可任意重新授权 |

本仓库的 MIT 许可只覆盖本项目自己新增的代码。外部依赖保留各自许可与作者信息；数据和权重另按发布方许可处理。未明确许可的参考仓库只提供上游链接和使用说明，不把上游内容重新标成 MIT。

## 4. 权重和数据确实可公开获取

审查通过 Hugging Face API 读取了文件清单、revision 和文件大小，没有在审查任务中下载大型权重或流场。

### 推荐默认权重

[thuerey-group/AeroTransformer](https://huggingface.co/thuerey-group/AeroTransformer)，固定 revision：

`698d70095a00d6f4a25f870e8de0772fc12b68f7`

| 文件夹 | 权重大小（bytes） | 内容 |
|---|---:|---|
| `ATsurf_S` | 4,331,444 | `model_config` + `best_model_weights` |
| `ATsurf_M` | 15,155,636 | 同上 |
| `ATsurf_L` | 58,218,804 | 同上 |

模型卡声明 MIT。这个发布仓库只有这三种表面模型，没有同名 CRM 微调实验的全部权重和训练划分。S 适合先排查安装，L 可作为服务器精度主实验的候选；大小差异不是精度实测结论。

### CRMpert 数据

[thuerey-group/CRMpert](https://huggingface.co/datasets/thuerey-group/CRMpert)，revision：

`88ece28b846fd1d9870933252db556cf97d30ae0`

模型任务数据包含 288 种几何、2,145 个流动样本，数据卡声明 CC-BY-SA-4.0。四个主要数组加元数据约 **2.14 GB（十进制）**：

| 文件 | bytes | 用途 |
|---|---:|---|
| `data.npy` | 1,686,896,768 | 参考表面上的三个流动通道 |
| `geom0.npy` | 226,492,544 | 每个几何的单元中心坐标 |
| `origingeom.npy` | 229,153,664 | 表面顶点，用于积分 |
| `index.npy` | 206,048 | 几何/工况映射、条件、参考量、系数 |
| `samples.parquet` | 742,044 | 带 CST、扭转、上反角等参数的元数据 |

只下载需要的白名单文件。不要直接全量下载 [SuperWing](https://huggingface.co/datasets/yunplus/SuperWing)：其当前完整发布约 3.49 TB，体积场远超本机剩余磁盘。第一轮并不需要重新预训练，也不需要体积数据。

### WebWing 使用的模型不是完全相同的发布入口

当前 floGen `SuperWingAPI` 默认读取 `yunplus/AeroTransformer` 的 `ATsurf_L_v1`；审查 revision 为 `2ef647c2183f1a40fd7f6afa7c511cd42fc229ba`。该发布还包含微调、集成与从零训练的变体，部分子目录没有独立 config。不要把它们和 `thuerey-group/ATsurf_L` 混用，也不要在未确认训练划分时声称其 CRM 测试样本独立。[固定 API 源码](https://github.com/YangYunjia/floGen/blob/ff3abda23e10e1073c07ffd78dad96979e940c77/flowvae/app/wing/api.py)

## 5. 接口必须真实表达物理量

直接模型输入为结构化表面坐标与 **`[AoA（度）, Mach]`**。AoA 不是目标升力；该权重也没有任意 Reynolds 数输入。要在固定升力下减阻，应在外层搜索 AoA 并检查预测/真实升力，不能把 `target_CL` 塞入 AoA 通道。

输出是三个表面通道：`Cp`、`150 * Cf_streamwise`、`300 * Cf_spanwise`，不是完整体积 CFD 状态。将三个通道还原并积分才得到 CL/CD/CM；不能只用 Cp 宣称总阻力，也不能把表面输出直接当 ADflow 体积 restart。单元中心数组为 `3 × 128 × 256`，顶点数组为 `3 × 129 × 257`。[数据定义](https://huggingface.co/datasets/yunplus/SuperWing)

源码审查发现以下需实测的适配点：

- 训练脚本写 `geom.npy`，发布数据叫 `geom0.npy`；路径可适配，但必须核对坐标处理，不能仅凭名字认为完全等价。
- floGen 的 `SuperWingAPI.predict` 会转动几何并调整 AoA，以消除基准根部扭转；`end2end_predict` 走另一条已重建几何路径。对公开已格式化数据不要重复转动。[处理代码](https://github.com/YangYunjia/floGen/blob/ff3abda23e10e1073c07ffd78dad96979e940c77/flowvae/post.py)
- CRMpert 卡片的 CM 描述写前缘参考点，而 `run-adflow.py` 写 `xRef=0.25`。必须保存实际参考面积、弦长和力矩参考点；CL/CD 可先对照，CM 不应在参考点未确认时混报。
- 同一 `index.npy` 同时有原 CFD 网格系数和参考表面积分系数。模型先与对应的参考积分标签比较；再单独列出对原求解器系数的总误差，把重采样/积分误差留在报告里。

统一 schema 允许某个 backend 的字段缺失并注明原因，比填零冒充支持更可靠。FEM 所需的节点载荷还需要压力/摩擦到力、面积、动压和保守映射；有 CL/CD 并不等于已实现气动结构耦合。

## 6. 两天里哪些结论成立

| 实际完成 | 可以说 | 不能说 |
|---|---|---|
| 软件接口与合成/缓存结果跑通 | 本地流程 smoke test 通过 | AI 比 CFD 快、物理精度已验证 |
| 真权重 + 公开 CFD 标签 | 该权重在指定数据上的误差与推理耗时 | 已经实测当前服务器 CFD 加速比 |
| 同几何、同工况的新跑 CFD | 指定案例上的误差与在线耗时比 | 完整 MDO 的总成本已降低 |
| 小型搜索 + CFD 复核最终设计 | 当前约束和范围内候选设计是否真实改善 | 任意机翼/结构/工况均有效、已达到 SOTA |
| 完整对照 + 多起点 + 全成本 | 在明确实验范围内的端到端优势 | 不同算力/不同约束下可直接沿用同一倍率 |

第一轮可用低维有界搜索绕开尚未验证的梯度接口。自动微分能给出代理模型自身的导数，不保证它与 CFD 导数一致；以后进入梯度优化前，要检查多个方向上的有限差分、可行性与 CFD 复核。接上不同 forward backend 时，不能默认保留原 CFD 伴随就获得一致梯度。

## 7. 成本与数据划分

保存每次运行的：代码/模型/数据 revision，样本与几何 ID，条件、网格、求解器设置、收敛状态、设备、MPI 进程数、预处理/模型/后处理/CFD 耗时，以及模型冷启动和热调用两种延迟。GPU 计时需要同步；训练、建数据、修正、失败重试、最终验证都要单列。

CPU core-hours 和 GPU-hours分别报告，不能直接相加变成没有定义的「等效 CPU 时间」。端到端优先报告相同服务器与调度条件下的墙钟时间；若要汇总货币或能耗成本，应明确单价或实测功率。并行任务的墙钟时间不等于各任务耗时之和。

微调时按**几何**划分训练/验证/测试，并保存清单；同一几何不同 AoA 随机拆分不足以证明新几何泛化。先用默认预训练权重评估，是清楚且容易审计的起点。之后才比较 frozen、线性/MLP head、轻量微调、全量微调及 probing。在取得公平基线前，不预写加速倍率或论文结论。

## 8. 研究计划中需要保留的边界

第一阶段交付为气动预测/验证工具。完整 MDO 还需结构、载荷传递、耦合收敛、约束、总导数与优化对照。STW 是后续问题定义，不是当前 demo 的名字；MACH-Aero 是气动软件框架，不能因名称就当作已经有完整 CFD–TACS 循环。

论文价值取决于：少量新 CFD 数据是否使 probing/适配比强基线更划算，以及优化器选出的设计能否在真实 CFD 下保持改善。模型直接推理复刻、漂亮压力图和模块接口是工程基础，不单独构成这一结论。
