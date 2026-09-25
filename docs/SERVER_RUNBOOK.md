# A6000：第一轮测试怎么跑

目标：先看到真模型效果和推理时间，再测真实 CFD，最后形成同一问题上的加速证据。

## 1. 安装与模型演示

机器：i9-14900、约 64 GB RAM、RTX A6000 48 GB。把仓库放在自己有写权限且有足够空间的目录。保留系统盘与 `/srv` 的余量，默认不要下载 SuperWing 全量数据。

```bash
git clone https://github.com/hongyi-lab/MDO.git
cd MDO
bash scripts/bootstrap_server.sh cuda
bash scripts/run_demo.sh cuda:0 ATsurf_S
```

默认在 `.venv` 安装 PyTorch 2.11.0 CUDA 12.8 wheel；驱动显示 CUDA 13.0 不妨碍运行受支持的较早 runtime。不会修改系统驱动。若没有 `venv`、Docker 或网络访问，请先解决系统环境，不要通过替换成假数据跳过。

上游版本由 `configs/upstream.lock.json` 固定。安装器遇到旧目录的本地修改或不同提交会停止，保留原文件。更改源码需明确记录，不要静默覆盖。

先查看 `results/doctor.json` 中 `torch.cuda_available`、GPU 名称、剩余磁盘，再打开生成的 `report.html`。

验收：模型 strict load 成功；8 个真实样本均返回有限的 Cp、Cf 与系数；JSON 记录模型 SHA256、数据样本 ID 与计时范围；图上能同时看到真值和预测。

如果小模型成功，再跑：

```bash
bash scripts/run_demo.sh cuda:0 ATsurf_L
```

两次结果使用不同目录。比较误差与端到端预测时间；大模型是否更准由结果决定。

## 2. 单独验证真实 CFD

```bash
bash scripts/bootstrap_cfd.sh
```

按 [CFD.md](CFD.md) 运行下载并校验过的 L3 教程机翼。先测试 8 个 MPI 进程，再视内存与耗时测试 16/24。i9 有性能核与能效核，不能把逻辑线程数直接等同于线性加速。

记录：成功收敛与否、总耗时、求解耗时、进程数、进程内存指标、网格 SHA、求解器/镜像版本。127 GB swap 不计入计划可用内存；若运行期间持续换页，缩小问题或降低并发。

该教程算例验证 CFD 能运行，**不是与模型相同的 CRMpert 几何**。不能把教程 CFD 时间除以 CRMpert 推理时间作为论文加速比。

## 3. 检查 AI 选出的候选是否靠谱

`evaluation.json` 列出每个样本的 Mach。使用实际存在的 Mach 和预先选定的最低升力要求：

```bash
source .venv/bin/activate
python -m mdo_demo screen \
  --evaluation results/<实际运行目录>/evaluation.json \
  --mach <该批样本实际Mach> --min-cl <实验规定的最低CL> \
  --output results/screening.json
```

它只在匹配 Mach 的有限候选中选最小预测阻力，再用公开 CFD 标签检查选中候选的升力、阻力和可行性。若样本工况没有足够可比候选，先用 `fetch_assets.py --sample-ids ...` 选定一组条件可比的数据。不要为了让结果好看而事后改变升力约束。

这个演示不等于连续形状优化，没有 FEM/厚度/体积约束；它可以暴露“模型觉得很好，真实标签不认可”的问题。

## 4. 加速结论的下一关

同一 CRMpert 机翼、AoA、Mach、Re、参考量与求解设置下重新运行 ADflow，且确认收敛，才能比较真实 CFD 在线耗时。当前公开数据的体积网格和完整网格生成参数不是一个现成可直接映射的下载包，具体缺口见 CFD.md。

成本分开记录：生成适配数据、训练/微调、模型加载、预热、预测、修正 CFD、失败重试、最终复核。CPU core-hours 与 GPU-hours分列；不要直接相加成“等效 CPU 小时”。初次离线数据或训练投入可能使少量查询暂时不划算。

## 5. 两天结束需要带回哪些文件

1. 模型运行目录中的 `evaluation.json`、`environment.json`、`report.html`、`case_*.npz`。
2. `results/environment.freeze.txt`。
3. CFD 请求 JSON、输出 JSON 和日志，以及固定镜像 digest。
4. 失败时的完整报错和失败输出记录。

拿到这些就能判断：预测误差在哪里、GPU 和 CPU 后处理谁耗时、真实 CFD 是否能跑、下一个阶段应该优先做模型适配还是同几何网格复核。
