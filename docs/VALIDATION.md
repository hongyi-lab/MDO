# 本地验证记录

日期：2026-09-25。以下为本地 Windows CPU 实测；**没有冒充 A6000 结果，也没有运行真实 ADflow 求解**。

## 真模型已跑通

- 官方 ATsurf_S，1,045,552 个参数，原模型定义和权重 `strict=True` 加载。
- Python 3.12.14、PyTorch 2.11.0+cpu，4 个 PyTorch CPU 线程。
- 固定版本 CRMpert 的 8 个真实机翼；原始样本 ID：`0,302,620,922,1233,1539,1842,2137`。
- 无本地训练或微调。这是不同几何的便利子集，不是官方测试划分。
- `pip check` 通过；模型、数组、下游表面积分、HTML/JSON/NPZ 全流程执行成功。

复现本次调用：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m mdo_demo evaluate \
  --data assets/CRMpert --checkpoint assets/AeroTransformer/ATsurf_S \
  --device cpu --limit 8 --warmup 2 --output results/new_cpu_run
```

| 这 8 个样本的结果 | 数值 |
|---|---:|
| 单次完整预测耗时中位数 | 0.0623 s |
| 单次完整预测耗时 P95 | 0.1024 s |
| 模型初始化 / 加载 | 2.149 s |
| Cp RMSE 的样本平均值 | 0.03305 |
| CL 绝对误差平均值 | 0.01282 |
| CD 绝对误差平均值 | 0.0007394（7.39 drag counts） |
| CD 绝对误差最大值 | 0.002280（22.80 drag counts） |

预测时间包含输入处理、模型推理、结果转回 CPU 和力积分；不包含模型加载、数据下载、数据读取、报告生成。P95 仅基于 8 次热调用，是初步诊断，不是稳定的性能基准。它既不是 CFD 加速比，也不是正式精度验收。

系数误差对发布的表面重积分标签计算。用当前固定后处理重新积分真值本身，仍有最大 `8.18e-5 CL / 6.85e-6 CD / 1.25e-3 CM` 的差异；报告另列 `integration_consistency_errors` 和相对重新积分真值的 `model_only_coefficient_errors`。不把这部分差异藏进模型误差。CM 参考点的历史约定需要进一步核对。

原始 CFD 数据：AeroTransformer 作者发布的 [CRMpert](https://huggingface.co/datasets/thuerey-group/CRMpert)，CC-BY-SA-4.0；上述数值是本仓库产生的初步评估。

## 软件检查

24 项标准库测试覆盖：原始样本 ID 与子集行号映射、几何/场形状、摩擦缩放、HTTP Range 失效时停止下载、非有限数拒绝、CFD 请求身份、残差/失败标记/升力收敛、CLI 不伪装真实求解、结果目录保护，以及模型错误导致候选不可行的情况。

Bash 脚本语法检查通过。固定上游源码 checkout 检查通过。真实报告在浏览器中检查过图表与表格。

## 待服务器验证

1. A6000 CUDA 预测时间、显存占用及 S/L 模型对比。
2. 官方 Docker 中的 ADflow 导入、真实 CFD 收敛、耗时和内存。
3. 同一机翼/工况/参考量下的 CFD 配对与在线速度比较。
4. 小型连续优化、真实 CFD 复核、适配方法对照与完整成本。

第 3 项之前，不产生“比 CFD 快多少倍”的正式结果；第 4 项之前，不声称完整 MDO 加速或达到 SOTA。
