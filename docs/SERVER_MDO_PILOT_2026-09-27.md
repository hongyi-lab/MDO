# 2026-09-27：A6000 上的 MDO 目标适配试验

用户调整了顺序：**直接尝试面向 MDO 的模型；普通微调以后再作消融。** 本轮不执行 v1 的普通 MSE 微调矩阵，不提前查看正式 Calibration/Test。

本轮配置是 [mdo_pilot_v2.json](../configs/mdo_pilot_v2.json)，沿用 v1 的几何隔离清单。这是开发试验，不能作为完成 v1 或取得盲测成绩的证明。

## 目录

服务器项目目录为 `~/Aero/cfd_fm_20260927`，与现有项目分开：

```text
code/MDO/       Git 仓库与固定上游源码
data/assets/    公开数据；仅450 Train +350 Dev场标签
models/         官方 Large 权重
envs/fm312/     独立 Python 3.12 / Torch CUDA 环境
cfd/            用户空间 CFD 环境、网格及匹配算例
runs/           每轮模型训练/评估结果与checkpoint
logs/           环境、下载、训练、CFD 的独立日志
manifests/      环境、版本与数据溯源
scripts/        该服务器项目的后台作业启动脚本
```

SSH 的地址、认证文件和密钥不放入代码仓库。下载和安装只写项目目录，不修改驱动或已有系统 Python。

## 模型与目标

- 官方 ATsurf_L，14,517,280 参数；与未适配 Large 做同集对照。
- 从同一官方 checkpoint 全参数适配，首轮固定 5 epochs，450 训练工况，共2250次更新。学习率 `1e-5`、Adam、FP32、batch1、seed7、clip1。
- 固定初始损失：`L_field + 0.01 L_CL + 0.01 L_CD + 0.1 L_span`。
- `L_field` 是原来的三个缩放表面通道 MSE；系数平方误差分别按 `0.01/0.0005` 归一化；`L_span` 是展向 `dCL/deta` 的宽度加权平方误差，按 `0.5` 归一化。
- 所有气动力/载荷目标都由同一真实 CFD 表面标签积分得到。没有新增伪标签；原数据的两套系数不混进训练目标。
- 先做5次预热、20次测量的训练资源探测；丢弃探测权重，正式训练从官方权重重启。
- 首轮固定最后一个epoch评价；调整损失或延长训练仅能依据 Dev，必须用新run ID记录，不能改用 Test选方案。

可微积分已与固定 cfdpost 的力积分实现核对，并通过梯度有限差分检查，见 [物理损失实现说明](PHYSICS_LOSS_NOTES.md)。这只证明数值接口实现一致，**不证明代理的几何导数等于 CFD 导数**。当前展向载荷仍是无量纲气动指标，完整有量纲 FEM 节点载荷传递尚未实现。

## 评价

开发集48个机翼、350工况；每个机翼先对工况平均，再对机翼等权汇总。报告CL/CD平均、尾部和最坏误差，Cp/Cf误差，展向载荷误差及分段耗时。原始预测数组和每个样本的误差均保留。

`evaluate_mdo_dev.py` 没有Test选项，拒绝含Calibration/Test样本的本轮数据目录。这里只展示开发效果，不计算本轮的正式置信放行率。

```bash
python scripts/evaluate_mdo_dev.py evaluate \
  --data assets/CRMpertTrainDev --checkpoint assets/AeroTransformer/ATsurf_L \
  --output /path/to/new-run/pretrained_dev --device cuda:0

# 训练入口及全部参数见 python scripts/train_mdo_adapter.py --help

python scripts/evaluate_mdo_dev.py evaluate \
  --data assets/CRMpertTrainDev --checkpoint /path/to/new-run/adapted_checkpoint \
  --output /path/to/new-run/adapted_dev --device cuda:0

python scripts/evaluate_mdo_dev.py compare \
  --before /path/to/new-run/pretrained_dev/evaluation.json \
  --after /path/to/new-run/adapted_dev/evaluation.json \
  --output /path/to/new-run/comparison.json
```

## 少量、匹配的真实 CFD

选择显式新定义的机翼案例（不是STW或公开CRMpert原案例的精确复刻），相同几何与工况同时交给FM和ADflow。第一例AoA约6.71°、Mach约0.8、Re2e7、T300K，采用8个MPI进程、81层原生网格，预计约352.8万单元。

不设十分钟墙钟上限。按预先指定的求解器残差/失败标志检查运行到收敛或明确失败；保留日志，先不生成完整CFD数据集。严格区分网格生成、初始化、求解和后处理耗时。

CFD的主翼表面范围、压力参考、参考面积及积分口径必须配套；FM未覆盖的翼尖/后缘表面不能悄悄计入一边的比较。模型训练和计时与CFD求解计时分时进行，避免争抢CPU后处理资源。

**同案例计时不自动构成等精度加速。** 只有参考CFD合格、定义匹配且FM误差满足事先约定要求后，才能报告指定案例的替代加速；否则分别报告误差与耗时，不宣称已解决完整MDO。
