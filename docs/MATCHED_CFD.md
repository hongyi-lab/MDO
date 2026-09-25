# 同一几何上的 FM / CFD 实验桥

这条路径生成一个**明确规定的新机翼算例**，让 ADflow 和 FM 使用同一个原生表面定义。
它不是 STW，也不是原始 CRMpert CFD 的精确复现；不会把不匹配案例的耗时相除后称为加速。

已完成模型环境安装后，在 Linux 服务器的一键入口：

```bash
bash scripts/run_live_case.sh cuda:0 8 81
```

脚本准备新几何、运行真实 FM、拉取/复用官方 CFD 镜像、生成体网格并求解、记录成对诊断。
81 是法向层数，默认约 350 万单元；安装调试可用 41，但它不代表相同精度。
这条 Linux 求解链还没有在用户服务器实测，不能把脚本生成视为 CFD 成功。

## 为什么没有直接把 origingeom.npy 送进 CFD

[原作者网格脚本](https://github.com/tum-pbs/AeroTransformer/blob/3dc350ff69e354d1451eae468c686a368bd6151f/simulation/gen-mesh.crmpert.py)
使用 13 个表面块，包括钝后缘和圆滑翼尖。公开 ML 数组是重新采样的主翼面，不完整包含这些块。
公开 Parquet 也不包含完整的原生 `input.json`，缺少固定平面形常数和完整约定。
因此，脚本明确要求完整参数，不自动补猜这些值。

`configs/new_crm_like_case.json` 是可以立即运行的**新示范几何**：

- CST 数字来自已固定版本的 CRMpert 表格第 0 行，按文档中的连续 7×10 顺序取值。
  原 Parquet 字段名称与这种顺序存在歧义；此处显式数字定义的是新形状，没有原样复现其 shape 0 的声明。
- 后掠、弦长、展长、扭转、上反和工况是明确选择的新算例参数，均写在配置中。
- 代码使用 SHA256 核验的作者原生函数生成 `wing.xyz` 和完整翼尖。
- FM 几何从这个**实际生成的原生主翼表面**插值得到，包含坐标约定和采样记录。
  这与发布数组采样方法不同，必须为新接口重新校准；既有 CRMpert 置信区间不自动适用。

配置包含来自 CC-BY-SA-4.0 数据的数值，其来源、修改及相同许可证写在配置里。
软件代码独立编写，维持仓库 MIT 许可证。

## 1. 本地或服务器准备几何、运行 FM

先完成仓库 `bootstrap_server.sh`（本地可使用已有独立 Python 环境），再安装小型几何依赖：

```bash
.venv/bin/python -m pip install \
  'git+https://github.com/YangYunjia/cst-modeling3d.git@9500b19a732463c26b32f6860ac6c517c7d212a5'

.venv/bin/python scripts/run_matched_cfd.py prepare \
  --input-json configs/new_crm_like_case.json \
  --output results/new_case_bundle --allow-reconstructed-case

.venv/bin/python scripts/run_matched_cfd.py validate \
  --request results/new_case_bundle/request.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/run_matched_cfd.py predict \
  --request results/new_case_bundle/request.json \
  --checkpoint assets/AeroTransformer/ATsurf_S --device cuda:0 \
  --output results/new_case_bundle/fm_prediction.json
```

`--allow-reconstructed-case` 是显式确认“新几何/新采样定义”的命令开关，防止误称原算例复现。
无需用户人工补 `input.json`，仓库示范配置已经完整。也可以替换为作者提供的完整原生参数。
准备阶段不依赖 MPI、pyHyp 或 ADflow，可以在 Windows 执行。它生成真实 13 块表面和
3×128×256 的 FM 输入；表面、FM 输入、配置及工况具有同一个 SHA256 身份记录。
移动整个 bundle 到服务器后仍可验证。任何修改都会使校验失败。

## 2. 服务器运行真实 CFD

先 `bash scripts/bootstrap_cfd.sh`。这一步拉取官方 MACH-Aero 镜像并记录实际 digest。
CFD 使用 CPU MPI；A6000 用于 FM，不把两者资源小时直接相加。

```bash
set -euo pipefail
source results/cfd_bootstrap/image.env
mkdir -p results/new_case_logs
docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$PWD,target=/home/mdolabuser/mount" \
  --workdir /home/mdolabuser/mount \
  --env "MDO_CFD_IMAGE_DIGEST=$MDO_CFD_IMAGE_DIGEST" \
  --env OMP_NUM_THREADS=1 \
  "$MDO_CFD_IMAGE_DIGEST" /bin/bash -lc \
  'source "${BASHRC_MDOLAB:?}" && mpirun -np 8 python scripts/run_matched_cfd.py run \
    --request results/new_case_bundle/request.json --output results/new_case_cfd' \
  2>&1 | tee results/new_case_logs/live.log
```

CFD 容器只读取准备好的 `wing.xyz`，无需再次安装 CST 包或 PyTorch。
默认 81 层的作者体网格设置（约 350 万单元，精确预估写入 bundle），不是低精度演示网格。为首次调试可以指定
`--wall-normal-layers 41`，但必须记录网格变化，不能据此声称相同精度加速。

条件固定采用作者的 RANS 配方：SA 湍流模型、Re=2×10⁷、Re 长度=1 m、T=300 K、
chordRef=1 m、力矩参考点 (0.25,0,0)、面积来自本次原生几何；Mach/AoA 来自配置。
体网格需要正体积和正质量；随后要求 ADflow 残差达到设定容差且没有 solve/fatal failure。
未收敛结果不会被比较工具接受。第一次运行本身也需要确认网格收敛性。

输出目录必须是新的空目录。`result.json` 记录：

- 同一 bundle 的 case hash、体网格 hash、完整物理条件和参考量。
- 真实 ADflow 全壁面及分组 CL/CD/CM、求解残差、状态、网格质量、MPI rank 数。
  `mainwing` 为 1/2/3/5/6/7 块，`trailingedge` 为 4/8 块，`tip` 为 9–13 块。
  pyHyp 将这些 family 写入 CGNS，ADflow `addFunction` 分别积分。
- 体网格时间、求解器时间和整体 live mesh+CFD 时间；模型下载/训练/启动另计。
- 原生压力/摩擦力表面文件、镜像 digest、solver 版本。

## 3. 比较当前已支持的结果

```bash
.venv/bin/python scripts/run_matched_cfd.py compare \
  --prediction results/new_case_bundle/fm_prediction.json \
  --cfd-result results/new_case_cfd/result.json \
  --output results/new_case_logs/paired_diagnostic.json
```

比较首先核验相同几何/条件身份以及 CFD 收敛，然后记录系数差和两边实测时间。
**比较使用相同物理主翼面**：FM 积分主翼面，对照 ADflow `group_coefficients.mainwing`，
不混用含翼尖/后缘的全壁面值。两边仍有离散表面采样/积分差异，差值不是纯模型误差。
原生 CFD 场文件保留，但映射到 FM 参考面的守恒插值、同定义分布载荷和导数还没有实现；
`structured_field_comparison.available=false`。
程序故意保持 `speedup=null` 和 `matched_speedup_eligible=false`，直到这些环节及新算例校准通过。
分组气动系数可以用于新算例的系数精度检查，但此结果尚不能替代完整场/载荷的自动 CFD 回退接口。

本地已执行并验证：原生表面生成、表面到 FM 输入、bundle 校验、真实权重推理及防混样测试。
**本地未执行** Linux pyHyp 体网格和 ADflow，服务器首次运行可能暴露网格/环境问题；不宣称 CFD 已通过。

参考：[作者求解器配方](https://github.com/tum-pbs/AeroTransformer/blob/3dc350ff69e354d1451eae468c686a368bd6151f/simulation/run-adflow.py)、
[pyHyp 参数](https://mdolab-pyhyp.readthedocs-hosted.com/en/latest/options.html)、
[pyHyp 网格质量字段](https://mdolab-pyhyp.readthedocs-hosted.com/en/latest/_modules/pyhyp/pyHyp.html)、
[ADflow API](https://mdolab-adflow.readthedocs-hosted.com/en/latest/API.html)、
[ADflow 压力力定义](https://mdolab-adflow.readthedocs-hosted.com/en/latest/costFunctions.html)
（压力力采用壁面压力减自由流压力，与 Cp 基准一致）。
