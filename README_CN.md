# ImprovedGS
中文 | [English](README.md)

*Xiaobin Deng, Changyu Diao, Min Li, Ruohan Yu, Duanqing Xu*
*Zhejiang University*
### [[arXiv](https://arxiv.org/abs/2508.12313)]  [[Project Page](https://xiaobin2001.github.io/improved-gs-web/)]  [[Results](https://pan.baidu.com/s/1NL5jTkJnzwO2KdfMFSIbvg?pwd=pz67)] [[Data](https://pan.baidu.com/s/1-qi3NE9CZKp8JPX6Y2ba_Q?pwd=45t2)]
---
本仓库是论文 **Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering**（CVPR 2026 Findings）的官方实现，可用于复现 ImprovedGS 全文方法，也提供原始 `3dgs`、`absgs`、`minigs`（Mini-Splatting-D）、`mcmc`（3DGS-MCMC）等方法的消融复现支持。

## 简介

3DGS 已经能做到高质量实时渲染，但原始致密化策略比较简单，影响了重建质量的上限。ImprovedGS 从三个问题出发改进致密化流程：**什么时候致密化、怎么致密化、以及如何减少过拟合**。

本项目面向刚入门 3D Gaussian Splatting 的研究者和开发者，也适合需要日常重建工具的用户。

| 功能 | 说明 |
| --- | --- |
| 论文复现 | 一键复现 ImprovedGS 方法和论文中的主要训练流程 |
| 组件消融 | 支持 LAS / RAP / EAS / MU 等 ImprovedGS 组件开关 |
| 方法对比 | 内置 `3dgs`、`absgs`、`minigs`、`mcmc`、`improvedgs` |
| 轻量化实验 | 提供 `gns` 剪枝模式，用于轻量化训练和剪枝实验 |
| 日常重建 | 支持训练、渲染、PSNR/SSIM/LPIPS/FPS 评测和批量汇总；添加了SpeedySplat的精准半包盒、自定义的CUDA_Adam等加速训练技术 |

本仓库构建自最新版 3DGS。相较于 2025 年 8 月构建自 TamingGS 的版本，当前版本基于 3DGS 原版代码库的最新改动重新整理，因此支持 Mip-Splatting、深度约束、曝光学习等功能。同时，当前版本加入了多项训练加速和批量复现实用改动。[GNS](https://xiaobin2001.github.io/GNS-web/) 是本团队关于 3DGS 轻量化的一项工作，方法简单有效，也已加入当前代码库，方便用户在日常训练测试中降低场景的存储开销。

具体的项目文档参考：[详细项目功能介绍](PROJECT_CN.md)

## 安装与环境配置

推荐环境（作者的开发版本）：

| 依赖 | 推荐版本 |
| --- | --- |
| Python | 3.10.19 |
| CUDA | 12.1 |
| PyTorch | 2.1.1 + cu121 |
| 编译工具 | 可正常编译 CUDA/C++ 扩展 |

更低或更高版本的 PyTorch、Python、CUDA 暂未系统测试。已知 NumPy 2.0 及以上版本可能与部分依赖不兼容，如果自行使用其他版本，需要根据报错调整依赖或兼容代码。

Windows 用户需要先安装 CUDA Toolkit 和 Visual Studio C++ 编译环境。例如：先安装 Visual Studio 2019/2022，并安装 C++ build tools；再安装 CUDA Toolkit，并确认VS和CUDA的相关目录已加入环境变量。Linux 用户需要确认 `nvcc`、`gcc/g++` 和 CUDA runtime 可用。

推荐使用 conda、miniconda 或 miniforge 创建独立环境：

```bash
conda create -n improvedgs python=3.10.19
conda activate improvedgs
```

下载项目并进入项目根目录：

```bash
git clone https://github.com/XiaoBin2001/Improved-GS.git
cd Improved-GS
```

安装 Python 依赖。`glances[gpu]` 可用于监控硬件使用情况，推荐安装：

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm plyfile glances[gpu]
pip install numpy==1.26.1 opencv-python==4.10.0.82 setuptools==69.5.1
```

也可以使用 `ninja` 编译：

```bash
pip install ninja
```

安装 CUDA 子模块：

```bash
pip install submodules/diff-gaussian-rasterization/ submodules/simple-knn/ submodules/fused-ssim/ --no-build-isolation
```

验证安装：

```bash
python -c "import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim; print(torch.cuda.is_available())"
```

## 训练数据组织格式

推荐使用 COLMAP 格式。`data_root` 是数据集根目录，配置中的每个 `scene.name` 都会对应到 `data_root/scene_name`。

```text
data_root/
  scene_name/
    images/
      000001.jpg
      000002.jpg
    sparse/0/
      cameras.bin 或 cameras.txt
      images.bin 或 images.txt
      points3D.bin 或 points3D.txt
      test.txt 可选
      depth_params.json 可选
    depths/ 可选
```

说明：

- 默认读取 `images/`，也可以通过配置里的 `images` 字段指定其他图片目录。
- COLMAP 的 `.bin` 文件优先读取；没有 `.bin` 时会回退到 `.txt`。
- `eval=true` 时默认按内部 LLFF hold 规则从排序后的图片中划分 test 视角；`eval=false` 时不划分 test 视角。
- 如果使用深度约束，需要提供 `depths/` 和 `sparse/0/depth_params.json`。

继承自 3DGS，本项目也兼容 Blender / NeRF synthetic 格式，例如 NeRF 合成数据集：

```text
scene_name/
  transforms_train.json
  transforms_test.json
  train/test 图片路径由 JSON 中的 file_path 指向
  depths/ 可选
```

## 训练脚本与使用方式

本项目推荐所有训练、后处理和批量评测都通过 config 入口执行，避免手写很长的命令行参数。`total_config.json` 和 `total_config.schema.json` 给出了所有可调参数，方便用户参考。在不指定时，默认使用 `training_config.json` 文件。

| 入口 | 作用 |
| --- | --- |
| `run.py` | 统一批量入口，读取 JSON 配置，执行训练、后处理、repeat 选优和 total CSV 汇总 |
| `train.py` | 单场景训练底层入口，通常由 `run.py` 调用 |
| `postprocess.py` | 后处理入口，渲染 train/test 视角并计算 PSNR、SSIM、LPIPS、FPS，通常由 `run.py` 调用 |

`configs/` 文件夹提供了复现 `3dgs`、`absgs`、`improvedgs`、`mcmc`、`minigs` 的配置示例。除 ImprovedGS 外，其他致密化方法由我们手动复现并添加，整体结果与原论文大致对齐；如果需要完全一致的严格复现，建议同时参考对应方法的原仓库。

运行示例：

```bash
python run.py
python run.py -c configs/improvedgs.json # 执行单个config
python run.py -c configs/ # 执行文件夹内的所有config
```

如果只想训练，可在 config 中设置：

```json
{
  "run_postprocess": false
}
```

如果只想评测已有模型，可以直接调用后处理入口：

```bash
python postprocess.py -s /path/to/scene -m output/scene
```


## 输出结果说明

单个场景的输出通常位于 `output_root/scene_name/`。如果配置了 `repeat > 1` 且启用了 `select_best_repeat_by_psnr`，脚本会保留 PSNR 最好的 repeat，并将其整理为基础场景名。

```text
output_root/
  improvedgs_result_test_total.csv
  improvedgs_result_train_total.csv
  scene_name/
    cfg_args
    training_parameters.json
    input.ply
    cameras.json
    log.txt
    exposure.json
    opacity_report.txt
    result_test.json
    result_train.json
    point_cloud/
      iteration_xxx/
        point_cloud.ply
    test/
      ours_xxx/
        renders/
        gt/
        per_view.json
    train/
      ours_xxx/
        renders/
        gt/
        per_view.json
```

主要文件说明：

| 文件或目录 | 说明 |
| --- | --- |
| `point_cloud/iteration_*/point_cloud.ply` | 训练得到的 Gaussian 点云模型 |
| `training_parameters.json` | 本次训练参数、开始时间、结束时间和训练耗时 |
| `cfg_args` | 兼容 3DGS 原始流程的参数快照 |
| `input.ply`、`cameras.json` | 初始点云和相机信息备份 |
| `result_test.json`、`result_train.json` | 单场景平均指标，包含 PSNR、SSIM、LPIPS、FPS、NUM、Training_time |
| `test/ours_<iter>/renders/` | test 视角渲染图 |
| `test/ours_<iter>/gt/` | test 视角 GT 图 |
| `test/ours_<iter>/per_view.json` | 每张图的 PSNR、SSIM、LPIPS |
| `<config_name>_result_*_total.csv` | 多场景批量汇总结果 |

## 未来计划

- 后续会陆续提供各种数据集的 `<config_name>_result_*_total.csv`，方便大家对齐和复现结果。
- 如果发现 bug，也会持续更新修复。
- 如果发现代码的任何问题均可在 `issue` 中提出。

## 致谢

本项目使用或参考了以下工作中的代码、方法或实现思路：

| 项目 | 链接 |
| --- | --- |
| 3D Gaussian Splatting | [GitHub](https://github.com/graphdeco-inria/gaussian-splatting) |
| Taming 3DGS | [GitHub](https://github.com/humansensinglab/taming-3dgs) |
| Mini-Splatting | [GitHub](https://github.com/fatPeter/mini-splatting) |
| 3DGS-MCMC | [GitHub](https://github.com/ubc-vision/3dgs-mcmc) |
| AbsGS | [GitHub](https://github.com/TY424/AbsGS) |
| Speedy-Splat | [GitHub](https://github.com/j-alex-hanson/speedy-splat) |

感谢这些开源项目和论文为 3DGS 社区提供的基础实现与研究思路。

## 引用

如果本项目对你的研究有帮助，请引用 ImprovedGS：

```bibtex
@InProceedings{Deng_2026_CVPR,
    author    = {Deng, Xiaobin and Diao, Changyu and Li, Min and Yu, Ruohan and Xu, Duanqing},
    title     = {Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {223-232}
}
```
