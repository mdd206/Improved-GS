# ImprovedGS
[中文](README_CN.md) | English

*Xiaobin Deng, Changyu Diao, Min Li, Ruohan Yu, Duanqing Xu*
*Zhejiang University*
### [[arXiv](https://arxiv.org/abs/2508.12313)]  [[Project Page](https://xiaobin2001.github.io/improved-gs-web/)]  [[Results](https://pan.baidu.com/s/1NL5jTkJnzwO2KdfMFSIbvg?pwd=pz67)] [[Data](https://pan.baidu.com/s/1-qi3NE9CZKp8JPX6Y2ba_Q?pwd=45t2)]
---
This repository is the official implementation of **Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering** (CVPR 2026 Findings). It can reproduce the full ImprovedGS method and also supports ablation reproduction for the original `3dgs`, `absgs`, `minigs` (Mini-Splatting-D), `mcmc` (3DGS-MCMC), and other methods.

## Introduction

3DGS can already achieve high-quality real-time rendering, but the original densification strategy is relatively simple and limits the upper bound of reconstruction quality. ImprovedGS improves the densification process from three questions: **when to densify, how to densify, and how to reduce overfitting**.

This project is intended for researchers and developers who are new to 3D Gaussian Splatting, and it is also suitable for users who need a daily reconstruction tool.

| Feature | Description |
| --- | --- |
| Paper reproduction | Reproduce the ImprovedGS method and the main training pipeline from the paper with one command |
| Component ablation | Supports switches for ImprovedGS components such as LAS / RAP / EAS / MU |
| Method comparison | Built-in support for `3dgs`, `absgs`, `minigs`, `mcmc`, and `improvedgs` |
| Lightweight experiments | Provides the `gns` pruning mode for lightweight training and pruning experiments |
| Daily reconstruction | Supports training, rendering, PSNR/SSIM/LPIPS/FPS evaluation, and batch aggregation; includes acceleration techniques such as SpeedySplat's accurate half bounding box and a custom CUDA_Adam |

This repository is built from the latest 3DGS. Compared with the version built from TamingGS in August 2025, the current version has been reorganized based on the latest changes from the original 3DGS codebase, so it supports Mip-Splatting, depth constraints, exposure learning, and other features. The current version also adds multiple practical changes for training acceleration and batch reproduction. [GNS](https://xiaobin2001.github.io/GNS-web/) is our team's work on lightweight 3DGS. The method is simple and effective, and it has also been added to this codebase so users can reduce scene storage cost during daily training and testing.

For detailed project documentation, see: [Detailed project usage guide](PROJECT.md)

## Installation and Environment Setup

Recommended environment (the authors' development version):

| Dependency | Recommended Version |
| --- | --- |
| Python | 3.10.19 |
| CUDA | 12.1 |
| PyTorch | 2.1.1 + cu121 |
| Build tools | Able to compile CUDA/C++ extensions |

Lower or higher versions of PyTorch, Python, and CUDA have not been systematically tested. NumPy 2.0 and later are known to be potentially incompatible with some dependencies. If you use other versions, you may need to adjust dependencies or compatibility code according to the errors.

Windows users need to install the CUDA Toolkit and Visual Studio C++ build environment first. For example: install Visual Studio 2019/2022 with C++ build tools, then install the CUDA Toolkit, and make sure the related VS and CUDA directories are added to environment variables. Linux users need to confirm that `nvcc`, `gcc/g++`, and the CUDA runtime are available.

We recommend using conda, miniconda, or miniforge to create an isolated environment:

```bash
conda create -n improvedgs python=3.10.19
conda activate improvedgs
```

Download the project and enter the project root:

```bash
git clone https://github.com/XiaoBin2001/Improved-GS.git
cd Improved-GS
```

Install Python dependencies. `glances[gpu]` can be used to monitor hardware usage and is recommended:

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm plyfile glances[gpu]
pip install numpy==1.26.1 opencv-python==4.10.0.82 setuptools==69.5.1
```

You can also use `ninja` for compilation:

```bash
pip install ninja
```

Install the CUDA submodules:

```bash
pip install submodules/diff-gaussian-rasterization/ submodules/simple-knn/ submodules/fused-ssim/ --no-build-isolation
```

Verify the installation:

```bash
python -c "import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim; print(torch.cuda.is_available())"
```

## Training Data Layout

COLMAP format is recommended. `data_root` is the dataset root directory, and each `scene.name` in the config corresponds to `data_root/scene_name`.

```text
data_root/
  scene_name/
    images/
      000001.jpg
      000002.jpg
    sparse/0/
      cameras.bin or cameras.txt
      images.bin or images.txt
      points3D.bin or points3D.txt
      test.txt optional
      depth_params.json optional
    depths/ optional
```

Notes:

- The project reads `images/` by default. You can also specify another image directory with the `images` field in the config.
- COLMAP `.bin` files are read first; if `.bin` files do not exist, the reader falls back to `.txt` files.
- When `eval=true`, test views are split from sorted images by the internal LLFF hold rule; when `eval=false`, no test views are split.
- If you use depth constraints, you need to provide `depths/` and `sparse/0/depth_params.json`.

Inherited from 3DGS, this project is also compatible with Blender / NeRF synthetic format, for example:

```text
scene_name/
  transforms_train.json
  transforms_test.json
  train/test image paths are specified by file_path in the JSON files
  depths/ optional
```

## Training Scripts and Usage

This project recommends running all training, post-processing, and batch evaluation through the config entry point to avoid manually writing very long command-line arguments. `total_config.json` and `total_config.schema.json` list all configurable parameters for reference. If no config is specified, `training_config.json` is used by default.

| Entry | Purpose |
| --- | --- |
| `run.py` | Unified batch entry point. Reads JSON configs and runs training, post-processing, repeat selection, and total CSV aggregation |
| `train.py` | Low-level single-scene training entry point, usually called by `run.py` |
| `postprocess.py` | Post-processing entry point. Renders train/test views and computes PSNR, SSIM, LPIPS, and FPS, usually called by `run.py` |

The `configs/` folder provides example configs for reproducing `3dgs`, `absgs`, `improvedgs`, `mcmc`, and `minigs`. Except for ImprovedGS, the other densification methods were manually reproduced and added by us, and their overall results roughly align with the original papers. If you need strict reproduction with exactly matching results, we recommend also referring to the corresponding original repositories.

Run examples:

```bash
python run.py
python run.py -c configs/improvedgs.json # run one config
python run.py -c configs/ # run all configs in the folder
```

If you only want to train, set this in the config:

```json
{
  "run_postprocess": false
}
```

If you only want to evaluate an existing model, call the post-processing entry point directly:

```bash
python postprocess.py -s /path/to/scene -m output/scene
```


## Output Description

The output of a single scene is usually placed under `output_root/scene_name/`. If `repeat > 1` is configured and `select_best_repeat_by_psnr` is enabled, the script keeps the repeat with the best PSNR and organizes it under the base scene name.

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

Main files:

| File or Directory | Description |
| --- | --- |
| `point_cloud/iteration_*/point_cloud.ply` | Trained Gaussian point-cloud model |
| `training_parameters.json` | Training parameters, start time, end time, and training duration for this run |
| `cfg_args` | Parameter snapshot compatible with the original 3DGS workflow |
| `input.ply`, `cameras.json` | Backup of the initial point cloud and camera information |
| `result_test.json`, `result_train.json` | Per-scene average metrics, including PSNR, SSIM, LPIPS, FPS, NUM, and Training_time |
| `test/ours_<iter>/renders/` | Rendered images for test views |
| `test/ours_<iter>/gt/` | Ground-truth images for test views |
| `test/ours_<iter>/per_view.json` | Per-image PSNR, SSIM, and LPIPS |
| `<config_name>_result_*_total.csv` | Batch aggregation results for multiple scenes |

## Roadmap

- We will gradually provide `<config_name>_result_*_total.csv` files for various datasets to make result alignment and reproduction easier.
- Bugs will continue to be fixed as they are discovered.
- If you find any problem in the code, please open an issue.

## Acknowledgements

This project uses or refers to code, methods, or implementation ideas from the following works:

| Project | Link |
| --- | --- |
| 3D Gaussian Splatting | [GitHub](https://github.com/graphdeco-inria/gaussian-splatting) |
| Taming 3DGS | [GitHub](https://github.com/humansensinglab/taming-3dgs) |
| Mini-Splatting | [GitHub](https://github.com/fatPeter/mini-splatting) |
| 3DGS-MCMC | [GitHub](https://github.com/ubc-vision/3dgs-mcmc) |
| AbsGS | [GitHub](https://github.com/TY424/AbsGS) |
| Speedy-Splat | [GitHub](https://github.com/j-alex-hanson/speedy-splat) |

We thank these open-source projects and papers for providing foundational implementations and research ideas to the 3DGS community.

## Citation

If this project is helpful to your research, please cite ImprovedGS:

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
