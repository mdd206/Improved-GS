# ImprovedGS 项目使用文档

[返回 README](README_CN.md)

本文作为 `README_CN.md` 的补充文档使用。安装、基础数据格式、入口脚本简介和输出目录总览已在 README 中说明，这里不重复展开；本文重点说明配置系统、批量运行、方法选择、参数含义和常见使用任务。

## 1. 推荐使用方式

本项目推荐统一通过 `run.py` + JSON 配置运行。`run.py` 会根据配置自动完成：

| 阶段 | 行为 |
| --- | --- |
| 路径解析 | 将 `data_root + scene.name` 作为输入路径，将 `output_root + scene.name` 作为输出路径 |
| 训练 | 为每个场景调用 `train.py` |
| 后处理 | 可选调用 `postprocess.py` 渲染并计算指标 |
| repeat | 可对同一场景重复训练多次 |
| 选优 | 可按 PSNR 保留最好一次 repeat |
| 汇总 | 多场景时生成 train/test total CSV |

推荐新实验从复制配置开始：

```bash
cp configs/improvedgs.json configs/my_exp.json
python run.py -c configs/my_exp.json --dry_run
python run.py -c configs/my_exp.json --only_scene bicycle
```

## 2. 配置文件结构

可运行配置需要有顶层 `scenes` 列表。一个最小示例：

```json
{
  "data_root": "/path/to/data",
  "output_root": "output/my_exp",
  "repeat": 1,
  "run_postprocess": true,
  "train_args": {
    "training_method": "improvedgs"
  },
  "postprocess_args": {
    "include_train": true,
    "no_save_train_images": true
  },
  "scenes": [
    {
      "name": "bicycle"
    }
  ]
}
```

### 2.1 顶层运行参数

| 参数 | 说明 |
| --- | --- |
| `data_root` | 数据集根目录 |
| `output_root` | 实验输出根目录 |
| `repeat` | 每个场景重复训练次数 |
| `python_executable` | 子进程使用的 Python 命令 |
| `gpu_auto_select` | 是否自动选择较空闲 GPU |
| `gpu_id` | 指定 GPU；为 `null` 时由自动选卡逻辑决定 |
| `stop_on_error` | 某个场景失败后是否停止整个配置 |
| `run_postprocess` | 训练后是否执行后处理 |
| `select_best_repeat_by_psnr` | repeat 多次时是否按 PSNR 保留最好结果 |
| `output_suffix` | 输出目录名后缀 |

### 2.2 参数覆盖规则

配置支持全局参数和场景级覆盖。场景级优先级更高。

```json
{
  "train_args": {
    "training_method": "improvedgs",
    "budget": 3000000
  },
  "scenes": [
    {
      "name": "bonsai",
      "train_args": {
        "budget": 1000000
      }
    }
  ]
}
```

上例中，`bonsai` 使用 `budget=1000000`，其他场景使用全局 `budget=3000000`。

可覆盖的字段：

| 字段 | 全局位置 | 场景级位置 |
| --- | --- | --- |
| 训练参数 | `train_args` | `scenes[].train_args` |
| 后处理参数 | `postprocess_args` | `scenes[].postprocess_args` |
| checkpoint 保存参数 | `checkpoint_args` | `scenes[].checkpoint_args` |
| 输出后缀 | `output_suffix` | `scenes[].output_suffix` |

### 2.3 输出命名规则

| 配置 | 输出目录 |
| --- | --- |
| 默认 | `output_root/scene_name` |
| 顶层 `output_suffix: "a"` | `output_root/scene_name-a` |
| 场景级 `output_suffix: "b"` | `output_root/scene_name-b` |
| 顶层和场景级都有 | `output_root/scene_name-a-b` |
| `repeat: 3` | `scene_name-1`、`scene_name-2`、`scene_name-3` |

## 3. `run.py` 命令

| 命令 | 说明 |
| --- | --- |
| `python run.py` | 运行默认 `training_config.json` |
| `python run.py -c configs/improvedgs.json` | 运行单个配置 |
| `python run.py -c configs/` | 按文件名顺序运行目录下所有可运行配置 |
| `python run.py -c xxx.json --dry_run` | 只打印命令，不执行 |
| `python run.py -c xxx.json --only_scene a,b` | 只运行指定场景 |
| `python run.py -c xxx.json --rebuild_total` | 从已有结果重建 total CSV |

目录模式会执行带有顶层 `scenes` 列表的 `.json` 文件，并跳过 `.schema.json`。

## 4. 训练方法

训练方法由 `train_args.training_method` 控制：

| 方法 | 说明 |
| --- | --- |
| `3dgs` | 原始 3DGS baseline |
| `absgs` | AbsGS 风格对比方法 |
| `minigs` | Mini-Splatting-D 风格对比方法 |
| `mcmc` | 3DGS-MCMC 风格对比方法 |
| `improvedgs` | ImprovedGS 主方法 |
| `gns` | ImprovedGS 组件加 GNS 轻量化设置 |

### 4.1 ImprovedGS

```json
{
  "train_args": {
    "training_method": "improvedgs",
    "use_las": true,
    "use_rap": true,
    "use_eas": true,
    "use_mu": true
  }
}
```

组件开关：

| 参数 | 说明 |
| --- | --- |
| `use_las` | 是否启用 LAS |
| `use_rap` | 是否启用 RAP |
| `use_eas` | 是否启用 EAS |
| `use_mu` | 是否启用 MU |

常用参数：

| 参数 | 说明 |
| --- | --- |
| `budget` | 目标高斯数量预算 |
| `budget_multiplier` | 预算 warmup 倍率 |
| `budget_warmup_until_offset` | 预算 warmup 结束偏移 |
| `edge_sample_cams` | EAS 采样相机数量 |
| `split_distance` | LAS 分裂位置系数 |
| `opacity_reduction` | 分裂后的 opacity 衰减系数 |
| `improvedgs_reset_min_opacity` | RAP 相关 opacity reset 下限 |
| `rap_prune_ratio` | RAP 剪枝比例 |
| `rap_prune_offset` | RAP reset 后延迟剪枝步数 |
| `rap_rounds` | RAP 最大轮数 |
| `mu_start_iter` | MU 第一阶段起始迭代 |
| `mu_interval` | MU 第一阶段更新间隔 |
| `mu_second_start_iter` | MU 第二阶段起始迭代 |
| `mu_second_interval` | MU 第二阶段更新间隔 |

### 4.2 GNS

```json
{
  "train_args": {
    "training_method": "gns",
    "final_budget_mode": "num",
    "final_budget": 600000
  }
}
```

| 参数 | 说明 |
| --- | --- |
| `gns_opacity_reg` | opacity 正则权重 |
| `gns_opacity_lr_scale` | opacity 学习率倍率 |
| `final_budget_mode` | 最终预算模式：`off`、`num`、`rate` |
| `final_budget` | `num` 模式下的目标高斯数量 |
| `final_rate` | `rate` 模式下的保留比例 |
| `reg_prune_from_iter` | 正则剪枝起始迭代 |
| `reg_prune_until_iter` | 正则剪枝结束迭代 |

### 4.3 MCMC

`mcmc` 需要设置 `budget > 0`。

```json
{
  "train_args": {
    "training_method": "mcmc",
    "budget": 3000000,
    "densify_until_iter": 25000
  }
}
```

| 参数 | 说明 |
| --- | --- |
| `budget` | 目标高斯数量 |
| `noise_lr` | 噪声强度系数 |
| `mcmc_noise_opacity_sharpness` | opacity 噪声门控参数 |
| `mcmc_scale_reg` | scale 正则权重 |
| `mcmc_opacity_reg` | opacity 正则权重 |

### 4.4 MiniGS

```json
{
  "train_args": {
    "training_method": "minigs",
    "minigs_num_depth": 3500000,
    "minigs_num_max": 4500000
  }
}
```

| 参数 | 说明 |
| --- | --- |
| `minigs_imp_metric` | 重要性指标类型 |
| `minigs_num_depth` | 深度回采样数量 |
| `minigs_num_max` | 最大高斯数量 |
| `minigs_reinit_interval` | 周期性 reinit 间隔 |
| `minigs_blur_screen_coverage_divisor` | blur screen coverage 阈值系数 |

## 5. 通用训练参数

这些参数写在 `train_args` 或 `scenes[].train_args` 中。

### 5.1 数据相关

| 参数 | 说明 |
| --- | --- |
| `images` | 图片目录名，默认 `images` |
| `depths` | 深度图目录名，空字符串表示不用深度 |
| `resolution` | 分辨率设置 |
| `white_background` | 是否使用白色背景 |
| `init_type` | COLMAP 初始化方式：`sfm` 或 `random` |
| `train_test_exp` | 是否启用 train/test exposure 设置 |
| `data_device` | 图像数据加载设备 |
| `eval` | 是否划分 test 视角 |

### 5.2 优化相关

| 参数 | 说明 |
| --- | --- |
| `iterations` | 总训练迭代数 |
| `position_lr_init` | 位置初始学习率 |
| `position_lr_final` | 位置最终学习率 |
| `feature_lr` | SH 特征学习率 |
| `opacity_lr` | opacity 学习率 |
| `initial_opacity` | 初始 opacity |
| `scaling_lr` | scale 学习率 |
| `rotation_lr` | rotation 学习率 |
| `lambda_dssim` | DSSIM loss 权重 |
| `densification_interval` | 致密化间隔 |
| `opacity_reset_interval` | opacity reset 间隔 |
| `densify_from_iter` | 开始致密化迭代 |
| `densify_until_iter` | 结束致密化迭代 |
| `densify_grad_threshold` | 致密化梯度阈值 |
| `min_opacity` | 低 opacity 剪枝阈值 |
| `random_background` | 是否使用随机背景 |
| `optimizer_type` | 优化器类型：`default` 或 `ours_adam` |

### 5.3 深度、抗锯齿和曝光

| 参数 | 说明 |
| --- | --- |
| `depth_l1_weight_init` | 深度 L1 权重初始值 |
| `depth_l1_weight_final` | 深度 L1 权重最终值 |
| `depth_ratio` | 深度混合相关参数 |
| `antialiasing` | 是否启用抗锯齿 |
| `exposure_lr_init` | 曝光学习率初始值 |
| `exposure_lr_final` | 曝光学习率最终值 |

## 6. 保存、日志和 checkpoint

| 参数 | 说明 |
| --- | --- |
| `test_iterations` | 训练中执行测试日志的迭代列表 |
| `save_iterations` | 保存 `point_cloud.ply` 的迭代列表；最终迭代会自动追加 |
| `checkpoint_iterations` | 保存 `chkpnt*.pth` 的迭代列表 |
| `start_checkpoint_file` | 从指定 checkpoint 恢复训练 |
| `start_checkpoint_dir` | checkpoint 所在目录；为空时默认在当前 `model_path` 下找 |
| `report_lpips_test` | 训练中 test 日志是否计算 LPIPS |
| `report_lpips_train` | 训练中 train 日志是否计算 LPIPS |
| `progress_bar_width` | 训练进度条宽度 |
| `empty_cache_interval` | 每隔多少迭代清理一次 CUDA cache，`0` 表示关闭 |

保存 checkpoint：

```json
{
  "checkpoint_args": {
    "checkpoint_iterations": [7000, 15000]
  }
}
```

从其他目录恢复：

```json
{
  "train_args": {
    "start_checkpoint_dir": "output/improvedgs/bicycle",
    "start_checkpoint_file": "chkpnt15000.pth"
  }
}
```

原地恢复时，只设置 `start_checkpoint_file`，不设置 `start_checkpoint_dir`。这种情况下不会自动归档当前输出目录。

## 7. 后处理参数

`postprocess_args` 会传给 `postprocess.py`。`run.py` 会自动补充 `source_path`、`model_path`，并同步必要的数据参数。

| 参数 | 说明 |
| --- | --- |
| `iteration` | 评测迭代；`-1` 表示自动加载最大保存迭代 |
| `include_train` | 是否评测 train split |
| `skip_test` | 是否跳过 test split |
| `no_save_test_images` | 是否不保存 test 渲染图和 GT |
| `no_save_train_images` | 是否不保存 train 渲染图和 GT |
| `fps_warmup_rounds` | FPS 测量前的 warmup 轮数 |
| `fps_measure_rounds` | FPS 正式测量轮数 |
| `quiet` | 是否减少随机状态输出 |

只评测已有模型：

```bash
python postprocess.py -s /path/to/scene -m output/improvedgs/bicycle
```

只评测 train：

```bash
python postprocess.py -s /path/to/scene -m output/improvedgs/bicycle \
  --skip_test true \
  --include_train true
```

不保存渲染图：

```json
{
  "postprocess_args": {
    "no_save_test_images": true,
    "no_save_train_images": true
  }
}
```

## 8. GPU、repeat 和汇总

### 8.1 GPU 选择

自动选卡：

```json
{
  "gpu_auto_select": true,
  "gpu_id": null
}
```

指定 GPU：

```json
{
  "gpu_auto_select": false,
  "gpu_id": 0
}
```

自动选卡依赖 `nvidia-smi`，并使用 `/tmp/3dgs_gpu_locks` 下的 lock 文件降低本地多任务抢卡概率。

### 8.2 repeat 选优

```json
{
  "repeat": 3,
  "select_best_repeat_by_psnr": true
}
```

行为：

1. 生成 `scene-1`、`scene-2`、`scene-3`。
2. 优先按 test PSNR 选择最好一次。
3. 没有 test 结果时退回使用 train PSNR。
4. 最优输出重命名为 `scene`。
5. 未选中的 repeat 输出会被删除。

开启 `select_best_repeat_by_psnr` 时，即使 `run_postprocess` 为 `false`，也会执行后处理，因为选优需要指标。

### 8.3 重建 total CSV

已有各场景 `result_train.json` 或 `result_test.json` 时，可直接重建总表：

```bash
python run.py -c configs/improvedgs.json --rebuild_total
```

## 9. 常见配置片段

### 9.1 只训练

```json
{
  "run_postprocess": false
}
```

### 9.2 只保存指标，不保存图片

```json
{
  "postprocess_args": {
    "no_save_test_images": true,
    "no_save_train_images": true
  }
}
```

### 9.3 使用自定义图片目录

```json
{
  "train_args": {
    "images": "images_4"
  }
}
```

### 9.4 随机初始化

```json
{
  "train_args": {
    "init_type": "random"
  }
}
```

### 9.5 深度约束

```json
{
  "train_args": {
    "depths": "depths",
    "depth_l1_weight_init": 1.0,
    "depth_l1_weight_final": 0.01
  }
}
```

### 9.6 白色背景

```json
{
  "train_args": {
    "white_background": true
  }
}
```

### 9.7 ImprovedGS 去掉某个组件

```json
{
  "train_args": {
    "training_method": "improvedgs",
    "use_las": false
  }
}
```

### 9.8 单场景不同预算

```json
{
  "train_args": {
    "training_method": "improvedgs",
    "budget": 3000000
  },
  "scenes": [
    {
      "name": "bonsai",
      "train_args": {
        "budget": 1000000
      }
    }
  ]
}
```

## 10. 结果字段补充

README 已说明输出目录结构。这里补充 total CSV 和 `result_*.json` 中常见字段含义：

| 字段 | 说明 |
| --- | --- |
| `config_name` | 配置文件名，不含 `.json` |
| `split` | `train` 或 `test` |
| `iteration` | 评测迭代 |
| `scene_name` | 场景名 |
| `PSNR` | 平均 PSNR，越高越好 |
| `SSIM` | 平均 SSIM，越高越好 |
| `LPIPS` | 平均 LPIPS，越低越好 |
| `FPS` | 渲染帧率 |
| `NUM` | 高斯数量 |
| `Training_time` | 训练耗时，单位分钟 |

## 11. 排查建议

| 现象 | 建议检查 |
| --- | --- |
| 配置没有被目录模式执行 | 确认 JSON 顶层存在 `scenes` 列表 |
| 找不到场景 | 检查 `data_root/scene.name` 是否存在 |
| 没有 test 指标 | 检查 `eval`、test 划分和 `skip_test` |
| 没有 train 指标 | 检查 `postprocess_args.include_train` |
| 没有渲染图 | 检查 `no_save_test_images`、`no_save_train_images` |
| MCMC 报 budget 错误 | 设置 `budget > 0` |
| 自动选卡失败 | 检查 `nvidia-smi`，或直接设置 `gpu_id` |
| repeat 选优失败 | 确认每个 repeat 都已生成可读取的 `result_test.json` 或 `result_train.json` |

## 12. 实验习惯

- 新实验优先复制 `configs/improvedgs.json` 或 `training_config.json`。
- 批量实验前先用 `--dry_run` 检查命令。
- 大规模实验前先用 `--only_scene` 跑一个场景。
- 对比实验保持相同的 `data_root`、`scenes`、`repeat` 和后处理设置。
- 论文复现优先查看 `*_result_test_total.csv`，再检查单场景 `result_test.json` 和渲染图。
