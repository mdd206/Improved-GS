# ImprovedGS Project Usage Documentation

[Back to README](README.md)

This document is a supplement to `README.md`. Installation, the basic data layout, entry-point overview, and output directory overview are already covered in the README, so they are not repeated here. This document focuses on the config system, batch execution, method selection, parameter meanings, and common usage tasks.

## 1. Recommended Usage

This project recommends running through `run.py` + JSON configs. `run.py` automatically performs the following steps according to the config:

| Stage | Behavior |
| --- | --- |
| Path resolution | Uses `data_root + scene.name` as the input path and `output_root + scene.name` as the output path |
| Training | Calls `train.py` for each scene |
| Post-processing | Optionally calls `postprocess.py` to render and compute metrics |
| repeat | Can train the same scene multiple times |
| Best selection | Can keep the repeat with the best PSNR |
| Aggregation | Generates train/test total CSV files for multi-scene runs |

For a new experiment, start by copying a config:

```bash
cp configs/improvedgs.json configs/my_exp.json
python run.py -c configs/my_exp.json --dry_run
python run.py -c configs/my_exp.json --only_scene bicycle
```

## 2. Config File Structure

A runnable config needs a top-level `scenes` list. Minimal example:

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

### 2.1 Top-Level Runtime Parameters

| Parameter | Description |
| --- | --- |
| `data_root` | Dataset root directory |
| `output_root` | Experiment output root directory |
| `repeat` | Number of repeated training runs for each scene |
| `python_executable` | Python command used by child processes |
| `gpu_auto_select` | Whether to automatically choose a less busy GPU |
| `gpu_id` | Specific GPU ID; when `null`, GPU selection is controlled by the auto-selection logic |
| `stop_on_error` | Whether to stop the whole config when one scene fails |
| `run_postprocess` | Whether to run post-processing after training |
| `select_best_repeat_by_psnr` | Whether to keep the best repeat by PSNR when running multiple repeats |
| `output_suffix` | Suffix for output directory names |

### 2.2 Parameter Override Rules

Configs support global parameters and scene-level overrides. Scene-level parameters have higher priority.

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

In this example, `bonsai` uses `budget=1000000`, while other scenes use the global `budget=3000000`.

Fields that can be overridden:

| Field | Global Position | Scene-Level Position |
| --- | --- | --- |
| Training parameters | `train_args` | `scenes[].train_args` |
| Post-processing parameters | `postprocess_args` | `scenes[].postprocess_args` |
| Checkpoint saving parameters | `checkpoint_args` | `scenes[].checkpoint_args` |
| Output suffix | `output_suffix` | `scenes[].output_suffix` |

### 2.3 Output Naming Rules

| Config | Output Directory |
| --- | --- |
| Default | `output_root/scene_name` |
| Top-level `output_suffix: "a"` | `output_root/scene_name-a` |
| Scene-level `output_suffix: "b"` | `output_root/scene_name-b` |
| Both top-level and scene-level suffixes | `output_root/scene_name-a-b` |
| `repeat: 3` | `scene_name-1`, `scene_name-2`, `scene_name-3` |

## 3. `run.py` Commands

| Command | Description |
| --- | --- |
| `python run.py` | Run the default `training_config.json` |
| `python run.py -c configs/improvedgs.json` | Run a single config |
| `python run.py -c configs/` | Run all runnable configs in the directory in filename order |
| `python run.py -c xxx.json --dry_run` | Print commands only, without executing them |
| `python run.py -c xxx.json --only_scene a,b` | Run only the specified scenes |
| `python run.py -c xxx.json --rebuild_total` | Rebuild total CSV files from existing results |

Directory mode runs `.json` files that contain a top-level `scenes` list and skips `.schema.json` files.

## 4. Training Methods

The training method is controlled by `train_args.training_method`:

| Method | Description |
| --- | --- |
| `3dgs` | Original 3DGS baseline |
| `absgs` | AbsGS-style comparison method |
| `minigs` | Mini-Splatting-D-style comparison method |
| `mcmc` | 3DGS-MCMC-style comparison method |
| `improvedgs` | Main ImprovedGS method |
| `gns` | ImprovedGS components plus GNS lightweight settings |

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

Component switches:

| Parameter | Description |
| --- | --- |
| `use_las` | Whether to enable LAS |
| `use_rap` | Whether to enable RAP |
| `use_eas` | Whether to enable EAS |
| `use_mu` | Whether to enable MU |

Common parameters:

| Parameter | Description |
| --- | --- |
| `budget` | Target Gaussian count budget |
| `budget_multiplier` | Budget warmup multiplier |
| `budget_warmup_until_offset` | Budget warmup end offset |
| `edge_sample_cams` | Number of cameras sampled by EAS |
| `split_distance` | LAS split position coefficient |
| `opacity_reduction` | Opacity decay coefficient after splitting |
| `improvedgs_reset_min_opacity` | RAP-related opacity reset lower bound |
| `rap_prune_ratio` | RAP pruning ratio |
| `rap_prune_offset` | Delayed pruning steps after RAP reset |
| `rap_rounds` | Maximum RAP rounds |
| `mu_start_iter` | MU first-stage start iteration |
| `mu_interval` | MU first-stage update interval |
| `mu_second_start_iter` | MU second-stage start iteration |
| `mu_second_interval` | MU second-stage update interval |

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

| Parameter | Description |
| --- | --- |
| `gns_opacity_reg` | Opacity regularization weight |
| `gns_opacity_lr_scale` | Opacity learning-rate multiplier |
| `final_budget_mode` | Final budget mode: `off`, `num`, or `rate` |
| `final_budget` | Target Gaussian count in `num` mode |
| `final_rate` | Retention ratio in `rate` mode |
| `reg_prune_from_iter` | Regularized pruning start iteration |
| `reg_prune_until_iter` | Regularized pruning end iteration |

### 4.3 MCMC

`mcmc` requires `budget > 0`.

```json
{
  "train_args": {
    "training_method": "mcmc",
    "budget": 3000000,
    "densify_until_iter": 25000
  }
}
```

| Parameter | Description |
| --- | --- |
| `budget` | Target Gaussian count |
| `noise_lr` | Noise strength coefficient |
| `mcmc_noise_opacity_sharpness` | Opacity noise gate parameter |
| `mcmc_scale_reg` | Scale regularization weight |
| `mcmc_opacity_reg` | Opacity regularization weight |

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

| Parameter | Description |
| --- | --- |
| `minigs_imp_metric` | Importance metric type |
| `minigs_num_depth` | Depth back-sampling count |
| `minigs_num_max` | Maximum Gaussian count |
| `minigs_reinit_interval` | Periodic reinit interval |
| `minigs_blur_screen_coverage_divisor` | Blur screen coverage threshold coefficient |

## 5. Common Training Parameters

These parameters are written in `train_args` or `scenes[].train_args`.

### 5.1 Data-Related

| Parameter | Description |
| --- | --- |
| `images` | Image directory name, default `images` |
| `depths` | Depth-map directory name; an empty string means no depth is used |
| `resolution` | Resolution setting |
| `white_background` | Whether to use a white background |
| `init_type` | COLMAP initialization type: `sfm` or `random` |
| `train_test_exp` | Whether to enable train/test exposure settings |
| `data_device` | Device for loading image data |
| `eval` | Whether to split test views |

### 5.2 Optimization-Related

| Parameter | Description |
| --- | --- |
| `iterations` | Total training iterations |
| `position_lr_init` | Initial position learning rate |
| `position_lr_final` | Final position learning rate |
| `feature_lr` | SH feature learning rate |
| `opacity_lr` | Opacity learning rate |
| `initial_opacity` | Initial opacity |
| `scaling_lr` | Scale learning rate |
| `rotation_lr` | Rotation learning rate |
| `lambda_dssim` | DSSIM loss weight |
| `densification_interval` | Densification interval |
| `opacity_reset_interval` | Opacity reset interval |
| `densify_from_iter` | Densification start iteration |
| `densify_until_iter` | Densification end iteration |
| `densify_grad_threshold` | Densification gradient threshold |
| `min_opacity` | Low-opacity pruning threshold |
| `random_background` | Whether to use a random background |
| `optimizer_type` | Optimizer type: `default` or `ours_adam` |

### 5.3 Depth, Antialiasing, and Exposure

| Parameter | Description |
| --- | --- |
| `depth_l1_weight_init` | Initial depth L1 weight |
| `depth_l1_weight_final` | Final depth L1 weight |
| `depth_ratio` | Depth blending related parameter |
| `antialiasing` | Whether to enable antialiasing |
| `exposure_lr_init` | Initial exposure learning rate |
| `exposure_lr_final` | Final exposure learning rate |

## 6. Saving, Logging, and Checkpoint

| Parameter | Description |
| --- | --- |
| `test_iterations` | Iteration list for test logging during training |
| `save_iterations` | Iteration list for saving `point_cloud.ply`; the final iteration is automatically appended |
| `checkpoint_iterations` | Iteration list for saving `chkpnt*.pth` |
| `start_checkpoint_file` | Resume training from the specified checkpoint |
| `start_checkpoint_dir` | Directory containing the checkpoint; when empty, the current `model_path` is used |
| `report_lpips_test` | Whether to compute LPIPS for test logs during training |
| `report_lpips_train` | Whether to compute LPIPS for train logs during training |
| `progress_bar_width` | Training progress-bar width |
| `empty_cache_interval` | CUDA cache cleanup interval; `0` disables cleanup |

Save checkpoints:

```json
{
  "checkpoint_args": {
    "checkpoint_iterations": [7000, 15000]
  }
}
```

Resume from another directory:

```json
{
  "train_args": {
    "start_checkpoint_dir": "output/improvedgs/bicycle",
    "start_checkpoint_file": "chkpnt15000.pth"
  }
}
```

For in-place resume, set only `start_checkpoint_file` and do not set `start_checkpoint_dir`. In this case, the current output directory is not automatically archived.

## 7. Post-Processing Parameters

`postprocess_args` is passed to `postprocess.py`. `run.py` automatically fills `source_path` and `model_path`, and synchronizes the necessary data parameters.

| Parameter | Description |
| --- | --- |
| `iteration` | Evaluation iteration; `-1` means automatically loading the largest saved iteration |
| `include_train` | Whether to evaluate the train split |
| `skip_test` | Whether to skip the test split |
| `no_save_test_images` | Whether not to save test rendered images and GT |
| `no_save_train_images` | Whether not to save train rendered images and GT |
| `fps_warmup_rounds` | Number of warmup rounds before FPS measurement |
| `fps_measure_rounds` | Number of measured rounds for FPS |
| `quiet` | Whether to reduce random-state output |

Evaluate an existing model only:

```bash
python postprocess.py -s /path/to/scene -m output/improvedgs/bicycle
```

Evaluate train only:

```bash
python postprocess.py -s /path/to/scene -m output/improvedgs/bicycle \
  --skip_test true \
  --include_train true
```

Do not save rendered images:

```json
{
  "postprocess_args": {
    "no_save_test_images": true,
    "no_save_train_images": true
  }
}
```

## 8. GPU, Repeat, and Aggregation

### 8.1 GPU Selection

Automatic GPU selection:

```json
{
  "gpu_auto_select": true,
  "gpu_id": null
}
```

Specify a GPU:

```json
{
  "gpu_auto_select": false,
  "gpu_id": 0
}
```

Automatic GPU selection depends on `nvidia-smi` and uses lock files under `/tmp/3dgs_gpu_locks` to reduce local multi-task GPU conflicts.

### 8.2 Repeat Best Selection

```json
{
  "repeat": 3,
  "select_best_repeat_by_psnr": true
}
```

Behavior:

1. Generate `scene-1`, `scene-2`, and `scene-3`.
2. Prefer selecting the best repeat by test PSNR.
3. If test results are unavailable, fall back to train PSNR.
4. Rename the best output to `scene`.
5. Delete the unselected repeat outputs.

When `select_best_repeat_by_psnr` is enabled, post-processing is run even if `run_postprocess` is `false`, because metrics are required for best selection.

### 8.3 Rebuild Total CSV

When each scene already has `result_train.json` or `result_test.json`, rebuild the total tables directly:

```bash
python run.py -c configs/improvedgs.json --rebuild_total
```

## 9. Common Config Snippets

### 9.1 Train Only

```json
{
  "run_postprocess": false
}
```

### 9.2 Save Metrics Only, Not Images

```json
{
  "postprocess_args": {
    "no_save_test_images": true,
    "no_save_train_images": true
  }
}
```

### 9.3 Use a Custom Image Directory

```json
{
  "train_args": {
    "images": "images_4"
  }
}
```

### 9.4 Random Initialization

```json
{
  "train_args": {
    "init_type": "random"
  }
}
```

### 9.5 Depth Constraints

```json
{
  "train_args": {
    "depths": "depths",
    "depth_l1_weight_init": 1.0,
    "depth_l1_weight_final": 0.01
  }
}
```

### 9.6 White Background

```json
{
  "train_args": {
    "white_background": true
  }
}
```

### 9.7 Remove One ImprovedGS Component

```json
{
  "train_args": {
    "training_method": "improvedgs",
    "use_las": false
  }
}
```

### 9.8 Different Budgets for Individual Scenes

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

## 10. Result Field Notes

The README already explains the output directory structure. This section adds the meanings of common fields in total CSV files and `result_*.json`:

| Field | Description |
| --- | --- |
| `config_name` | Config filename without `.json` |
| `split` | `train` or `test` |
| `iteration` | Evaluation iteration |
| `scene_name` | Scene name |
| `PSNR` | Average PSNR; higher is better |
| `SSIM` | Average SSIM; higher is better |
| `LPIPS` | Average LPIPS; lower is better |
| `FPS` | Rendering frame rate |
| `NUM` | Gaussian count |
| `Training_time` | Training duration, in minutes |

## 11. Troubleshooting

| Symptom | Suggested Check |
| --- | --- |
| Config is not run in directory mode | Confirm that the JSON has a top-level `scenes` list |
| Scene not found | Check whether `data_root/scene.name` exists |
| No test metrics | Check `eval`, test split, and `skip_test` |
| No train metrics | Check `postprocess_args.include_train` |
| No rendered images | Check `no_save_test_images` and `no_save_train_images` |
| MCMC reports a budget error | Set `budget > 0` |
| Automatic GPU selection fails | Check `nvidia-smi`, or set `gpu_id` directly |
| Repeat best selection fails | Confirm that every repeat has readable `result_test.json` or `result_train.json` |

## 12. Experiment Habits

- For new experiments, copy `configs/improvedgs.json` or `training_config.json` first.
- Before batch experiments, use `--dry_run` to check commands.
- Before large-scale experiments, use `--only_scene` to run one scene first.
- For comparison experiments, keep the same `data_root`, `scenes`, `repeat`, and post-processing settings.
- For paper reproduction, first check `*_result_test_total.csv`, then inspect per-scene `result_test.json` and rendered images.
