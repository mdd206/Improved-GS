# Pipeline VAI cho ImprovedGS

Tai lieu nay mo ta luong xu ly du lieu Viettel AI Race (VAI) da duoc tich hop vao ImprovedGS. Pipeline giu nguyen pose COLMAP, xu ly camera `SIMPLE_RADIAL`, render dung `test_poses.csv`, danh gia public ground truth va tao ZIP submission.

## 1. Luong xu ly

```text
VAI raw scene
  -> COLMAP image_undistorter
  -> anh PINHOLE RGBA + alpha mask
  -> ImprovedGS train
  -> render test_poses.csv tren canvas undistort
  -> SIMPLE_RADIAL redistort + crop
  -> sharpen mot lan
  -> JPEG dung ten CSV + PNG lossless
  -> SSIM / PSNR / LPIPS / weighted score tren JPEG
  -> validate + ZIP JPEG rieng + ZIP PNG rieng
```

`3dgs-origin` khong duoc import hay sua doi. Tat ca code VAI moi nam trong package `vai/` va cac CLI `vai_*.py` cua repository nay.

## 2. Preprocess HCM0204

COLMAP CLI phai co trong `PATH`. Lenh sau chi xu ly HCM0204 va tao scene chuan tai `/kaggle/working/vai_cleaned/HCM0204`:

```bash
python vai_preprocess.py \
  --input /kaggle/input/datasets/xuanph/phase1/phase1/public_set \
  --output /kaggle/working/vai_cleaned \
  --subset HCM0204
```

Output co layout:

```text
vai_cleaned/HCM0204/
  images/                 # PNG RGBA da undistort
  sparse/0/               # COLMAP PINHOLE da loc theo anh train
  test/images/            # Public ground truth neu co
  test/test_poses.csv
  vai_metadata.json       # Camera SIMPLE_RADIAL goc va camera PINHOLE moi
```

Preprocess tao scene trong thu muc tam, validate xong moi thay output dich. Neu scene dich da ton tai, lenh se dung; chi dung `--overwrite` khi muon tao lai scene do.

Kiem tra lai output ma khong preprocess:

```bash
python vai_preprocess.py \
  --input /kaggle/input/datasets/xuanph/phase1/phase1/public_set \
  --output /kaggle/working/vai_cleaned \
  --subset HCM0204 \
  --validate_only
```

## 3. Train, render va evaluate

Khi chay notebook Kaggle, sua truc tiep dictionary `VAI_CONFIG` trong cell co tag
`parameters`. Notebook ghi dictionary nay thanh
`/kaggle/working/vai_<set_name>.runtime.json`; dry-run va train deu dung file runtime do,
khong doc config HCM0204 trong source repo. File
[configs/vai_hcm0204.json](configs/vai_hcm0204.json) chi la template cho cach chay CLI
ngoai notebook.

Config mac dinh trong notebook da dat:

- `training_method=improvedgs`.
- `coarse_to_fine=true`: train 1/4 resolution den iteration 2.000, 1/2 den 5.000, sau do dung full resolution.
- `pose_aware_sampling=true`: giu moi train camera mot lan va lap them toi da mot lan cho khoang 25% camera gan cac test pose thua coverage.
- `eval=false` de dung toan bo 240 anh train, khong LLFF-hold anh.
- `data_device=cpu` de 240 anh va edge map khong chiem bo nho GPU Kaggle.
- `postprocess_script=vai_render.py`.
- Render JPEG theo dung ten `.JPG` trong CSV vao `/kaggle/working/vai_renders/<set>/<scene>`.
- Neu `save_png=true`, luu them PNG lossless vao
  `/kaggle/working/vai_png/<set>/<scene>`.
- Redistort bang bicubic interpolation.
- Unsharp mask voi `amount=1.0`, `sigma=0.60`.
- Luu JPEG voi `quality=95`, `subsampling=2` (4:2:0).
- Danh gia public GT vao `/kaggle/working/vai_eval/<set>/<scene>.json`.
- Ghi summary tuong thich batch runner vao `result_test.json` cua model.

Kiem tra command truoc:

```bash
python run.py -c configs/vai_hcm0204.json --dry_run
```

Chay pipeline:

```bash
python run.py -c configs/vai_hcm0204.json
```

Co the render lai checkpoint ma khong train:

```bash
python vai_render.py \
  -s /kaggle/working/vai_cleaned/HCM0204 \
  -m /kaggle/working/vai_models/HCM0204 \
  --output_root /kaggle/working/vai_renders \
  --eval_root /kaggle/working/vai_eval \
  --output_extension csv \
  --save_png true \
  --png_root /kaggle/working/vai_png \
  --redistort_interpolation bicubic \
  --sharpen_amount 1.0 \
  --sharpen_sigma 0.60 \
  --jpeg_quality 95 \
  --jpeg_subsampling 2 \
  --evaluate true \
  --require_gt true \
  --overwrite true
```

Voi private set khong co ground truth, dat `evaluate=false` va `require_gt=false`.
Renderer van sinh day du JPEG va PNG.

## 4. Danh gia lai anh co san

```bash
python vai_evaluate.py \
  --source_path /kaggle/working/vai_cleaned/HCM0204 \
  --render_dir /kaggle/working/vai_renders/HCM0204 \
  --output /kaggle/working/vai_eval/HCM0204.json \
  --output_extension csv \
  --lpips_net alex \
  --psnr_max 40
```

Weighted score duoc tinh bang:

```text
0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * clamp(PSNR / 40, 0, 1)
```

## 5. Validate va tao ZIP

```bash
python vai_package.py \
  --phase_dir /kaggle/input/datasets/xuanph/phase1/phase1 \
  --set_name public_set \
  --submission_dir /kaggle/working/vai_renders \
  --zip_path /kaggle/working/HCM0204_render.zip \
  --subset HCM0204 \
  --output_extension csv

python vai_package.py \
  --phase_dir /kaggle/input/datasets/xuanph/phase1/phase1 \
  --set_name public_set \
  --submission_dir /kaggle/working/vai_png \
  --zip_path /kaggle/working/public_set_png.zip \
  --subset HCM0204 \
  --output_extension png
```

Tool se tu choi tao ZIP neu thieu anh, sai kich thuoc, sai ten hoac co file thua
trong thu muc scene. JPEG va PNG nam trong hai root rieng, nen moi ZIP chi chua
dung dinh dang da duoc doi chieu voi `test_poses.csv`.

## 6. Notebook Kaggle

[notebooks/vai_hcm0204.ipynb](notebooks/vai_hcm0204.ipynb) gom cac cell clone, cai
dependency, preprocess, dry-run, train/render/evaluate va dong goi ZIP JPEG,
ZIP PNG va evaluation.

Tat ca lua chon dataset/scene nam trong cell `parameters`:

```python
SET_NAME = "public_set"
SCENE_NAMES = ["HCM0204"]                 # mot scene
SCENE_NAMES = ["HCM0204", "SCENE_KHAC"]  # nhieu scene
SCENE_NAMES = []                          # tat ca scene trong set
EVALUATE = SET_NAME == "public_set"
REQUIRE_GT = EVALUATE
SAVE_PNG = True
```

De chay private, doi `SET_NAME` thanh ten thu muc private, vi du
`private_set1`. Notebook tu tao `VAI_CONFIG["scenes"]`, preprocess dung danh
sach da chon, batch runner train/render tung scene, va package chung cac scene
vao `public_set_jpeg.zip` va `public_set_png.zip` (ten thay doi theo `SET_NAME`).
Public set tao them ZIP evaluation; private set bo qua evaluation.

De thay doi iterations, budget Gaussian, duong dan output, tham so sharpen, JPEG,
evaluation hoac cac train/render argument khac, chi sua cell `VAI_CONFIG` o dau
notebook. Co the them argument moi vao `train_args` hoac `postprocess_args` ngay trong
cell nay ma khong can sua file Python hay JSON trong repository.
