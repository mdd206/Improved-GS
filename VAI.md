# Pipeline VAI cho ImprovedGS

Tai lieu nay mo ta luong xu ly du lieu Viettel AI Race (VAI) da duoc tich hop vao ImprovedGS. Pipeline giu nguyen pose COLMAP, xu ly camera `SIMPLE_RADIAL`, render dung `test_poses.csv`, danh gia public ground truth va tao ZIP submission.

## 1. Luong xu ly

```text
VAI raw scene
  -> copy anh RGB + camera SIMPLE_RADIAL native
  -> MVSplat pinhole context tam thoi (khong ghi de anh train)
  -> sparse COLMAP + point MVSplat da loc -> points3D.ply
  -> ImprovedGS thuan train
  -> render test_poses.csv voi SIMPLE_RADIAL native
  -> sharpen mot lan
  -> JPEG dung ten CSV + PNG lossless
  -> SSIM / PSNR / LPIPS / weighted score tren JPEG
  -> validate + ZIP JPEG rieng + ZIP PNG rieng
```

`3dgs-origin` khong duoc import hay sua doi. Tat ca code VAI moi nam trong package `vai/` va cac CLI `vai_*.py` cua repository nay.

## 2. Preprocess HCM0204

Thi nghiem MVSplat-init khong can COLMAP CLI va khong bat fixed-pose
retriangulation P1. Anh RGB va camera `SIMPLE_RADIAL` duoc giu nguyen. Output
rieng nam tai
`/kaggle/working/vai_mvsplat_init_native_simple_radial/public_set/HCM0204`:

```bash
python vai_preprocess.py \
  --input /kaggle/input/datasets/xuanph/phase1/phase1/public_set \
  --output /kaggle/working/vai_mvsplat_init_native_simple_radial/public_set \
  --subset HCM0204 \
  --overwrite \
  --native_simple_radial
```

Output co layout:

```text
vai_mvsplat_init_native_simple_radial/public_set/HCM0204/
  images/                 # Anh RGB raw
  sparse/0/               # SIMPLE_RADIAL + sparse goc + MVSplat points3D.ply
  test/images/            # Public ground truth neu co
  test/test_poses.csv
  vai_metadata.json       # Camera va thong ke preprocess
```

Preprocess tao scene trong thu muc tam, validate xong moi thay output dich. Neu scene dich da ton tai, lenh se dung; chi dung `--overwrite` khi muon tao lai scene do.

Root rieng la bat buoc de thi nghiem khong tai su dung scene P1 hoac D3 cu.
Cell preprocess co `--overwrite`, `--native_simple_radial` va khong truyen bat
ky flag `--fixed_pose_retriangulation` nao.

Kiem tra lai output ma khong preprocess:

```bash
python vai_preprocess.py \
  --input /kaggle/input/datasets/xuanph/phase1/phase1/public_set \
  --output /kaggle/working/vai_mvsplat_init_native_simple_radial/public_set \
  --subset HCM0204 \
  --validate_only
```

## 3. Train, render va evaluate

Khi chay notebook Kaggle, sua truc tiep dictionary `VAI_CONFIG` trong cell co tag
`parameters`. Notebook ghi dictionary nay thanh
`/kaggle/working/vai_<set_name>_mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025.runtime.json`;
dry-run va
train deu dung file runtime do,
khong doc config HCM0204 trong source repo. File
[configs/vai_hcm0204.json](configs/vai_hcm0204.json) chi la template cho cach chay CLI
ngoai notebook.

Config mac dinh trong notebook da dat:

- `training_method=improvedgs`.
- `iterations=30000`; luu point cloud tai 30.000.
- PLY duoc ghi binary theo chunk 65.536 Gaussian va chi replace file dich sau
  khi ghi xong, tranh peak RAM khi model gan budget 5,5 trieu. Thi nghiem khong
  luu optimizer checkpoint mac dinh vi state nay rat lon.
- `position_lr_max_steps=30000`.
- `coarse_to_fine=false`.
- `pose_aware_sampling=false`.
- `densify_grad_threshold=0.00025` va `budget=5500000`.
- `eval=false` de dung toan bo 240 anh train, khong LLFF-hold anh.
- `data_device=cpu` de 240 anh va edge map khong chiem bo nho GPU Kaggle.
- `postprocess_script=vai_render.py`.
- Render JPEG theo dung ten `.JPG` trong CSV vao
  `/kaggle/working/vai_renders/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/<set>/<scene>`.
- Neu `save_png=true`, luu them PNG lossless vao
  `/kaggle/working/vai_png/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/<set>/<scene>`.
- Redistort bang bicubic interpolation.
- Unsharp mask voi `amount=1.0`, `sigma=0.60`.
- Luu JPEG voi `quality=95`, `subsampling=2` (4:2:0).
- Danh gia public GT vao
  `/kaggle/working/vai_eval/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/<set>/<scene>.json`.
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
  -s /kaggle/working/vai_mvsplat_init_native_simple_radial/public_set/HCM0204 \
  -m /kaggle/working/vai_models/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/public_set/HCM0204 \
  --iteration 30000 \
  --output_root /kaggle/working/vai_renders/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/public_set \
  --eval_root /kaggle/working/vai_eval/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/public_set \
  --output_extension csv \
  --save_png true \
  --png_root /kaggle/working/vai_png/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/public_set \
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
  --source_path /kaggle/working/vai_mvsplat_init_native_simple_radial/public_set/HCM0204 \
  --render_dir /kaggle/working/vai_renders/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/public_set/HCM0204 \
  --output /kaggle/working/vai_eval/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/public_set/HCM0204.json \
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
  --submission_dir /kaggle/working/vai_renders/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/public_set \
  --zip_path /kaggle/working/public_set_mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025_jpeg.zip \
  --subset HCM0204 \
  --output_extension csv

python vai_package.py \
  --phase_dir /kaggle/input/datasets/xuanph/phase1/phase1 \
  --set_name public_set \
  --submission_dir /kaggle/working/vai_png/mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025/public_set \
  --zip_path /kaggle/working/public_set_mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025_png.zip \
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
EXPERIMENT_NAME = "mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025"
```

De chay private, doi `SET_NAME` thanh ten thu muc private, vi du
`private_set1`. Notebook tu tao `VAI_CONFIG["scenes"]`, preprocess dung danh
sach da chon, batch runner train/render tung scene, va package chung cac scene
vao ZIP JPEG va PNG co ten theo `SET_NAME` va `EXPERIMENT_NAME`.
Public set tao them ZIP evaluation; private set bo qua evaluation.

De thay doi iterations, budget Gaussian, duong dan output, tham so sharpen, JPEG,
evaluation hoac cac train/render argument khac, chi sua cell `VAI_CONFIG` o dau
notebook. Co the them argument moi vao `train_args` hoac `postprocess_args` ngay trong
cell nay ma khong can sua file Python hay JSON trong repository.

## 7. MVSplat-init + ImprovedGS thuan

Nhanh `agent/mvsplat-init-improvedgs` dung MVSplat nhu mot buoc khoi tao hinh
hoc duy nhat. MVSplat khong nam trong training loop: CLI chi ghi
`sparse/0/points3D.ply` gom XYZ/RGB, sau do `GaussianModel.create_from_pcd`
cua ImprovedGS tu khoi tao scale, rotation, opacity va SH. Vi vay LAS, EAS, RAP
va MU van la ImprovedGS thuan; khong co loss hay densification cua HF-GS.

Pipeline giu bo anh/camera `SIMPLE_RADIAL` cho train va render. Moi context cua
MVSplat duoc undistort tam thoi trong RAM thanh pinhole 256x256. Cac cap anh
duoc chon bang shared COLMAP tracks, baseline theo robust scene diagonal va
phan bo deu tren trajectory. Near/far duoc suy ra theo sparse depth cua tung
camera. Point du doan phai qua cac bo loc:

- opacity cua MVSplat;
- valid SIMPLE_RADIAL-to-PINHOLE mask;
- near/far camera;
- relative-depth consistency voi view con lai;
- expanded sparse-scene bounds;
- voxel dedup, trong do sparse COLMAP goc luon la anchor.

Chay rieng initializer sau `vai_preprocess.py --native_simple_radial`:

```bash
python -u vai_mvsplat_init.py \
  --data_root /kaggle/working/vai_mvsplat_init_native_simple_radial/public_set \
  --mvsplat_repo /kaggle/working/mvsplat \
  --checkpoint /kaggle/working/mvsplat_checkpoints/re10k.ckpt \
  --checkpoint_sha256 83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a \
  --subset HCM0204 \
  --image_size 256 \
  --max_pairs 32 \
  --min_shared_tracks 24 \
  --opacity_threshold 0.35 \
  --consistency_rel_error 0.15 \
  --voxel_divisor 6000 \
  --max_points 600000 \
  --overwrite
```

Notebook pin MVSplat tai commit
`01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01`, kiem tra SHA256 checkpoint,
khong cai COLMAP/OpenCV va khong pin NumPy. Cau hinh training sau initializer
la `training_method=improvedgs`, 30.000 iteration, budget 5,5 trieu va
`densify_grad_threshold=0.00025`.
