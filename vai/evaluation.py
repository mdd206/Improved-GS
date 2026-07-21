"""Danh gia anh render VAI bang SSIM, PSNR, LPIPS va weighted score."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS
from utils.loss_utils import psnr, ssim
from vai.common import output_name_for_pose


def compute_weighted_score(
    ssim_value: float,
    psnr_value: float,
    lpips_value: float,
    psnr_max: float = 40.0,
) -> tuple[float, float]:
    """Tinh diem VAI tu ba metric theo cong thuc cua ban to chuc."""
    if psnr_max <= 0:
        raise ValueError("psnr_max phai lon hon 0")
    psnr_norm = max(0.0, min(float(psnr_value) / float(psnr_max), 1.0))
    weighted_score = (
        0.4 * (1.0 - float(lpips_value))
        + 0.3 * float(ssim_value)
        + 0.3 * psnr_norm
    )
    return weighted_score, psnr_norm


def _files_by_stem(folder: Path) -> dict[str, Path]:
    """Lap bang file anh theo stem de ghep JPG ground truth voi PNG render."""
    result: dict[str, Path] = {}
    for path in sorted(item for item in folder.iterdir() if item.is_file()):
        if path.stem in result:
            raise ValueError(f"Trung stem anh trong {folder}: {path.stem}")
        result[path.stem] = path
    return result


def _load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Doc anh RGB thanh tensor batch tren device danh gia."""
    with Image.open(path) as image:
        tensor = tf.to_tensor(image.convert("RGB")).unsqueeze(0)
    return tensor.to(device)


def evaluate_rendered_scene(
    gt_dir: str | Path,
    render_dir: str | Path,
    pose_rows: list[dict[str, str]],
    output_extension: str = "csv",
    psnr_max: float = 40.0,
    lpips_net: str = "alex",
    device: str = "cuda",
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Danh gia day du cac pose va tra ve summary cung metric tung anh."""
    gt_dir = Path(gt_dir)
    render_dir = Path(render_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay ground truth: {gt_dir}")
    if not render_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay anh render: {render_dir}")
    if lpips_net not in {"alex", "squeeze", "vgg"}:
        raise ValueError(f"LPIPS network khong duoc ho tro: {lpips_net}")

    torch_device = torch.device(device)
    lpips_model = LPIPS(lpips_net).to(torch_device).eval()
    gt_by_stem = _files_by_stem(gt_dir)
    render_by_stem = _files_by_stem(render_dir)

    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    per_view: dict[str, dict[str, float]] = {}
    for row in tqdm(pose_rows, desc="Evaluating VAI", dynamic_ncols=True):
        stem = Path(row["image_name"]).stem
        gt_path = gt_by_stem.get(stem)
        render_path = render_by_stem.get(stem)
        expected_render_name = output_name_for_pose(row["image_name"], output_extension)
        if gt_path is None:
            raise FileNotFoundError(f"Thieu ground truth cho pose: {row['image_name']}")
        if render_path is None or render_path.name != expected_render_name:
            raise FileNotFoundError(f"Thieu anh render: {render_dir / expected_render_name}")

        expected_size = (int(float(row["width"])), int(float(row["height"])))
        with Image.open(gt_path) as gt_image:
            gt_size = gt_image.size
        with Image.open(render_path) as render_image:
            render_size = render_image.size
        if gt_size != expected_size or render_size != expected_size:
            raise ValueError(
                f"Sai kich thuoc {stem}: gt={gt_size} render={render_size} expected={expected_size}"
            )

        gt_tensor = _load_rgb_tensor(gt_path, torch_device)
        render_tensor = _load_rgb_tensor(render_path, torch_device)
        with torch.inference_mode():
            ssim_value = float(ssim(render_tensor, gt_tensor).item())
            psnr_value = float(psnr(render_tensor, gt_tensor).item())
            lpips_value = float(lpips_model(render_tensor, gt_tensor).item())
        total_ssim += ssim_value
        total_psnr += psnr_value
        total_lpips += lpips_value
        per_view[expected_render_name] = {
            "SSIM": ssim_value,
            "PSNR": psnr_value,
            "LPIPS": lpips_value,
        }

    image_count = len(pose_rows)
    mean_ssim = total_ssim / image_count
    mean_psnr = total_psnr / image_count
    mean_lpips = total_lpips / image_count
    weighted_score, psnr_norm = compute_weighted_score(
        mean_ssim,
        mean_psnr,
        mean_lpips,
        psnr_max,
    )
    summary = {
        "image_count": image_count,
        "num_images": image_count,
        "SSIM": mean_ssim,
        "PSNR": mean_psnr,
        "LPIPS": mean_lpips,
        "psnr_norm": psnr_norm,
        "psnr_max": float(psnr_max),
        "weighted_score": weighted_score,
        "lpips_net": lpips_net,
    }
    return summary, per_view
