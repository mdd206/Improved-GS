"""Khoi tao point cloud ImprovedGS bang MVSplat tren cac cap anh co pose co dinh."""
from __future__ import annotations

import hashlib
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image as PilImage

from vai import VAI_METADATA_FILENAME
from vai.colmap_io import (
    Camera,
    Image,
    Point3D,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3d_binary,
    write_point_cloud_ply,
)
from vai.common import load_vai_metadata, save_json


MVSPLAT_UPSTREAM = "https://github.com/donydchen/mvsplat.git"
MVSPLAT_TESTED_COMMIT = "01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01"


@dataclass(frozen=True)
class PairSpec:
    """Mot cap context MVSplat da duoc xep hang bang overlap va baseline."""

    first_id: int
    second_id: int
    shared_tracks: int
    baseline_ratio: float
    score: float


@dataclass(frozen=True)
class PreparedView:
    """Anh pinhole 256x256 va camera dung truc tiep cho MVSplat."""

    image_id: int
    image_name: str
    image: torch.Tensor
    valid_mask: torch.Tensor
    c2w: np.ndarray
    w2c: np.ndarray
    intrinsics: np.ndarray
    near: float
    far: float


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Doi quaternion COLMAP (w, x, y, z) thanh world-to-camera rotation."""
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion COLMAP khong hop le")
    w, x, y, z = qvec / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_matrices(image: Image) -> tuple[np.ndarray, np.ndarray]:
    """Tra ve (c2w, w2c) theo convention OpenCV ma MVSplat su dung."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = _qvec_to_rotmat(image.qvec)
    w2c[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return np.linalg.inv(w2c), w2c


def _robust_scene_bounds(points: dict[int, Point3D]) -> tuple[np.ndarray, np.ndarray, float]:
    if not points:
        raise ValueError("MVSplat-init can sparse points3D.bin lam neo hinh hoc")
    xyz = np.stack([points[key].xyz for key in sorted(points)]).astype(np.float64)
    robust_min = np.quantile(xyz, 0.01, axis=0)
    robust_max = np.quantile(xyz, 0.99, axis=0)
    diagonal = float(np.linalg.norm(robust_max - robust_min))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        diagonal = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("Sparse point cloud khong co kich thuoc hinh hoc")
    return robust_min, robust_max, diagonal


def _registered_track_ids(image: Image) -> set[int]:
    return {int(value) for value in image.point3D_ids if int(value) >= 0}


def select_context_pairs(
    images: dict[int, Image],
    points: dict[int, Point3D],
    max_pairs: int = 32,
    min_shared_tracks: int = 24,
    min_baseline_ratio: float = 0.005,
    max_baseline_ratio: float = 0.30,
    target_baseline_ratio: float = 0.05,
) -> list[PairSpec]:
    """Chon cap anh phu deu trajectory, uu tien overlap va baseline vua phai."""
    if int(max_pairs) <= 0:
        raise ValueError("mvsplat_max_pairs phai duong")
    if int(min_shared_tracks) < 1:
        raise ValueError("mvsplat_min_shared_tracks phai it nhat 1")
    if not 0.0 < float(min_baseline_ratio) < float(max_baseline_ratio):
        raise ValueError("Khoang baseline MVSplat khong hop le")
    if not float(min_baseline_ratio) <= float(target_baseline_ratio) <= float(max_baseline_ratio):
        raise ValueError("mvsplat_target_baseline_ratio phai nam trong khoang baseline")

    _, _, scene_diagonal = _robust_scene_bounds(points)
    ordered = sorted(images.values(), key=lambda item: (item.name, item.id))
    if len(ordered) < 2:
        raise ValueError("MVSplat-init can it nhat hai anh registered")
    track_sets = {image.id: _registered_track_ids(image) for image in ordered}
    centers = {image.id: _camera_matrices(image)[0][:3, 3] for image in ordered}

    candidates: list[PairSpec] = []
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            shared = len(track_sets[first.id] & track_sets[second.id])
            if shared < int(min_shared_tracks):
                continue
            baseline_ratio = float(
                np.linalg.norm(centers[first.id] - centers[second.id]) / scene_diagonal
            )
            if not float(min_baseline_ratio) <= baseline_ratio <= float(max_baseline_ratio):
                continue
            baseline_quality = math.exp(
                -abs(math.log(max(baseline_ratio, 1e-12) / float(target_baseline_ratio)))
            )
            score = float(shared) * (0.25 + 0.75 * baseline_quality)
            candidates.append(
                PairSpec(
                    first_id=int(first.id),
                    second_id=int(second.id),
                    shared_tracks=int(shared),
                    baseline_ratio=baseline_ratio,
                    score=score,
                )
            )
    if not candidates:
        raise ValueError(
            "Khong tim duoc cap MVSplat hop le. Thu giam --min_shared_tracks hoac "
            "mo rong khoang --min_baseline_ratio/--max_baseline_ratio."
        )

    candidates.sort(
        key=lambda item: (
            item.score,
            item.shared_tracks,
            -abs(item.baseline_ratio - float(target_baseline_ratio)),
            -item.first_id,
            -item.second_id,
        ),
        reverse=True,
    )
    by_image: dict[int, list[PairSpec]] = {image.id: [] for image in ordered}
    for candidate in candidates:
        by_image[candidate.first_id].append(candidate)
        by_image[candidate.second_id].append(candidate)

    anchor_count = min(int(max_pairs), len(ordered))
    anchor_indices = np.unique(
        np.rint(np.linspace(0, len(ordered) - 1, anchor_count)).astype(np.int64)
    )
    selected: list[PairSpec] = []
    selected_keys: set[tuple[int, int]] = set()

    def add(candidate: PairSpec) -> bool:
        key = (min(candidate.first_id, candidate.second_id), max(candidate.first_id, candidate.second_id))
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected.append(candidate)
        return True

    for anchor_index in anchor_indices:
        anchor_id = ordered[int(anchor_index)].id
        for candidate in by_image[anchor_id]:
            if add(candidate):
                break
        if len(selected) >= int(max_pairs):
            break
    for candidate in candidates:
        if len(selected) >= int(max_pairs):
            break
        add(candidate)
    return selected


def estimate_depth_bounds(
    image: Image,
    points: dict[int, Point3D],
    scene_diagonal: float,
    near_quantile: float = 0.02,
    far_quantile: float = 0.98,
    near_padding: float = 0.75,
    far_padding: float = 1.25,
) -> tuple[float, float]:
    """Suy ra near/far theo depth sparse point nhin thay trong tung camera."""
    visible_ids = sorted(_registered_track_ids(image) & set(points))
    visible = (
        np.stack([points[point_id].xyz for point_id in visible_ids]).astype(np.float64)
        if visible_ids
        else np.empty((0, 3), dtype=np.float64)
    )
    _, w2c = _camera_matrices(image)

    def positive_depths(xyz: np.ndarray) -> np.ndarray:
        if len(xyz) == 0:
            return np.empty((0,), dtype=np.float64)
        depths = (xyz @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
        return depths[np.isfinite(depths) & (depths > 0.0)]

    depths = positive_depths(visible)
    if len(depths) < 16:
        all_xyz = np.stack([points[key].xyz for key in sorted(points)]).astype(np.float64)
        depths = positive_depths(all_xyz)
    if len(depths) < 2:
        raise ValueError(f"Khong du sparse depth duong cho camera {image.name}")

    lower = float(np.quantile(depths, float(near_quantile))) * float(near_padding)
    upper = float(np.quantile(depths, float(far_quantile))) * float(far_padding)
    minimum_near = max(float(scene_diagonal) * 1e-4, 1e-5)
    near = max(lower, minimum_near)
    far = max(upper, near * 2.0)
    if not np.isfinite(near) or not np.isfinite(far):
        raise ValueError(f"Near/far khong huu han cho camera {image.name}")
    return float(near), float(far)


def prepare_pinhole_view(
    image_path: str | Path,
    camera: Camera,
    image: Image,
    near: float,
    far: float,
    image_size: int = 256,
    focal_scale: float = 1.0,
) -> PreparedView:
    """Resize/crop mot view PINHOLE RGBA da preprocess vao context vuong cua MVSplat."""
    if camera.model == "PINHOLE" and len(camera.params) == 4:
        focal_x, focal_y, center_x, center_y = [
            float(value) for value in camera.params
        ]
    elif camera.model == "SIMPLE_PINHOLE" and len(camera.params) == 3:
        focal, center_x, center_y = [float(value) for value in camera.params]
        focal_x = focal_y = focal
    else:
        raise ValueError(
            "MVSplat-init tren main yeu cau PINHOLE/SIMPLE_PINHOLE, "
            f"nhan duoc {camera.model}"
        )
    if int(image_size) <= 0 or int(image_size) % 4 != 0:
        raise ValueError("mvsplat_image_size phai duong va chia het cho 4")
    if float(focal_scale) <= 0.0:
        raise ValueError("mvsplat_focal_scale phai duong")

    with PilImage.open(image_path) as source_image:
        if source_image.mode != "RGBA":
            raise ValueError(
                f"Anh MVSplat-init phai la RGBA tu main preprocess: {image_path}"
            )
        source_rgba = np.asarray(source_image, dtype=np.float32) / 255.0
    height, width = source_rgba.shape[:2]
    if (width, height) != (int(camera.width), int(camera.height)):
        raise ValueError(
            f"Kich thuoc anh {image.name} khong khop cameras.bin: "
            f"{width}x{height} != {camera.width}x{camera.height}"
        )

    target_size = int(image_size)
    source_scale = float(min(width, height))
    normalized_focal_x = focal_x / source_scale * float(focal_scale)
    normalized_focal_y = focal_y / source_scale * float(focal_scale)
    intrinsics = np.asarray(
        [
            [normalized_focal_x, 0.0, 0.5],
            [0.0, normalized_focal_y, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    coordinates = (torch.arange(target_size, dtype=torch.float32) + 0.5) / target_size
    ys, xs = torch.meshgrid(coordinates, coordinates, indexing="ij")
    xu = (xs - float(intrinsics[0, 2])) / float(intrinsics[0, 0])
    yu = (ys - float(intrinsics[1, 2])) / float(intrinsics[1, 1])
    source_x = focal_x * xu + center_x
    source_y = focal_y * yu + center_y
    bounds_mask = (
        (source_x >= 0.0)
        & (source_x <= max(width - 1, 0))
        & (source_y >= 0.0)
        & (source_y <= max(height - 1, 0))
    )
    grid_x = source_x * (2.0 / max(width - 1, 1)) - 1.0
    grid_y = source_y * (2.0 / max(height - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    source_tensor = torch.from_numpy(source_rgba).permute(2, 0, 1).unsqueeze(0)
    sampled_rgba = functional.grid_sample(
        source_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    valid_mask = bounds_mask & (sampled_rgba[3] > 0.5)
    pinhole = sampled_rgba[:3] * valid_mask.unsqueeze(0)
    c2w, w2c = _camera_matrices(image)
    return PreparedView(
        image_id=int(image.id),
        image_name=image.name,
        image=pinhole.contiguous(),
        valid_mask=valid_mask.contiguous(),
        c2w=c2w.astype(np.float32),
        w2c=w2c.astype(np.float32),
        intrinsics=intrinsics,
        near=float(near),
        far=float(far),
    )


def sha256_file(path: str | Path) -> str:
    """Tinh SHA256 theo chunk de notebook kiem tra dung checkpoint."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _register_namespace_package(name: str, path: Path) -> None:
    """Bo qua __init__ side-effect cua upstream, nhung van giu relative import."""
    existing = sys.modules.get(name)
    if existing is not None:
        existing_paths = [Path(value).resolve() for value in getattr(existing, "__path__", [])]
        if path.resolve() not in existing_paths:
            raise RuntimeError(
                f"Python da import namespace {name} tu noi khac: {existing_paths}"
            )
        return
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path.resolve())]
    sys.modules[name] = module


def load_mvsplat_encoder(
    mvsplat_repo: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> tuple[torch.nn.Module, int]:
    """Khoi tao encoder Re10K chinh thuc va nap rieng cac key `encoder.*`."""
    mvsplat_repo = Path(mvsplat_repo).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    for required in (
        mvsplat_repo / "src" / "model" / "encoder" / "encoder_costvolume.py",
        checkpoint_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Thieu MVSplat artifact: {required}")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MVSplat-init can GPU CUDA")

    repo_text = str(mvsplat_repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    imported_src = sys.modules.get("src")
    if imported_src is not None:
        imported_file = getattr(imported_src, "__file__", None)
        imported_paths = [
            Path(value).resolve() for value in getattr(imported_src, "__path__", [])
        ]
        if imported_file is not None:
            imported_paths.append(Path(imported_file).resolve())
        if not any(
            path == mvsplat_repo / "src" or mvsplat_repo in path.parents
            for path in imported_paths
        ):
            raise RuntimeError(
                f"Python da import package `src` tu noi khac: {imported_paths}. "
                "Hay chay MVSplat-init trong process moi."
            )

    try:
        from src.global_cfg import set_cfg

        # Hai __init__.py nay import toan bo dataset/evaluation/visualizer, keo theo
        # Lightning, sk-video, OpenCV va CUDA rasterizer du khong dung. Dang ky
        # namespace package cho phep Python nap dung cac submodule encoder can thiet.
        _register_namespace_package("src.model", mvsplat_repo / "src" / "model")
        _register_namespace_package(
            "src.model.encoder",
            mvsplat_repo / "src" / "model" / "encoder",
        )
        _register_namespace_package("src.dataset", mvsplat_repo / "src" / "dataset")
        from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
        from src.model.encoder.encoder_costvolume import (
            EncoderCostVolume,
            EncoderCostVolumeCfg,
            OpacityMappingCfg,
        )
        from src.model.encoder.visualization.encoder_visualizer_costvolume_cfg import (
            EncoderVisualizerCostVolumeCfg,
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Thieu dependency MVSplat. Cai cac goi jaxtyping, einops, e3nn "
            "va hydra-core truoc khi chay."
        ) from error

    set_cfg(
        SimpleNamespace(
            mode="test",
            dataset=SimpleNamespace(
                view_sampler=SimpleNamespace(num_context_views=2),
            ),
        )
    )
    encoder_cfg = EncoderCostVolumeCfg(
        name="costvolume",
        d_feature=128,
        num_depth_candidates=128,
        num_surfaces=1,
        visualizer=EncoderVisualizerCostVolumeCfg(
            num_samples=8,
            min_resolution=256,
            export_ply=False,
        ),
        gaussian_adapter=GaussianAdapterCfg(
            gaussian_scale_min=0.5,
            gaussian_scale_max=15.0,
            sh_degree=4,
        ),
        opacity_mapping=OpacityMappingCfg(initial=0.0, final=0.0, warm_up=1),
        gaussians_per_pixel=1,
        unimatch_weights_path=None,
        downscale_factor=4,
        shim_patch_size=4,
        multiview_trans_attn_split=2,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=[1, 1, 1],
        costvolume_unet_attn_res=[4],
        depth_unet_feat_dim=32,
        depth_unet_attn_res=[16],
        depth_unet_channel_mult=[1, 1, 1, 1, 1],
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_backbone_cross_attn=False,
        wo_cost_volume_refine=False,
        use_epipolar_trans=False,
    )
    encoder = EncoderCostVolume(encoder_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError("Checkpoint MVSplat khong co key `encoder.*`")
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint MVSplat khong khop encoder Re10K: missing={} unexpected={}".format(
                incompatible.missing_keys[:10],
                incompatible.unexpected_keys[:10],
            )
        )
    encoder = encoder.eval().to(torch.device(device))
    return encoder, int(checkpoint.get("global_step", 300_000))


def infer_pair(
    encoder: torch.nn.Module,
    first: PreparedView,
    second: PreparedView,
    global_step: int,
    device: str = "cuda",
    mixed_precision: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Chay encoder mot lan va tra means/opacities theo [view, pixel]."""
    views = (first, second)
    context = {
        "image": torch.stack([view.image for view in views])[None].to(device),
        "extrinsics": torch.from_numpy(np.stack([view.c2w for view in views]))[None].to(device),
        "intrinsics": torch.from_numpy(np.stack([view.intrinsics for view in views]))[None].to(device),
        "near": torch.tensor([[view.near for view in views]], dtype=torch.float32, device=device),
        "far": torch.tensor([[view.far for view in views]], dtype=torch.float32, device=device),
        "index": torch.tensor([[view.image_id for view in views]], dtype=torch.int64, device=device),
    }
    if mixed_precision not in {"none", "fp16", "bf16"}:
        raise ValueError("mvsplat_mixed_precision phai la none, fp16 hoac bf16")
    autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    with torch.inference_mode(), torch.autocast(
        device_type=torch.device(device).type,
        dtype=autocast_dtype,
        enabled=mixed_precision != "none",
    ):
        gaussians = encoder(
            context,
            int(global_step),
            deterministic=True,
            scene_names=[first.image_name],
        )
    point_count = first.image.shape[-2] * first.image.shape[-1]
    means = gaussians.means[0].detach().float().cpu().numpy()
    opacities = gaussians.opacities[0].detach().float().cpu().numpy()
    if means.shape != (2 * point_count, 3) or opacities.shape != (2 * point_count,):
        raise ValueError(
            f"Shape MVSplat bat ngo: means={means.shape}, opacities={opacities.shape}"
        )
    return means.reshape(2, point_count, 3), opacities.reshape(2, point_count)


def _camera_depth(points: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    return (points @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]


def filter_pair_predictions(
    means: np.ndarray,
    opacities: np.ndarray,
    views: tuple[PreparedView, PreparedView],
    opacity_threshold: float = 0.35,
    consistency_rel_error: float = 0.15,
    target_opacity_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Loc confidence, valid mask, depth bound va consistency cheo hai view."""
    if means.ndim != 3 or means.shape[0] != 2 or means.shape[-1] != 3:
        raise ValueError(f"means MVSplat sai shape: {means.shape}")
    if opacities.shape != means.shape[:2]:
        raise ValueError("means va opacities MVSplat khong khop")
    if not 0.0 <= float(opacity_threshold) <= 1.0:
        raise ValueError("mvsplat_opacity_threshold phai nam trong [0, 1]")
    if float(consistency_rel_error) <= 0.0:
        raise ValueError("mvsplat_consistency_rel_error phai duong")

    height, width = views[0].image.shape[-2:]
    point_count = height * width
    if means.shape[1] != point_count:
        raise ValueError("So Gaussian moi view khong khop kich thuoc context")
    accepted_xyz: list[np.ndarray] = []
    accepted_rgb: list[np.ndarray] = []
    accepted_confidence: list[np.ndarray] = []
    stats: dict[str, Any] = {"predicted_points": int(means.shape[0] * means.shape[1])}

    depth_maps = [
        _camera_depth(means[view_index], views[view_index].w2c)
        for view_index in range(2)
    ]
    for source_index in range(2):
        target_index = 1 - source_index
        source_view = views[source_index]
        target_view = views[target_index]
        source_points = means[source_index]
        source_confidence = opacities[source_index]
        source_depth = depth_maps[source_index]
        target_depth = depth_maps[target_index]

        mask = np.isfinite(source_points).all(axis=1)
        mask &= np.isfinite(source_confidence)
        mask &= source_confidence >= float(opacity_threshold)
        mask &= source_view.valid_mask.reshape(-1).cpu().numpy()
        mask &= source_depth >= source_view.near
        mask &= source_depth <= source_view.far

        target_camera_points = (
            source_points @ target_view.w2c[:3, :3].T + target_view.w2c[:3, 3]
        )
        projected_depth = target_camera_points[:, 2]
        safe_depth = np.maximum(projected_depth, 1e-12)
        normalized_x = (
            target_view.intrinsics[0, 0] * target_camera_points[:, 0] / safe_depth
            + target_view.intrinsics[0, 2]
        )
        normalized_y = (
            target_view.intrinsics[1, 1] * target_camera_points[:, 1] / safe_depth
            + target_view.intrinsics[1, 2]
        )
        finite_projection = (
            np.isfinite(projected_depth)
            & np.isfinite(normalized_x)
            & np.isfinite(normalized_y)
        )
        target_x = np.rint(
            np.where(finite_projection, normalized_x * width - 0.5, 0.0)
        ).astype(np.int64)
        target_y = np.rint(
            np.where(finite_projection, normalized_y * height - 0.5, 0.0)
        ).astype(np.int64)
        in_frame = (
            finite_projection
            & (projected_depth > 0.0)
            & (target_x >= 0)
            & (target_x < width)
            & (target_y >= 0)
            & (target_y < height)
        )
        safe_x = np.clip(target_x, 0, width - 1)
        safe_y = np.clip(target_y, 0, height - 1)
        target_flat_index = safe_y * width + safe_x
        matched_depth = target_depth[target_flat_index]
        matched_confidence = opacities[target_index, target_flat_index]
        matched_valid = (
            target_view.valid_mask.reshape(-1).cpu().numpy()[target_flat_index]
        )
        relative_scale = np.maximum(np.maximum(projected_depth, matched_depth), 1e-6)
        consistent = (
            in_frame
            & np.isfinite(matched_depth)
            & (matched_depth > 0.0)
            & matched_valid
            & (matched_confidence >= float(opacity_threshold) * float(target_opacity_ratio))
            & (
                np.abs(projected_depth - matched_depth)
                <= float(consistency_rel_error) * relative_scale
            )
        )
        mask &= consistent

        colors = (
            source_view.image.permute(1, 2, 0)
            .reshape(-1, 3)
            .mul(255.0)
            .round()
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        accepted_xyz.append(source_points[mask].astype(np.float64))
        accepted_rgb.append(colors[mask])
        accepted_confidence.append(source_confidence[mask].astype(np.float32))
        stats[f"accepted_view_{source_index}"] = int(np.count_nonzero(mask))

    xyz = np.concatenate(accepted_xyz, axis=0)
    rgb = np.concatenate(accepted_rgb, axis=0)
    confidence = np.concatenate(accepted_confidence, axis=0)
    stats["accepted_points"] = int(len(xyz))
    return xyz, rgb, confidence, stats


class VoxelPointMerger:
    """Giu nguyen sparse point goc va chen MVSplat point tot nhat moi voxel."""

    def __init__(
        self,
        original_points: dict[int, Point3D],
        voxel_divisor: float = 6_000.0,
        max_points: int = 600_000,
        bounds_margin: float = 0.25,
    ) -> None:
        if float(voxel_divisor) <= 0.0:
            raise ValueError("mvsplat_voxel_divisor phai duong")
        if int(max_points) <= 0:
            raise ValueError("mvsplat_max_points phai duong")
        self.original_xyz = np.stack(
            [original_points[key].xyz for key in sorted(original_points)]
        ).astype(np.float64)
        self.original_rgb = np.stack(
            [original_points[key].rgb for key in sorted(original_points)]
        ).astype(np.uint8)
        if len(self.original_xyz) > int(max_points):
            raise ValueError(
                f"mvsplat_max_points={max_points} nho hon sparse goc={len(self.original_xyz)}"
            )
        self.robust_min, self.robust_max, self.scene_diagonal = _robust_scene_bounds(
            original_points
        )
        self.voxel_size = self.scene_diagonal / float(voxel_divisor)
        self.bounds_min = self.robust_min - float(bounds_margin) * self.scene_diagonal
        self.bounds_max = self.robust_max + float(bounds_margin) * self.scene_diagonal
        self.new_limit = int(max_points) - len(self.original_xyz)
        self.original_voxels = {
            self._voxel_key(point) for point in self.original_xyz
        }
        self.candidates: dict[
            tuple[int, int, int], tuple[float, int, np.ndarray, np.ndarray]
        ] = {}
        self.sequence = 0
        self.stats = {
            "input_mvsplat_points": 0,
            "outside_scene_bounds": 0,
            "overlap_original_voxel": 0,
            "replaced_duplicate_voxel": 0,
        }

    def _voxel_key(self, point: np.ndarray) -> tuple[int, int, int]:
        index = np.floor((point - self.robust_min) / self.voxel_size).astype(np.int64)
        return int(index[0]), int(index[1]), int(index[2])

    def _prune(self) -> None:
        if self.new_limit <= 0:
            self.candidates.clear()
            return
        ranked = sorted(
            self.candidates.items(),
            key=lambda item: (item[1][0], -item[1][1]),
            reverse=True,
        )[: self.new_limit]
        self.candidates = dict(ranked)

    def add(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if not (len(xyz) == len(rgb) == len(confidence)):
            raise ValueError("MVSplat xyz/rgb/confidence khong khop")
        self.stats["input_mvsplat_points"] += int(len(xyz))
        for point, color, score in zip(xyz, rgb, confidence):
            self.sequence += 1
            if (
                not np.isfinite(point).all()
                or np.any(point < self.bounds_min)
                or np.any(point > self.bounds_max)
            ):
                self.stats["outside_scene_bounds"] += 1
                continue
            key = self._voxel_key(point)
            if key in self.original_voxels:
                self.stats["overlap_original_voxel"] += 1
                continue
            current = self.candidates.get(key)
            candidate = (float(score), self.sequence, point.copy(), color.copy())
            if current is None or (candidate[0], -candidate[1]) > (current[0], -current[1]):
                if current is not None:
                    self.stats["replaced_duplicate_voxel"] += 1
                self.candidates[key] = candidate
        if self.new_limit > 0 and len(self.candidates) > 2 * self.new_limit:
            self._prune()

    def finalize(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self._prune()
        values = sorted(
            self.candidates.values(),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        if values:
            new_xyz = np.stack([item[2] for item in values]).astype(np.float64)
            new_rgb = np.stack([item[3] for item in values]).astype(np.uint8)
            merged_xyz = np.concatenate([self.original_xyz, new_xyz], axis=0)
            merged_rgb = np.concatenate([self.original_rgb, new_rgb], axis=0)
        else:
            merged_xyz = self.original_xyz.copy()
            merged_rgb = self.original_rgb.copy()
        result_stats = {
            **self.stats,
            "original_points": int(len(self.original_xyz)),
            "added_voxel_points": int(len(values)),
            "merged_points": int(len(merged_xyz)),
            "growth_ratio": float(len(values) / max(len(self.original_xyz), 1)),
            "voxel_size": float(self.voxel_size),
            "robust_scene_diagonal": float(self.scene_diagonal),
        }
        return merged_xyz, merged_rgb, result_stats


def initialize_scene(
    scene_path: str | Path,
    encoder: torch.nn.Module,
    encoder_global_step: int,
    *,
    device: str = "cuda",
    mixed_precision: str = "none",
    image_size: int = 256,
    focal_scale: float = 1.0,
    max_pairs: int = 32,
    min_shared_tracks: int = 24,
    min_baseline_ratio: float = 0.005,
    max_baseline_ratio: float = 0.30,
    target_baseline_ratio: float = 0.05,
    opacity_threshold: float = 0.35,
    consistency_rel_error: float = 0.15,
    voxel_divisor: float = 6_000.0,
    max_points: int = 600_000,
    bounds_margin: float = 0.25,
    checkpoint_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Chay MVSplat theo cap, hop nhat point va thay PLY khoi tao cua mot scene."""
    scene_path = Path(scene_path)
    image_dir = scene_path / "images"
    sparse_dir = scene_path / "sparse" / "0"
    metadata_path = scene_path / VAI_METADATA_FILENAME
    for required in (
        image_dir,
        sparse_dir / "cameras.bin",
        sparse_dir / "images.bin",
        sparse_dir / "points3D.bin",
        metadata_path,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Scene MVSplat-init thieu: {required}")
    metadata = load_vai_metadata(scene_path)
    if bool(metadata.get("native_simple_radial", False)):
        raise ValueError(
            "MVSplat-init ImprovedGS thuan khong nhan scene native SIMPLE_RADIAL"
        )
    if metadata.get("fixed_pose_retriangulation", {}).get("enabled"):
        raise ValueError("Khong ket hop P1 voi MVSplat-init trong cung mot ablation")
    undistorted_camera = metadata.get("undistorted_camera", {})
    if undistorted_camera.get("model") not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError(
            "Scene MVSplat-init phai den tu main preprocess PINHOLE RGBA"
        )
    output_ply = sparse_dir / "points3D.ply"
    if output_ply.exists() and not overwrite:
        raise FileExistsError(f"{output_ply} da ton tai; dung --overwrite de tao lai")

    cameras = read_intrinsics_binary(sparse_dir / "cameras.bin")
    if len(cameras) != 1:
        raise ValueError(f"VAI MVSplat-init yeu cau mot camera, nhan {len(cameras)}")
    camera = next(iter(cameras.values()))
    if camera.model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError(
            f"Camera MVSplat-init phai la PINHOLE, nhan duoc {camera.model}"
        )
    images = read_extrinsics_binary(sparse_dir / "images.bin")
    images = {
        key: image
        for key, image in images.items()
        if (image_dir / image.name).is_file()
    }
    points = read_points3d_binary(sparse_dir / "points3D.bin")
    _, _, scene_diagonal = _robust_scene_bounds(points)
    pairs = select_context_pairs(
        images,
        points,
        max_pairs=max_pairs,
        min_shared_tracks=min_shared_tracks,
        min_baseline_ratio=min_baseline_ratio,
        max_baseline_ratio=max_baseline_ratio,
        target_baseline_ratio=target_baseline_ratio,
    )
    merger = VoxelPointMerger(
        points,
        voxel_divisor=voxel_divisor,
        max_points=max_points,
        bounds_margin=bounds_margin,
    )
    prepared: dict[int, PreparedView] = {}
    bounds: dict[int, tuple[float, float]] = {}
    pair_results: list[dict[str, Any]] = []

    def get_view(image_id: int) -> PreparedView:
        if image_id not in prepared:
            image_record = images[image_id]
            if image_id not in bounds:
                bounds[image_id] = estimate_depth_bounds(
                    image_record,
                    points,
                    scene_diagonal,
                )
            near, far = bounds[image_id]
            prepared[image_id] = prepare_pinhole_view(
                image_dir / image_record.name,
                camera,
                image_record,
                near,
                far,
                image_size=image_size,
                focal_scale=focal_scale,
            )
        return prepared[image_id]

    for pair_index, pair in enumerate(pairs, start=1):
        first = get_view(pair.first_id)
        second = get_view(pair.second_id)
        print(
            "  MVSplat pair {}/{}: {} + {} (shared={}, baseline={:.4f})".format(
                pair_index,
                len(pairs),
                first.image_name,
                second.image_name,
                pair.shared_tracks,
                pair.baseline_ratio,
            ),
            flush=True,
        )
        means, opacities = infer_pair(
            encoder,
            first,
            second,
            encoder_global_step,
            device=device,
            mixed_precision=mixed_precision,
        )
        xyz, rgb, confidence, filter_stats = filter_pair_predictions(
            means,
            opacities,
            (first, second),
            opacity_threshold=opacity_threshold,
            consistency_rel_error=consistency_rel_error,
        )
        merger.add(xyz, rgb, confidence)
        pair_results.append(
            {
                "first": first.image_name,
                "second": second.image_name,
                "shared_tracks": pair.shared_tracks,
                "baseline_ratio": pair.baseline_ratio,
                "near_far": [
                    [first.near, first.far],
                    [second.near, second.far],
                ],
                **filter_stats,
            }
        )
        print(
            f"    accepted={filter_stats['accepted_points']}/"
            f"{filter_stats['predicted_points']}",
            flush=True,
        )

    merged_xyz, merged_rgb, merge_stats = merger.finalize()
    if merge_stats["added_voxel_points"] <= 0:
        raise RuntimeError(
            "MVSplat-init khong them duoc point nao sau loc. "
            "Kiem tra camera/near-far hoac giam opacity/consistency threshold."
        )
    temporary_ply = output_ply.with_suffix(output_ply.suffix + ".tmp")
    write_point_cloud_ply(temporary_ply, merged_xyz, merged_rgb)
    os.replace(temporary_ply, output_ply)

    mvsplat_metadata = {
        "enabled": True,
        "mode": "geometry_only",
        "upstream": MVSPLAT_UPSTREAM,
        "tested_commit": MVSPLAT_TESTED_COMMIT,
        "checkpoint_sha256": checkpoint_sha256,
        "encoder_global_step": int(encoder_global_step),
        "image_size": int(image_size),
        "focal_scale": float(focal_scale),
        "max_pairs": int(max_pairs),
        "selected_pairs": int(len(pairs)),
        "min_shared_tracks": int(min_shared_tracks),
        "min_baseline_ratio": float(min_baseline_ratio),
        "max_baseline_ratio": float(max_baseline_ratio),
        "target_baseline_ratio": float(target_baseline_ratio),
        "opacity_threshold": float(opacity_threshold),
        "consistency_rel_error": float(consistency_rel_error),
        "voxel_divisor": float(voxel_divisor),
        "max_points": int(max_points),
        "bounds_margin": float(bounds_margin),
        "mixed_precision": mixed_precision,
        **merge_stats,
        "pairs": pair_results,
    }
    metadata["mvsplat_init"] = mvsplat_metadata
    save_json(metadata_path, metadata)
    return {
        "scene_name": scene_path.name,
        "output_ply": str(output_ply),
        **{key: value for key, value in merge_stats.items() if key != "pairs"},
        "selected_pairs": int(len(pairs)),
    }


def initialize_dataset(
    data_root: str | Path,
    subset: list[str],
    mvsplat_repo: str | Path,
    checkpoint_path: str | Path,
    **options: Any,
) -> list[dict[str, Any]]:
    """Nap encoder mot lan va khoi tao tat ca scene duoc chon."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Khong tim thay data root: {data_root}")
    scene_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    requested = set(subset)
    missing = sorted(requested - {path.name for path in scene_dirs})
    if missing:
        raise ValueError(f"Khong tim thay scene da preprocess: {missing}")
    selected = [path for path in scene_dirs if not requested or path.name in requested]
    if not selected:
        raise ValueError(f"Khong co scene nao trong {data_root}")

    checkpoint_digest = sha256_file(checkpoint_path)
    expected_digest = options.pop("expected_checkpoint_sha256", None)
    if expected_digest and checkpoint_digest.lower() != str(expected_digest).lower():
        raise ValueError(
            f"SHA256 checkpoint sai: {checkpoint_digest} != {expected_digest}"
        )
    device = str(options.get("device", "cuda"))
    encoder, global_step = load_mvsplat_encoder(
        mvsplat_repo,
        checkpoint_path,
        device=device,
    )
    results = []
    for scene_dir in selected:
        print(f"MVSplat-init scene {scene_dir.name}...", flush=True)
        result = initialize_scene(
            scene_dir,
            encoder,
            global_step,
            checkpoint_sha256=checkpoint_digest,
            **options,
        )
        results.append(result)
        print(
            "  OK: original={} + mvsplat={} -> {} points".format(
                result["original_points"],
                result["added_voxel_points"],
                result["merged_points"],
            ),
            flush=True,
        )
    return results
