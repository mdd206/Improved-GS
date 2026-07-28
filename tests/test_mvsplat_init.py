"""Kiem thu phan MVSplat-init khong can checkpoint hoac GPU."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image as PilImage

from vai.colmap_io import (
    Camera,
    Image,
    Point3D,
    write_extrinsics_binary,
    write_intrinsics_binary,
    write_points3d_binary,
)
from vai.mvsplat_init import (
    PreparedView,
    VoxelPointMerger,
    filter_pair_predictions,
    initialize_scene,
    select_context_pairs,
)


def make_point(
    point_id: int,
    xyz: tuple[float, float, float],
    image_ids: tuple[int, ...],
    rgb: tuple[int, int, int] = (128, 128, 128),
) -> Point3D:
    return Point3D(
        id=point_id,
        xyz=np.asarray(xyz, dtype=np.float64),
        rgb=np.asarray(rgb, dtype=np.uint8),
        error=0.5,
        image_ids=np.asarray(image_ids, dtype=np.int32),
        point2D_idxs=np.arange(len(image_ids), dtype=np.int32),
    )


def make_image(
    image_id: int,
    name: str,
    center_x: float,
    point_ids: tuple[int, ...],
) -> Image:
    return Image(
        id=image_id,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        tvec=np.asarray([-center_x, 0.0, 0.0], dtype=np.float64),
        camera_id=1,
        name=name,
        xys=np.zeros((len(point_ids), 2), dtype=np.float64),
        point3D_ids=np.asarray(point_ids, dtype=np.int64),
    )


def make_view(
    image_id: int,
    image_name: str,
    c2w: np.ndarray,
    height: int = 2,
    width: int = 2,
) -> PreparedView:
    intrinsics = np.asarray(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return PreparedView(
        image_id=image_id,
        image_name=image_name,
        image=torch.ones((3, height, width), dtype=torch.float32) * 0.5,
        valid_mask=torch.ones((height, width), dtype=torch.bool),
        c2w=c2w.astype(np.float32),
        w2c=np.linalg.inv(c2w).astype(np.float32),
        intrinsics=intrinsics,
        near=1.0,
        far=3.0,
    )


class PairSelectionTests(unittest.TestCase):
    def test_pairs_cover_ordered_trajectory_and_require_shared_tracks(self) -> None:
        points = {
            1: make_point(1, (-1.0, -0.2, 3.0), (1, 2)),
            2: make_point(2, (-0.5, 0.1, 3.1), (1, 2)),
            3: make_point(3, (0.0, -0.1, 3.2), (1, 2, 3)),
            4: make_point(4, (0.5, 0.2, 3.3), (1, 2, 3)),
            5: make_point(5, (1.0, -0.2, 3.4), (3, 4)),
            6: make_point(6, (1.5, 0.1, 3.5), (3, 4)),
        }
        images = {
            1: make_image(1, "000.JPG", 0.00, (1, 2, 3, 4)),
            2: make_image(2, "001.JPG", 0.10, (1, 2, 3, 4)),
            3: make_image(3, "002.JPG", 0.20, (3, 4, 5, 6)),
            4: make_image(4, "003.JPG", 0.30, (5, 6)),
        }

        pairs = select_context_pairs(
            images,
            points,
            max_pairs=2,
            min_shared_tracks=2,
            min_baseline_ratio=0.001,
            max_baseline_ratio=0.5,
            target_baseline_ratio=0.04,
        )

        self.assertEqual(len(pairs), 2)
        selected_images = {
            image_id
            for pair in pairs
            for image_id in (pair.first_id, pair.second_id)
        }
        self.assertIn(1, selected_images)
        self.assertIn(4, selected_images)
        self.assertTrue(all(pair.shared_tracks >= 2 for pair in pairs))


class ConsistencyFilterTests(unittest.TestCase):
    def test_cross_view_depth_rejects_outlier(self) -> None:
        c2w = np.eye(4, dtype=np.float32)
        views = (make_view(1, "a.JPG", c2w), make_view(2, "b.JPG", c2w))
        pixel_centers = (
            (-0.5, -0.5, 2.0),
            (0.5, -0.5, 2.0),
            (-0.5, 0.5, 2.0),
            (0.5, 0.5, 2.0),
        )
        means = np.asarray([pixel_centers, pixel_centers], dtype=np.float32)
        opacities = np.full((2, 4), 0.9, dtype=np.float32)

        xyz, rgb, confidence, stats = filter_pair_predictions(
            means,
            opacities,
            views,
            opacity_threshold=0.35,
            consistency_rel_error=0.05,
        )
        self.assertEqual(len(xyz), 8)
        self.assertEqual(rgb.shape, (8, 3))
        self.assertEqual(confidence.shape, (8,))

        outlier_means = means.copy()
        outlier_means[1, 0, 2] = 2.8
        filtered_xyz, _, _, filtered_stats = filter_pair_predictions(
            outlier_means,
            opacities,
            views,
            opacity_threshold=0.35,
            consistency_rel_error=0.05,
        )
        self.assertLess(len(filtered_xyz), len(xyz))
        self.assertEqual(
            filtered_stats["accepted_points"],
            len(filtered_xyz),
        )
        self.assertEqual(stats["predicted_points"], 8)


class VoxelMergeTests(unittest.TestCase):
    def test_original_points_are_anchors_and_best_new_confidence_wins(self) -> None:
        original = {
            1: make_point(1, (0.0, 0.0, 0.0), (1, 2), (255, 0, 0)),
            2: make_point(2, (10.0, 10.0, 10.0), (1, 2), (0, 255, 0)),
        }
        merger = VoxelPointMerger(
            original,
            voxel_divisor=10.0,
            max_points=4,
            bounds_margin=0.5,
        )
        merger.add(
            np.asarray(
                [
                    [0.01, 0.01, 0.01],
                    [4.0, 4.0, 4.0],
                    [4.1, 4.1, 4.1],
                    [7.0, 7.0, 7.0],
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    [1, 1, 1],
                    [10, 20, 30],
                    [40, 50, 60],
                    [70, 80, 90],
                ],
                dtype=np.uint8,
            ),
            np.asarray([0.99, 0.4, 0.9, 0.8], dtype=np.float32),
        )
        xyz, rgb, stats = merger.finalize()

        self.assertEqual(stats["original_points"], 2)
        self.assertEqual(stats["added_voxel_points"], 2)
        self.assertEqual(stats["merged_points"], 4)
        np.testing.assert_allclose(xyz[:2], np.asarray([[0, 0, 0], [10, 10, 10]]))
        colors = {tuple(color.tolist()) for color in rgb}
        self.assertIn((255, 0, 0), colors)
        self.assertIn((0, 255, 0), colors)
        self.assertIn((40, 50, 60), colors)
        self.assertIn((70, 80, 90), colors)
        self.assertNotIn((1, 1, 1), colors)


class SceneIntegrationTests(unittest.TestCase):
    def test_scene_initializer_writes_merged_ply_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene = Path(temp_dir) / "HCM0204"
            image_dir = scene / "images"
            sparse_dir = scene / "sparse" / "0"
            image_dir.mkdir(parents=True)
            sparse_dir.mkdir(parents=True)
            PilImage.new("RGB", (8, 8), (100, 120, 140)).save(image_dir / "a.JPG")
            PilImage.new("RGB", (8, 8), (110, 130, 150)).save(image_dir / "b.JPG")
            camera = Camera(
                id=1,
                model="SIMPLE_RADIAL",
                width=8,
                height=8,
                params=np.asarray([8.0, 4.0, 4.0, 0.0], dtype=np.float64),
            )
            write_intrinsics_binary({1: camera}, sparse_dir / "cameras.bin")
            points = {
                index: make_point(
                    index,
                    (x, y, z),
                    (1, 2),
                    (20 * index, 30, 40),
                )
                for index, (x, y, z) in enumerate(
                    (
                        (-0.8, -0.6, 2.2),
                        (0.8, -0.6, 2.4),
                        (-0.8, 0.6, 2.6),
                        (0.8, 0.6, 2.8),
                    ),
                    start=1,
                )
            }
            images = {
                1: make_image(1, "a.JPG", 0.0, tuple(points)),
                2: make_image(2, "b.JPG", 0.1, tuple(points)),
            }
            write_extrinsics_binary(images, sparse_dir / "images.bin")
            write_points3d_binary(points, sparse_dir / "points3D.bin")
            with open(scene / "vai_metadata.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "format_version": 1,
                        "native_simple_radial": True,
                        "fixed_pose_retriangulation": {"enabled": False},
                    },
                    handle,
                )

            def fake_infer(
                encoder,
                first,
                second,
                global_step,
                device="cuda",
                mixed_precision="none",
            ):
                del encoder, global_step, device, mixed_precision
                means = []
                for view in (first, second):
                    height, width = view.image.shape[-2:]
                    ys, xs = np.meshgrid(
                        (np.arange(height) + 0.5) / height,
                        (np.arange(width) + 0.5) / width,
                        indexing="ij",
                    )
                    coordinates = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
                    camera_rays = coordinates @ np.linalg.inv(view.intrinsics).T
                    camera_points = camera_rays.reshape(-1, 3) * 2.5
                    world_points = (
                        camera_points @ view.c2w[:3, :3].T + view.c2w[:3, 3]
                    )
                    means.append(world_points.astype(np.float32))
                return np.stack(means), np.full((2, 64), 0.9, dtype=np.float32)

            with patch("vai.mvsplat_init.infer_pair", side_effect=fake_infer):
                result = initialize_scene(
                    scene,
                    encoder=object(),
                    encoder_global_step=300_000,
                    device="cpu",
                    image_size=8,
                    max_pairs=1,
                    min_shared_tracks=2,
                    min_baseline_ratio=0.001,
                    max_baseline_ratio=0.5,
                    target_baseline_ratio=0.05,
                    consistency_rel_error=0.20,
                    voxel_divisor=100.0,
                    max_points=100,
                    bounds_margin=2.0,
                    checkpoint_sha256="test",
                    overwrite=True,
                )

            self.assertGreater(result["added_voxel_points"], 0)
            with open(sparse_dir / "points3D.ply", "rb") as handle:
                header_lines = []
                while True:
                    line = handle.readline().decode("ascii").strip()
                    header_lines.append(line)
                    if line == "end_header":
                        break
            vertex_line = next(
                line for line in header_lines if line.startswith("element vertex ")
            )
            self.assertEqual(
                int(vertex_line.rsplit(" ", 1)[-1]),
                result["merged_points"],
            )
            with open(scene / "vai_metadata.json", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertTrue(metadata["mvsplat_init"]["enabled"])
            self.assertEqual(metadata["mvsplat_init"]["mode"], "geometry_only")
            self.assertEqual(metadata["mvsplat_init"]["selected_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
