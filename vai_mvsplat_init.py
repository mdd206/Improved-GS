"""CLI khoi tao point cloud VAI bang MVSplat truoc khi train ImprovedGS thuan."""
from __future__ import annotations

import argparse
import json

from vai.mvsplat_init import initialize_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MVSplat geometry-only initializer for main ImprovedGS",
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--mvsplat_repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument("--subset", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mixed_precision",
        choices=("none", "fp16", "bf16"),
        default="none",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--focal_scale", type=float, default=1.0)
    parser.add_argument("--max_pairs", type=int, default=32)
    parser.add_argument("--min_shared_tracks", type=int, default=24)
    parser.add_argument("--min_baseline_ratio", type=float, default=0.005)
    parser.add_argument("--max_baseline_ratio", type=float, default=0.30)
    parser.add_argument("--target_baseline_ratio", type=float, default=0.05)
    parser.add_argument("--opacity_threshold", type=float, default=0.35)
    parser.add_argument("--consistency_rel_error", type=float, default=0.15)
    parser.add_argument("--voxel_divisor", type=float, default=6_000.0)
    parser.add_argument("--max_points", type=int, default=600_000)
    parser.add_argument("--bounds_margin", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = initialize_dataset(
        data_root=args.data_root,
        subset=args.subset,
        mvsplat_repo=args.mvsplat_repo,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
        mixed_precision=args.mixed_precision,
        image_size=args.image_size,
        focal_scale=args.focal_scale,
        max_pairs=args.max_pairs,
        min_shared_tracks=args.min_shared_tracks,
        min_baseline_ratio=args.min_baseline_ratio,
        max_baseline_ratio=args.max_baseline_ratio,
        target_baseline_ratio=args.target_baseline_ratio,
        opacity_threshold=args.opacity_threshold,
        consistency_rel_error=args.consistency_rel_error,
        voxel_divisor=args.voxel_divisor,
        max_points=args.max_points,
        bounds_margin=args.bounds_margin,
        overwrite=args.overwrite,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
