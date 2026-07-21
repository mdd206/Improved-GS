"""CLI danh gia lai mot thu muc render VAI ma khong train lai."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vai.common import load_vai_metadata, read_pose_rows, save_json
from vai.evaluation import evaluate_rendered_scene


def main() -> int:
    """Doc scene da preprocess va tinh metric cho anh render."""
    parser = argparse.ArgumentParser(description="Evaluate VAI rendered images")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--output_extension", default="csv")
    parser.add_argument("--psnr_max", type=float, default=40.0)
    parser.add_argument("--lpips_net", choices=("alex", "squeeze", "vgg"), default="alex")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source_path = Path(args.source_path)
    metadata = load_vai_metadata(source_path)
    pose_rows = read_pose_rows(source_path / metadata.get("test_poses", "test/test_poses.csv"))
    summary, per_view = evaluate_rendered_scene(
        gt_dir=source_path / metadata.get("test_images", "test/images"),
        render_dir=args.render_dir,
        pose_rows=pose_rows,
        output_extension=args.output_extension,
        psnr_max=args.psnr_max,
        lpips_net=args.lpips_net,
        device=args.device,
    )
    result = {"scene_name": metadata.get("scene_name", source_path.name), **summary, "per_view": per_view}
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
