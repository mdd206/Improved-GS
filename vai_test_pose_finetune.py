"""CLI for independent per-test-pose ImprovedGS fine-tuning."""
from __future__ import annotations

from argparse import ArgumentParser

from arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    parse_bool_arg,
)
from utils.general_utils import safe_state


def build_parser() -> tuple[
    ArgumentParser,
    ModelParams,
    OptimizationParams,
    PipelineParams,
]:
    parser = ArgumentParser(
        description=(
            "Load one base ImprovedGS PLY independently per test pose, rank all "
            "train views, fine-tune on top-K, and render that pose."
        )
    )
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--base_iteration", default=30_000, type=int)
    parser.add_argument("--test_poses", default="")
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--pose_start_index", default=0, type=int)
    parser.add_argument("--pose_count", default=-1, type=int)
    parser.add_argument("--fine_tune_steps", default=3_000, type=int)
    parser.add_argument("--split_from_step", default=0, type=int)
    parser.add_argument("--split_until_step", default=1_500, type=int)
    parser.add_argument("--top_k", default=25, type=int)
    parser.add_argument("--sigma_multiplier", default=3.0, type=float)
    parser.add_argument(
        "--save_pose_models",
        default=False,
        nargs="?",
        const=True,
        type=parse_bool_arg,
    )
    parser.add_argument("--output_root", default="")
    parser.add_argument("--output_extension", default="csv")
    parser.add_argument(
        "--save_png",
        default=False,
        nargs="?",
        const=True,
        type=parse_bool_arg,
    )
    parser.add_argument("--png_root", default="")
    parser.add_argument(
        "--redistort_interpolation",
        default="bicubic",
        choices=("bilinear", "bicubic"),
    )
    parser.add_argument("--sharpen_amount", default=1.0, type=float)
    parser.add_argument("--sharpen_sigma", default=0.6, type=float)
    parser.add_argument("--jpeg_quality", default=95, type=int)
    parser.add_argument(
        "--jpeg_subsampling",
        default=2,
        type=int,
        choices=(0, 1, 2),
    )
    parser.add_argument(
        "--evaluate",
        default=True,
        nargs="?",
        const=True,
        type=parse_bool_arg,
    )
    parser.add_argument(
        "--require_gt",
        default=False,
        nargs="?",
        const=True,
        type=parse_bool_arg,
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        nargs="?",
        const=True,
        type=parse_bool_arg,
    )
    parser.add_argument(
        "--resume",
        default=False,
        nargs="?",
        const=True,
        type=parse_bool_arg,
        help=(
            "Giu PNG/manifest da hoan tat va bo qua cac pose hop le khi chay lai."
        ),
    )
    parser.add_argument("--psnr_max", default=40.0, type=float)
    parser.add_argument(
        "--lpips_net",
        default="alex",
        choices=("alex", "squeeze", "vgg"),
    )
    parser.add_argument("--progress_bar_width", default=100, type=int)
    parser.add_argument("--empty_cache_interval", default=200, type=int)
    parser.add_argument(
        "--quiet",
        default=False,
        nargs="?",
        const=True,
        type=parse_bool_arg,
    )
    return parser, model, optimization, pipeline


def main(argv: list[str] | None = None) -> int:
    parser, model, optimization, pipeline = build_parser()
    args = parser.parse_args(argv)
    safe_state(bool(args.quiet))

    if str(args.source_path).strip() == "":
        parser.error("--source_path is required.")
    if str(args.model_path).strip() == "":
        parser.error("--model_path is required as the output root.")
    dataset = model.extract(args)

    # Import after argument parsing so `--help` stays usable without CUDA builds.
    from vai.pose_finetuning import run_test_pose_finetuning

    run_test_pose_finetuning(
        dataset=dataset,
        opt=optimization.extract(args),
        pipeline=pipeline.extract(args),
        runtime_args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
