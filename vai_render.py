"""CLI render va danh gia test pose VAI bang model ImprovedGS."""
from __future__ import annotations

from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args, parse_bool_arg
from utils.general_utils import safe_state


def main() -> int:
    """Nap tham so model va chay VAI postprocess."""
    parser = ArgumentParser(description="Render VAI test poses")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--output_root", default="")
    parser.add_argument("--eval_root", default="")
    parser.add_argument("--output_extension", default="csv")
    parser.add_argument(
        "--redistort_interpolation",
        default="bicubic",
        choices=("bilinear", "bicubic"),
    )
    parser.add_argument("--sharpen_amount", default=1.0, type=float)
    parser.add_argument("--sharpen_sigma", default=0.6, type=float)
    parser.add_argument("--jpeg_quality", default=95, type=int)
    parser.add_argument("--jpeg_subsampling", default=2, type=int, choices=(0, 1, 2))
    parser.add_argument("--evaluate", default=True, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--require_gt", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--overwrite", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--psnr_max", default=40.0, type=float)
    parser.add_argument("--lpips_net", default="alex", choices=("alex", "squeeze", "vgg"))
    parser.add_argument("--quiet", default=False, nargs="?", const=True, type=parse_bool_arg)
    args = get_combined_args(parser)
    safe_state(args.quiet)
    # Import tre de --help khong yeu cau CUDA extension da duoc cai.
    from vai.rendering import render_vai_scene

    render_vai_scene(
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        iteration=args.iteration,
        scene_name=args.scene_name,
        output_root=args.output_root,
        eval_root=args.eval_root,
        output_extension=args.output_extension,
        redistort_interpolation=args.redistort_interpolation,
        sharpen_amount=args.sharpen_amount,
        sharpen_sigma=args.sharpen_sigma,
        jpeg_quality=args.jpeg_quality,
        jpeg_subsampling=args.jpeg_subsampling,
        evaluate=args.evaluate,
        require_gt=args.require_gt,
        overwrite=args.overwrite,
        psnr_max=args.psnr_max,
        lpips_net=args.lpips_net,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
