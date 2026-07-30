"""Independent ImprovedGS fine-tuning for every VAI test pose."""
from __future__ import annotations

import copy
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from gaussian_renderer import render
from scene import Scene
from scene.methods.densification_stage import run_3dgs_densification_method
from scene.methods.initialization_3dgs import build_gaussian_model_3dgs
from scene.methods.optimization_methods import run_3dgs_optimization_method
from scene.methods.pruning_methods import run_3dgs_pruning_method
from scene.training_context import (
    attach_scene_and_gaussians_to_context,
    build_training_context,
)
from scene.training_loop import (
    run_3dgs_parameter_update_method,
    should_run_parameter_update,
    update_3dgs_training_schedule,
)
from scene.training_runtime import (
    build_training_loop_state,
    prepare_output_and_logger,
    synchronized_timestamp,
)
from utils.experiment_utils import finalize_training_parameters
from utils.general_utils import searchForMaxIteration
from utils.pose_aware_sampling import pose_from_csv_row
from utils.test_pose_finetune import (
    configure_local_improvedgs_options,
    pose_output_directory_name,
    select_top_training_views,
    validate_finetune_schedule,
)
from vai.common import (
    load_vai_metadata,
    output_name_for_pose,
    read_pose_rows,
    save_json,
    slice_pose_rows,
)
from vai.distortion import redistort_and_crop
from vai.evaluation import evaluate_rendered_scene
from vai.finetune_progress import (
    load_json_object,
    save_progress,
    valid_completed_pose_result,
    validate_resume_manifest,
)
from vai.image_processing import save_render_image, sharpen_image
from vai.rendering import (
    _original_radial_camera,
    _prepare_scene_output,
    _single_undistorted_camera,
    _validate_pose_intrinsics,
    camera_from_pose_row,
)


def resolve_base_iteration(base_model_path: str | Path, iteration: int) -> int:
    """Resolve and validate the immutable source PLY iteration."""
    base_model_path = Path(base_model_path)
    resolved_iteration = int(iteration)
    if resolved_iteration == -1:
        resolved_iteration = int(
            searchForMaxIteration(str(base_model_path / "point_cloud"))
        )
    if resolved_iteration < 1:
        raise ValueError("base_iteration must be a positive iteration or -1.")
    ply_path = (
        base_model_path
        / "point_cloud"
        / "iteration_{}".format(resolved_iteration)
        / "point_cloud.ply"
    )
    if not ply_path.is_file():
        raise FileNotFoundError("Could not find base model PLY: {}".format(ply_path))
    return resolved_iteration


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_path_overlap(output_path: Path, protected_paths: list[Path]) -> None:
    """Reject outputs that could overwrite or be written inside protected inputs."""
    resolved_output = output_path.resolve()
    for protected_path in protected_paths:
        resolved_protected = protected_path.resolve()
        if (
            resolved_output == resolved_protected
            or _path_contains(resolved_output, resolved_protected)
            or _path_contains(resolved_protected, resolved_output)
        ):
            raise ValueError(
                "Output path {} overlaps protected input {}".format(
                    resolved_output,
                    resolved_protected,
                )
            )


def _prepare_model_output_root(
    output_root: Path,
    base_model_path: Path,
    source_path: Path,
    overwrite: bool,
    preserve_existing: bool = False,
) -> None:
    """Create a clean output root without ever touching the source model."""
    resolved_output = output_root.resolve()
    _reject_path_overlap(output_root, [base_model_path, source_path])
    if output_root.exists() and not output_root.is_dir():
        raise FileExistsError(
            "Fine-tune output path is not a directory: {}".format(output_root)
        )
    if output_root.exists() and any(output_root.iterdir()):
        if preserve_existing:
            return
        if not overwrite:
            raise FileExistsError(
                "Fine-tune output already exists: {}. Use --overwrite true "
                "or choose another --model_path.".format(output_root)
            )
        if output_root.parent.resolve() == resolved_output:
            raise ValueError("Refusing to remove unsafe output root: {}".format(output_root))
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _copy_scene_reproducibility_files(source_scene: Scene, target_path: Path) -> None:
    """Reuse camera metadata without decoding all train images again."""
    source_path = Path(source_scene.model_path)
    for file_name in ("input.ply", "cameras.json"):
        source_file = source_path / file_name
        if not source_file.is_file():
            raise FileNotFoundError(
                "Missing reusable scene file: {}".format(source_file)
            )
        shutil.copy2(source_file, target_path / file_name)


def _load_fresh_base_model(
    dataset: Any,
    opt: Any,
    pose_model_path: Path,
    base_model_path: Path,
    base_iteration: int,
    camera_template: Scene | None,
    copy_scene_files: bool,
) -> tuple[Any, Scene, Scene]:
    """Load a fresh base PLY while reusing immutable train-camera objects."""
    gaussians = build_gaussian_model_3dgs(dataset, opt)
    if camera_template is None:
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=base_iteration,
            shuffle=False,
            resolution_scales=[1.0],
            load_model_path=str(base_model_path),
        )
        camera_template = scene
        return gaussians, scene, camera_template

    scene = copy.copy(camera_template)
    scene.model_path = str(pose_model_path)
    scene.load_model_path = str(base_model_path)
    scene.loaded_iter = int(base_iteration)
    scene.gaussians = gaussians
    base_ply = (
        base_model_path
        / "point_cloud"
        / "iteration_{}".format(base_iteration)
        / "point_cloud.ply"
    )
    gaussians.load_ply(str(base_ply), bool(dataset.train_test_exp))
    gaussians.spatial_lr_scale = float(scene.cameras_extent)
    gaussians.initialize_exposure_parameters(
        scene.getTrainCameras(1.0),
        str(base_model_path / "exposure.json"),
    )
    if copy_scene_files:
        _copy_scene_reproducibility_files(camera_template, pose_model_path)
    return gaussians, scene, camera_template


def _render_target_pose(
    row: dict[str, str],
    gaussians: Any,
    dataset: Any,
    pipeline: Any,
    background: torch.Tensor,
    undistorted_camera: dict[str, Any],
    radial_k: float,
    output_path: Path,
    png_output_path: Path | None,
    redistort_interpolation: str,
    sharpen_amount: float,
    sharpen_sigma: float,
    jpeg_quality: int,
    jpeg_subsampling: int,
) -> None:
    """Render and post-process exactly one pose with its dedicated model."""
    with torch.inference_mode():
        camera = camera_from_pose_row(row, undistorted_camera)
        rendering = render(
            camera,
            gaussians,
            pipeline,
            background,
            use_trained_exp=bool(dataset.train_test_exp),
            track_gradients=False,
            inference_only=True,
        )["render"]
        rendering = redistort_and_crop(
            rendering,
            focal=float(undistorted_camera["fx"]),
            render_cx=float(undistorted_camera["cx"]),
            render_cy=float(undistorted_camera["cy"]),
            radial_k=float(radial_k),
            target_cx=float(row["cx"]),
            target_cy=float(row["cy"]),
            target_width=int(float(row["width"])),
            target_height=int(float(row["height"])),
            interpolation=redistort_interpolation,
        )
        rendering = sharpen_image(
            rendering,
            amount=float(sharpen_amount),
            sigma=float(sharpen_sigma),
        )
        save_render_image(
            rendering,
            output_path,
            jpeg_quality=int(jpeg_quality),
            jpeg_subsampling=int(jpeg_subsampling),
        )
        if png_output_path is not None:
            save_render_image(rendering, png_output_path)


def _run_local_finetune(
    context: Any,
    gaussians: Any,
    selected_cameras: list[Any],
    fine_tune_steps: int,
    progress_bar_width: int,
    empty_cache_interval: int,
    progress_description: str,
) -> tuple[float, int]:
    """Run exactly `fine_tune_steps` optimizer updates on the selected pool."""
    gaussians.training_setup(context.opt)
    loop_state = build_training_loop_state(context, False, False)
    viewpoint_stack = loop_state["viewpoint_stack"]
    ema_loss = 0.0
    progress_bar = tqdm(
        total=int(fine_tune_steps),
        desc=progress_description,
        ncols=int(progress_bar_width),
        dynamic_ncols=False,
    )
    active_start = synchronized_timestamp()
    optimizer_updates = 0
    try:
        for local_step in range(1, int(fine_tune_steps) + 1):
            if (
                int(empty_cache_interval) > 0
                and local_step % int(empty_cache_interval) == 0
            ):
                torch.cuda.empty_cache()
            update_3dgs_training_schedule(context, gaussians, local_step)
            if not viewpoint_stack:
                viewpoint_stack.extend(selected_cameras)
            camera_index = int(
                torch.randint(0, len(viewpoint_stack), (1,)).item()
            )
            viewpoint_camera = viewpoint_stack.pop(camera_index)
            optimization_outputs = run_3dgs_optimization_method(
                context,
                gaussians,
                loop_state,
                local_step,
                viewpoint_camera,
            )
            loss = optimization_outputs["loss"]
            render_state = optimization_outputs["render_state"]
            with torch.no_grad():
                ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss
                will_update = should_run_parameter_update(
                    context.opt,
                    local_step,
                    str(context.method_config["training_method"]),
                    bool(context.method_config.get("use_mu", False)),
                )
                run_3dgs_parameter_update_method(
                    context,
                    gaussians,
                    local_step,
                    render_state["radii"],
                )
                if will_update:
                    optimizer_updates += 1
                run_3dgs_densification_method(
                    context,
                    gaussians,
                    local_step,
                    render_state,
                )
                run_3dgs_pruning_method(
                    context,
                    gaussians,
                    local_step,
                )
            if local_step % 10 == 0 or local_step == int(fine_tune_steps):
                progress_bar.set_postfix(
                    {
                        "Loss": "{:.4f}".format(ema_loss),
                        "N_GS": str(int(gaussians.get_xyz.shape[0])),
                    }
                )
            progress_bar.update(1)
    finally:
        progress_bar.close()
    if optimizer_updates != int(fine_tune_steps):
        raise RuntimeError(
            "Fine-tune phai co dung {} optimizer update, nhan duoc {}".format(
                fine_tune_steps,
                optimizer_updates,
            )
        )
    return max(synchronized_timestamp() - active_start, 0.0), optimizer_updates


def run_test_pose_finetuning(
    dataset: Any,
    opt: Any,
    pipeline: Any,
    runtime_args: Namespace,
) -> dict[str, Any]:
    """Fine-tune a fresh 30k model per pose, render, and optionally evaluate."""
    fine_tune_steps = int(runtime_args.fine_tune_steps)
    split_from_step = int(runtime_args.split_from_step)
    split_until_step = int(runtime_args.split_until_step)
    top_k = int(runtime_args.top_k)
    sigma_multiplier = float(runtime_args.sigma_multiplier)
    save_pose_models = bool(runtime_args.save_pose_models)
    resume_enabled = bool(getattr(runtime_args, "resume", False))
    validate_finetune_schedule(
        fine_tune_steps,
        split_from_step,
        split_until_step,
        top_k,
        sigma_multiplier,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Test-pose ImprovedGS fine-tuning requires a CUDA GPU.")
    if bool(dataset.train_test_exp):
        raise ValueError(
            "This VAI test-pose experiment requires train_test_exp=false because "
            "test pose image names do not have learned exposure rows."
        )

    source_path = Path(dataset.source_path)
    base_model_path = Path(runtime_args.base_model_path)
    output_root = Path(dataset.model_path)
    if str(dataset.model_path).strip() == "":
        raise ValueError("--model_path is required as the per-pose model output root.")
    base_iteration = resolve_base_iteration(
        base_model_path,
        int(runtime_args.base_iteration),
    )
    _prepare_model_output_root(
        output_root,
        base_model_path,
        source_path,
        bool(runtime_args.overwrite),
        preserve_existing=resume_enabled,
    )

    metadata = load_vai_metadata(source_path)
    scene_name = str(runtime_args.scene_name).strip() or str(
        metadata.get("scene_name") or source_path.name
    )
    configured_pose_path = str(runtime_args.test_poses).strip()
    pose_path = (
        Path(configured_pose_path)
        if configured_pose_path
        else source_path / metadata.get("test_poses", "test/test_poses.csv")
    )
    if configured_pose_path and not pose_path.is_absolute():
        pose_path = source_path / pose_path
    all_pose_rows = read_pose_rows(pose_path)
    pose_start_index = int(runtime_args.pose_start_index)
    pose_rows = slice_pose_rows(
        all_pose_rows,
        start_index=pose_start_index,
        pose_count=int(runtime_args.pose_count),
    )
    pose_end_index = pose_start_index + len(pose_rows)
    # Validate the requested extension before any per-pose GPU training starts.
    output_name_for_pose(
        pose_rows[0]["image_name"],
        str(runtime_args.output_extension),
    )
    undistorted_camera = _single_undistorted_camera(source_path)
    original_camera = _original_radial_camera(metadata)
    _validate_pose_intrinsics(pose_rows, original_camera)

    render_root = (
        Path(runtime_args.output_root)
        if str(runtime_args.output_root).strip()
        else output_root / "vai_submission"
    )
    png_root = (
        Path(runtime_args.png_root)
        if str(runtime_args.png_root).strip()
        else output_root / "vai_png"
    )
    if bool(runtime_args.save_png) and png_root.resolve() == render_root.resolve():
        raise ValueError("png_root must differ from output_root.")
    _reject_path_overlap(render_root, [base_model_path, source_path])
    if bool(runtime_args.save_png):
        _reject_path_overlap(png_root, [base_model_path, source_path])
    scene_render_path = _prepare_scene_output(
        render_root,
        scene_name,
        bool(runtime_args.overwrite),
        preserve_existing=resume_enabled,
    )
    scene_png_path = (
        _prepare_scene_output(
            png_root,
            scene_name,
            bool(runtime_args.overwrite),
            preserve_existing=resume_enabled,
        )
        if bool(runtime_args.save_png)
        else None
    )
    background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(
        background_color,
        dtype=torch.float32,
        device="cuda",
    )

    local_opt = configure_local_improvedgs_options(
        opt,
        fine_tune_steps,
        split_from_step,
        split_until_step,
    )
    manifest: dict[str, Any] = {
        "experiment": "independent_test_pose_finetuning",
        "method": "improvedgs",
        "scene_name": scene_name,
        "source_path": str(source_path),
        "base_model_path": str(base_model_path),
        "base_iteration": int(base_iteration),
        "saved_iteration": (
            int(base_iteration + fine_tune_steps) if save_pose_models else None
        ),
        "save_pose_models": save_pose_models,
        "fine_tune_steps": fine_tune_steps,
        "optimizer_updates_per_pose": fine_tune_steps,
        "split_from_step": split_from_step,
        "split_until_step": split_until_step,
        "budget_warmup": False,
        "coarse_to_fine": False,
        "pose_aware_sampling": False,
        "top_k": top_k,
        "sigma_multiplier": sigma_multiplier,
        "source_test_pose_count": len(all_pose_rows),
        "pose_start_index": pose_start_index,
        "pose_end_index_exclusive": pose_end_index,
        "test_pose_count": len(pose_rows),
        "pose_indices": list(range(pose_start_index, pose_end_index)),
        "render_dir": str(scene_render_path),
        "output_extension": str(runtime_args.output_extension),
        "resume_enabled": resume_enabled,
        "progress_file": str(
            render_root / "{}_finetune_progress.json".format(scene_name)
        ),
        "poses": [],
    }
    manifest_path = output_root / "test_pose_finetune_manifest.json"
    progress_paths = [
        output_root / "finetune_progress.json",
        render_root / "{}_finetune_progress.json".format(scene_name),
    ]

    # Khi resume, chi tin pose co manifest hop le va file anh doc duoc.
    if resume_enabled and manifest_path.is_file():
        previous_manifest = load_json_object(manifest_path)
        validate_resume_manifest(previous_manifest, manifest)
        previous_by_index = {
            int(pose_result["pose_index"]): pose_result
            for pose_result in previous_manifest.get("poses", [])
            if isinstance(pose_result, dict) and "pose_index" in pose_result
        }
        resumed_results = []
        for batch_pose_index, row in enumerate(pose_rows):
            pose_index = pose_start_index + batch_pose_index
            previous_result = previous_by_index.get(pose_index)
            if previous_result is None:
                continue
            render_name = output_name_for_pose(
                row["image_name"],
                str(runtime_args.output_extension),
            )
            if valid_completed_pose_result(
                previous_result,
                row,
                pose_index,
                scene_render_path / render_name,
                fine_tune_steps,
                save_pose_models,
            ):
                resumed_results.append(previous_result)
        manifest["poses"] = sorted(
            resumed_results,
            key=lambda item: int(item["pose_index"]),
        )

    manifest["completed_pose_count"] = len(manifest["poses"])
    manifest["total_training_seconds"] = float(
        sum(float(pose["training_seconds"]) for pose in manifest["poses"])
    )
    save_json(manifest_path, manifest)
    save_progress(progress_paths, manifest, "running")
    completed_pose_indices = {
        int(pose["pose_index"]) for pose in manifest["poses"]
    }
    if completed_pose_indices:
        print(
            "[{}] Resume: da co {}/{} pose hop le, se bo qua: {}".format(
                scene_name,
                len(completed_pose_indices),
                len(pose_rows),
                sorted(index + 1 for index in completed_pose_indices),
            ),
            flush=True,
        )

    camera_template: Scene | None = None
    total_training_seconds = float(manifest["total_training_seconds"])
    final_iteration = int(base_iteration + fine_tune_steps)
    for batch_pose_index, row in enumerate(pose_rows):
        pose_index = pose_start_index + batch_pose_index
        pose_position = batch_pose_index + 1
        if pose_index in completed_pose_indices:
            print(
                "[{}] Pose {}/{} (scene {}/{}): da co PNG hop le, bo qua.".format(
                    scene_name,
                    pose_position,
                    len(pose_rows),
                    pose_index + 1,
                    len(all_pose_rows),
                ),
                flush=True,
            )
            continue

        current_pose = {
            "pose_index": pose_index,
            "pose_number_in_scene": pose_index + 1,
            "pose_number_in_run": pose_position,
            "test_image_name": row["image_name"],
            "stage": "loading_base_model",
        }
        save_progress(
            progress_paths,
            manifest,
            "running",
            current_pose,
        )
        print(
            "\n[{}] Bat dau pose {}/{} (scene {}/{}): {}".format(
                scene_name,
                pose_position,
                len(pose_rows),
                pose_index + 1,
                len(all_pose_rows),
                row["image_name"],
            ),
            flush=True,
        )
        pose_directory_name = pose_output_directory_name(
            pose_index,
            row["image_name"],
        )
        pose_model_path = output_root / "poses" / pose_directory_name
        pose_dataset = copy.copy(dataset)
        pose_dataset.model_path = str(pose_model_path)
        pose_dataset.eval = False
        pose_runtime_args = copy.copy(runtime_args)
        pose_runtime_args.model_path = str(pose_model_path)
        pose_runtime_args.iterations = fine_tune_steps
        pose_runtime_args.training_method = "improvedgs"
        pose_runtime_args.coarse_to_fine = False
        pose_runtime_args.pose_aware_sampling = False
        pose_runtime_args.test_iterations = []
        pose_runtime_args.save_iterations = []
        pose_runtime_args.checkpoint_iterations = []
        prepare_output_and_logger(pose_dataset, pose_runtime_args)

        pose_opt = copy.copy(local_opt)
        gaussians, scene, camera_template = _load_fresh_base_model(
            pose_dataset,
            pose_opt,
            pose_model_path,
            base_model_path,
            base_iteration,
            camera_template,
            copy_scene_files=save_pose_models,
        )
        base_gaussian_count = int(gaussians.get_xyz.shape[0])
        if int(pose_opt.budget) < base_gaussian_count:
            raise ValueError(
                "Configured --budget ({}) is below the base model count ({}). "
                "Pass the original 30k model budget so ImprovedGS can split "
                "without an invalid local budget.".format(
                    int(pose_opt.budget),
                    base_gaussian_count,
                )
            )
        all_train_cameras = scene.getTrainCameras(1.0)
        selection = select_top_training_views(
            all_train_cameras,
            pose_from_csv_row(row),
            top_k=top_k,
            sigma_multiplier=sigma_multiplier,
        )
        selected_cameras = [
            all_train_cameras[index] for index in selection.selected_indices
        ]
        selection_payload = {
            "pose_index": pose_index,
            "batch_pose_index": batch_pose_index,
            "test_image_name": row["image_name"],
            **selection.to_dict(),
        }
        save_json(pose_model_path / "view_selection.json", selection_payload)
        current_pose["stage"] = "fine_tuning"
        current_pose["selected_top_k"] = len(selected_cameras)
        save_progress(
            progress_paths,
            manifest,
            "running",
            current_pose,
        )
        print(
            "[{}] Pose {}/{} {}: selected {}/{} views, sigma={:.6f}, "
            "score range [{:.6g}, {:.6g}]".format(
                scene_name,
                pose_position,
                len(pose_rows),
                row["image_name"],
                len(selected_cameras),
                len(all_train_cameras),
                selection.sigma,
                selection.views[-1].score,
                selection.views[0].score,
            ),
            flush=True,
        )

        context = build_training_context(
            pose_dataset,
            pose_opt,
            pipeline,
            pose_runtime_args,
        )
        context = attach_scene_and_gaussians_to_context(
            context,
            scene,
            gaussians,
            train_cameras_override=selected_cameras,
        )
        training_seconds, optimizer_updates = _run_local_finetune(
            context,
            gaussians,
            selected_cameras,
            fine_tune_steps,
            int(runtime_args.progress_bar_width),
            int(runtime_args.empty_cache_interval),
            "{} pose {}/{}".format(
                scene_name,
                pose_position,
                len(pose_rows),
            ),
        )
        total_training_seconds += training_seconds
        if save_pose_models:
            scene.save(final_iteration)
        finalize_training_parameters(str(pose_model_path), training_seconds)

        render_name = output_name_for_pose(
            row["image_name"],
            str(runtime_args.output_extension),
        )
        png_name = output_name_for_pose(row["image_name"], "png")
        current_pose["stage"] = "rendering_png"
        current_pose["training_seconds"] = float(training_seconds)
        current_pose["optimizer_updates"] = int(optimizer_updates)
        save_progress(
            progress_paths,
            manifest,
            "running",
            current_pose,
        )
        print(
            "[{}] Fine-tune xong pose {}/{} trong {:.1f}s; dang render PNG...".format(
                scene_name,
                pose_position,
                len(pose_rows),
                training_seconds,
            ),
            flush=True,
        )
        _render_target_pose(
            row=row,
            gaussians=gaussians,
            dataset=pose_dataset,
            pipeline=pipeline,
            background=background,
            undistorted_camera=undistorted_camera,
            radial_k=float(original_camera["radial_k"]),
            output_path=scene_render_path / render_name,
            png_output_path=(
                scene_png_path / png_name if scene_png_path is not None else None
            ),
            redistort_interpolation=str(runtime_args.redistort_interpolation),
            sharpen_amount=float(runtime_args.sharpen_amount),
            sharpen_sigma=float(runtime_args.sharpen_sigma),
            jpeg_quality=int(runtime_args.jpeg_quality),
            jpeg_subsampling=int(runtime_args.jpeg_subsampling),
        )

        pose_result = {
            "pose_index": pose_index,
            "batch_pose_index": batch_pose_index,
            "test_image_name": row["image_name"],
            "model_path": str(pose_model_path),
            "model_saved": save_pose_models,
            "point_cloud_path": (
                str(
                    pose_model_path
                    / "point_cloud"
                    / "iteration_{}".format(final_iteration)
                    / "point_cloud.ply"
                )
                if save_pose_models
                else ""
            ),
            "render_path": str(scene_render_path / render_name),
            "training_seconds": float(training_seconds),
            "optimizer_updates": int(optimizer_updates),
            "gaussian_count": int(gaussians.get_xyz.shape[0]),
            "selection": selection_payload,
        }
        manifest["poses"].append(pose_result)
        manifest["poses"].sort(key=lambda item: int(item["pose_index"]))
        manifest["completed_pose_count"] = len(manifest["poses"])
        manifest["total_training_seconds"] = float(total_training_seconds)
        save_json(manifest_path, manifest)
        completed_pose_indices.add(pose_index)
        save_progress(progress_paths, manifest, "running")
        print(
            "[{}] Hoan tat pose {}/{} -> {} | tong da xong: {}/{}".format(
                scene_name,
                pose_position,
                len(pose_rows),
                scene_render_path / render_name,
                len(completed_pose_indices),
                len(pose_rows),
            ),
            flush=True,
        )

        # Keep decoded camera images on CPU, but release every pose-specific
        # Gaussian model and optimizer before loading the immutable base PLY again.
        if camera_template is scene:
            camera_template.gaussians = None
        scene.gaussians = None
        del context, scene, gaussians
        torch.cuda.empty_cache()

    gt_dir = source_path / metadata.get("test_images", "test/images")
    if bool(runtime_args.evaluate):
        if not gt_dir.is_dir():
            if bool(runtime_args.require_gt):
                raise FileNotFoundError(
                    "Could not find public ground truth: {}".format(gt_dir)
                )
            print("No ground truth; skipping evaluation: {}".format(gt_dir))
        else:
            summary, per_view = evaluate_rendered_scene(
                gt_dir=gt_dir,
                render_dir=scene_render_path,
                pose_rows=pose_rows,
                output_extension=str(runtime_args.output_extension),
                psnr_max=float(runtime_args.psnr_max),
                lpips_net=str(runtime_args.lpips_net),
                device="cuda",
            )
            manifest["evaluation"] = summary
            save_json(output_root / "vai_per_view.json", per_view)
            save_json(output_root / "result_test.json", {**summary, **manifest})

    save_json(manifest_path, manifest)
    save_progress(progress_paths, manifest, "completed")
    return manifest
