"""CLI validate va dong goi anh submission VAI."""
from __future__ import annotations

import argparse

from vai.packaging import package_submission


def main() -> int:
    """Kiem tra scene da render roi tao ZIP submission."""
    parser = argparse.ArgumentParser(description="Validate and package VAI submission")
    parser.add_argument("--phase_dir", required=True)
    parser.add_argument("--set_name", default="private_set1")
    parser.add_argument("--submission_dir", required=True)
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--subset", nargs="*", default=[])
    parser.add_argument("--output_extension", default="csv")
    parser.add_argument("--keep_csv_extension", action="store_true")
    parser.add_argument("--allow_extra", action="store_true")
    args = parser.parse_args()

    output_extension = "csv" if args.keep_csv_extension else args.output_extension
    counts = package_submission(
        phase_dir=args.phase_dir,
        set_name=args.set_name,
        submission_root=args.submission_dir,
        zip_path=args.zip_path,
        subset=args.subset,
        output_extension=output_extension,
        allow_extra=args.allow_extra,
    )
    for scene_name, count in counts.items():
        print(f"{scene_name}: {count} images OK")
    print(f"Packed {sum(counts.values())} images -> {args.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
