#!/usr/bin/env python3
"""Prepare a YOLOv8 classification dataset with balanced train symlinks.

PyTorch models use WeightedRandomSampler, but YOLO's training API reads folders.
So for YOLO only, this script creates a separate folder:

    data/yolo_cls_balanced/
      train/class_name/*.jpg symlinks
      val/class_name/*.jpg symlinks
      test/class_name/*.jpg symlinks

Train is balanced by symlink oversampling/downsampling. Validation and test stay
natural and unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path('/backup/Intern/combine_skindiseaseclassifier_devraj')
SOURCE_SPLITS = PROJECT_ROOT / 'data' / 'splits'
OUTPUT_ROOT = PROJECT_ROOT / 'data' / 'yolo_cls_balanced'
REPORT_ROOT = PROJECT_ROOT / 'reports' / 'yolo_dataset'
MARKDOWN_ROOT = PROJECT_ROOT / 'markdown_reports'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_TARGET = 1727
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE_SPLITS)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--reports', type=Path, default=REPORT_ROOT)
    parser.add_argument('--target-train-count', type=int, default=DEFAULT_TARGET)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--force', action='store_true', help='Delete existing output folder before writing.')
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.rglob('*')
        if (path.is_file() or path.is_symlink()) and path.suffix.lower() in IMAGE_EXTS
    )


def reset_output(output: Path, force: bool) -> None:
    if output.exists():
        if not force:
            raise FileExistsError(f'Output exists. Use --force to recreate: {output}')
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def link_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def choose_balanced_sources(images: list[Path], target: int, rng: random.Random) -> list[Path]:
    if not images:
        return []
    shuffled = images[:]
    rng.shuffle(shuffled)
    if len(shuffled) >= target:
        return shuffled[:target]

    chosen = []
    while len(chosen) < target:
        round_images = images[:]
        rng.shuffle(round_images)
        chosen.extend(round_images)
    return chosen[:target]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(split: str, source: Path, output: Path, target: int, rng: random.Random) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    split_source = source / split
    split_output = output / split

    for class_dir in sorted(path for path in split_source.iterdir() if path.is_dir()):
        class_name = class_dir.name
        images = list_images(class_dir)
        original_count = len(images)

        if split == 'train':
            selected = choose_balanced_sources(images, target, rng)
        else:
            selected = images

        source_counter: Counter[str] = Counter()
        for index, src in enumerate(selected):
            source_counter[str(src.resolve())] += 1
            duplicate_index = source_counter[str(src.resolve())] - 1
            safe_name = f'{index:06d}__dup{duplicate_index:02d}__{src.name}'
            dst = split_output / class_name / safe_name
            link_image(src, dst)
            manifest_rows.append({
                'split': split,
                'class_name': class_name,
                'source_path': str(src.resolve()),
                'yolo_path': str(dst),
                'duplicate_index': duplicate_index,
            })

        final_count = len(selected)
        summary_rows.append({
            'split': split,
            'class_name': class_name,
            'original_count': original_count,
            'final_count': final_count,
            'added_by_oversampling': max(final_count - original_count, 0) if split == 'train' else 0,
            'downsampled_removed': max(original_count - final_count, 0) if split == 'train' else 0,
        })

    return manifest_rows, summary_rows


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    reset_output(args.output, args.force)
    args.reports.mkdir(parents=True, exist_ok=True)
    MARKDOWN_ROOT.mkdir(parents=True, exist_ok=True)

    all_manifest_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    for split in ['train', 'val', 'test']:
        manifest_rows, summary_rows = prepare_split(split, args.source, args.output, args.target_train_count, rng)
        all_manifest_rows.extend(manifest_rows)
        all_summary_rows.extend(summary_rows)

    write_csv(
        args.reports / 'yolo_dataset_manifest.csv',
        all_manifest_rows,
        ['split', 'class_name', 'source_path', 'yolo_path', 'duplicate_index'],
    )
    write_csv(
        args.reports / 'yolo_dataset_summary.csv',
        all_summary_rows,
        ['split', 'class_name', 'original_count', 'final_count', 'added_by_oversampling', 'downsampled_removed'],
    )

    totals = Counter()
    for row in all_summary_rows:
        totals[str(row['split'])] += int(row['final_count'])

    summary_json = {
        'source_splits': str(args.source),
        'output_root': str(args.output),
        'reports_root': str(args.reports),
        'target_train_count_per_class': args.target_train_count,
        'seed': args.seed,
        'train_balancing': 'symlink oversampling/downsampling to fixed target per class',
        'validation_balancing': False,
        'test_balancing': False,
        'totals': dict(totals),
    }
    (args.reports / 'yolo_dataset_summary.json').write_text(json.dumps(summary_json, indent=2), encoding='utf-8')

    md_path = MARKDOWN_ROOT / 'yolo_dataset_summary.md'
    with md_path.open('w', encoding='utf-8') as file:
        file.write('# YOLO Classification Dataset Summary\n\n')
        file.write(f'Source split folder: `{args.source}`\n\n')
        file.write(f'YOLO dataset folder: `{args.output}`\n\n')
        file.write(f'Train target per class: **{args.target_train_count:,}** symlinks.\n\n')
        file.write('Validation and test were not balanced; they remain natural.\n\n')
        file.write('## Split Totals\n\n')
        file.write('| Split | Images |\n|---|---:|\n')
        for split in ['train', 'val', 'test']:
            file.write(f'| {split} | {totals[split]:,} |\n')
        file.write('\n## Class Summary\n\n')
        file.write('| Split | Class | Original | Final | Added | Downsampled Removed |\n')
        file.write('|---|---|---:|---:|---:|---:|\n')
        for row in all_summary_rows:
            file.write(
                f"| {row['split']} | {row['class_name']} | {int(row['original_count']):,} | "
                f"{int(row['final_count']):,} | {int(row['added_by_oversampling']):,} | "
                f"{int(row['downsampled_removed']):,} |\n"
            )

    updater = PROJECT_ROOT / 'scripts' / 'update_project_structure.py'
    if updater.exists():
        subprocess.run([sys.executable, str(updater)], cwd=PROJECT_ROOT, check=False)

    print('=' * 72)
    print('YOLO CLASSIFICATION DATASET READY')
    print('=' * 72)
    print(f'Output: {args.output}')
    print(f'Reports: {args.reports}')
    print(f'Markdown: {md_path}')
    for split in ['train', 'val', 'test']:
        print(f'{split}: {totals[split]:,}')


if __name__ == '__main__':
    main()
