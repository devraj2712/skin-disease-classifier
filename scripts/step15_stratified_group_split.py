#!/usr/bin/env python3
"""Create leakage-safe stratified train/val/test split.

Rules:
1. Keep class proportions as close as possible to 70/15/15.
2. Keep every phash_group_id entirely in one split.

The output uses symlinks to avoid duplicating image files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path('/backup/Intern/combine_skindiseaseclassifier_devraj')
IMAGES_ROOT = PROJECT_ROOT / 'data/selected_images'
PHASH_MANIFEST = PROJECT_ROOT / 'reports/perceptual_hash/perceptual_hash_manifest.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data/splits'
DEFAULT_REPORTS = PROJECT_ROOT / 'reports/split'
SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--reports', type=Path, default=DEFAULT_REPORTS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--copy', action='store_true', help='Copy image files instead of symlinking.')
    parser.add_argument('--force', action='store_true', help='Delete existing output split folder before writing.')
    return parser.parse_args()


def read_phash_manifest() -> list[dict[str, str]]:
    if not PHASH_MANIFEST.exists():
        raise FileNotFoundError(f'Missing perceptual hash manifest: {PHASH_MANIFEST}')
    with PHASH_MANIFEST.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def image_exists(row: dict[str, str]) -> bool:
    return Path(row['image_path']).exists()


def assign_groups_for_class(groups: list[dict[str, object]], total: int, rng: random.Random) -> None:
    targets = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    current = {split: 0 for split in SPLIT_RATIOS}

    # Larger groups first prevents one big group from badly overshooting a split later.
    rng.shuffle(groups)
    groups.sort(key=lambda group: int(group['size']), reverse=True)

    for group in groups:
        size = int(group['size'])
        best_split = min(
            SPLIT_RATIOS,
            key=lambda split: (
                (current[split] + size - targets[split]) / max(targets[split], 1),
                current[split] / max(targets[split], 1),
            ),
        )
        group['split'] = best_split
        current[best_split] += size


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def refresh_project_structure() -> None:
    updater = PROJECT_ROOT / 'scripts' / 'update_project_structure.py'
    if updater.exists():
        subprocess.run([sys.executable, str(updater)], check=False)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    reports = args.reports.resolve()
    rng = random.Random(args.seed)

    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_phash_manifest() if image_exists(row)]
    if not rows:
        raise RuntimeError('No valid images found in perceptual hash manifest.')

    by_class_group: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_class_group[row['class_name']][row['phash_group_id']].append(row)

    group_records: list[dict[str, object]] = []
    for class_name, groups in sorted(by_class_group.items()):
        class_group_records = []
        total = sum(len(items) for items in groups.values())
        for group_id, items in groups.items():
            class_group_records.append(
                {
                    'class_name': class_name,
                    'phash_group_id': group_id,
                    'size': len(items),
                    'items': items,
                    'split': '',
                }
            )
        assign_groups_for_class(class_group_records, total, rng)
        group_records.extend(class_group_records)

    split_manifest_rows: list[dict[str, str]] = []
    split_counts: Counter[tuple[str, str]] = Counter()
    group_split_rows: list[dict[str, str]] = []

    for group in group_records:
        split = str(group['split'])
        class_name = str(group['class_name'])
        group_id = str(group['phash_group_id'])
        items = group['items']
        assert isinstance(items, list)
        group_split_rows.append(
            {
                'class_name': class_name,
                'phash_group_id': group_id,
                'split': split,
                'group_size': str(len(items)),
            }
        )
        for row in items:
            src = Path(row['image_path']).resolve()
            dst = output / split / class_name / src.name
            link_or_copy(src, dst, args.copy)
            split_manifest_rows.append(
                {
                    'split': split,
                    'class_name': class_name,
                    'phash_group_id': group_id,
                    'source_image_path': str(src),
                    'split_image_path': str(dst),
                    'dhash': row.get('dhash', ''),
                    'phash': row.get('phash', ''),
                }
            )
            split_counts[(split, class_name)] += 1

    # Leakage check: each phash_group_id must appear in only one split.
    group_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in split_manifest_rows:
        group_to_splits[row['phash_group_id']].add(row['split'])
    leakage_rows = [
        {
            'phash_group_id': group_id,
            'splits': '|'.join(sorted(splits)),
            'leakage': 'yes' if len(splits) > 1 else 'no',
        }
        for group_id, splits in sorted(group_to_splits.items())
        if len(splits) > 1
    ]

    split_manifest_fields = [
        'split', 'class_name', 'phash_group_id', 'source_image_path',
        'split_image_path', 'dhash', 'phash'
    ]
    write_csv(reports / 'split_manifest.csv', split_manifest_fields, split_manifest_rows)
    write_csv(reports / 'group_split_assignments.csv', ['class_name', 'phash_group_id', 'split', 'group_size'], group_split_rows)
    write_csv(reports / 'leakage_check.csv', ['phash_group_id', 'splits', 'leakage'], leakage_rows)

    class_names = sorted({row['class_name'] for row in split_manifest_rows})
    split_summary_rows = []
    for class_name in class_names:
        class_total = sum(split_counts[(split, class_name)] for split in SPLIT_RATIOS)
        row = {'class_name': class_name, 'total': str(class_total)}
        for split in SPLIT_RATIOS:
            count = split_counts[(split, class_name)]
            row[f'{split}_count'] = str(count)
            row[f'{split}_pct'] = f'{(count / class_total * 100) if class_total else 0:.2f}'
        split_summary_rows.append(row)

    write_csv(
        reports / 'split_class_summary.csv',
        ['class_name', 'total', 'train_count', 'train_pct', 'val_count', 'val_pct', 'test_count', 'test_pct'],
        split_summary_rows,
    )

    total_by_split = Counter(row['split'] for row in split_manifest_rows)
    summary = {
        'project_root': str(PROJECT_ROOT),
        'input_images_root': str(IMAGES_ROOT),
        'output_split_root': str(output),
        'reports_root': str(reports),
        'ratios': SPLIT_RATIOS,
        'seed': args.seed,
        'mode': 'copy' if args.copy else 'symlink',
        'total_images': len(split_manifest_rows),
        'total_groups': len(group_records),
        'total_classes': len(class_names),
        'total_by_split': dict(sorted(total_by_split.items())),
        'leakage_groups_found': len(leakage_rows),
    }
    (reports / 'split_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    with (reports / 'split_summary.md').open('w', encoding='utf-8') as file:
        file.write('# Stratified Group Split Summary\n\n')
        file.write(f'Input images: `{IMAGES_ROOT}`\n\n')
        file.write(f'Output split folder: `{output}`\n\n')
        file.write('Split rule: approximately **70% train / 15% validation / 15% test**.\n\n')
        file.write('Leakage rule: each `phash_group_id` is kept in only one split.\n\n')
        file.write(f'Total images: **{len(split_manifest_rows):,}**\n\n')
        file.write(f'Total classes: **{len(class_names)}**\n\n')
        file.write(f'Total perceptual-hash groups: **{len(group_records):,}**\n\n')
        file.write(f'Leakage groups found after split: **{len(leakage_rows)}**\n\n')
        file.write('## Split Totals\n\n')
        file.write('| Split | Images | Percent |\n|---|---:|---:|\n')
        total = len(split_manifest_rows)
        for split in ['train', 'val', 'test']:
            count = total_by_split[split]
            file.write(f'| {split} | {count:,} | {count / total * 100:.2f}% |\n')
        file.write('\n## Class-Wise Split Counts\n\n')
        file.write('| Class | Total | Train | Val | Test |\n|---|---:|---:|---:|---:|\n')
        for row in split_summary_rows:
            file.write(
                f"| {row['class_name']} | {int(row['total']):,} | "
                f"{int(row['train_count']):,} | {int(row['val_count']):,} | {int(row['test_count']):,} |\n"
            )

    print('=' * 72)
    print('STEP 15 - STRATIFIED GROUP SPLIT COMPLETE')
    print('=' * 72)
    print(f'Output split folder : {output}')
    print(f'Reports folder      : {reports}')
    print(f'Total images        : {len(split_manifest_rows):,}')
    print(f'Total groups        : {len(group_records):,}')
    print(f'Total classes       : {len(class_names)}')
    for split in ['train', 'val', 'test']:
        print(f'{split:<5}: {total_by_split[split]:>7,} images')
    print(f'Leakage groups found: {len(leakage_rows)}')
    refresh_project_structure()
    print(f'Summary: {PROJECT_ROOT / "markdown_reports" / "split_summary.md"}')
    print(f'Project structure updated: {PROJECT_ROOT / "PROJECT_STRUCTURE.txt"}')


if __name__ == '__main__':
    main()
