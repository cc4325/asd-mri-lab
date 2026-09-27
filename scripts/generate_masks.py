"""为已完成预处理的影像缓存独立 SynthStrip 脑掩膜。"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def build_mask(subject_dir: Path, model: Path) -> str:
    destination = subject_dir / 'synthstrip_mask_native.nii.gz'
    if not destination.exists():
        from asd_mri.synthstrip import synthstrip_mask
        synthstrip_mask(subject_dir / 't1_n4_native.nii.gz', model, destination)
    return subject_dir.name


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model = root / 'resources/synthstrip.1.pt'
    subjects = []
    for path in sorted((root / 'outputs/preprocessed').glob('*/*/metadata.json')):
        metadata = json.loads(path.read_text(encoding='utf-8'))
        if metadata.get('status') == 'complete':
            subjects.append(path.parent)
    print(f'已完成预处理 {len(subjects)} 例；开始缓存独立脑掩膜', flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(build_mask, subject, model) for subject in subjects]
        for index, future in enumerate(as_completed(futures), 1):
            print(f'{index}/{len(subjects)} {future.result()}', flush=True)


if __name__ == '__main__':
    main()
