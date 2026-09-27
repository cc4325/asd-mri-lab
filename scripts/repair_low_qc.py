"""仅对模板相关性未达标的影像尝试 SynthStrip 脑提取和重新配准。"""
from __future__ import annotations

import json
from pathlib import Path

from asd_mri.data import ImageRecord
from asd_mri.preprocessing import build_qc_table, preprocess_subject
from asd_mri.synthstrip import synthstrip_mask


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs' / 'preprocessed'
    model = root / 'resources' / 'synthstrip.1.pt'
    if not model.exists():
        raise FileNotFoundError('缺少官方 SynthStrip 权重 synthstrip.1.pt')
    count = 0
    for metadata_path in sorted(output.glob('*/*/metadata.json')):
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        ncc = metadata.get('ncc_to_template')
        if metadata.get('status') == 'complete' and ncc is not None and float(ncc) >= 0.45:
            continue
        if 'brain_mask_sha256' in metadata.get('parameters', {}):
            print(f"{metadata['subject_id']}: 已使用 SynthStrip 修复但仍未过质控，请人工复核", flush=True)
            continue
        subject_dir = metadata_path.parent
        subject_id = metadata['subject_id']
        split = metadata['split']
        original = root / split / f'{subject_id}_T1w.nii.gz'
        if not original.exists():
            raise FileNotFoundError(original)
        corrected = subject_dir / 't1_n4_native.nii.gz'
        if not corrected.exists():
            print(f'{subject_id}: 缺少 N4 输出；先运行常规预处理', flush=True)
            continue
        mask = subject_dir / 'synthstrip_mask_native.nii.gz'
        if not mask.exists():
            synthstrip_mask(corrected, model, mask)
        result = preprocess_subject(
            ImageRecord(subject_id, original, split), output,
            root / 'resources' / 'mni152_t1_2mm.nii.gz',
            root / 'resources' / 'mni152_brain_mask_2mm.nii.gz',
            brain_mask_path=mask,
            overwrite=True,
        )
        count += 1
        print(f"{subject_id}: {result['status']}, NCC={result.get('ncc_to_template')}, "
              f"error={result.get('error')}", flush=True)
    build_qc_table(output, root / 'outputs' / 'qc_summary.csv')
    print(f'尝试修复 {count} 例', flush=True)


if __name__ == '__main__':
    main()
