from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_to_img


def prepare_resources(resources_dir: Path) -> dict[str, Path]:
    """准备固定模板、脑掩膜和 100 区 Schaefer 图谱。"""
    resources_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = resources_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    template_path = resources_dir / "mni152_t1_2mm.nii.gz"
    mask_path = resources_dir / "mni152_brain_mask_2mm.nii.gz"
    atlas_path = resources_dir / "schaefer100_2mm.nii.gz"
    labels_path = resources_dir / "schaefer100_labels.json"

    if not template_path.exists():
        nib.save(datasets.load_mni152_template(resolution=2), template_path)
    if not mask_path.exists():
        nib.save(datasets.load_mni152_brain_mask(resolution=2), mask_path)
    if not atlas_path.exists() or not labels_path.exists():
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=100,
            yeo_networks=7,
            resolution_mm=2,
            data_dir=str(cache_dir),
        )
        aligned = resample_to_img(nib.load(atlas.maps), nib.load(template_path), interpolation="nearest", force_resample=True, copy_header=True)
        nib.save(aligned, atlas_path)
        labels = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in atlas.labels
        ]
        labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "template": template_path,
        "mask": mask_path,
        "atlas": atlas_path,
        "labels": labels_path,
    }
