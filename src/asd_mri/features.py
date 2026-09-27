from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _load_float(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(path)
    return image.get_fdata(dtype=np.float32), image


def _masked_native_volumes(subject_dir: Path, model_path: Path) -> dict[str, float]:
    """用独立脑掩膜核算体积，防止模板反变换掩膜误收颅骨。"""
    mask_path = subject_dir / 'synthstrip_mask_native.nii.gz'
    if not mask_path.exists():
        from .synthstrip import synthstrip_mask
        synthstrip_mask(subject_dir / 't1_n4_native.nii.gz', model_path, mask_path)
    mask_image = nib.load(mask_path)
    mask = mask_image.get_fdata(dtype=np.float32) > 0
    mask_volume = float(mask.sum() * abs(np.linalg.det(mask_image.affine[:3, :3])))
    if not 600_000 <= mask_volume <= 2_300_000:
        raise ValueError(f'{subject_dir.name} 独立脑掩膜体积异常：{mask_volume:.0f} mm³')
    volumes = {}
    for name in ('gm', 'wm', 'csf'):
        probability, reference = _load_float(subject_dir / f'{name}_native.nii.gz')
        if mask_image.shape != reference.shape or not np.allclose(mask_image.affine, reference.affine):
            from nibabel.processing import resample_from_to
            native_mask = resample_from_to(mask_image, reference, order=0).get_fdata(dtype=np.float32) > 0
        else:
            native_mask = mask
        voxel_volume = float(abs(np.linalg.det(reference.affine[:3, :3])))
        volumes[name] = float(np.clip(probability, 0, 1)[native_mask].sum() * voxel_volume)
    if not 500_000 <= sum(volumes.values()) <= 2_300_000:
        raise ValueError(f'{subject_dir.name} 掩膜内组织体积异常：{sum(volumes.values()):.0f} mm³')
    return volumes


def extract_subject_features(subject_dir: Path, atlas_path: Path, labels_path: Path) -> dict[str, object]:
    metadata = json.loads((subject_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"{subject_dir.name} 尚未成功完成预处理")
    if metadata.get('ncc_to_template',0) < 0.45:
        raise ValueError(f'{subject_dir.name} 配准相关性低于质控门槛 0.45，需修复后再提取')
    gm, reference = _load_float(subject_dir / "gm_probability_smooth.nii.gz")
    wm, _ = _load_float(subject_dir / "wm_probability.nii.gz")
    csf, _ = _load_float(subject_dir / "csf_probability.nii.gz")
    atlas_image = nib.load(atlas_path)
    from nibabel.processing import resample_from_to
    if atlas_image.shape != reference.shape or not np.allclose(atlas_image.affine, reference.affine):
        atlas_image = resample_from_to(atlas_image, reference, order=0)
    atlas = np.asarray(atlas_image.dataobj, dtype=np.int16)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = [label for label in labels if label.lower() != 'background']
    if gm.shape != atlas.shape:
        raise ValueError(f"图谱与处理影像尺寸不一致：{atlas.shape} != {gm.shape}")
    voxel_volume = float(abs(np.linalg.det(reference.affine[:3, :3])))
    volumes = _masked_native_volumes(subject_dir, atlas_path.parent / 'synthstrip.1.pt')
    row: dict[str, object] = {
        "subject_id": subject_dir.name,
        "split": metadata["split"],
        "qc_ncc": float(metadata["ncc_to_template"]),
        "global_gm_volume_mm3": volumes['gm'],
        "global_wm_volume_mm3": volumes['wm'],
        "global_csf_volume_mm3": volumes['csf'],
    }
    row["estimated_tiv_mm3"] = (
        row["global_gm_volume_mm3"] + row["global_wm_volume_mm3"] + row["global_csf_volume_mm3"]
    )
    for roi_index, label in enumerate(labels, start=1):
        roi = atlas == roi_index
        safe_label = label.replace("7Networks_", "").replace("-", "_")
        if not np.any(roi):
            roi_mean = 0.0
            roi_volume = 0.0
        else:
            roi_mean = float(gm[roi].mean())
            roi_volume = float(gm[roi].sum() * voxel_volume)
        row[f"gm_mean__{safe_label}"] = roi_mean
    for name in ('gm','wm','csf'):
        row[f'global_{name}_fraction'] = row[f'global_{name}_volume_mm3'] / row['estimated_tiv_mm3']
    # Schaefer 左右 ROI 数目并不一一相同，因此仅在网络层面比较左右均值。
    for network in ['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']:
        sides=[]
        for side in ['LH','RH']:
            ids=[i for i,label in enumerate(labels,1) if f'_{side}_{network}_' in label]
            values=gm[np.isin(atlas,ids)]
            sides.append(float(values.mean()) if values.size else 0.0)
        row[f'network_asymmetry__{network}']=(sides[0]-sides[1])/(sides[0]+sides[1]+1e-8)
    warp_path=subject_dir/'transforms/forward_0.nii.gz'
    if warp_path.exists():
        import ants
        jac_path=subject_dir/'log_jacobian_mni.nii.gz'
        if not jac_path.exists():
            domain=ants.image_read(str(subject_dir/'t1_mni.nii.gz'))
            jac=ants.create_jacobian_determinant_image(domain,str(warp_path),do_log=True,geom=True)
            ants.image_write(jac,str(jac_path))
        jac=nib.load(jac_path).get_fdata(dtype=np.float32)
        for i,label in enumerate(labels,1):
            roi=atlas==i
            row[f'log_jacobian__{label}']=float(jac[roi].mean())
    return row


def build_feature_table(preprocessed_root: Path, atlas_path: Path, labels_path: Path, output_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for subject_dir in sorted(preprocessed_root.glob("*/*")):
        if not (subject_dir / "metadata.json").exists():
            continue
        try:
            rows.append(extract_subject_features(subject_dir, atlas_path, labels_path))
        except Exception as exc:
            failures.append(f"{subject_dir}: {exc}")
    if failures:
        raise RuntimeError('特征提取失败，不允许静默丢弃受试者：' + '\n'.join(failures))
    if not rows:
        raise RuntimeError("没有可用于特征提取的预处理结果")
    table = pd.DataFrame(rows).sort_values(["split", "subject_id"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    return table
