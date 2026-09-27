from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import ants
import matplotlib
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from .data import ImageRecord, inspect_nifti, sha256_file

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _robust_normalize(image: ants.ANTsImage) -> ants.ANTsImage:
    data = image.numpy().astype(np.float32)
    foreground = data[data > 0]
    if foreground.size == 0:
        raise ValueError("影像没有正值体素")
    lower, upper = np.percentile(foreground, [1.0, 99.0])
    if upper <= lower:
        raise ValueError("影像强度范围异常")
    data = np.clip((data - lower) / (upper - lower), 0.0, 1.0)
    return image.new_image_like(data)


def _normalized_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(a) & np.isfinite(b)
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    if av.size < 100 or av.std() == 0 or bv.std() == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _save_qc(
    registered: np.ndarray,
    template: np.ndarray,
    segmentation: np.ndarray,
    output_path: Path,
    subject_id: str,
) -> None:
    slices = [registered.shape[index] // 2 for index in range(3)]
    registered_views = [
        np.rot90(registered[slices[0], :, :]),
        np.rot90(registered[:, slices[1], :]),
        np.rot90(registered[:, :, slices[2]]),
    ]
    template_views = [
        np.rot90(template[slices[0], :, :]),
        np.rot90(template[:, slices[1], :]),
        np.rot90(template[:, :, slices[2]]),
    ]
    segmentation_views = [
        np.rot90(segmentation[slices[0], :, :]),
        np.rot90(segmentation[:, slices[1], :]),
        np.rot90(segmentation[:, :, slices[2]]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, image, overlay in zip(axes[0], template_views, registered_views, strict=True):
        axis.imshow(image, cmap="gray")
        axis.imshow(overlay, cmap="magma", alpha=0.35)
        axis.axis("off")
    for axis, image, labels in zip(axes[1], registered_views, segmentation_views, strict=True):
        axis.imshow(image, cmap="gray")
        axis.imshow(np.ma.masked_where(labels == 0, labels), cmap="viridis", alpha=0.35)
        axis.axis("off")
    axes[0, 0].set_title("MNI 与配准结果叠加")
    axes[1, 0].set_title("CSF / GM / WM 分割")
    fig.suptitle(f"{subject_id} 预处理质量控制")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def preprocess_subject(
    record: ImageRecord,
    output_root: Path,
    template_path: Path,
    mask_path: Path,
    transform: str = "antsRegistrationSyNQuickRepro[s]",
    n4_shrink_factor: int = 4,
    smoothing_fwhm_mm: float = 4.0,
    brain_mask_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """处理单例 T1，并返回可序列化的状态信息。"""
    started = time.perf_counter()
    subject_dir = output_root / record.split / record.subject_id
    done_path = subject_dir / "metadata.json"
    parameters={'transform':transform,'n4_shrink_factor':n4_shrink_factor,'smoothing_fwhm_mm':smoothing_fwhm_mm}
    if brain_mask_path is not None:
        parameters['brain_mask_sha256'] = sha256_file(brain_mask_path)
    input_hash=sha256_file(record.path)
    if done_path.exists() and not overwrite:
        previous = json.loads(done_path.read_text(encoding="utf-8"))
        previous_parameters=previous.get('parameters',{'transform':'antsRegistrationSyNQuickRepro[s]','n4_shrink_factor':4,'smoothing_fwhm_mm':4.0})
        parameters_match = previous_parameters == parameters
        if brain_mask_path is None and 'brain_mask_sha256' in previous_parameters:
            saved_mask = subject_dir / 'synthstrip_mask_native.nii.gz'
            base_parameters = {key: value for key, value in previous_parameters.items() if key != 'brain_mask_sha256'}
            parameters_match = (base_parameters == parameters and saved_mask.exists()
                                and sha256_file(saved_mask) == previous_parameters['brain_mask_sha256'])
        valid_outputs=all(Path(path).exists() for path in previous.get('outputs',{}).values())
        if previous.get("status") == "complete" and previous.get('pipeline_version') == '2' and parameters_match and previous.get('input_sha256')==input_hash and valid_outputs:
            return previous

    subject_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "subject_id": record.subject_id,
        "split": record.split,
        "input_path": str(record.path),
        "status": "running",
        "pipeline_version": "2",
        "parameters": parameters,
    }
    try:
        ants.config.set_ants_deterministic(True, 20260924)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
        metadata["input_header"] = inspect_nifti(record.path)
        metadata["input_sha256"] = input_hash
        image = ants.image_read(str(record.path), reorient="RAS")
        template = ants.image_read(str(template_path), reorient="RAS")
        mask = ants.image_read(str(mask_path), reorient="RAS")

        corrected = ants.n4_bias_field_correction(
            image,
            shrink_factor=int(n4_shrink_factor),
            convergence={"iters": [30, 20, 10], "tol": 1e-5},
            verbose=False,
        )
        normalized = _robust_normalize(corrected)
        if brain_mask_path is not None:
            extracted_mask = ants.image_read(str(brain_mask_path), reorient='RAS')
            if not ants.image_physical_space_consistency(normalized, extracted_mask):
                raise ValueError('脑掩膜与原始影像的物理空间不一致')
            moving_for_registration = normalized * (extracted_mask > 0)
        else:
            extracted_mask = None
            moving_for_registration = normalized
        print(f"{record.subject_id}: N4 complete; registration", flush=True)
        registration = ants.registration(
            fixed=template,
            moving=moving_for_registration,
            type_of_transform=transform,
            verbose=False,
        )
        registered = registration["warpedmovout"]
        brain = registered * mask
        print(f"{record.subject_id}: registration complete; segmentation", flush=True)

        native_mask = ants.apply_transforms(fixed=normalized, moving=mask,
            transformlist=registration['invtransforms'], interpolator='nearestNeighbor')
        if extracted_mask is not None:
            native_mask = native_mask * (extracted_mask > 0)
        native_brain = normalized * native_mask
        ants.image_write(native_mask, str(subject_dir / 'brain_mask_native.nii.gz'))
        ants.image_write(corrected, str(subject_dir / 't1_n4_native.nii.gz'))
        segmentation_result = ants.atropos(
            a=native_brain,
            x=native_mask,
            i="KMeans[3]",
            m="[0.1,1x1x1]",
            c="[5,0]",
            verbose=0,
        )
        raw_segmentation = segmentation_result["segmentation"]
        raw_labels = raw_segmentation.numpy().astype(np.uint8)
        brain_array = native_brain.numpy()
        means = {
            label: float(brain_array[raw_labels == label].mean())
            for label in (1, 2, 3)
            if np.any(raw_labels == label)
        }
        if len(means) != 3:
            raise RuntimeError(f"组织分割未生成三个类别：{means}")
        ordered = sorted(means, key=means.get)
        tissue_names = ["csf", "gm", "wm"]
        probabilities: dict[str, ants.ANTsImage] = {}
        native_probabilities = {}
        final_seg = np.zeros_like(raw_labels)
        for final_label, (name, raw_label) in enumerate(zip(tissue_names, ordered, strict=True), 1):
            native_probability = segmentation_result["probabilityimages"][raw_label - 1]
            native_probabilities[name] = native_probability
            ants.image_write(native_probability, str(subject_dir / f'{name}_native.nii.gz'))
            probabilities[name] = ants.apply_transforms(fixed=template, moving=native_probability,
                transformlist=registration['fwdtransforms'], interpolator='linear') * mask
            final_seg[raw_labels == raw_label] = final_label
        native_seg = normalized.new_image_like(final_seg)
        ants.image_write(native_seg, str(subject_dir / 'tissue_labels_native.nii.gz'))
        final_seg_image = ants.apply_transforms(fixed=template, moving=native_seg,
            transformlist=registration['fwdtransforms'], interpolator='nearestNeighbor')
        final_seg = final_seg_image.numpy()

        # 仅对特征用概率图平滑；原始概率图仍单独保存。
        sigma_voxels = smoothing_fwhm_mm / 2.355 / np.asarray(registered.spacing)
        smoothed_gm = probabilities["gm"].new_image_like(
            gaussian_filter(probabilities["gm"].numpy(), sigma=sigma_voxels)
        )

        outputs = {
            "registered": subject_dir / "t1_mni.nii.gz",
            "brain": subject_dir / "t1_brain_mni.nii.gz",
            "segmentation": subject_dir / "tissue_labels.nii.gz",
            "csf": subject_dir / "csf_probability.nii.gz",
            "gm": subject_dir / "gm_probability.nii.gz",
            "gm_smoothed": subject_dir / "gm_probability_smooth.nii.gz",
            "wm": subject_dir / "wm_probability.nii.gz",
        }
        ants.image_write(registered, str(outputs["registered"]))
        ants.image_write(brain, str(outputs["brain"]))
        ants.image_write(final_seg_image, str(outputs["segmentation"]))
        ants.image_write(probabilities["csf"], str(outputs["csf"]))
        ants.image_write(probabilities["gm"], str(outputs["gm"]))
        ants.image_write(smoothed_gm, str(outputs["gm_smoothed"]))
        ants.image_write(probabilities["wm"], str(outputs["wm"]))

        transform_dir = subject_dir / "transforms"
        transform_dir.mkdir(exist_ok=True)
        copied_transforms = []
        for index, transform_path in enumerate(registration["fwdtransforms"]):
            source = Path(transform_path)
            target = transform_dir / f"forward_{index}{''.join(source.suffixes)}"
            shutil.copy2(source, target)
            copied_transforms.append(str(target))

        template_array = template.numpy()
        mask_array = mask.numpy() > 0
        ncc = _normalized_correlation(registered.numpy(), template_array, mask_array)
        voxel_volume = float(np.prod(normalized.spacing))
        volumes = {
            name: float(image.numpy().sum() * voxel_volume)
            for name, image in native_probabilities.items()
        }
        _save_qc(
            registered.numpy(),
            template_array,
            final_seg,
            subject_dir / "qc.png",
            record.subject_id,
        )

        metadata.update(
            {
                "status": "complete",
                "transform": transform,
                "ncc_to_template": ncc,
                "tissue_intensity_order": {name: means[label] for name, label in zip(tissue_names, ordered, strict=True)},
                "tissue_volumes_mm3": volumes,
                "outputs": {key: str(value) for key, value in outputs.items()},
                "forward_transforms": copied_transforms,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    done_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _preprocess_worker(kwargs: dict[str, object]) -> dict[str, object]:
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    return preprocess_subject(**kwargs)  # type: ignore[arg-type]


def preprocess_many(
    records: Iterable[ImageRecord],
    output_root: Path,
    template_path: Path,
    mask_path: Path,
    workers: int = 1,
    callback: Callable[[dict[str, object]], None] | None = None,
    **parameters: object,
) -> list[dict[str, object]]:
    jobs = [
        {
            "record": record,
            "output_root": output_root,
            "template_path": template_path,
            "mask_path": mask_path,
            **parameters,
        }
        for record in records
    ]
    results: list[dict[str, object]] = []
    if workers <= 1:
        for job in jobs:
            result = _preprocess_worker(job)
            results.append(result)
            if callback:
                callback(result)
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_preprocess_worker, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if callback:
                callback(result)
    return results


def build_qc_table(output_root: Path, destination: Path) -> None:
    import pandas as pd

    rows = []
    for metadata_path in sorted(output_root.glob("*/*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "subject_id": metadata.get("subject_id"),
                "split": metadata.get("split"),
                "status": metadata.get("status"),
                "ncc_to_template": metadata.get("ncc_to_template"),
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "error": metadata.get("error", ""),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)

