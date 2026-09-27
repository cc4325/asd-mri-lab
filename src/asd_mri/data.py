from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ImageRecord:
    subject_id: str
    path: Path
    split: str


def subject_id_from_path(path: Path) -> str:
    suffix = "_T1w.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"文件名不符合约定：{path.name}")
    return path.name[: -len(suffix)]


def discover_images(directory: Path, split: str) -> list[ImageRecord]:
    return [
        ImageRecord(subject_id_from_path(path), path.resolve(), split)
        for path in sorted(directory.glob("*_T1w.nii.gz"))
    ]


def validate_dataset(train_dir: Path, test_dir: Path, labels_csv: Path) -> pd.DataFrame:
    train = discover_images(train_dir, "train")
    test = discover_images(test_dir, "test")
    labels = pd.read_csv(labels_csv, dtype={"subject_id": str, "label": int})
    expected = {record.subject_id for record in train}
    if labels.subject_id.duplicated().any():
        raise ValueError('标签文件存在重复 subject_id')
    received = set(labels["subject_id"])
    if expected != received:
        raise ValueError(
            f"训练影像与标签不匹配，缺失标签={sorted(expected - received)}，"
            f"多余标签={sorted(received - expected)}"
        )
    if not set(labels["label"]).issubset({1, 2}):
        raise ValueError("标签只能是 1（ASD）或 2（HC）")
    if len({record.subject_id for record in train + test}) != len(train) + len(test):
        raise ValueError("训练集与测试集出现重复 subject_id")
    return labels.sort_values("subject_id").reset_index(drop=True)


def inspect_nifti(path: Path) -> dict[str, object]:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    finite = np.isfinite(data)
    return {
        "shape": list(image.shape),
        "zooms": [float(v) for v in image.header.get_zooms()[:3]],
        "dtype": str(data.dtype),
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "finite_fraction": float(finite.mean()),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

