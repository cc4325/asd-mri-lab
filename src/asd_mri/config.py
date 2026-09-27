from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    train_dir: Path
    test_dir: Path
    labels_csv: Path
    submission_template: Path
    resources_dir: Path
    output_dir: Path
    preprocessing: dict[str, Any]
    modeling: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        root_value = Path(raw.get("project_root", "."))
        root = (config_path.parent.parent / root_value).resolve()

        def resolve(key: str) -> Path:
            return (root / raw[key]).resolve()

        return cls(
            root=root,
            train_dir=resolve("train_dir"),
            test_dir=resolve("test_dir"),
            labels_csv=resolve("labels_csv"),
            submission_template=resolve("submission_template"),
            resources_dir=resolve("resources_dir"),
            output_dir=resolve("output_dir"),
            preprocessing=raw["preprocessing"],
            modeling=raw["modeling"],
        )

