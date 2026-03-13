from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


REQUIRED_COLUMNS = [
    "image_id",
    "file_path",
    "split",
    "scenario",
    "lighting",
    "robot_task",
    "label",
    "annotator_id",
    "camera_id",
    "width",
    "height",
    "blur_score",
    "is_occluded",
    "bbox_area_ratio",
]


class DatasetValidator:
    def __init__(self, metadata_path: Path, blur_threshold: float = 25.0, min_bbox_ratio: float = 0.03):
        self.metadata_path = metadata_path
        self.blur_threshold = blur_threshold
        self.min_bbox_ratio = min_bbox_ratio
        self.df = pd.read_csv(metadata_path)

    def run_all_checks(self) -> Dict[str, object]:
        results = {
            "row_count": int(len(self.df)),
            "column_check": self._check_required_columns(),
            "missing_labels": self._check_missing_labels(),
            "duplicate_ids": self._check_duplicate_ids(),
            "missing_files": self._check_missing_files(),
            "blurry_images": self._check_blurry_images(),
            "tiny_boxes": self._check_tiny_boxes(),
            "class_distribution": self._class_distribution(),
            "split_distribution": self._split_distribution(),
            "scenario_distribution": self._scenario_distribution(),
        }
        results["summary"] = self._build_summary(results)
        return results

    def _check_required_columns(self) -> Dict[str, object]:
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        return {"passed": len(missing_cols) == 0, "missing_columns": missing_cols}

    def _check_missing_labels(self) -> Dict[str, object]:
        missing = self.df[self.df["label"].isna()]
        return {"count": int(len(missing)), "rows": missing[["image_id", "file_path"]].to_dict("records")}

    def _check_duplicate_ids(self) -> Dict[str, object]:
        duplicates = self.df[self.df.duplicated(subset=["image_id"], keep=False)]
        return {"count": int(len(duplicates)), "rows": duplicates[["image_id", "file_path"]].to_dict("records")}

    def _check_missing_files(self) -> Dict[str, object]:
        missing_mask = ~self.df["file_path"].apply(lambda x: Path(x).exists())
        missing = self.df[missing_mask]
        return {"count": int(len(missing)), "rows": missing[["image_id", "file_path"]].to_dict("records")}

    def _check_blurry_images(self) -> Dict[str, object]:
        blurry = self.df[self.df["blur_score"] < self.blur_threshold]
        return {
            "count": int(len(blurry)),
            "threshold": self.blur_threshold,
            "rows": blurry[["image_id", "blur_score", "file_path"]].to_dict("records"),
        }

    def _check_tiny_boxes(self) -> Dict[str, object]:
        tiny = self.df[self.df["bbox_area_ratio"] < self.min_bbox_ratio]
        return {
            "count": int(len(tiny)),
            "threshold": self.min_bbox_ratio,
            "rows": tiny[["image_id", "bbox_area_ratio", "file_path"]].to_dict("records"),
        }

    def _class_distribution(self) -> Dict[str, int]:
        return self.df["robot_task"].fillna("MISSING").value_counts().to_dict()

    def _split_distribution(self) -> Dict[str, int]:
        return self.df["split"].value_counts().to_dict()

    def _scenario_distribution(self) -> Dict[str, int]:
        return self.df["scenario"].value_counts().to_dict()

    @staticmethod
    def _build_summary(results: Dict[str, object]) -> Dict[str, object]:
        issue_counts = {
            "missing_labels": results["missing_labels"]["count"],
            "duplicate_ids": results["duplicate_ids"]["count"],
            "missing_files": results["missing_files"]["count"],
            "blurry_images": results["blurry_images"]["count"],
            "tiny_boxes": results["tiny_boxes"]["count"],
        }
        total_issues = sum(issue_counts.values())
        severity = "low"
        if total_issues > 30:
            severity = "high"
        elif total_issues > 10:
            severity = "medium"
        return {"total_issues": total_issues, "severity": severity, "issue_counts": issue_counts}


if __name__ == "__main__":
    validator = DatasetValidator(Path("data/raw/dataset_metadata.csv"))
    results = validator.run_all_checks()
    print(results["summary"])
