from __future__ import annotations

from pathlib import Path

from src.build_dashboard import build_dashboard
from src.generate_sample_data import GenerationConfig, generate_dataset
from src.validate_dataset import DatasetValidator


def run_pipeline() -> None:
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    report_dir = project_root / "data" / "reports"

    print("[1/3] Generating synthetic Optimus-style vision dataset...")
    generate_dataset(GenerationConfig(output_dir=raw_dir, num_samples=500, seed=42))

    metadata_path = raw_dir / "dataset_metadata.csv"

    print("[2/3] Running validation checks...")
    validation = DatasetValidator(metadata_path=metadata_path).run_all_checks()

    print("[3/3] Building dashboard and artifacts...")
    dashboard_path = build_dashboard(metadata_path, validation, report_dir)

    print("\nPipeline completed successfully.")
    print(f"Metadata CSV: {metadata_path}")
    print(f"Dashboard HTML: {dashboard_path}")
    print(f"Validation summary: {validation['summary']}")


if __name__ == "__main__":
    run_pipeline()
