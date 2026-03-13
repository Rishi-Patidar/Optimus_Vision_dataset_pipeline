from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_bar_chart(series: pd.Series, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    series.sort_values(ascending=False).plot(kind="bar")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_dashboard(metadata_path: Path, validation_results: dict, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metadata_path)

    class_chart = report_dir / "class_distribution.png"
    split_chart = report_dir / "split_distribution.png"
    scenario_chart = report_dir / "scenario_distribution.png"

    _save_bar_chart(df["robot_task"].value_counts(), "Robot Task Distribution", class_chart)
    _save_bar_chart(df["split"].value_counts(), "Data Split Distribution", split_chart)
    _save_bar_chart(df["scenario"].value_counts(), "Scenario Distribution", scenario_chart)

    issue_counts = validation_results["summary"]["issue_counts"]
    issues_chart = report_dir / "issue_summary.png"
    _save_bar_chart(pd.Series(issue_counts), "Validation Issue Summary", issues_chart)

    html_path = report_dir / "validation_dashboard.html"
    html = f"""
    <html>
    <head>
        <title>Optimus Vision Dataset Validation Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
            .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(2, minmax(300px, 1fr)); gap: 20px; }}
            img {{ width: 100%; max-width: 720px; border: 1px solid #ddd; }}
            code, pre {{ background: #f7f7f7; padding: 12px; display: block; overflow-x: auto; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        </style>
    </head>
    <body>
        <h1>Optimus Vision Dataset Validation Dashboard</h1>
        <div class="card">
            <h2>Executive Summary</h2>
            <p><strong>Rows:</strong> {validation_results['row_count']}</p>
            <p><strong>Total Issues:</strong> {validation_results['summary']['total_issues']}</p>
            <p><strong>Severity:</strong> {validation_results['summary']['severity']}</p>
        </div>
        <div class="grid">
            <div class="card"><h3>Task Distribution</h3><img src="{class_chart.name}" /></div>
            <div class="card"><h3>Split Distribution</h3><img src="{split_chart.name}" /></div>
            <div class="card"><h3>Scenario Distribution</h3><img src="{scenario_chart.name}" /></div>
            <div class="card"><h3>Validation Issue Summary</h3><img src="{issues_chart.name}" /></div>
        </div>
        <div class="card">
            <h2>Validation Results (JSON)</h2>
            <pre>{json.dumps(validation_results, indent=2, default=str)}</pre>
        </div>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")
    return html_path


if __name__ == "__main__":
    from validate_dataset import DatasetValidator

    metadata = Path("data/raw/dataset_metadata.csv")
    report_dir = Path("data/reports")
    validation = DatasetValidator(metadata).run_all_checks()
    path = build_dashboard(metadata, validation, report_dir)
    print(f"Dashboard written to {path}")
