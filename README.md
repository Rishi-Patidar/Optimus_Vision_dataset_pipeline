# Optimus Vision Dataset Collection Pipeline

A complete project that demonstrates how to plan, generate, validate, and report on a vision dataset pipeline for an AI / robotics use case.

## What this project does

This project simulates a small version of the workflow a Data Operations PM would coordinate for an AI team:

1. Generate a synthetic image dataset and metadata file.
2. Run data quality checks on the dataset.
3. Build a dashboard that summarizes dataset quality and composition.
4. Produce artifacts that can be shared with engineering, analytics, and leadership.

## Data sources used in this version

This project uses **synthetic data generated locally**. That choice is intentional so the project is fully runnable from scratch without external downloads.

### Included data source
- Synthetic robot-action images generated with Pillow
- Synthetic metadata with labels, scenarios, camera IDs, blur scores, occlusion flags, and bounding-box area ratios

### Public datasets you can plug in later
- COCO
- Open Images
- Roboflow datasets
- Kaggle computer vision datasets
- Custom warehouse / factory images collected internally

## Project structure

```text
optimus_vision_dataset_pipeline/
├── main.py
├── requirements.txt
├── README.md
├── docs/
│   └── IMPLEMENTATION_GUIDE.md
├── src/
│   ├── generate_sample_data.py
│   ├── validate_dataset.py
│   └── build_dashboard.py
└── data/
    ├── raw/
    └── reports/
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## Output artifacts

After running the project, you will get:

- `data/raw/dataset_metadata.csv`
- `data/reports/validation_dashboard.html`
- `data/reports/*.png` charts

## Example PM talking points for your resume

- Planned and governed an AI dataset collection pipeline with validation checkpoints and dashboard reporting.
- Coordinated quality controls for dataset completeness, duplication, blur detection, and annotation integrity.
- Delivered stakeholder-ready reporting that translated technical quality metrics into action items.
