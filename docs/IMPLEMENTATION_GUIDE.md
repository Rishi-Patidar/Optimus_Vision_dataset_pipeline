# Implementation Guide: Optimus Vision Dataset Collection Pipeline

This guide explains the project as if you are building it from scratch in an AI / robotics organization.

## 1. Problem statement

You need a repeatable way to collect and validate image data for model training. The pipeline must answer five questions:

1. How is data being collected?
2. Is the metadata complete and consistent?
3. Are files actually present?
4. Are there quality problems such as blur or tiny objects?
5. Can stakeholders understand progress and issues quickly?

## 2. Architecture

The architecture has three layers.

### Layer A: Collection
- Synthetic image creation
- Metadata capture
- File storage

### Layer B: Validation
- Required column checks
- Missing label checks
- Duplicate ID checks
- Missing file checks
- Blur threshold checks
- Bounding box size checks

### Layer C: Reporting
- Class distribution chart
- Split distribution chart
- Scenario distribution chart
- Issue summary chart
- HTML dashboard

## 3. Step-by-step build process

### Step 1: Define the metadata schema

Before collecting data, decide what each row should contain. In this project, the schema is:

- `image_id`
- `file_path`
- `split`
- `scenario`
- `lighting`
- `robot_task`
- `label`
- `annotator_id`
- `capture_date`
- `camera_id`
- `width`
- `height`
- `blur_score`
- `is_occluded`
- `bbox_area_ratio`

Why this matters:
- Without a schema, collection becomes inconsistent.
- Model training suffers when metadata is missing or unreliable.
- PMs need stable definitions to align engineering and data operations teams.

### Step 2: Generate raw data

The generator creates a simple robot-like stick figure for five actions:
- walk
- lift
- reach
- sit
- inspect

Each image is saved as a PNG. Metadata is saved to `dataset_metadata.csv`.

Mechanism:
- A label is chosen randomly.
- Scenario, lighting, camera, and split are assigned.
- A robot pose is drawn onto a blank image.
- Blur and occlusion are simulated to mimic real-world data imperfections.
- A blur score is calculated.
- One metadata row is written for each image.

### Step 3: Inject realistic issues

This is an important teaching mechanism. Real datasets are messy, so the project deliberately injects a few issues:

- one missing label
- one duplicate image ID
- one missing file path
- one very blurry image
- one tiny bounding box ratio

Why this matters:
- Validation code needs to be tested against failures, not only perfect data.
- This shows how to build controls before the dataset reaches model training.

### Step 4: Run validations

The validator loads the CSV and performs checks.

#### Required columns
Confirms the schema exists.

#### Missing labels
Flags rows that cannot be used for supervised learning.

#### Duplicate IDs
Detects identifier collisions that would corrupt downstream tracking.

#### Missing files
Confirms every metadata row points to a real file.

#### Blur threshold
Flags low-quality images below the configured sharpness threshold.

#### Tiny boxes
Flags images where the object occupies too little of the scene.

### Step 5: Generate a dashboard

The dashboard converts validation outputs into charts and an HTML report.

Charts included:
- task distribution
- split distribution
- scenario distribution
- issue summary

Why this matters:
- Engineers need the raw issue lists.
- PMs need concise progress and quality signals.
- Leaders need a dashboard that explains whether the dataset is ready.

## 4. How to run the project

```bash
pip install -r requirements.txt
python main.py
```

## 5. How I implemented each module

### `generate_sample_data.py`
Purpose:
- Create images
- Produce metadata
- Simulate realistic quality issues

Key implementation ideas:
- Pillow is used for image generation.
- NumPy is used to estimate blur score.
- Pandas writes the final metadata table.

### `validate_dataset.py`
Purpose:
- Validate metadata integrity and file quality

Key implementation ideas:
- A class-based validator keeps logic modular.
- Each check returns both counts and affected rows.
- Summary output aggregates issue counts for dashboarding.

### `build_dashboard.py`
Purpose:
- Convert validation results into stakeholder-friendly reporting

Key implementation ideas:
- Matplotlib saves charts as PNG files.
- A lightweight HTML file displays charts and the full JSON validation summary.

### `main.py`
Purpose:
- Orchestrate the full flow end to end

Mechanism:
1. generate synthetic data
2. validate metadata and files
3. produce dashboard

## 6. What a PM is actually doing in this project

Although this repo contains code, the delivery ownership mirrors Project Management responsibilities:

- define the data quality standard
- align metadata definitions
- set validation rules
- monitor issue trends
- convert metrics into decision-ready reporting
- coordinate remediation priorities

## 7. How to extend this project

### Option A: Use real images
Replace the synthetic generator with a loader for a public dataset.

### Option B: Add annotation review
Add fields like `review_status`, `annotator_confidence`, and `reviewer_id`.

### Option C: Add automation
Schedule daily validation runs with GitHub Actions or Airflow.

### Option D: Add BI tooling
Replace the static dashboard with Streamlit or Power BI.

## 8. Suggested next enhancements

- Add train / val leakage checks
- Add label taxonomy validation
- Add per-camera quality drift detection
- Add annotation agreement metrics
- Add issue severity SLA tracking

## 9. Resume-ready summary

You can describe this project as:

Planned and governed an AI vision dataset pipeline that generated structured metadata, ran automated validation checks, and published dashboard-based quality reporting to support model training readiness.
