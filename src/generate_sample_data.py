from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter


@dataclass
class GenerationConfig:
    output_dir: Path
    num_samples: int = 500
    seed: int = 42
    image_size: tuple[int, int] = (256, 256)


LABELS = ["walk", "lift", "reach", "sit", "inspect"]
SCENARIOS = ["warehouse", "factory_floor", "lab", "outdoor_test"]
LIGHTING = ["bright", "dim", "mixed"]
CAMERAS = ["cam_a", "cam_b", "cam_c"]
SPLITS = ["train", "train", "train", "val", "test"]


def _draw_robot_shape(draw: ImageDraw.ImageDraw, label: str, size: tuple[int, int]) -> None:
    w, h = size
    # Head
    draw.ellipse((w * 0.4, h * 0.12, w * 0.6, h * 0.3), outline="black", width=3)
    # Torso
    draw.line((w * 0.5, h * 0.3, w * 0.5, h * 0.62), fill="black", width=4)

    if label == "walk":
        draw.line((w * 0.5, h * 0.38, w * 0.34, h * 0.48), fill="black", width=4)
        draw.line((w * 0.5, h * 0.38, w * 0.66, h * 0.44), fill="black", width=4)
        draw.line((w * 0.5, h * 0.62, w * 0.40, h * 0.84), fill="black", width=4)
        draw.line((w * 0.5, h * 0.62, w * 0.64, h * 0.82), fill="black", width=4)
    elif label == "lift":
        draw.line((w * 0.5, h * 0.38, w * 0.34, h * 0.28), fill="black", width=4)
        draw.line((w * 0.5, h * 0.38, w * 0.66, h * 0.28), fill="black", width=4)
        draw.rectangle((w * 0.66, h * 0.22, w * 0.78, h * 0.34), outline="red", width=3)
        draw.line((w * 0.5, h * 0.62, w * 0.42, h * 0.86), fill="black", width=4)
        draw.line((w * 0.5, h * 0.62, w * 0.58, h * 0.86), fill="black", width=4)
    elif label == "reach":
        draw.line((w * 0.5, h * 0.38, w * 0.26, h * 0.36), fill="black", width=4)
        draw.line((w * 0.5, h * 0.38, w * 0.72, h * 0.34), fill="black", width=4)
        draw.line((w * 0.5, h * 0.62, w * 0.42, h * 0.86), fill="black", width=4)
        draw.line((w * 0.5, h * 0.62, w * 0.58, h * 0.86), fill="black", width=4)
    elif label == "sit":
        draw.line((w * 0.5, h * 0.38, w * 0.36, h * 0.50), fill="black", width=4)
        draw.line((w * 0.5, h * 0.38, w * 0.64, h * 0.50), fill="black", width=4)
        draw.line((w * 0.5, h * 0.62, w * 0.62, h * 0.72), fill="black", width=4)
        draw.line((w * 0.62, h * 0.72, w * 0.62, h * 0.86), fill="black", width=4)
        draw.line((w * 0.5, h * 0.62, w * 0.42, h * 0.78), fill="black", width=4)
        draw.line((w * 0.42, h * 0.78, w * 0.32, h * 0.78), fill="black", width=4)
    else:  # inspect
        draw.line((w * 0.5, h * 0.38, w * 0.36, h * 0.48), fill="black", width=4)
        draw.line((w * 0.5, h * 0.38, w * 0.70, h * 0.26), fill="black", width=4)
        draw.ellipse((w * 0.70, h * 0.22, w * 0.82, h * 0.34), outline="blue", width=3)
        draw.line((w * 0.5, h * 0.62, w * 0.44, h * 0.86), fill="black", width=4)
        draw.line((w * 0.5, h * 0.62, w * 0.56, h * 0.86), fill="black", width=4)


def _compute_blur_score(arr: np.ndarray) -> float:
    # variance of gradient magnitude as a simple sharpness proxy
    gx = np.diff(arr.astype(float), axis=1)
    gy = np.diff(arr.astype(float), axis=0)
    gmag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    return float(np.var(gmag))


def generate_dataset(config: GenerationConfig) -> pd.DataFrame:
    random.seed(config.seed)
    np.random.seed(config.seed)

    image_dir = config.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    for idx in range(config.num_samples):
        label = random.choice(LABELS)
        scenario = random.choice(SCENARIOS)
        lighting = random.choice(LIGHTING)
        camera = random.choice(CAMERAS)
        split = random.choice(SPLITS)
        is_occluded = random.random() < 0.18
        bbox_ratio = round(random.uniform(0.08, 0.42), 3)

        bg_tone = random.randint(180, 250)
        image = Image.new("RGB", config.image_size, color=(bg_tone, bg_tone, bg_tone))
        draw = ImageDraw.Draw(image)
        _draw_robot_shape(draw, label, config.image_size)

        # Draw scene label
        draw.text((10, 10), f"{scenario} | {label}", fill="black")

        # Add some synthetic sensor noise / blur
        if lighting == "dim":
            image = image.point(lambda p: int(p * 0.75))
        if random.random() < 0.12:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 2.5)))
        if is_occluded:
            occ_x = random.randint(40, 140)
            occ_y = random.randint(80, 170)
            ImageDraw.Draw(image).rectangle((occ_x, occ_y, occ_x + 45, occ_y + 35), fill="gray")

        image_id = f"img_{idx:05d}"
        file_name = f"{image_id}.png"
        file_path = image_dir / file_name
        image.save(file_path)

        gray = np.array(image.convert("L"))
        blur_score = round(_compute_blur_score(gray), 3)

        rows.append(
            {
                "image_id": image_id,
                "file_path": str(file_path.resolve()),
                "split": split,
                "scenario": scenario,
                "lighting": lighting,
                "robot_task": label,
                "label": label,
                "annotator_id": f"ann_{random.randint(1, 8):02d}",
                "capture_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=random.randint(0, 60)),
                "camera_id": camera,
                "width": config.image_size[0],
                "height": config.image_size[1],
                "blur_score": blur_score,
                "is_occluded": is_occluded,
                "bbox_area_ratio": bbox_ratio,
            }
        )

    df = pd.DataFrame(rows)

    # Introduce a few realistic quality issues so validations have something to catch.
    if len(df) >= 8:
        df.loc[3, "label"] = None
        df.loc[4, "image_id"] = df.loc[1, "image_id"]  # duplicate
        df.loc[5, "file_path"] = str((config.output_dir / "images" / "missing_img.png").resolve())
        df.loc[6, "blur_score"] = 5.0
        df.loc[7, "bbox_area_ratio"] = 0.01

    metadata_path = config.output_dir / "dataset_metadata.csv"
    df.to_csv(metadata_path, index=False)
    return df


if __name__ == "__main__":
    cfg = GenerationConfig(output_dir=Path("data/raw"))
    result = generate_dataset(cfg)
    print(f"Generated dataset with {len(result)} rows at {cfg.output_dir}")
