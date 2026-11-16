"""构建 CPLID 绝缘子完整性二分类实验数据。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_DIR / "CPLID"
SEED = 6047
REFERENCE_IDS = {
    "Normal": ["0049", "0060"],
    "Major": ["000", "150"],
}
TRAIN_PER_CLASS = 30
BLUE = (0, 128, 255)
IMAGE_DIRS = {
    "reference": PROJECT_DIR / "reference image",
    "train": PROJECT_DIR / "train image",
    "test": PROJECT_DIR / "test image",
}
JSON_ROOT = PROJECT_DIR / "json"


def load_insulator_boxes(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        if (obj.findtext("name") or "").strip().lower() != "insulator":
            continue
        box = obj.find("bndbox")
        boxes.append(
            tuple(
                int(float(box.findtext(key)))
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
        )
    if not boxes:
        raise ValueError(f"未找到 insulator 框：{xml_path}")
    return boxes


def draw_blue_box(image_path: Path, box, output_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(3, round(min(image.size) / 250))
    draw.rectangle(box, outline=BLUE, width=width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=94)


def collect_samples(source: Path):
    specs = {
        "Normal": (
            source / "Normal_Insulators" / "images",
            source / "Normal_Insulators" / "labels",
        ),
        "Major": (
            source / "Defective_Insulators" / "images",
            source / "Defective_Insulators" / "labels" / "insulator",
        ),
    }
    samples = {}
    for grade, (image_dir, label_dir) in specs.items():
        items = []
        for image_path in sorted(image_dir.glob("*.jpg")):
            xml_path = label_dir / f"{image_path.stem}.xml"
            if not xml_path.is_file():
                raise FileNotFoundError(f"缺少 XML：{xml_path}")
            items.append(
                {
                    "id": image_path.stem,
                    "grade": grade,
                    "source_image": image_path,
                    "source_xml": xml_path,
                    "boxes": load_insulator_boxes(xml_path),
                }
            )
        samples[grade] = items
    return samples


def assign_splits(samples):
    rng = random.Random(SEED)
    assigned = []
    for grade in ("Normal", "Major"):
        lookup = {sample["id"]: sample for sample in samples[grade]}
        missing = [item_id for item_id in REFERENCE_IDS[grade] if item_id not in lookup]
        if missing:
            raise ValueError(f"参考图不存在：{grade} {missing}")

        reference_set = set(REFERENCE_IDS[grade])
        for item_id in REFERENCE_IDS[grade]:
            assigned.append({**lookup[item_id], "split": "reference"})

        remaining = [item for item in samples[grade] if item["id"] not in reference_set]
        train_candidates = (
            [item for item in remaining if len(item["boxes"]) == 1]
            if grade == "Normal"
            else list(remaining)
        )
        rng.shuffle(train_candidates)
        train_items = train_candidates[:TRAIN_PER_CLASS]
        train_ids = {item["id"] for item in train_items}
        for item in train_items:
            assigned.append({**item, "split": "train"})
        for item in remaining:
            if item["id"] in train_ids:
                continue
            assigned.append({**item, "split": "test"})
    return assigned


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="构建绝缘子 Normal/Major 数据集")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--clean", action="store_true", help="重建前删除既有生成数据")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.clean:
        for image_dir in IMAGE_DIRS.values():
            image_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                for image_path in image_dir.glob(pattern):
                    image_path.unlink()
        if JSON_ROOT.exists():
            shutil.rmtree(JSON_ROOT)
        old_preview = PROJECT_DIR / "参考图预览.jpg"
        if old_preview.exists():
            old_preview.unlink()

    assigned = assign_splits(collect_samples(source))
    manifest_records = []

    processed = 0
    for sample in assigned:
        split = sample["split"]
        grade = sample["grade"]
        if split in {"reference", "train"} and len(sample["boxes"]) != 1:
            raise ValueError(f"{split} 样本必须只有一个框：{sample['source_image']}")
        for box_index, box in enumerate(sample["boxes"], start=1):
            suffix = "" if len(sample["boxes"]) == 1 else f"_box{box_index}"
            stem = f"{split}_{grade}_{sample['id']}{suffix}"
            output_image = IMAGE_DIRS[split] / f"{stem}.jpg"
            draw_blue_box(sample["source_image"], box, output_image)
            manifest_records.append(
                {
                    "id": sample["id"],
                    "box_index": box_index,
                    "box": list(box),
                    "grade": grade,
                    "split": split,
                    "source_image": sample["source_image"].relative_to(source).as_posix(),
                    "source_xml": sample["source_xml"].relative_to(source).as_posix(),
                    "output_image": output_image.relative_to(PROJECT_DIR).as_posix(),
                }
            )
            processed += 1
            if processed % 100 == 0:
                print(f"已处理 {processed} 个单框样本")

    expected_images = {Path(item["output_image"]).resolve() for item in manifest_records}
    for image_dir in IMAGE_DIRS.values():
        for image_path in image_dir.glob("*.jpg"):
            if image_path.resolve() not in expected_images:
                image_path.unlink()

    counts = {}
    for split in ("reference", "train", "test"):
        counts[split] = {
            grade: sum(
                item["split"] == split and item["grade"] == grade
                for item in manifest_records
            )
            for grade in ("Normal", "Major")
        }
        counts[split]["total"] = sum(counts[split].values())

    manifest = {
        "dataset": "CPLID 绝缘子完整性二分类",
        "seed": SEED,
        "reference_ids": REFERENCE_IDS,
        "train_per_class": TRAIN_PER_CLASS,
        "counts": counts,
        "records": manifest_records,
    }
    write_json(PROJECT_DIR / "split_manifest.json", manifest)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
