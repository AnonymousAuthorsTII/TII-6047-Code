"""将参考图文对和训练图文对构造成Qwen风格多模态微调JSON。"""

import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_config import (
    prompt1,
    prompt_inference,
    reference_result,
    reference_start,
    system_prompt,
)


REFERENCE_DIR = ROOT / "reference image"
TRAIN_DIR = ROOT / "train image"
OUTPUT_PATH = ROOT / "insulator_60.json"
RANDOM_SEED = 6047


def relative_path(path):
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_pairs(folder, prefix, expected_count):
    pairs = []
    for image in sorted(folder.glob(f"{prefix}_*.jpg")):
        json_path = image.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"缺少同名JSON：{json_path}")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if set(data) != {"has_insulator", "grade_result"}:
            raise ValueError(f"JSON字段不正确：{json_path}")
        pairs.append((image, data))
    if len(pairs) != expected_count:
        raise ValueError(f"{prefix}图文对应为{len(pairs)}组，预期{expected_count}组")
    return pairs


def grade(data):
    return str(data["grade_result"]).split("|", 1)[0].strip()


def build_user_content(reference_pairs):
    parts = [prompt1(), reference_start()]
    for index, (_, data) in enumerate(reference_pairs, start=1):
        parts.append("<image>")
        parts.append(
            reference_result(
                index=index,
                grade=grade(data),
                data=data,
                has_next=index < len(reference_pairs),
            )
        )
    parts.extend(["<image>", prompt_inference()])
    return "\n\n".join(parts)


def main():
    references = load_pairs(REFERENCE_DIR, "reference", 4)
    grade_order = {"Normal": 0, "Major": 1}
    references.sort(key=lambda item: (grade_order[grade(item[1])], item[0].name))
    training_pairs = load_pairs(TRAIN_DIR, "train", 60)
    user_content = build_user_content(references)
    reference_images = [relative_path(image) for image, _ in references]

    samples = []
    for image, answer in training_pairs:
        images = reference_images + [relative_path(image)]
        if user_content.count("<image>") != len(images):
            raise ValueError(f"图片占位符不匹配：{image.name}")
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": json.dumps(answer, ensure_ascii=False, indent=2),
                    },
                ],
                "images": images,
            }
        )

    random.Random(RANDOM_SEED).shuffle(samples)
    OUTPUT_PATH.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成：{OUTPUT_PATH}")
    print(f"样本数：{len(samples)}")
    print(f"每条样本图片数：{len(samples[0]['images'])}")
    print(f"每条样本<image>数：{samples[0]['messages'][1]['content'].count('<image>')}")
    (ROOT / "微调语料提示词预览.txt").write_text(
        "[System Prompt]\n"
        + system_prompt()
        + "\n\n[User Prompt]\n"
        + user_content
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
