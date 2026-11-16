# -*- coding: utf-8 -*-
"""使用4组参考图文对，为60张训练图生成同名JSON。"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_config import (
    prompt1,
    prompt_json_COT,
    reference_result,
    reference_start,
    system_prompt,
)


REFERENCE_DIR = ROOT / "reference image"
TRAIN_DIR = ROOT / "train image"


def load_env(path):
    if not path or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def b64(path):
    return base64.b64encode(path.read_bytes()).decode("ascii")


def text_message(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def image_message(path):
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + b64(path)},
            }
        ],
    }


def grade_from_name(path, prefix):
    match = re.match(rf"{prefix}_(Normal|Major)_", path.stem)
    if not match:
        raise ValueError(f"无法从文件名读取等级：{path.name}")
    return match.group(1)


def read_reference_pairs():
    pairs = []
    for image in sorted(REFERENCE_DIR.glob("reference_*.jpg")):
        json_path = image.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"缺少参考JSON：{json_path}")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        pairs.append((image, grade_from_name(image, "reference"), data))
    if len(pairs) != 4:
        raise ValueError(f"参考图文对必须为4组，当前为 {len(pairs)} 组")
    return pairs


def build_messages(target, target_grade, reference_pairs):
    messages = [
        {"role": "system", "content": system_prompt()},
        text_message(prompt1() + "\n\n" + reference_start()),
    ]
    for index, (image, grade, data) in enumerate(reference_pairs, start=1):
        messages.append(image_message(image))
        messages.append(
            text_message(
                reference_result(
                    index=index,
                    grade=grade,
                    data=data,
                    has_next=index < len(reference_pairs),
                )
            )
        )
    messages.append(image_message(target))
    messages.append(text_message(prompt_json_COT(target_grade)))
    return messages


def parse_json(text, expected_grade):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    data = json.loads(cleaned[cleaned.index("{") : cleaned.rindex("}") + 1])
    if set(data) != {"has_insulator", "grade_result"}:
        raise ValueError(f"JSON字段不正确：{list(data)}")
    grade = str(data["grade_result"]).split("|", 1)[0].strip()
    if grade != expected_grade:
        raise ValueError(f"模型等级 {grade} 与人工等级 {expected_grade} 不一致")
    return data


def main():
    parser = argparse.ArgumentParser(description="生成60张训练图的同名JSON")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env(args.env_file)

    references = read_reference_pairs()
    targets = sorted(TRAIN_DIR.glob("train_*.jpg"))
    if len(targets) != 60:
        raise ValueError(f"训练图应为60张，当前为 {len(targets)} 张")
    if args.limit is not None:
        targets = targets[: args.limit]

    if args.dry_run:
        for target in targets:
            grade = grade_from_name(target, "train")
            print(target.name, grade, len(build_messages(target, grade, references)))
        return

    api_key = os.environ.get("COMMERCIAL_MLLM_API_KEY", "").strip()
    base_url = os.environ.get("COMMERCIAL_MLLM_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError("缺少 COMMERCIAL_MLLM_API_KEY 或 COMMERCIAL_MLLM_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    completed = 0
    for index, target in enumerate(targets, start=1):
        output = target.with_suffix(".json")
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(targets)}] 跳过：{output.name}")
            continue
        grade = grade_from_name(target, "train")
        for attempt in range(1, 6):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=build_messages(target, grade, references),
                    stream=False,
                    temperature=0.0,
                )
                data = parse_json(response.choices[0].message.content, grade)
                output.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                completed += 1
                print(f"[{index}/{len(targets)}] 完成：{target.name} → {output.name}")
                break
            except Exception as error:
                if attempt == 5:
                    raise
                print(f"[{index}/{len(targets)}] 第 {attempt} 次失败：{error}")
                time.sleep(3)
    print(f"本次新增：{completed}；训练JSON总数：{len(list(TRAIN_DIR.glob('train_*.json')))}")


if __name__ == "__main__":
    main()
