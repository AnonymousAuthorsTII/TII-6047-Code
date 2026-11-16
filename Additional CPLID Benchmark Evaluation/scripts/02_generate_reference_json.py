# -*- coding: utf-8 -*-
"""先为4张参考图片生成同名 JSON；消息结构严格采用文字→图片→文字。"""

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

from prompt_config import prompt_json_COT, reference_bootstrap_intro, system_prompt


REFERENCE_DIR = ROOT / "reference image"


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


def text_message(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def grade_from_name(path):
    match = re.match(r"reference_(Normal|Major)_", path.stem)
    if not match:
        raise ValueError(f"无法从文件名读取等级：{path.name}")
    return match.group(1)


def parse_json(text, expected_grade):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    data = json.loads(cleaned[cleaned.index("{") : cleaned.rindex("}") + 1])
    if set(data) != {"has_insulator", "grade_result"}:
        raise ValueError(f"JSON 字段不正确：{list(data)}")
    grade = str(data["grade_result"]).split("|", 1)[0].strip()
    if grade != expected_grade:
        raise ValueError(f"模型等级 {grade} 与人工等级 {expected_grade} 不一致")
    return data


def messages_for(image_path, grade):
    return [
        {"role": "system", "content": system_prompt()},
        text_message(reference_bootstrap_intro()),
        image_message(image_path),
        text_message(prompt_json_COT(grade)),
    ]


def main():
    parser = argparse.ArgumentParser(description="生成参考图片同名 JSON")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env(args.env_file)

    images = sorted(REFERENCE_DIR.glob("reference_*.jpg"))
    if len(images) != 4:
        raise ValueError(f"参考图片应为4张，当前为 {len(images)} 张")
    if args.dry_run:
        for image in images:
            print(image.name, grade_from_name(image), len(messages_for(image, grade_from_name(image))))
        return

    api_key = os.environ.get("COMMERCIAL_MLLM_API_KEY", "").strip()
    base_url = os.environ.get("COMMERCIAL_MLLM_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError("缺少 COMMERCIAL_MLLM_API_KEY 或 COMMERCIAL_MLLM_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    for index, image in enumerate(images, start=1):
        output = image.with_suffix(".json")
        if output.exists() and not args.overwrite:
            print(f"[{index}/4] 跳过：{output.name}")
            continue
        grade = grade_from_name(image)
        for attempt in range(1, 6):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages_for(image, grade),
                    stream=False,
                    temperature=0.0,
                )
                data = parse_json(response.choices[0].message.content, grade)
                output.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"[{index}/4] 完成：{image.name} → {output.name}")
                break
            except Exception as error:
                if attempt == 5:
                    raise
                print(f"第 {attempt} 次失败：{error}")
                time.sleep(3)


if __name__ == "__main__":
    main()
