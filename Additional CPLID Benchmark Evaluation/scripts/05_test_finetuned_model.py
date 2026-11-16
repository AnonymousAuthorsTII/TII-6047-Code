# -*- coding: utf-8 -*-
"""在服务器上使用 Ollama 测试绝缘子完整性微调模型。"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


# ========================= Portable defaults =========================
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "paper_gllm_insulator_60_language_e300:latest"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
REFERENCE_DIR = PROJECT_DIR / "reference image"
TEST_DIR = PROJECT_DIR / "test image"
RESULT_ROOT = PROJECT_DIR / "results"

REFERENCE_ITEMS = (
    (
        "reference_Normal_0049.jpg",
        "Normal",
        {
            "has_insulator": "存在|蓝色框内可辨认出由多片盘形绝缘子连续排列组成的绝缘子串，片体轮廓及片间连接清晰可见。",
            "grade_result": "Normal|蓝色框内所有可见绝缘子片排列连续，片体轮廓完整，未见明确缺失、断裂或破损；局部片体相互重叠，但未见可确认为完整性缺陷的证据。",
        },
    ),
    (
        "reference_Normal_0060.jpg",
        "Normal",
        {
            "has_insulator": "存在|蓝色框内可清楚辨认由左下方向右上方延伸的绝缘子串，多片棕红色伞盘沿串排列。",
            "grade_result": "Normal|蓝色框内所有可见绝缘子片排列连续，伞盘轮廓完整，未见明确的绝缘子片缺失、断裂或破损，符合Normal标准。",
        },
    ),
    (
        "reference_Major_000.jpg",
        "Major",
        {
            "has_insulator": "存在|蓝色框内可辨认出两串近水平排列的盘形绝缘子，绝缘子片及其连接关系清晰可见。",
            "grade_result": "Major|蓝色框内下方绝缘子串靠右端可见绝缘子片伞裙明显残缺，原本应呈完整盘状的部位仅剩较窄的残余部分，与相邻完整绝缘子片形成明显差异，属于明确破损，符合至少存在一个绝缘子片缺失、断裂或破损的Major判据。",
        },
    ),
    (
        "reference_Major_150.jpg",
        "Major",
        {
            "has_insulator": "存在|蓝色框内可辨认出由多片盘形绝缘子组成的绝缘子串，沿左上至右下方向排列。",
            "grade_result": "Major|蓝色框内中部偏左、上排绝缘子在相邻完整盘片之间存在明显破损：该处盘形伞裙大面积缺失，仅见中央连接部位及不完整残片，形成明显缺口，符合至少一个绝缘子片明确缺失或破损的判定条件。",
        },
    ),
)

SYSTEM_PROMPT = (
    "【角色】你是一名电网输电设备缺陷定级专家。你的任务是仅依据图像中蓝色框内的"
    "可见证据和专家决策树，对绝缘子完整性进行可靠定级。不得根据文件名、背景或不可见信息推测。"
)

TASK_PROMPT = """【缺陷定级任务】
本次需要完成绝缘子完整性缺陷定级，等级包括 Normal 和 Major。

【前置说明】
1. 每张输入图片只有一个蓝色框，只判断该蓝色框内的绝缘子串。
2. 根据绝缘子片的排列连续性、轮廓完整性和局部缺口进行判断。
3. 不得把遮挡、模糊、反光、背景结构或蓝色框线误判为绝缘子缺陷。

【定级决策树】
1. has_insulator：蓝色框内是否存在可辨认的绝缘子串？
   ├─ 不存在 → 图像无有效目标，无法进行 Normal/Major 定级。
   └─ 存在 → 进入步骤2。

2. grade_result：是否至少存在一个明确缺失、断裂或破损的绝缘子片？
   ├─ 否 → grade_result=Normal。
   └─ 是 → grade_result=Major。

【输出格式】
必须输出合法 JSON，字段不可增减；每项值使用“结论|充分理由”格式。不得输出 Markdown 代码块或额外文字。
{
  "has_insulator": "<存在|不存在>|<蓝色框内是否存在可辨认绝缘子串的理由>",
  "grade_result": "<Normal|Major>|<依据绝缘子片完整性给出的完整理由>"
}

【图片参考示例】下面依次提供4张人工确认的参考图片及其定级 JSON，请学习其判断方式和输出格式。

示例1："""

INFERENCE_PROMPT = "请按照上述前置说明、决策树和参考示例，独立输出待测试图片的定级 JSON。只输出合法 JSON。"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VALID_GRADES = ("Normal", "Major")


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def image_message(path: Path) -> dict[str, Any]:
    return {"role": "user", "images": [encode_image(path)]}


def text_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def reference_result_text(index: int, grade: str, result: dict[str, str]) -> str:
    """生成参考图后的文字；实际请求与提示词预览共用，避免二者漂移。"""
    text = (
        f"这是一张被人工定级为“{grade}”的参考图片，定级过程和结果如下：\n"
        + json.dumps(result, ensure_ascii=False, indent=2)
    )
    if index < len(REFERENCE_ITEMS):
        text += f"\n\n示例{index + 1}："
    else:
        text += "\n\n【待处理图片】下面是需要进行缺陷定级的图片："
    return text


def build_messages(reference_dir: Path, test_image: Path) -> list[dict[str, Any]]:
    """按照 grade_rarity 的文字/图片交替方式构造 Ollama 消息。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        text_message(TASK_PROMPT),
    ]
    for index, (filename, grade, result) in enumerate(REFERENCE_ITEMS, start=1):
        messages.append(image_message(reference_dir / filename))
        messages.append(text_message(reference_result_text(index, grade, result)))
    messages.append(image_message(test_image))
    messages.append(text_message(INFERENCE_PROMPT))
    return messages


def expected_grade(path: Path) -> str:
    match = re.match(r"test_(Normal|Major)_", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"无法从文件名解析真实等级：{path.name}")
    return match.group(1).title()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """允许模型偶尔输出代码块，但最终只提取第一个可解析 JSON 对象。"""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def extract_grade(text: str, parsed: dict[str, Any] | None) -> str:
    if parsed:
        value = str(parsed.get("grade_result", ""))
        match = re.search(r"\b(Normal|Major)\b", value, flags=re.I)
        if match:
            return match.group(1).title()
    labels = re.findall(r"\b(Normal|Major)\b", text, flags=re.I)
    return labels[-1].title() if labels else ""


def request_ollama(
    session: requests.Session,
    api_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    retries: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 2048,
            "num_ctx": 50000,
        },
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "")
            if not content.strip():
                raise RuntimeError("模型返回了空内容")
            return content, result
        except (requests.RequestException, ValueError, RuntimeError) as error:
            last_error = error
            print(f"  请求失败（{attempt}/{retries}）：{error}", flush=True)
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"达到最大重试次数：{last_error}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    """将汇总指标写成长表 CSV；逐图模型输出仍保留为 JSON。"""
    rows: list[tuple[str, str, Any, str]] = [
        ("dataset", "total_samples", metrics["total"], "count"),
        ("dataset", "correct_samples", metrics["correct"], "count"),
        ("dataset", "unparsed_samples", metrics["unparsed"], "count"),
        ("overall", "accuracy", metrics["accuracy"] * 100, "percent"),
        ("overall", "macro_precision", metrics["macro_precision"] * 100, "percent"),
        ("overall", "macro_recall", metrics["macro_recall"] * 100, "percent"),
        ("overall", "macro_f1", metrics["macro_f1"] * 100, "percent"),
        ("severity", "major_recall", metrics["major_recall"] * 100, "percent"),
        ("severity", "over_grading_rate", metrics["over_grading_rate"] * 100, "percent"),
        ("severity", "under_grading_rate", metrics["under_grading_rate"] * 100, "percent"),
        ("severity", "nmae", metrics["nmae"] * 100, "percent"),
    ]
    for label in VALID_GRADES:
        values = metrics["per_class"][label]
        for name in ("precision", "recall", "f1", "support"):
            value = values[name] if name == "support" else values[name] * 100
            rows.append((label, name, value, "count" if name == "support" else "percent"))
    for true_label, predictions in metrics["confusion_matrix"].items():
        for predicted_label, value in predictions.items():
            rows.append(("confusion_matrix", f"actual_{true_label}_pred_{predicted_label}", value, "count"))
    for name, value in metrics.get("run_counts", {}).items():
        rows.append(("run", name, value, "count"))
    if "elapsed_seconds" in metrics:
        rows.append(("run", "elapsed_seconds", metrics["elapsed_seconds"], "seconds"))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("scope", "metric", "value", "unit"))
        writer.writerows(rows)


def calculate_metrics(result_dir: Path, model: str) -> dict[str, Any]:
    records = []
    for path in sorted(result_dir.glob("test_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("true_grade") in VALID_GRADES:
                records.append(record)
        except (OSError, json.JSONDecodeError) as error:
            print(f"警告：无法读取结果 {path.name}：{error}", flush=True)

    matrix = {
        true: {pred: 0 for pred in (*VALID_GRADES, "Unparsed")}
        for true in VALID_GRADES
    }
    for record in records:
        predicted = record.get("answer")
        if predicted not in VALID_GRADES:
            predicted = "Unparsed"
        matrix[record["true_grade"]][predicted] += 1

    per_class: dict[str, Any] = {}
    for label in VALID_GRADES:
        tp = matrix[label][label]
        fn = sum(matrix[label].values()) - tp
        fp = sum(matrix[other][label] for other in VALID_GRADES if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    total = len(records)
    correct = sum(matrix[label][label] for label in VALID_GRADES)
    macro_precision = sum(per_class[label]["precision"] for label in VALID_GRADES) / 2
    macro_recall = sum(per_class[label]["recall"] for label in VALID_GRADES) / 2
    normal_support = sum(matrix["Normal"].values())
    major_support = sum(matrix["Major"].values())
    over_grading = matrix["Normal"]["Major"]
    under_grading = matrix["Major"]["Normal"]
    ordinal_error = over_grading + under_grading
    return {
        "model": model,
        "total": total,
        "correct": correct,
        "unparsed": sum(matrix[label]["Unparsed"] for label in VALID_GRADES),
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": sum(per_class[label]["f1"] for label in VALID_GRADES) / 2,
        "major_recall": per_class["Major"]["recall"],
        "over_grading_rate": over_grading / total if total else 0.0,
        "under_grading_rate": under_grading / total if total else 0.0,
        "nmae": (
            (over_grading / normal_support if normal_support else 0.0)
            + (under_grading / major_support if major_support else 0.0)
        ) / 2,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def validate_inputs(reference_dir: Path, test_dir: Path) -> list[Path]:
    missing = [reference_dir / item[0] for item in REFERENCE_ITEMS if not (reference_dir / item[0]).is_file()]
    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"缺少参考图片：\n{names}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"测试图片目录不存在：{test_dir}")
    images = sorted(
        path for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and path.name.startswith("test_")
    )
    if not images:
        raise FileNotFoundError(f"测试图片目录中没有 test_* 图片：{test_dir}")
    for image in images:
        expected_grade(image)
    return images


def save_prompt_preview(path: Path) -> None:
    """按实际 Ollama 消息顺序输出文本预览，图片位置用文件名占位。"""
    parts = ["[system]\n" + SYSTEM_PROMPT, "[user]\n" + TASK_PROMPT]
    for index, (filename, grade, result) in enumerate(REFERENCE_ITEMS, start=1):
        parts.append(f"[user image]\n<{filename}>")
        parts.append("[user]\n" + reference_result_text(index, grade, result))
    parts.extend(("[user image]\n<待测试图片>", "[user]\n" + INFERENCE_PROMPT))
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试绝缘子完整性微调模型")
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama 模型名称")
    parser.add_argument("--api-url", default=OLLAMA_API_URL)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--limit", type=int, help="仅测试前 N 张，便于小规模检查")
    parser.add_argument("--timeout", type=int, default=1200, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有同名结果")
    parser.add_argument("--dry-run", action="store_true", help="只检查路径和提示词，不调用模型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = validate_inputs(args.reference_dir, args.test_dir)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit 必须大于 0")
        images = images[: args.limit]

    result_dir = args.result_root / "predictions"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_prompt_preview(args.result_root / "prompt_preview.txt")

    normal_count = sum(expected_grade(image) == "Normal" for image in images)
    major_count = len(images) - normal_count
    print(f"模型：{args.model}")
    print(f"测试图片：{len(images)}（Normal={normal_count}, Major={major_count}）")
    print(f"结果目录：{result_dir}")
    if args.dry_run:
        sample_messages = build_messages(args.reference_dir, images[0])
        print(f"Dry run 通过；示例消息段数：{len(sample_messages)}")
        return

    counts = {"completed": 0, "skipped": 0, "failed": 0}
    session = requests.Session()
    started_at = time.time()

    for index, image in enumerate(images, start=1):
        output_path = result_dir / f"{image.stem}.json"
        if output_path.exists() and not args.overwrite:
            counts["skipped"] += 1
            print(f"[{index}/{len(images)}] 跳过：{image.name}", flush=True)
            continue

        truth = expected_grade(image)
        print(f"[{index}/{len(images)}] 测试：{image.name}（真实={truth}）", flush=True)
        try:
            response_text, raw_result = request_ollama(
                session=session,
                api_url=args.api_url,
                model=args.model,
                messages=build_messages(args.reference_dir, image),
                timeout=args.timeout,
                retries=args.retries,
            )
            parsed = extract_json_object(response_text)
            answer = extract_grade(response_text, parsed)
            record = {
                "picture": image.name,
                "true_grade": truth,
                "response": response_text,
                "parsed_json": parsed,
                "answer": answer,
                "result": 1 if answer == truth else (0 if answer in VALID_GRADES else -1),
                "correct": int(answer == truth),
                "model": args.model,
                "prompt_eval_count": raw_result.get("prompt_eval_count"),
                "eval_count": raw_result.get("eval_count"),
                "total_duration": raw_result.get("total_duration"),
            }
            write_json(output_path, record)
            counts["completed"] += 1
            print(f"  预测={answer or '无法解析'}，correct={record['correct']}", flush=True)
        except Exception as error:  # 保存失败记录，便于之后定位并用 --overwrite 重跑
            write_json(
                output_path,
                {
                    "picture": image.name,
                    "true_grade": truth,
                    "response": "",
                    "parsed_json": None,
                    "answer": "",
                    "result": -1,
                    "correct": 0,
                    "model": args.model,
                    "error": str(error),
                },
            )
            counts["failed"] += 1
            print(f"  失败：{error}", flush=True)

        metrics = calculate_metrics(result_dir, args.model)
        metrics["run_counts"] = counts
        metrics["elapsed_seconds"] = round(time.time() - started_at, 2)
        write_metrics_csv(args.result_root / "metrics.csv", metrics)

    metrics = calculate_metrics(result_dir, args.model)
    metrics["run_counts"] = counts
    metrics["elapsed_seconds"] = round(time.time() - started_at, 2)
    write_metrics_csv(args.result_root / "metrics.csv", metrics)
    print("\n测试完成：")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
