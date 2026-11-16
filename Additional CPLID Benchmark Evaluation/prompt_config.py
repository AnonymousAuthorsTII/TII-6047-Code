"""按照 rarity 项目的分段方式提供中文提示词。"""

import json


def system_prompt():
    return (
        "【角色】你是一名电网输电设备缺陷定级专家。你的任务是仅依据图像中蓝色框内的"
        "可见证据和专家决策树，对绝缘子完整性进行可靠定级。不得根据文件名、背景或不可见信息推测。"
    )


def prompt1():
    return """【缺陷定级任务】
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
}"""


def reference_start():
    return "【图片参考示例】下面依次提供4张人工确认的参考图片及其定级 JSON，请学习其判断方式和输出格式。\n\n示例1："


def reference_result(index, grade, data, has_next=True):
    text = (
        f"这是一张被人工定级为“{grade}”的参考图片，定级过程和结果如下：\n"
        + json.dumps(data, ensure_ascii=False, indent=2)
    )
    if has_next:
        text += f"\n\n示例{index + 1}："
    else:
        text += "\n\n【待处理图片】下面是需要进行缺陷定级的图片："
    return text


def prompt_json_COT(grade):
    return f"""该图片的人工确认真实等级为“{grade}”。
请严格保持该等级不变，根据前置说明和决策树，为该图片生成定级 JSON。
理由必须来自蓝色框内的可见证据，不得提及文件名、XML、红框、数据目录或“已知标签”。
只输出合法 JSON。"""


def prompt_inference():
    return "请按照上述前置说明、决策树和参考示例，独立输出待测试图片的定级 JSON。只输出合法 JSON。"


def q1_task_prompt():
    return """【缺陷定级任务】
本次需要判断蓝色框内绝缘子串的完整性，等级包括 Normal 和 Major。

请注意：
1. 每张输入图片只有一个蓝色框，只判断该蓝色框内的绝缘子串。
2. Normal 表示所有可见绝缘子片均连续且完整，没有明确缺失、断裂或破损。
3. Major 表示至少存在一个明确缺失、断裂或破损的绝缘子片。
4. 请根据绝缘子片的连续性、轮廓完整性和局部缺口判断，不得把遮挡、模糊、反光、背景结构或蓝色框线误判为缺陷。"""


def q1_objective_prompt():
    return "【待定级图片】下面是需要进行缺陷定级的图片："


def q1_question():
    return "请判断待定级图片的缺陷等级，只输出最终等级，格式必须为 ***Normal*** 或 ***Major***。"


def q1_system_prompt():
    return (
        "【角色】你是一名电网输电设备缺陷定级专家。请仅依据图像中蓝色框内的可见证据，"
        "对绝缘子完整性进行可靠定级。不得根据文件名、背景或不可见信息推测。"
    )


def reference_bootstrap_intro():
    return prompt1() + "\n\n【待生成参考 JSON 的图片】下面提供一张人工选定的参考图片："
