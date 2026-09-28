#!/usr/bin/env python3
"""Generate a visual Bai He Liang Chi example of the MLLM temporal-query workflow."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import generate_long_term_aqa_pipeline_flowchart as base


HERE = Path(__file__).resolve().parent
base.SVG_PATH = HERE / "mllm_temporal_query_baiheliangchi.svg"
base.PNG_PATH = HERE / "mllm_temporal_query_baiheliangchi.png"
C = base.COLORS

TEACHER_IMAGE = Path("/home/sqw/VisualSearch/aqa/FrameData/KeyPose/3_baiheliangchi/00026.jpg")
STUDENT_IMAGE = HERE / "examples" / "baiheliangchi_student_example.jpg"


def photo(f, path, x, y, w, h, crop="landscape", border="#D8DEE7"):
    image = Image.open(path).convert("RGB")
    if crop == "portrait":
        target_ratio = w / h
        src_ratio = image.width / image.height
        if src_ratio > target_ratio:
            crop_w = int(image.height * target_ratio)
            left = (image.width - crop_w) // 2
            image = image.crop((left, 0, left + crop_w, image.height))
    fitted = ImageOps.fit(image, (w, h), method=Image.Resampling.LANCZOS)
    f.draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=base.rgba("#FFFFFF"))
    f.image.paste(fitted, (x, y))
    f.draw.rounded_rectangle((x, y, x + w, y + h), radius=8, outline=base.rgba(border), width=3)

    buffer = BytesIO()
    fitted.save(buffer, "JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    f.add_svg(
        f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
        f'href="data:image/jpeg;base64,{encoded}" preserveAspectRatio="none"/>'
    )
    f.add_svg(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
        f'fill="none" stroke="{border}" stroke-width="3"/>'
    )


def stage(f, x, w, number, title, accent):
    f.rect(x, 244, w, 646, C["panel"], C["line_light"], 2, 8, shadow=True)
    f.rect(x, 244, w, 64, accent, radius=8)
    f.add_svg(f'<circle cx="{x + 34}" cy="276" r="20" fill="#FFFFFF" fill-opacity="0.20"/>')
    f.draw.ellipse((x + 14, 256, x + 54, 296), fill=base.rgba("#FFFFFF", 52))
    f.text(x + 34, 276, number, 21, color=C["white"], bold=True)
    f.text(x + 68, 276, title, 22, color=C["white"], bold=True, anchor="start")


def dot(f, x, y, color, radius=7):
    f.add_svg(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}"/>')
    f.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=base.rgba(color))


def build() -> None:
    f = base.Flowchart()
    f.text(62, 54, "以“白鹤亮翅”为例：MLLM 时序查询如何生成训练反馈", 40, bold=True, anchor="start")
    f.text(64, 104, "将动作要领变成可执行查询，再用10个参考教学视频的量化证据判断学生动作", 21, color=C["muted"], anchor="start")

    f.rect(62, 139, 1796, 72, C["panel"], C["line_light"], 2, 8, shadow=True)
    f.text(94, 175, "已完成", 18, color=C["phase1"], bold=True, anchor="start")
    f.rect(195, 157, 252, 34, C["data_fill"], C["data_stroke"], 1, 6)
    f.text(321, 174, "白鹤亮翅招式边界分割", 16, color=C["phase1"], bold=True)
    f.rect(472, 157, 330, 34, C["algo_fill"], C["algo_stroke"], 1, 6)
    f.text(637, 174, "教师与学生帧序列 DTW 对齐", 16, color=C["phase2"], bold=True)
    # f.text(1815, 175, "案例中的指标分布仅用于说明流程", 16, color=C["muted"], anchor="end")

    widths = [320, 290, 390, 330, 290]
    xs = [62, 414, 736, 1158, 1520]
    colors = [C["phase1"], C["phase2"], C["phase1"], C["phase2"], C["output_stroke"]]
    titles = ["动作要领", "生成时序查询", "建立教师参考", "映射学生阶段", "比较并反馈"]
    for i, (x, w, title, color) in enumerate(zip(xs, widths, titles, colors), start=1):
        stage(f, x, w, str(i), title, color)

    # 1. Technique and an actual teacher key pose.
    photo(f, TEACHER_IMAGE, 82, 330, 280, 205)
    f.rect(82, 550, 280, 150, C["soft_blue"], C["line_light"], 1, 7)
    f.text(102, 576, "白鹤亮翅 · 定势", 19, color=C["phase1"], bold=True, anchor="start")
    f.text(102, 617, ["右手位于额头右前上方", "沉肘曲臂，右臂自然弯曲"], 20, bold=True, anchor="start", line_gap=13)
    f.text(102, 675, "自然弧度，而不是把手臂锁直", 16, color=C["muted"], anchor="start")
    f.rect(82, 727, 280, 126, C["panel"], C["line_light"], 1, 7)
    f.text(102, 754, "输入给 MLLM", 17, color=C["muted"], bold=True, anchor="start")
    f.text(102, 805, ["动作名称", "+ 动作要领", "+ 可用指标库"], 21, bold=True, anchor="start", line_gap=9)

    # 2. Query generation with only the key facts.
    f.text(559, 348, "MLLM", 26, color=C["phase2"], bold=True)
    f.text(559, 380, "不直接猜帧号", 17, color=C["muted"], bold=True)
    f.rect(444, 418, 230, 310, C["soft_green"], C["algo_stroke"], 2, 8)
    query_items = [
        ("何时", "定势阶段"),
        ("看哪里", "右肩—右肘—右腕"),
        ("算什么", "右肘夹角"),
    ]
    for idx, (label, value) in enumerate(query_items):
        cy = 468 + idx * 86
        dot(f, 470, cy, C["phase2"], 13)
        f.text(470, cy, str(idx + 1), 15, color=C["white"], bold=True)
        f.text(496, cy - 12, label, 16, color=C["muted"], bold=True, anchor="start")
        f.text(496, cy + 17, value, 19, bold=True, anchor="start")
    f.rect(444, 754, 230, 86, C["panel"], C["line_light"], 1, 7)
    f.text(559, 781, "输出：结构化查询", 18, color=C["phase2"], bold=True)
    f.text(559, 814, "交给程序执行", 17, color=C["muted"])

    # 3. Ten teacher values form a reference interval.
    f.text(931, 345, "对10个参考教学视频", 21, bold=True)
    f.text(931, 379, "查询同一动作阶段与同一指标", 17, color=C["muted"])
    f.rect(770, 414, 322, 152, C["soft_blue"], C["line_light"], 1, 7)
    f.text(792, 439, "10个教师参考值", 17, color=C["phase1"], bold=True, anchor="start")
    teacher_xs = [804, 836, 862, 889, 919, 947, 973, 1001, 1032, 1060]
    teacher_ys = [500, 486, 510, 492, 504, 480, 509, 490, 502, 484]
    for idx, (px, py) in enumerate(zip(teacher_xs, teacher_ys), start=1):
        dot(f, px, py, C["data_stroke"], 8)
        f.text(px, 536, str(idx), 13, color=C["muted"])
    f.rect(770, 594, 322, 168, C["panel"], C["line_light"], 1, 7)
    f.text(792, 621, "右肘夹角的合理范围", 18, bold=True, anchor="start")
    f.add_svg('<line x1="801" y1="684" x2="1064" y2="684" stroke="#7C8998" stroke-width="3"/>')
    f.draw.line((801, 684, 1064, 684), fill=base.rgba(C["line"]), width=3)
    f.rect(850, 663, 158, 42, C["data_fill"], C["data_stroke"], 2, 8)
    f.text(929, 684, "教师参考范围", 16, color=C["phase1"], bold=True)
    # for px in teacher_xs:
    #     mapped_x = 850 + (px - 804) / (1060 - 804) * 158
    #     dot(f, int(mapped_x), 684, C["data_stroke"], 5)
    f.text(931, 735, "由10个视频统计，而非单一教师", 16, color=C["muted"])
    f.rect(797, 792, 268, 48, C["data_fill"], C["data_stroke"], 1, 7)
    f.text(931, 816, "输出：指标参考区间", 18, color=C["phase1"], bold=True)

    # 4. DTW maps the teacher stage to a student stage.
    f.text(1323, 344, "沿已建立的 DTW 对应关系", 19, bold=True)
    f.text(1323, 375, "教师目标阶段 → 学生对应阶段", 17, color=C["muted"])
    photo(f, TEACHER_IMAGE, 1182, 414, 132, 225, crop="portrait")
    photo(f, STUDENT_IMAGE, 1350, 414, 132, 225, crop="portrait", border=C["phase2"])
    f.arrow([(1318, 526), (1345, 526)], color=C["phase2"], width=4)
    f.text(1248, 663, "教师阶段", 16, color=C["phase1"], bold=True)
    f.text(1416, 663, "学生阶段", 16, color=C["phase2"], bold=True)
    f.rect(1182, 705, 300, 106, C["soft_green"], C["algo_stroke"], 1, 7)
    f.text(1204, 734, "在学生对应帧计算", 17, color=C["phase2"], bold=True, anchor="start")
    f.text(1204, 773, "右手臂关节夹角，相对高度", 20, bold=True, anchor="start")
    # f.text(1323, 844, "不在学生视频中另挑“最好看”的帧", 15, color=C["muted"])

    # 5. Compare with the distribution and generate evidence-grounded feedback.
    f.text(1665, 348, "学生值", 21, bold=True)
    f.text(1665, 378, "与教师范围比较", 17, color=C["muted"])
    f.rect(1546, 418, 238, 154, C["panel"], C["line_light"], 1, 7)
    f.add_svg('<line x1="1570" y1="486" x2="1760" y2="486" stroke="#7C8998" stroke-width="3"/>')
    f.draw.line((1570, 486, 1760, 486), fill=base.rgba(C["line"]), width=3)
    f.rect(1600, 465, 104, 42, C["data_fill"], C["data_stroke"], 2, 8)
    f.text(1652, 486, "参考范围", 15, color=C["phase1"], bold=True)
    dot(f, 1733, 486, C["output_stroke"], 9)
    f.text(1733, 528, "学生", 15, color=C["output_stroke"], bold=True)
    f.rect(1546, 598, 238, 92, C["output_fill"], C["output_stroke"], 2, 7)
    f.text(1665, 625, "示例判断", 16, color=C["output_stroke"], bold=True)
    f.text(1665, 660, "右手可能过低", 23, color=C["output_stroke"], bold=True)
    f.rect(1546, 716, 238, 132, C["panel"], C["output_stroke"], 2, 7)
    f.text(1566, 742, "教练式反馈", 16, color=C["output_stroke"], bold=True, anchor="start")
    f.text(1566, 791, ["右手相对头部的高度偏低, ", "建议调整右手位置。"], 18, bold=True, anchor="start", line_gap=10)

    # Flow arrows between the five visual stages.
    arrow_y = 566
    for start, end, color in [(382, 414, C["line"]), (704, 736, C["line"]), (1126, 1158, C["line"]), (1488, 1520, C["line"])]:
        f.arrow([(start, arrow_y), (end, arrow_y)], color=color, width=4)

    f.rect(62, 926, 1796, 102, C["panel"], C["line_light"], 2, 8, shadow=True)
    f.text(92, 953, "职责分工", 17, color=C["muted"], bold=True, anchor="start")
    f.rect(218, 946, 714, 60, C["soft_green"], C["algo_stroke"], 1, 7)
    f.text(575, 976, "MLLM：动作要领 → 时序查询；量化证据 → 自然语言反馈", 20, color=C["phase2"], bold=True)
    f.rect(960, 946, 852, 60, C["soft_blue"], C["data_stroke"], 1, 7)
    f.text(1386, 976, "程序：DTW定位 · 指标计算 · 10教师统计 · 范围判断", 20, color=C["phase1"], bold=True)

    f.save()


if __name__ == "__main__":
    build()
    print(f"wrote {base.SVG_PATH}")
    print(f"wrote {base.PNG_PATH}")
