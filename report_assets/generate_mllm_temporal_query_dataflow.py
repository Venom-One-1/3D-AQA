#!/usr/bin/env python3
"""Generate the MLLM temporal-query qualitative AQA data-flow diagram."""

from pathlib import Path

import generate_long_term_aqa_pipeline_flowchart as base


HERE = Path(__file__).resolve().parent
base.SVG_PATH = HERE / "mllm_temporal_query_dataflow.svg"
base.PNG_PATH = HERE / "mllm_temporal_query_dataflow.png"
C = base.COLORS


def lane(f, y, h, number, title, subtitle, accent, fill):
    f.rect(62, y, 1796, h, fill, C["line_light"], 2, 8)
    f.rect(62, y, 148, h, accent, radius=8)
    f.add_svg(
        f'<circle cx="104" cy="{y + 42}" r="23" fill="#FFFFFF" fill-opacity="0.18" '
        f'stroke="#FFFFFF" stroke-width="2"/>'
    )
    f.draw.ellipse(
        (80, y + 18, 128, y + 66),
        fill=base.rgba("#FFFFFF", 45),
        outline=base.rgba(C["white"]),
        width=2,
    )
    f.text(104, y + 42, number, 24, color=C["white"], bold=True)
    f.text(136, y + 96, title, 23, color=C["white"], bold=True)
    f.text(136, y + 137, subtitle, 16, color=C["white"])


def small_tag(f, x, y, w, label, color):
    f.rect(x, y, w, 30, C["panel"], color, 1, 5)
    f.text(x + w / 2, y + 15, label, 15, color=color, bold=True)


def build() -> None:
    f = base.Flowchart()

    f.text(64, 54, "MLLM 时序查询驱动的定性动作质量分析", 42, bold=True, anchor="start")
    f.text(66, 105, "用文字推理生成可执行查询，用量化证据完成定位、判断与训练反馈", 21, color=C["muted"], anchor="start")

    f.rect(64, 139, 1792, 72, C["panel"], C["line_light"], 2, 8, shadow=True)
    f.text(92, 175, "前置结果", 19, color=C["phase1"], bold=True, anchor="start")
    small_tag(f, 220, 157, 250, "24式招式边界已分割", C["phase1"])
    small_tag(f, 494, 157, 340, "教师与学生帧序列已通过 DTW 对齐", C["phase2"])
    f.text(1818, 175, "以下流程在每一式内部独立执行", 17, color=C["muted"], anchor="end")

    lane(f, 234, 230, "1", "教师标准", "10个参考视频", C["phase1"], C["soft_blue"])
    lane(f, 484, 230, "2", "时序查询", "动作要领 → 查询", C["phase2"], C["soft_green"])
    lane(f, 734, 230, "3", "学生诊断", "对齐 → 判断 → 反馈", C["phase3"], C["soft_amber"])

    f.box(242, 286, 205, 104, ["10个参考", "教学视频"], "data", size=24, bold=True)
    f.box(495, 286, 230, 104, ["同一招式", "已分割片段"], "data", size=23, bold=True)
    f.box(773, 274, 270, 128, ["逐帧计算量化指标", "关节夹角", "归一化相对距离 · …"], "algo", size=20, bold=True)
    f.box(1093, 286, 242, 104, ["执行时序查询", "定位目标阶段"], "algo", size=23, bold=True)
    f.box(1383, 274, 246, 128, ["10位教师在目标阶段", "对应指标值", "与局部时间序列"], "data", size=20, bold=True)
    f.database(1668, 274, 162, 128, ["教师参考范围", "10视频统计"], size=20)
    f.arrow([(447, 338), (495, 338)])
    f.arrow([(725, 338), (773, 338)])
    f.arrow([(1043, 338), (1093, 338)])
    f.arrow([(1335, 338), (1383, 338)])
    f.arrow([(1629, 338), (1668, 338)])
    f.text(1749, 429, ["保留10个原始值", "并估计合理区间"], 16, color=C["muted"])

    f.box(242, 536, 205, 104, ["该式动作要领", "文字知识"], "data", size=23, bold=True)
    f.database(495, 524, 230, 128, ["通用指标库", "角度 · 距离 · 方向", "身体部位"], size=19)
    f.box(773, 524, 270, 128, ["MLLM 生成", "结构化时序查询"], "algo", size=24, bold=True)
    f.box(1093, 512, 300, 152, ["查询描述", "何时：阶段 / 事件 / 条件", "看哪里：身体部位", "算什么：指标与方向"], "output", size=19, bold=True)
    f.box(1470, 536, 280, 104, ["程序解释并执行", "不让模型直接猜帧号"], "algo", size=21, bold=True)
    f.arrow([(447, 588), (773, 588)])
    f.arrow([(725, 588), (773, 588)])
    f.arrow([(1043, 588), (1093, 588)])
    f.arrow([(1393, 588), (1470, 588)])
    f.arrow([(1610, 536), (1610, 462), (1214, 462), (1214, 390)], "查询定位", (1430, 462), color=C["phase2"], width=4)

    f.box(242, 786, 205, 104, ["学生同一招式", "已分割片段"], "data", size=23, bold=True)
    f.box(495, 774, 230, 128, ["逐帧计算", "同一组量化指标"], "algo", size=22, bold=True)
    f.box(773, 774, 270, 128, ["DTW 对应关系", "教师目标阶段", "→ 学生对应阶段"], "algo", size=21, bold=True)
    f.box(1093, 786, 242, 104, ["学生阶段", "指标值与时间序列"], "data", size=21, bold=True)
    f.diamond(1502, 838, 270, 142, ["是否落在", "10位教师参考范围？"], size=21)
    f.box(1662, 774, 176, 128, ["结构化证据", "偏差方向", "发生阶段"], "output", size=19, bold=True)
    f.arrow([(447, 838), (495, 838)])
    f.arrow([(725, 838), (773, 838)])
    f.arrow([(1043, 838), (1093, 838)])
    f.arrow([(1335, 838), (1367, 838)])
    f.arrow([(1637, 838), (1662, 838)])
    f.arrow([(1214, 390), (1214, 440), (908, 440), (908, 774)], "教师目标阶段", (1015, 440), dashed=True, color=C["phase1"])
    f.arrow([(1749, 402), (1749, 690), (1502, 690), (1502, 767)], "10教师统计", (1640, 690), dashed=True, color=C["phase1"])

    f.rect(210, 989, 1648, 68, C["panel"], C["line_light"], 2, 8, shadow=True)
    f.text(246, 1023, "输出", 18, color=C["output_stroke"], bold=True, anchor="start")
    f.text(333, 1023, "动作要领 + 结构化证据", 22, bold=True, anchor="start")
    f.arrow([(662, 1023), (760, 1023)], color=C["output_stroke"], width=4)
    f.text(785, 1023, "MLLM 基于事实推理", 22, bold=True, anchor="start")
    f.arrow([(1060, 1023), (1158, 1023)], color=C["output_stroke"], width=4)
    f.rect(1184, 1000, 622, 46, C["output_fill"], C["output_stroke"], 2, 7)
    f.text(1495, 1023, "教练式反馈：问题位置 · 偏差事实 · 可执行纠正建议", 21, color=C["output_stroke"], bold=True)
    f.arrow([(1750, 902), (1750, 964), (1495, 964), (1495, 1000)], color=C["output_stroke"], width=4)

    f.save()


if __name__ == "__main__":
    build()
    print(f"wrote {base.SVG_PATH}")
    print(f"wrote {base.PNG_PATH}")
