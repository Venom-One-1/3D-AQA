#!/usr/bin/env python3
"""Generate a Chinese explainer for SMPL-pose DTW in the Tai Chi project."""

from __future__ import annotations

import math
from pathlib import Path

import generate_long_term_aqa_pipeline_flowchart as base


WIDTH, HEIGHT = 1920, 1080
HERE = Path(__file__).resolve().parent
SVG_PATH = HERE / "dtw_algorithm_flow_taichi.svg"
PNG_PATH = HERE / "dtw_algorithm_flow_taichi.png"

C = base.COLORS
C.update(
    {
        "navy": "#294F6B",
        "teal": "#287C71",
        "amber": "#A56E1B",
        "coral": "#B85E4C",
        "matrix_0": "#27766F",
        "matrix_1": "#4A9189",
        "matrix_2": "#79AAA4",
        "matrix_3": "#A9C8C3",
        "matrix_4": "#D5E5E2",
        "matrix_5": "#EEF4F2",
    }
)


def line(
    f: base.Flowchart,
    points: list[tuple[float, float]],
    color: str,
    width: int = 3,
    dashed: bool = False,
) -> None:
    coords = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    f.add_svg(
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash}/>'
    )
    if dashed:
        for p1, p2 in zip(points, points[1:]):
            f._dashed_line(p1, p2, color, width)
    else:
        f.draw.line(points, fill=base.rgba(color), width=width)


def circle(
    f: base.Flowchart,
    cx: float,
    cy: float,
    r: float,
    fill: str,
    stroke: str | None = None,
    width: int = 2,
) -> None:
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    f.add_svg(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{stroke_attr}/>')
    f.draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=base.rgba(fill),
        outline=base.rgba(stroke) if stroke else None,
        width=width,
    )


def pose(
    f: base.Flowchart,
    cx: float,
    cy: float,
    phase: float,
    scale: float,
    color: str,
) -> None:
    """Draw one simple Tai Chi pose; phase changes arm and stance positions."""
    head = (cx, cy - 25 * scale)
    neck = (cx, cy - 16 * scale)
    hip = (cx + (phase - 0.5) * 4 * scale, cy + 5 * scale)
    shoulder_l = (cx - 9 * scale, cy - 12 * scale)
    shoulder_r = (cx + 9 * scale, cy - 12 * scale)
    hand_l = (cx - (10 + 16 * phase) * scale, cy + (-8 + 18 * phase) * scale)
    hand_r = (cx + (25 - 13 * phase) * scale, cy + (-3 - 12 * phase) * scale)
    elbow_l = ((shoulder_l[0] + hand_l[0]) / 2, (shoulder_l[1] + hand_l[1]) / 2 - 3 * scale)
    elbow_r = ((shoulder_r[0] + hand_r[0]) / 2, (shoulder_r[1] + hand_r[1]) / 2 + 2 * scale)
    foot_l = (cx - (13 + 5 * phase) * scale, cy + 31 * scale)
    foot_r = (cx + (16 + 5 * phase) * scale, cy + 31 * scale)
    knee_l = ((hip[0] + foot_l[0]) / 2 - 2 * scale, cy + 17 * scale)
    knee_r = ((hip[0] + foot_r[0]) / 2 + 2 * scale, cy + 17 * scale)
    skeleton = [
        (neck, hip),
        (shoulder_l, neck, shoulder_r),
        (shoulder_l, elbow_l, hand_l),
        (shoulder_r, elbow_r, hand_r),
        (hip, knee_l, foot_l),
        (hip, knee_r, foot_r),
    ]
    for segment in skeleton:
        line(f, list(segment), color, max(2, int(2.2 * scale)))
    circle(f, head[0], head[1], 5 * scale, C["panel"], color, max(1, int(1.7 * scale)))
    for joint in (neck, hip, elbow_l, elbow_r, knee_l, knee_r):
        circle(f, joint[0], joint[1], 2.1 * scale, color)


def arrow_icon(
    f: base.Flowchart,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
) -> None:
    f.arrow([(int(x1), int(y1)), (int(x2), int(y2))], color=color, width=3)


def phase_panel(
    f: base.Flowchart,
    x: int,
    width: int,
    color: str,
    number: str,
    title: str,
) -> None:
    f.rect(x, 165, width, 735, C["panel"], C["line_light"], 2, 8)
    f.rect(x, 165, width, 66, color, radius=8)
    circle(f, x + 33, 198, 20, color, C["white"], 2)
    f.text(x + 33, 197, number, 22, color=C["white"], bold=True)
    f.text(x + 65, 198, title, 25, color=C["white"], bold=True, anchor="start")


def draw_sequence(
    f: base.Flowchart,
    y: int,
    label: str,
    note: str,
    phases: list[float],
    color: str,
) -> None:
    f.text(82, y - 38, label, 20, color=color, bold=True, anchor="start")
    f.text(415, y - 60, note, 14, color=C["muted"], anchor="end")
    start_x = 94
    gap = 45 if len(phases) > 6 else 61
    for index, phase in enumerate(phases):
        x = start_x + index * gap
        f.rect(x - 20, y - 27, 40, 66, C["bg"], C["line_light"], 1, 5)
        pose(f, x, y + 1, phase, 0.72, color)
    line(f, [(82, y + 49), (419, y + 49)], C["line_light"], 2)
    arrow_icon(f, 394, y + 49, 419, y + 49, C["line"])


def draw_matrix(f: base.Flowchart) -> None:
    rows, cols, cell = 9, 10, 42
    x0, y0 = 562, 302
    path_forward = [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (6, 7),
        (7, 8),
        (8, 9),
    ]
    path_set = set(path_forward)
    palette = [C[f"matrix_{i}"] for i in range(6)]

    f.text(x0 + cols * cell / 2, 262, "教师参考帧  rj   →", 19, color=C["navy"], bold=True)
    f.rect(972, 247, 63, 28, C["algo_fill"], C["algo_stroke"], 1, 5)
    f.text(1003, 261, "M × N", 15, color=C["teal"], bold=True)
    f.text(495, 458, ["学生帧", "si", "↓"], 19, color=C["teal"], bold=True)
    for row in range(rows):
        expected = row * (cols - 1) / (rows - 1)
        if row in (1, 4, 6):
            expected -= 0.6
        for col in range(cols):
            level = min(5, int(abs(col - expected) * 1.35))
            fill = palette[level]
            x, y = x0 + col * cell, y0 + row * cell
            f.rect(x, y, cell, cell, fill, C["white"], 1, 0)
            if (row, col) in path_set:
                circle(f, x + cell / 2, y + cell / 2, 5, C["white"], C["coral"], 2)

    # Draw the optimal path in backtracking order: bottom-right to top-left.
    centers = [
        (x0 + col * cell + cell / 2, y0 + row * cell + cell / 2)
        for row, col in reversed(path_forward)
    ]
    line(f, centers, C["coral"], 4)
    f._arrowhead(centers[-2], centers[-1], C["coral"], 14)
    f.add_svg(
        f'<polygon points="{centers[-1][0]},{centers[-1][1]} '
        f'{centers[-1][0]+13},{centers[-1][1]+4} {centers[-1][0]+5},{centers[-1][1]+14}" '
        f' fill="{C["coral"]}"/>'
    )

    f.text(773, 705, "右下角 → 左上角回溯", 18, color=C["coral"], bold=True)
    f.rect(510, 738, 505, 136, C["algo_fill"], C["algo_stroke"], 2, 8)
    f.text(535, 762, "Local cost（逐帧姿态差异）", 18, color=C["teal"], bold=True, anchor="start")
    f.text(535, 795, "C(i,j) = (1/23) Σ(k=1...23)", 18, bold=True, anchor="start")
    f.text(535, 823, "dSO(3)( S[i,k], R[j,k] )", 18, bold=True, anchor="start")
    f.text(535, 846, "距离越小 → 颜色越深 → 两帧动作越接近", 15, color=C["muted"], anchor="start")


def draw_dp(f: base.Flowchart) -> None:
    f.rect(1118, 258, 325, 150, C["decision_fill"], C["decision_stroke"], 2, 8)
    f.text(1280, 282, "动态规划累计代价", 18, color=C["amber"], bold=True)
    f.text(1280, 319, "D(i,j) = C(i,j) + min [", 16, bold=True)
    f.text(1280, 350, "D(i-1,j),  D(i,j-1),", 16, bold=True)
    f.text(1280, 380, "D(i-1,j-1) ]", 16, bold=True)

    f.text(1120, 430, "每个格点只允许三种前向步进", 18, color=C["muted"], bold=True, anchor="start")
    moves = [
        (1130, "↓", "学生前进", "教师保持"),
        (1238, "→", "教师前进", "学生保持"),
        (1346, "↘", "双方同步", "共同前进"),
    ]
    for x, symbol, main, sub in moves:
        f.rect(x, 458, 96, 118, C["soft_amber"], C["decision_stroke"], 1, 7)
        f.text(x + 48, 482, symbol, 30, color=C["amber"], bold=True)
        f.text(x + 48, 523, main, 17, bold=True)
        f.text(x + 48, 551, sub, 14, color=C["muted"])

    f.rect(1118, 615, 325, 124, C["output_fill"], C["output_stroke"], 2, 8)
    f.text(1280, 643, "回溯得到最优 DTW 路径", 14, color=C["coral"], bold=True)
    f.text(1280, 685, ["路径可弯曲：适配速度变化", "并可吸收局部停顿", "路径单调：只向前，不允许倒流"], 13, color=C["ink"], line_gap=6)

    f.rect(1118, 773, 325, 82, C["soft_green"], C["algo_stroke"], 2, 8)
    f.text(1280, 798, "全局最小累计代价", 18, color=C["teal"], bold=True)
    f.text(1280, 830, "≠ 每一帧各自贪心匹配", 15, color=C["muted"])


def draw_boundary_mapping(f: base.Flowchart) -> None:
    x_left, x_right = 1534, 1830
    f.text(1518, 270, "教师 24式 Ground Truth 边界", 19, color=C["navy"], bold=True, anchor="start")
    line(f, [(x_left, 326), (x_right, 326)], C["navy"], 3)
    for idx in range(13):
        x = x_left + idx * (x_right - x_left) / 12
        h = 15 if idx in (0, 4, 8, 12) else 8
        line(f, [(x, 326 - h), (x, 326 + h)], C["navy"], 2)
    f.text(x_left, 356, "第1式", 14, color=C["muted"], anchor="start")
    f.text(x_right, 356, "第24式", 14, color=C["muted"], anchor="end")
    boundary_x = x_left + 7 * (x_right - x_left) / 12
    circle(f, boundary_x, 326, 8, C["coral"], C["white"], 2)
    f.text(boundary_x, 292, "教师边界 rj*", 16, color=C["coral"], bold=True)

    f.text(1682, 406, "沿 DTW path 投影", 18, color=C["teal"], bold=True)
    f.arrow([(int(boundary_x), 338), (int(boundary_x), 430)], color=C["teal"], width=3)

    # One reference boundary may intersect several student frames on the path.
    candidate_y = 492
    f.text(1518, 455, "同一教师边界可能对应多个学生候选帧", 16, color=C["muted"], anchor="start")
    candidates = [(1616, "8.2°", False), (1682, "3.1°", True), (1748, "6.7°", False)]
    for x, cost, selected in candidates:
        line(f, [(boundary_x, 432), (x, candidate_y - 13)], C["line"], 2, dashed=True)
        circle(
            f,
            x,
            candidate_y,
            11 if selected else 8,
            C["coral"] if selected else C["matrix_3"],
            C["white"],
            2,
        )
        f.text(x, candidate_y + 29, cost, 14, color=C["coral"] if selected else C["muted"], bold=selected)

    f.rect(1518, 548, 328, 108, C["decision_fill"], C["decision_stroke"], 2, 8)
    f.text(1682, 570, "候选选择规则", 15, color=C["amber"], bold=True)
    f.text(1682, 618, ["沿 path 收集学生候选帧", "比较 local Geodesic Distance", "取距离最小者"], 12, bold=True, line_gap=4)
    f.arrow([(1682, 514), (1682, 548)], color=C["decision_stroke"], width=3)

    f.text(1518, 660, "学生时间轴", 18, color=C["teal"], bold=True, anchor="start")
    line(f, [(x_left, 696), (x_right, 696)], C["teal"], 3)
    for idx in range(13):
        x = x_left + idx * (x_right - x_left) / 12
        h = 14 if idx in (0, 5, 9, 12) else 7
        line(f, [(x, 696 - h), (x, 696 + h)], C["teal"], 2)
    selected_x = 1682
    circle(f, selected_x, 696, 9, C["coral"], C["white"], 2)
    f.arrow([(1682, 656), (1682, 682)], color=C["coral"], width=3)
    f.text(selected_x, 729, "对应学生帧 si*", 16, color=C["coral"], bold=True)

    f.rect(1518, 772, 328, 84, C["output_fill"], C["output_stroke"], 2, 8, shadow=True)
    f.text(1682, 799, "输出：学生24式边界", 15, bold=True)
    f.text(1682, 831, "B_student = {b1 ... b24}", 13, bold=True)
    f.arrow([(1682, 738), (1682, 772)], color=C["output_stroke"], width=3)


def build() -> None:
    base.SVG_PATH = SVG_PATH
    base.PNG_PATH = PNG_PATH
    f = base.Flowchart()

    f.text(58, 51, "DTW 如何对齐不同速度的太极拳动作？", 42, bold=True, anchor="start")
    f.text(60, 101, "以 SMPL 局部姿态距离为代价，寻找教师帧与学生帧的单调最优对应路径", 21, color=C["muted"], anchor="start")
    f.rect(1517, 66, 343, 47, C["algo_fill"], C["algo_stroke"], 1, 7)
    f.text(1688, 89, "项目用途：完整视频边界迁移", 17, color=C["teal"], bold=True)

    phase_panel(f, 55, 390, C["navy"], "1", "输入序列与姿态特征")
    phase_panel(f, 465, 600, C["teal"], "2", "构造 Local Cost Matrix")
    phase_panel(f, 1085, 390, C["amber"], "3", "动态规划与最优路径")
    phase_panel(f, 1495, 370, C["coral"], "4", "投影 24式边界")

    draw_sequence(f, 315, "教师参考  R=(r1 ... rN)", "较快 / N帧", [0.0, 0.25, 0.5, 0.75, 1.0], C["navy"])
    draw_sequence(f, 445, "学生练习  S=(s1 ... sM)", "较慢 + 局部停顿 / M帧", [0.0, 0.18, 0.38, 0.5, 0.5, 0.72, 1.0], C["teal"])
    f.text(250, 528, "同一动作阶段，但持续时间与速度不同", 17, color=C["muted"], bold=True)
    f.arrow([(250, 548), (250, 578)], color=C["line"], width=3)

    f.rect(85, 578, 330, 144, C["algo_fill"], C["algo_stroke"], 2, 8)
    pose(f, 125, 648, 0.58, 1.15, C["teal"])
    f.text(166, 608, "每帧提取 SMPL body_pose", 19, bold=True, anchor="start")
    f.text(166, 646, "23 个局部关节旋转矩阵", 20, color=C["teal"], bold=True, anchor="start")
    f.text(166, 684, "global_orient 不参与距离计算", 16, color=C["coral"], bold=True, anchor="start")
    f.box(106, 756, 288, 86, ["局部姿态特征序列", "R_feat：N × 23", "S_feat：M × 23"], "data", size=15, bold=True)

    draw_matrix(f)
    draw_dp(f)
    draw_boundary_mapping(f)

    # High-level flow arrows between the four stages.
    f.arrow([(394, 799), (465, 799)], color=C["line"], width=3)
    f.arrow([(1044, 350), (1085, 350)], color=C["line"], width=3)
    f.arrow([(1443, 680), (1470, 680), (1470, 406), (1495, 406)], color=C["line"], width=3)

    f.rect(55, 916, 1810, 132, C["panel"], C["line_light"], 2, 8, shadow=True)
    f.rect(55, 916, 16, 132, C["coral"], radius=8)
    f.text(99, 948, "核心直觉", 18, color=C["coral"], bold=True, anchor="start")
    f.text(99, 994, "DTW 寻找的是“动作阶段对应关系”，不是简单比较相同时间戳。", 27, bold=True, anchor="start")
    f.text(1788, 984, ["允许变速 / 停顿", "保持时间单调"], 16, color=C["muted"], bold=True, anchor="end", line_gap=10)

    f.save()


if __name__ == "__main__":
    HERE.mkdir(parents=True, exist_ok=True)
    build()
    print(f"wrote {SVG_PATH}")
    print(f"wrote {PNG_PATH}")
