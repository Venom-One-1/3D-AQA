#!/usr/bin/env python3
"""Generate the long-term 24-form Tai Chi AQA pipeline flowchart."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1080
HERE = Path(__file__).resolve().parent
SVG_PATH = HERE / "long_term_aqa_pipeline_flowchart.svg"
PNG_PATH = HERE / "long_term_aqa_pipeline_flowchart.png"

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

COLORS = {
    "bg": "#F5F7FA",
    "panel": "#FFFFFF",
    "ink": "#172433",
    "muted": "#627083",
    "line": "#7C8998",
    "line_light": "#D8DEE7",
    "phase1": "#326A95",
    "phase2": "#287C71",
    "phase3": "#9A6A22",
    "data_fill": "#E7F0F8",
    "data_stroke": "#4A7CA5",
    "algo_fill": "#E0F2EE",
    "algo_stroke": "#2F8277",
    "decision_fill": "#FFF1CE",
    "decision_stroke": "#B87B20",
    "output_fill": "#FBE8E4",
    "output_stroke": "#B85E4C",
    "lane_offline": "#E8EEF5",
    "lane_online": "#E5F1ED",
    "white": "#FFFFFF",
    "soft_blue": "#F2F7FB",
    "soft_green": "#F1F8F6",
    "soft_amber": "#FBF7ED",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    return (*hex_rgb(value), alpha)


def rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = None,
    width: int = 1,
) -> None:
    """Draw a rounded rectangle on Pillow versions predating rounded_rectangle."""
    x0, y0, x1, y1 = box
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        return
    draw.rectangle((x0 + radius, y0, x1 - radius, y1), fill=fill)
    draw.rectangle((x0, y0 + radius, x1, y1 - radius), fill=fill)
    draw.pieslice((x0, y0, x0 + 2 * radius, y0 + 2 * radius), 180, 270, fill=fill)
    draw.pieslice((x1 - 2 * radius, y0, x1, y0 + 2 * radius), 270, 360, fill=fill)
    draw.pieslice((x0, y1 - 2 * radius, x0 + 2 * radius, y1), 90, 180, fill=fill)
    draw.pieslice((x1 - 2 * radius, y1 - 2 * radius, x1, y1), 0, 90, fill=fill)
    if outline:
        for offset in range(width):
            r = radius - offset
            draw.line((x0 + r, y0 + offset, x1 - r, y0 + offset), fill=outline)
            draw.line((x0 + r, y1 - offset, x1 - r, y1 - offset), fill=outline)
            draw.line((x0 + offset, y0 + r, x0 + offset, y1 - r), fill=outline)
            draw.line((x1 - offset, y0 + r, x1 - offset, y1 - r), fill=outline)
            draw.arc((x0 + offset, y0 + offset, x0 + 2 * r, y0 + 2 * r), 180, 270, fill=outline)
            draw.arc((x1 - 2 * r, y0 + offset, x1 - offset, y0 + 2 * r), 270, 360, fill=outline)
            draw.arc((x0 + offset, y1 - 2 * r, x0 + 2 * r, y1 - offset), 90, 180, fill=outline)
            draw.arc((x1 - 2 * r, y1 - 2 * r, x1 - offset, y1 - offset), 0, 90, fill=outline)


class Flowchart:
    def __init__(self) -> None:
        self.svg: list[str] = []
        self.image = Image.new("RGB", (WIDTH, HEIGHT), hex_rgb(COLORS["bg"]))
        self.draw = ImageDraw.Draw(self.image, "RGBA")

    def add_svg(self, value: str) -> None:
        self.svg.append(value)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str | None = None,
        width: int = 2,
        radius: int = 8,
        shadow: bool = False,
    ) -> None:
        effect = ' filter="url(#shadow)"' if shadow else ""
        stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
        self.add_svg(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}"{stroke_attr}{effect}/>'
        )
        if shadow:
            rounded_rectangle(self.draw, (x + 3, y + 5, x + w + 3, y + h + 5), radius, (25, 42, 58, 22))
        rounded_rectangle(
            self.draw,
            (x, y, x + w, y + h),
            radius,
            rgba(fill),
            rgba(stroke) if stroke else None,
            width,
        )

    def text(
        self,
        x: float,
        y: float,
        lines: str | list[str],
        size: int,
        color: str = COLORS["ink"],
        bold: bool = False,
        anchor: str = "middle",
        line_gap: int = 6,
    ) -> None:
        if isinstance(lines, str):
            lines = [lines]
        line_height = size + line_gap
        total_height = len(lines) * line_height - line_gap
        top = y - total_height / 2
        svg_anchor = {"middle": "middle", "start": "start", "end": "end"}[anchor]
        family = "Noto Sans CJK SC, Noto Sans CJK, sans-serif"
        weight = "700" if bold else "400"
        tspans = []
        for index, line in enumerate(lines):
            dy = "0" if index == 0 else str(line_height)
            tspans.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
        self.add_svg(
            f'<text x="{x}" y="{top + size * 0.84}" text-anchor="{svg_anchor}" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}">{"".join(tspans)}</text>'
        )
        pil_anchor = {"middle": "mm", "start": "lm", "end": "rm"}[anchor]
        for index, line in enumerate(lines):
            cy = top + index * line_height + size / 2
            self.draw.text(
                (x, cy),
                line,
                font=font(size, bold),
                fill=rgba(color),
                anchor=pil_anchor,
            )

    def box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        lines: list[str],
        kind: str,
        size: int = 23,
        bold: bool = False,
        shadow: bool = True,
    ) -> None:
        fill = COLORS[f"{kind}_fill"]
        stroke = COLORS[f"{kind}_stroke"]
        self.rect(x, y, w, h, fill, stroke, 2, 8, shadow)
        self.text(x + w / 2, y + h / 2, lines, size, bold=bold)

    def diamond(
        self,
        cx: int,
        cy: int,
        w: int,
        h: int,
        lines: list[str],
        size: int = 22,
    ) -> None:
        points = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
        path = " ".join(f"{x},{y}" for x, y in points)
        self.add_svg(
            f'<polygon points="{path}" fill="{COLORS["decision_fill"]}" '
            f'stroke="{COLORS["decision_stroke"]}" stroke-width="2" filter="url(#shadow)"/>'
        )
        shadow_points = [(x + 3, y + 5) for x, y in points]
        self.draw.polygon(shadow_points, fill=(25, 42, 58, 22))
        self.draw.polygon(points, fill=rgba(COLORS["decision_fill"]))
        self.draw.line(points + [points[0]], fill=rgba(COLORS["decision_stroke"]), width=2)
        self.text(cx, cy, lines, size, bold=True, line_gap=5)

    def database(
        self, x: int, y: int, w: int, h: int, lines: list[str], size: int = 22
    ) -> None:
        fill, stroke = COLORS["data_fill"], COLORS["data_stroke"]
        ry = 13
        self.add_svg(
            f'<path d="M {x} {y + ry} A {w/2} {ry} 0 0 1 {x+w} {y+ry} '
            f'L {x+w} {y+h-ry} A {w/2} {ry} 0 0 1 {x} {y+h-ry} Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2" filter="url(#shadow)"/>'
        )
        self.add_svg(
            f'<ellipse cx="{x+w/2}" cy="{y+ry}" rx="{w/2}" ry="{ry}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        self.add_svg(
            f'<path d="M {x} {y+h-ry} A {w/2} {ry} 0 0 0 {x+w} {y+h-ry}" '
            f'fill="none" stroke="{stroke}" stroke-width="2"/>'
        )
        rounded_rectangle(self.draw, (x + 3, y + 5, x + w + 3, y + h + 5), 8, (25, 42, 58, 22))
        self.draw.rectangle((x, y + ry, x + w, y + h - ry), fill=rgba(fill), outline=rgba(stroke), width=2)
        self.draw.ellipse((x, y, x + w, y + 2 * ry), fill=rgba(fill), outline=rgba(stroke), width=2)
        self.draw.arc((x, y + h - 2 * ry, x + w, y + h), 0, 180, fill=rgba(stroke), width=2)
        self.text(x + w / 2, y + h / 2 + 5, lines, size, bold=True)

    def arrow(
        self,
        points: list[tuple[int, int]],
        label: str | None = None,
        label_xy: tuple[int, int] | None = None,
        dashed: bool = False,
        color: str = COLORS["line"],
        width: int = 3,
    ) -> None:
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="8 7"' if dashed else ""
        self.add_svg(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash} marker-end="url(#arrow)"/>'
        )
        if dashed:
            for p1, p2 in zip(points, points[1:]):
                self._dashed_line(p1, p2, color, width)
        else:
            self.draw.line(points, fill=rgba(color), width=width, joint="curve")
        self._arrowhead(points[-2], points[-1], color)
        if label and label_xy:
            label_font = font(18, True)
            if hasattr(self.draw, "textbbox"):
                tw = self.draw.textbbox((0, 0), label, font=label_font)[2]
            else:
                tw = self.draw.textsize(label, font=label_font)[0]
            self.rect(label_xy[0] - tw / 2 - 8, label_xy[1] - 13, tw + 16, 27, COLORS["bg"], radius=4)
            self.text(label_xy[0], label_xy[1], label, 18, color=color, bold=True)

    def _dashed_line(
        self, p1: tuple[int, int], p2: tuple[int, int], color: str, width: int
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        length = math.hypot(x2 - x1, y2 - y1)
        if not length:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        pos = 0.0
        while pos < length:
            end = min(pos + 8, length)
            self.draw.line(
                (x1 + ux * pos, y1 + uy * pos, x1 + ux * end, y1 + uy * end),
                fill=rgba(color),
                width=width,
            )
            pos += 15

    def _arrowhead(
        self, p1: tuple[int, int], p2: tuple[int, int], color: str, size: int = 12
    ) -> None:
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        points = [
            p2,
            (
                p2[0] - size * math.cos(angle - math.pi / 6),
                p2[1] - size * math.sin(angle - math.pi / 6),
            ),
            (
                p2[0] - size * math.cos(angle + math.pi / 6),
                p2[1] - size * math.sin(angle + math.pi / 6),
            ),
        ]
        self.draw.polygon(points, fill=rgba(color))

    def save(self) -> None:
        defs = f"""
<defs>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="3" dy="5" stdDeviation="4" flood-color="#192A3A" flood-opacity="0.12"/>
  </filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{COLORS['line']}"/>
  </marker>
</defs>"""
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}">\n{defs}\n'
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["bg"]}"/>\n'
            + "\n".join(self.svg)
            + "\n</svg>\n"
        )
        SVG_PATH.write_text(svg, encoding="utf-8")
        self.image.save(PNG_PATH, "PNG", optimize=True)


def build() -> None:
    f = Flowchart()

    f.text(58, 52, "长期目标：搭建24式太极拳完整视频动作质量评估 Pipeline", 42, bold=True, anchor="start")
    f.text(60, 101, "从完整视频自动分式，以可解释量化指标驱动逐式纠错与完整报告", 21, color=COLORS["muted"], anchor="start")
    legend = [("data", "数据/标准"), ("algo", "算法/计算"), ("decision", "判断"), ("output", "反馈/输出")]
    lx = 1325
    for kind, label in legend:
        f.rect(lx, 83, 23, 15, COLORS[f"{kind}_fill"], COLORS[f"{kind}_stroke"], 2, 3)
        f.text(lx + 32, 91, label, 17, color=COLORS["muted"], anchor="start")
        lx += 138

    panels = [
        (228, 505, COLORS["soft_blue"], COLORS["phase1"], "1", "24式时序分割"),
        (744, 505, COLORS["soft_green"], COLORS["phase2"], "2", "关键姿势定量分析"),
        (1260, 600, COLORS["soft_amber"], COLORS["phase3"], "3", "规则驱动反馈"),
    ]
    for x, w, body, accent, number, title in panels:
        f.rect(x, 166, w, 714, body, COLORS["line_light"], 2, 8)
        f.rect(x, 166, w, 70, accent, radius=8)
        f.add_svg(f'<circle cx="{x+35}" cy="201" r="21" fill="{accent}" stroke="#FFFFFF" stroke-width="2"/>')
        f.draw.ellipse((x + 14, 180, x + 56, 222), fill=rgba(accent), outline=rgba(COLORS["white"]), width=2)
        f.text(x + 35, 200, number, 23, color=COLORS["white"], bold=True)
        f.text(x + 70, 201, title, 27, color=COLORS["white"], bold=True, anchor="start")

    f.rect(58, 254, 145, 244, COLORS["lane_offline"], COLORS["line_light"], 2, 8)
    f.text(130, 343, ["离线参考", "构建"], 27, bold=True)
    f.text(130, 414, ["沉淀标准", "一次构建"], 18, color=COLORS["muted"])
    f.rect(58, 526, 145, 336, COLORS["lane_online"], COLORS["line_light"], 2, 8)
    f.text(130, 640, ["在线学生", "评估"], 27, bold=True)
    f.text(130, 723, ["输入完整视频", "自动逐式分析"], 18, color=COLORS["muted"])
    f.text(130, 817, "运行时", 18, color=COLORS["phase2"], bold=True)

    f.add_svg(f'<line x1="228" y1="513" x2="1860" y2="513" stroke="{COLORS["line_light"]}" stroke-width="2"/>')
    f.draw.line((228, 513, 1860, 513), fill=rgba(COLORS["line_light"]), width=2)
    f.text(242, 264, "OFFLINE", 15, color=COLORS["muted"], bold=True, anchor="start")
    f.text(242, 536, "ONLINE", 15, color=COLORS["muted"], bold=True, anchor="start")

    f.box(270, 294, 180, 72, ["多个参考", "教学视频"], "data", bold=True)
    f.box(496, 294, 194, 72, ["人工标注", "24式动作边界"], "algo", bold=True)
    f.box(382, 410, 200, 68, ["参考边界 +", "时序动作模板"], "data", size=21, bold=True)
    f.arrow([(450, 330), (496, 330)])
    f.arrow([(593, 366), (593, 386), (482, 386), (482, 410)])

    f.box(270, 570, 180, 72, ["练习者", "完整视频"], "data", bold=True)
    f.box(496, 570, 194, 72, ["提取时序", "动作特征"], "algo", bold=True)
    f.box(365, 690, 230, 82, ["DTW时序对齐", "映射24式边界"], "algo", bold=True)
    f.box(382, 808, 200, 56, ["24式分段结果"], "output", size=22, bold=True)
    f.arrow([(450, 606), (496, 606)])
    f.arrow([(593, 642), (593, 666), (480, 666), (480, 690)])
    f.arrow([(482, 478), (482, 690)], "模板约束", (530, 530), dashed=True)
    f.arrow([(480, 772), (480, 808)])

    f.box(790, 294, 180, 72, ["每式选取", "关键姿势"], "algo", bold=True)
    f.box(1014, 286, 190, 88, ["多参考视频", "计算参考指标"], "algo", size=21, bold=True)
    f.box(900, 410, 194, 68, ["标准值 /", "合理范围"], "data", size=22, bold=True)
    f.arrow([(582, 444), (714, 444), (714, 330), (790, 330)])
    f.arrow([(970, 330), (1014, 330)])
    f.arrow([(1109, 374), (1109, 390), (997, 390), (997, 410)])

    f.box(790, 570, 180, 72, ["逐式定位", "关键姿势"], "algo", bold=True)
    f.box(1010, 556, 196, 100, ["计算定量指标", "关节夹角", "归一化相对距离"], "algo", size=20, bold=True)
    f.box(858, 704, 278, 76, ["与标准值 /", "合理范围比较"], "decision", size=22, bold=True)
    f.box(900, 812, 194, 52, ["逐式指标偏差"], "output", size=21, bold=True)
    f.arrow([(582, 836), (710, 836), (710, 606), (790, 606)])
    f.arrow([(970, 606), (1010, 606)])
    f.arrow([(1108, 656), (1108, 680), (997, 680), (997, 704)])
    f.arrow([(997, 478), (997, 704)], "标准参照", (1045, 531), dashed=True)
    f.arrow([(997, 780), (997, 812)])

    f.box(1310, 294, 206, 72, ["整理24式", "动作要领"], "data", bold=True)
    f.database(1570, 286, 240, 92, ["动作要领–指标–反馈", "规则库"], size=20)
    f.arrow([(1516, 330), (1570, 330)])

    f.box(1305, 568, 220, 78, ["规则匹配与归因", "定位动作要领"], "algo", size=21, bold=True)
    f.diamond(1680, 607, 250, 126, ["指标是否超出", "合理范围？"], size=21)
    f.arrow([(1094, 838), (1235, 838), (1235, 607), (1305, 607)])
    f.arrow([(1690, 378), (1690, 438), (1415, 438), (1415, 568)], "规则约束", (1540, 438), dashed=True)
    f.arrow([(1525, 607), (1555, 607)])

    f.box(1300, 738, 205, 80, ["达标确认", "保持与巩固提示"], "output", size=20, bold=True)
    f.box(1560, 724, 265, 108, ["针对性教练式反馈", "问题定位 · 纠正方法", "专项训练提示"], "output", size=19, bold=True)
    f.arrow([(1650, 663), (1650, 690), (1402, 690), (1402, 738)], "否", (1488, 690))
    f.arrow([(1710, 666), (1710, 724)], "是", (1742, 693), color=COLORS["output_stroke"])
    f.box(1395, 844, 320, 50, ["招式级反馈（第1式—第24式）"], "output", size=20, bold=True)
    f.arrow([(1402, 818), (1402, 830), (1515, 830), (1515, 844)])
    f.arrow([(1692, 832), (1692, 838), (1595, 838), (1595, 844)])

    f.rect(228, 928, 1632, 112, COLORS["panel"], COLORS["line_light"], 2, 8, shadow=True)
    f.rect(228, 928, 16, 112, COLORS["output_stroke"], radius=8)
    f.arrow([(1555, 894), (1555, 928)], color=COLORS["output_stroke"])
    f.text(286, 960, "最终输出", 18, color=COLORS["output_stroke"], bold=True, anchor="start")
    f.text(286, 1002, "完整视频动作质量评估报告", 30, bold=True, anchor="start")
    chips = [(885, "24式分割"), (1085, "指标证据"), (1285, "逐式建议"), (1485, "总体总结")]
    for x, label in chips:
        f.rect(x, 960, 170, 46, COLORS["output_fill"], COLORS["output_stroke"], 1, 7)
        f.text(x + 85, 983, label, 19, color=COLORS["output_stroke"], bold=True)
    f.text(1778, 984, "可追溯 · 可解释", 17, color=COLORS["muted"], bold=True, anchor="end")

    f.save()


if __name__ == "__main__":
    HERE.mkdir(parents=True, exist_ok=True)
    build()
    print(f"wrote {SVG_PATH}")
    print(f"wrote {PNG_PATH}")
