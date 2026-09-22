"""Standalone figures and a Chinese review report for endpoint measurements."""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from aqa3d.endpoint_metrics import MOVE_METRICS, LABELS
from visualize_tas_boundary_frames import load_font, fit_frame, read_video_frames


def fmt(value):
    return "缺失" if value is None or not np.isfinite(value) else f"{value:.3f}"


def comparison_grid(output, records, title, columns=3):
    cell_w, cell_h = 400, 296
    image = Image.new("RGB", (columns*cell_w, 54+math.ceil(len(records)/columns)*cell_h), "white")
    draw = ImageDraw.Draw(image); font = load_font(18)
    draw.text((12,12), title, font=load_font(23), fill="black")
    for i, record in enumerate(records):
        left, top = (i%columns)*cell_w, 54+(i//columns)*cell_h
        draw.text((left+10,top+5), record["label"], font=font, fill="black")
        draw.text((left+10,top+30), record["detail"], font=load_font(15), fill="#333333")
        if record["image"] and Path(record["image"]).is_file():
            with Image.open(record["image"]) as frame:
                image.paste(fit_frame(frame,380,220),(left+10,top+60))
        else:
            draw.text((left+10,top+85), "无有效截图", font=font, fill="#a32020")
    image.save(output, quality=95)


def export_report(out, all_rows, references, comparisons, rules, student_ids):
    out = Path(out); figures = out/"figures"; figures.mkdir(exist_ok=True)
    font = FontProperties(fname=load_font(16).path)
    ref_lookup = {(r["move_id"],r["metric_id"]):r for r in references}
    lookup = {(r["subject_id"],r["move_id"],r["metric_id"]):r for r in comparisons}
    status_names = {"within":"区间内", "above":"高于区间", "below":"低于区间",
                   "invalid_student":"学生数据无效", "insufficient_teachers":"教师数不足",
                   "degenerate_reference_interval":"教师区间退化"}
    matrix = []
    report = ["# 前三式结束定势指标对比", "",
        "学生："+"、".join(student_ids)+"。每项保留中心帧值与窗口中位数；以下比较使用窗口中位数。",
        "参考范围来自原有 10 名教师，每位教师每项贡献一个值。BV1WE411W7JB 仅提供事件锚点，不加入范围统计。",
        "默认范围为 median ± 2×MAD（MAD 不乘 1.4826）；同时报告 P10–P90。至少 7 名教师有效才作比较。",
        "数值偏离是待人工判断的诊断信息，不是合格结论或总分，也不据此给学生排名。",
        "身体上方/前后方向采用躯干与髋肩定义的局部坐标。它不能衡量地面高度、真实竖直方向或承重。",
        "窗口半径为配置的 0.2 秒，并裁至当前招式：在结束边界实际只使用边界前约 0.2 秒及边界本帧。",
        "膝肘夹角单位为度，180°表示伸直。所有距离均按躯干长、肩宽或腿长归一化。",
        "图中实心点为窗口中位数、空心圆为中心帧值；灰色点为 10 名教师原始值；蓝色带为 median ± 2×MAD，虚线为 P10/P90。", ""]
    import json
    manifest = json.loads((out/"reference_manifest_snapshot.json").read_text())
    anchor_video = Path(manifest["reference"]["video_path"])
    anchor_frames = read_video_frames(anchor_video,[e["source_frame_index_0based"] for e in manifest["endpoint_keyposes"][:3]])
    for pose in rules["poses"]:
        mid=pose["move_id"]; metrics=MOVE_METRICS[mid]
        ncols=3; nrows=math.ceil(len(metrics)/ncols)
        fig, axs=plt.subplots(nrows,ncols,figsize=(16,3.8*nrows),squeeze=False,constrained_layout=True)
        for ax,metric in zip(axs.flat,metrics):
            ref=ref_lookup[(mid,metric)]
            valid_teachers=[r["value"] for r in ref["teacher_values"] if r["value"] is not None]
            if valid_teachers:
                ax.scatter(valid_teachers, -1+np.linspace(-.12,.12,len(valid_teachers)),color="#777777",s=20)
                if ref["status"]=="valid":
                    ax.axvspan(ref["median_minus_2mad"],ref["median_plus_2mad"],color="#b9dce9",alpha=.55)
                    ax.axvline(ref["p10"],ls="--",color="#47758a",lw=1)
                    ax.axvline(ref["p90"],ls="--",color="#47758a",lw=1)
            for si,sid in enumerate(student_ids):
                r=lookup[(sid,mid,metric)]
                if r["value"] is not None:
                    color="#16746e" if r["median_2mad_status"]=="within" else "#b74638"
                    ax.scatter(r["value"],si,color=color,s=40,zorder=3)
                if r["center_value"] is not None:
                    ax.scatter(r["center_value"],si,facecolors="none",edgecolors="#222222",s=85,zorder=4)
            ax.set_title(LABELS[metric],fontproperties=font,fontsize=11)
            ax.set_xlabel(ref["unit"]); ax.set_yticks([-1]+list(range(len(student_ids))),["Teachers"]+student_ids)
            ax.set_ylim(len(student_ids)-.5,-1.5); ax.grid(axis="x",alpha=.2)
        for ax in list(axs.flat)[len(metrics):]: ax.axis("off")
        fig.suptitle(f'{mid}. {pose["move_name_zh"]} | 中心帧与窗口中位数',fontproperties=font,fontsize=18)
        fig.savefig(figures/f"{mid:02d}_metric_comparison.png",dpi=160);plt.close(fig)
        anchor=manifest["endpoint_keyposes"][mid-1]
        anchor_path=figures/f"{mid:02d}_reference.jpg";anchor_frames[anchor["source_frame_index_0based"]].save(anchor_path,quality=95)
        student_pictures=[{"label":"参考锚点 BV1WE411W7JB", "detail":f'{anchor["boundary_time_seconds"]:.3f}s',"image":str(anchor_path)}]
        for sid in student_ids:
            r=lookup[(sid,mid,metrics[0])]
            student_pictures.append({"label":"Student "+sid,"detail":fmt(r["boundary_time_seconds"])+"s", "image":r["image"]})
        comparison_grid(figures/f"{mid:02d}_student_keyposes.jpg",student_pictures,pose["move_name_zh"]+"：参考与五名学生")
        teacher_pictures=[]
        for t in ref_lookup[(mid,metrics[0])]["teacher_values"]:
            teacher_pictures.append({"label":t["subject_id"],"detail":fmt(t["boundary_time_seconds"])+"s", "image":t["image"]})
        comparison_grid(figures/f"{mid:02d}_teacher_keyposes.jpg",teacher_pictures,pose["move_name_zh"]+"：10 名教师",columns=5)
        report += [f'## {mid}. {pose["move_name_zh"]}', "", pose["final_technique_step"], "",
            f'![学生定势](figures/{mid:02d}_student_keyposes.jpg)', "",
            f'[10 名教师定势](figures/{mid:02d}_teacher_keyposes.jpg) | [指标分布图](figures/{mid:02d}_metric_comparison.png)', "",
            "单元格：窗口中位数（中心帧值）；相对默认区间的方向。", "",
            "| 指标 | 教师 median ± 2MAD | 教师 P10–P90 | 有效教师数 | "+" | ".join(student_ids)+" |",
            "|"+"---|"*(4+len(student_ids))]
        for metric in metrics:
            ref=ref_lookup[(mid,metric)]
            bounds=f'{fmt(ref["median_minus_2mad"])}–{fmt(ref["median_plus_2mad"])}'
            percent=f'{fmt(ref["p10"])}–{fmt(ref["p90"])}'
            cells=[]; wide={"move_id":mid,"pose_id":pose["pose_id"],"metric_id":metric,
                "label_zh":LABELS[metric],"unit":ref["unit"],"teacher_median_minus_2mad":ref["median_minus_2mad"],
                "teacher_median_plus_2mad":ref["median_plus_2mad"],"teacher_p10":ref["p10"],"teacher_p90":ref["p90"]}
            for sid in student_ids:
                r=lookup[(sid,mid,metric)]
                status=status_names[r["median_2mad_status"]]
                cells.append(f'{fmt(r["value"])} ({fmt(r["center_value"])})；{status}')
                for key in ("value","center_value","median_2mad_status","p10_p90_status"):
                    wide[f'student_{sid}_{key}']=r[key]
            matrix.append(wide)
            report.append(f'| {LABELS[metric]} ({ref["unit"]}) | {bounds} | {percent} | {ref["valid_teacher_count"]}/10 | '+" | ".join(cells)+" |")
        report.append("")
    pd.DataFrame(matrix).to_csv(out/"comparison_matrix.csv",index=False)
    (out/"comparison_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
