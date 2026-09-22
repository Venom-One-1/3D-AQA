"""Conservative, evidence-linked feedback for endpoint metric comparisons."""

from dataclasses import asdict, dataclass
import math

from .endpoint_metrics import MOVE_METRICS, compare_interval, teacher_interval


STATUS_LABELS = {
    "feedback_candidate": "训练建议（待人工确认）",
    "needs_review": "需复核，暂不纠正",
    "observed_difference": "仅记录差异",
    "within_reference": "本指标未发现区间偏离",
}
REASONS = {
    "invalid_window": "有效窗口数据不足或数值无效",
    "missing_center": "中心帧无有效值",
    "center_shift": "中心帧与窗口中位数差异较大",
    "window_spread": "窗口内指标波动较大",
    "invalid_reference": "教师数据不足或参考区间退化",
    "interval_disagreement": "两种教师参考区间判断不一致",
    "center_direction_disagreement": "中心帧与窗口中位数的区间判断不一致",
    "unstable_teacher_reference": "参与参考统计的教师值存在窗口稳定性警告",
}


@dataclass(frozen=True)
class FeedbackConfig:
    center_delta_degree: float = 5.0
    center_delta_ratio: float = 0.05
    window_spread_degree: float = 10.0
    window_spread_ratio: float = 0.10
    minimum_teachers: int = 7
    minimum_valid_frames: int = 3
    minimum_valid_ratio: float = 0.7

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid configuration: {key}")
        if self.minimum_valid_ratio > 1 or self.minimum_teachers > 10:
            raise ValueError("Invalid coverage or teacher count")


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def stability_reasons(row: dict, config: FeedbackConfig) -> list[str]:
    reasons = []
    count, requested = row.get("valid_frame_count"), row.get("requested_frame_count")
    if (row.get("status") != "valid" or not finite(row.get("value"))
            or not finite(count) or not finite(requested) or requested <= 0
            or count < config.minimum_valid_frames or count > requested
            or count / requested < config.minimum_valid_ratio):
        reasons.append("invalid_window")
    center, value = row.get("center_value"), row.get("value")
    if not finite(center):
        reasons.append("missing_center")
    unit = row.get("unit")
    if unit not in ("degree", "ratio"):
        raise ValueError(f"Unsupported unit: {unit}")
    delta = config.center_delta_degree if unit == "degree" else config.center_delta_ratio
    spread = config.window_spread_degree if unit == "degree" else config.window_spread_ratio
    if finite(center) and finite(value) and abs(center-value) > delta:
        reasons.append("center_shift")
    lo, hi = row.get("window_min"), row.get("window_max")
    if not finite(lo) or not finite(hi) or hi < lo:
        if "invalid_window" not in reasons:
            reasons.append("invalid_window")
    elif hi-lo > spread:
        reasons.append("window_spread")
    return reasons


def make_rules(metric_rules: dict) -> dict:
    """Bind supported geometry directions to reviewed endpoint event IDs, not old image IDs."""
    templates = {}

    def add(mid, metric, below=None, above=None):
        templates[(mid, metric)] = {k: v for k, v in (("below", below), ("above", above)) if v}

    for side, label in (("left", "左"), ("right", "右")):
        add(1, f"{side}_knee_angle", above=f"估计的{label}膝屈曲幅度可能不足。请结合画面检查起势末尾是否完成屈膝下蹲，避免为了达到数值而强行加深。")
        add(1, f"{side}_elbow_angle", above=f"估计的{label}肘较教师更接近伸直。请检查下按时是否保留自然弯曲，不要将肘部僵直撑开。")
        add(1, f"{side}_wrist_pelvis_up_torso_ratio", above=f"估计的{label}腕相对躯干位置偏高。请检查该手是否已随动作下按至腹前，不要只依据三维数值继续下压。")
        add(1, f"{side}_wrist_pelvis_forward_torso_ratio", below=f"估计的{label}腕相对骨盆向前伸出较少。请检查手是否位于腹前，而不是收在身体侧后方。")
    add(1, "stance_width_shoulder_ratio", "估计的站距相对肩宽偏小。请检查两脚开立间距，结合自身条件保持与肩宽相称。", "估计的站距相对肩宽偏大。请检查两脚是否开得过宽，结合自身条件调整开立步。")
    add(2, "left_knee_angle", above="估计的左膝屈曲幅度可能不足。请检查左弓步是否形成，保持屈膝过程自然，不要为追求角度强压膝部。")
    add(2, "right_knee_angle", below="估计的右膝比教师弯曲更多。请检查左弓步时右腿是否自然舒展，不要刻意锁死膝关节。")
    for side, label in (("left", "左"), ("right", "右")):
        add(2, f"{side}_elbow_angle", above=f"估计的{label}肘较教师更接近伸直。请检查分手定势时该臂是否保留自然弧形，避免僵直撑臂。")
    add(2, "left_wrist_head_up_torso_ratio", "左腕相对头关节的上方位置估计偏低。请在画面中核对左手是否举至动作要求的位置；头关节不是眼睛，不能直接据此判定低于眼睛。")
    add(2, "left_wrist_head_forward_torso_ratio", "左腕相对头关节向前伸出估计较少。请检查左手是否在身体左前方展开，不要仅按数值增加伸手距离。")
    add(2, "right_wrist_hip_distance_torso_ratio", above="估计的右腕离右髋较远。请结合画面检查右手是否已落至右胯旁，而不是停在远离身体的位置。")
    add(3, "right_knee_angle", above="估计的右膝屈曲幅度可能不足。请检查白鹤亮翅末尾右腿的屈膝姿态；这个角度不能说明实际承重分配。")
    add(3, "right_elbow_angle", above="估计的右肘较教师更接近伸直。请检查右臂是否保持舒展的弧形和自然弯曲；这不等于已经发生肘关节过伸。")
    add(3, "right_wrist_head_up_torso_ratio", "右腕相对头关节的上方位置估计偏低。请核对右手定势位置；不以此要求掌心精确对齐额头，也不因略高于教师就要求降低。")
    add(3, "right_wrist_head_forward_torso_ratio", "右腕相对头关节向前伸出估计较少。请结合侧向画面检查右手是否保持在头部右前方。")
    add(3, "right_wrist_head_right_torso_ratio", "右腕相对头关节向右展开估计较少。请结合画面检查右手是否保持在头部右前方，而不是过度收到左侧。")
    add(3, "left_wrist_hip_distance_torso_ratio", above="估计的左腕离左髋较远。请检查左手是否已落至左胯前，不要仅按距离把手贴紧身体。")
    rules = []
    for pose in metric_rules["poses"]:
        for check in pose["checks"]:
            mid, metric = pose["move_id"], check["metric_id"]
            aspect = check["technique_aspect"]
            if metric == "right_wrist_hip_distance_torso_ratio":
                aspect = "右手落于右胯旁（以腕到髋的归一化距离作为位置代理）"
            elif metric == "left_wrist_hip_distance_torso_ratio":
                aspect = "左手落于左胯前（以腕到髋的归一化距离作为位置代理）"
            elif mid == 3 and metric == "left_elbow_angle":
                aspect = "左手落于左胯前时的左肘姿态（夹角仅供对照）"
            rules.append({"rule_id": f"{pose['pose_id']}:{metric}", "move_id": mid,
                "pose_id": pose["pose_id"], "metric_id": metric,
                "technique_aspect": aspect,
                "final_technique_step": pose["final_technique_step"],
                "feedback_by_direction": templates.get((mid, metric), {}),
                "unsupported_direction_policy": "record_only",
                "limitation": "几何代理指标；稳定不代表重建准确；不能单独判定真实承重、肌肉放松、掌心朝向或整体动作合格。"})
    return {"schema_version": "1.0", "status": "experimental_pending_manual_review",
            "rules": rules, "excluded_assessments": ["actual_weight_bearing", "muscle_relaxation",
                "palm_orientation", "gravity_relative_trunk_verticality", "overall_quality_score"]}


def validate_rules(rules: dict, metric_rules: dict) -> dict:
    expected = {(m, metric) for m, metrics in MOVE_METRICS.items() for metric in metrics}
    events = {p["move_id"]: p for p in metric_rules["poses"]}
    lookup = {}
    for rule in rules["rules"]:
        key = (rule["move_id"], rule["metric_id"])
        if key in lookup or key not in expected:
            raise ValueError(f"Duplicate or unknown feedback rule: {key}")
        event = events[key[0]]
        if rule["pose_id"] != event["pose_id"] or rule["final_technique_step"] != event["final_technique_step"]:
            raise ValueError(f"Feedback rule does not match endpoint manifest: {key}")
        if not set(rule["feedback_by_direction"]).issubset({"above", "below"}):
            raise ValueError(f"Unknown feedback direction: {key}")
        if any(not isinstance(v, str) or not v.strip() for v in rule["feedback_by_direction"].values()):
            raise ValueError(f"Empty feedback template: {key}")
        lookup[key] = rule
    if set(lookup) != expected:
        raise ValueError("Incomplete first-three-form feedback rules")
    return lookup


def evaluate_feedback(student: dict, teachers: list[dict], rule: dict,
                      config: FeedbackConfig) -> dict:
    """Keep all valid teacher medians; warnings abstain rather than silently filter teachers."""
    stats = teacher_interval([r.get("value") if r.get("status") == "valid" else None for r in teachers],
                             config.minimum_teachers)
    directions = {method: compare_interval(student.get("value"), stats, method)[0]
                  for method in ("median_2mad", "p10_p90")}
    center_directions = {method: compare_interval(student.get("center_value"), stats, method)[0]
                         for method in directions}
    reasons = stability_reasons(student, config)
    valid_states = {"within", "above", "below"}
    if any(x not in valid_states for x in directions.values()):
        reasons.append("invalid_reference" if stats["status"] != "valid" or
                       "degenerate_reference_interval" in directions.values() else "invalid_window")
    if all(x in valid_states for x in directions.values()) and len(set(directions.values())) > 1:
        reasons.append("interval_disagreement")
    if finite(student.get("center_value")) and any(center_directions[m] != directions[m] for m in directions):
        reasons.append("center_direction_disagreement")
    teacher_warnings = []
    for teacher in teachers:
        flags = stability_reasons(teacher, config)
        if flags:
            teacher_warnings.append({"subject_id": teacher["subject_id"], "reasons": flags,
                "included_in_reference": teacher.get("status") == "valid" and finite(teacher.get("value"))})
    if any(w["included_in_reference"] for w in teacher_warnings):
        reasons.append("unstable_teacher_reference")
    reasons = list(dict.fromkeys(reasons))
    direction = directions["median_2mad"]
    text = ""
    if reasons:
        decision = "needs_review"
    elif direction == "within":
        decision = "within_reference"
    elif direction in rule["feedback_by_direction"]:
        decision = "feedback_candidate"
        text = rule["feedback_by_direction"][direction]
    else:
        decision = "observed_difference"
    return {"record_id": f"{student['subject_id']}:{rule['rule_id']}",
        "subject_id": student["subject_id"], "move_id": student["move_id"],
        "move_name_zh": student["move_name_zh"], "move_name_pinyin": student["move_name_pinyin"],
        "pose_id": rule["pose_id"], "rule_id": rule["rule_id"], "metric_id": student["metric_id"],
        "metric_label_zh": student["metric_label_zh"], "technique_aspect": rule["technique_aspect"],
        "final_technique_step": rule["final_technique_step"], "unit": student["unit"],
        "decision": decision, "decision_zh": STATUS_LABELS[decision],
        "direction": direction, "interval_directions": directions, "center_directions": center_directions,
        "review_reasons": reasons, "review_reasons_zh": [REASONS[x] for x in reasons],
        "teacher_warnings": teacher_warnings, "teacher_reference": stats,
        "teacher_evidence": teachers, "student_evidence": student,
        "feedback": text, "limitation": rule["limitation"],
        "manual_review": {"verdict": "pending", "cause": "", "notes": ""}}


def coach_summary(records: list[dict]) -> str:
    suggestions = [r for r in records if r["decision"] == "feedback_candidate"]
    review_count = sum(r["decision"] == "needs_review" for r in records)
    if not suggestions:
        lead = "本式暂没有通过可靠性检查的纠正建议，这不代表动作已经标准。"
    else:
        lead = "以下为结束定势的几何估计提示，需结合画面核验：" + " ".join(r["feedback"] for r in suggestions[:2])
        if len(suggestions) > 2:
            lead += f" 另有{len(suggestions)-2}条候选建议见明细；展示顺序不表示问题严重程度。"
    if review_count:
        lead += f" 本式有{review_count}项指标需复核，暂不据此纠正。"
    return lead
