#!/usr/bin/env python
"""Build the complete 24-form technique-metric-feedback rule table."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
FIRST_THREE_PATH = PROJECT_ROOT / "technique_feedback_rules_first3.json"
OUTPUT_PATH = PROJECT_ROOT / "technique_feedback_rules.json"


ADDITIONAL_METRICS = {
    "torso_yaw_from_move_start": {
        "type": "signed_projected_vector_angle",
        "unit": "degree",
        "calculation": "signed_angle(move_start_forward_axis, body_forward_axis, vertical_axis)",
        "range": [
            -180.0,
            180.0,
        ],
        "existing_metric": False,
    },
    "left_wrist_shoulder_height_offset": {
        "type": "normalized_signed_height_difference",
        "unit": "normalized_length",
        "calculation": "dot(L_Wrist - L_Shoulder, vertical_axis) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "right_wrist_shoulder_height_offset": {
        "type": "normalized_signed_height_difference",
        "unit": "normalized_length",
        "calculation": "dot(R_Wrist - R_Shoulder, vertical_axis) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "left_wrist_pelvis_height_offset": {
        "type": "normalized_signed_height_difference",
        "unit": "normalized_length",
        "calculation": "dot(L_Wrist - Pelvis, vertical_axis) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "right_wrist_pelvis_height_offset": {
        "type": "normalized_signed_height_difference",
        "unit": "normalized_length",
        "calculation": "dot(R_Wrist - Pelvis, vertical_axis) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "left_arm_lateral_reach": {
        "type": "normalized_signed_projection",
        "unit": "normalized_length",
        "calculation": "dot(L_Wrist - L_Shoulder, lateral_axis) / shoulder_width",
        "normalizer": "shoulder_width",
        "existing_metric": False,
    },
    "right_arm_lateral_reach": {
        "type": "normalized_signed_projection",
        "unit": "normalized_length",
        "calculation": "dot(R_Shoulder - R_Wrist, lateral_axis) / shoulder_width",
        "normalizer": "shoulder_width",
        "existing_metric": False,
    },
    "left_arm_forward_reach": {
        "type": "normalized_signed_projection",
        "unit": "normalized_length",
        "calculation": "dot(L_Wrist - L_Shoulder, forward_axis) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "right_arm_forward_reach": {
        "type": "normalized_signed_projection",
        "unit": "normalized_length",
        "calculation": "dot(R_Wrist - R_Shoulder, forward_axis) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "left_knee_height_above_hip": {
        "type": "normalized_signed_height_difference",
        "unit": "normalized_length",
        "calculation": "dot(L_Knee - L_Hip, vertical_axis) / average_leg_length",
        "normalizer": "average_leg_length",
        "existing_metric": False,
    },
    "right_knee_height_above_hip": {
        "type": "normalized_signed_height_difference",
        "unit": "normalized_length",
        "calculation": "dot(R_Knee - R_Hip, vertical_axis) / average_leg_length",
        "normalizer": "average_leg_length",
        "existing_metric": False,
    },
    "left_support_balance_offset": {
        "type": "normalized_projected_distance",
        "unit": "normalized_length",
        "calculation": "norm(project_ground(Pelvis - L_Ankle)) / average_leg_length",
        "normalizer": "average_leg_length",
        "existing_metric": False,
    },
    "right_support_balance_offset": {
        "type": "normalized_projected_distance",
        "unit": "normalized_length",
        "calculation": "norm(project_ground(Pelvis - R_Ankle)) / average_leg_length",
        "normalizer": "average_leg_length",
        "existing_metric": False,
    },
    "right_wrist_to_left_elbow_distance": {
        "type": "normalized_joint_distance",
        "unit": "normalized_length",
        "calculation": "distance(R_Wrist, L_Elbow) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "left_wrist_to_right_elbow_distance": {
        "type": "normalized_joint_distance",
        "unit": "normalized_length",
        "calculation": "distance(L_Wrist, R_Elbow) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "wrist_midpoint_chest_height_offset": {
        "type": "normalized_signed_height_difference",
        "unit": "normalized_length",
        "calculation": "dot(0.5 * (L_Wrist + R_Wrist) - Spine3, vertical_axis) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "shoulder_height_difference": {
        "type": "normalized_signed_height_difference",
        "unit": "normalized_length",
        "calculation": "dot(L_Shoulder - R_Shoulder, vertical_axis) / shoulder_width",
        "normalizer": "shoulder_width",
        "existing_metric": False,
    },
    "hip_height_difference": {
        "type": "normalized_signed_height_difference",
        "unit": "normalized_length",
        "calculation": "dot(L_Hip - R_Hip, vertical_axis) / shoulder_width",
        "normalizer": "shoulder_width",
        "existing_metric": False,
    },
    "left_wrist_to_head_distance": {
        "type": "normalized_joint_distance",
        "unit": "normalized_length",
        "calculation": "distance(L_Wrist, Head) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
    "right_wrist_to_head_distance": {
        "type": "normalized_joint_distance",
        "unit": "normalized_length",
        "calculation": "distance(R_Wrist, Head) / torso_length",
        "normalizer": "torso_length",
        "existing_metric": False,
    },
}


def check(
    pose_id: str,
    suffix: str,
    metric_id: str,
    *,
    below: str | None = None,
    above: str | None = None,
) -> dict:
    feedback = {}
    if below:
        feedback["below"] = below
    if above:
        feedback["above"] = above
    if not feedback:
        raise ValueError(f"{pose_id}:{metric_id} has no feedback.")
    return {
        "check_id": f"{pose_id}_{suffix}",
        "metric_id": metric_id,
        "reference_key": f"{pose_id}:{metric_id}",
        "feedback": feedback,
    }


def unique_checks(items: Iterable[dict]) -> list[dict]:
    result = []
    seen = set()
    for item in items:
        if item["metric_id"] in seen:
            continue
        seen.add(item["metric_id"])
        result.append(item)
    return result


def unsupported(requirement: str, reason: str) -> dict:
    return {"requirement": requirement, "reason": reason}


PALM_GAZE = unsupported(
    "掌形、掌心或视线方向",
    "SMPL-24关节点不能可靠表达手掌形状、掌心朝向和视线方向。",
)
TEMPORAL_COORDINATION = unsupported(
    "手法、步法、转腰和重心移动的协调顺序",
    "动作先后、速度和协调关系需要连续帧与相位分析。",
)
FOOT_CONTACT = unsupported(
    "脚掌或脚跟真实着地顺序",
    "单帧SMPL关节只能提供踝部姿势代理，不能确认真实地面接触顺序。",
)
HAND_SHAPE = unsupported(
    "勾手、拳形及手指细节",
    "SMPL-24没有足够的手指关节来判断手型。",
)


def trunk(pose_id: str, context: str = "动作") -> list[dict]:
    return [
        check(
            pose_id,
            "trunk_upright",
            "trunk_vertical_angle",
            above=f"{context}时上体倾斜。请保持立身中正，避免前俯、后仰或侧倾。",
        )
    ]


def relaxed_elbows(pose_id: str) -> list[dict]:
    return [
        check(
            pose_id,
            "left_elbow_relaxed",
            "left_elbow_angle",
            below="左肘弯曲过紧。请放松肩肘并保持手臂圆活。",
            above="左臂过直。请避免锁死左肘，保持自然弧形。",
        ),
        check(
            pose_id,
            "right_elbow_relaxed",
            "right_elbow_angle",
            below="右肘弯曲过紧。请放松肩肘并保持手臂圆活。",
            above="右臂过直。请避免锁死右肘，保持自然弧形。",
        ),
    ]


def support_side(pose_id: str, side: str, context: str) -> list[dict]:
    if side == "left":
        feedback = {
            "above": f"{context}时重心向左腿转移不足。请先稳定左腿承重，再完成手法或步法。"
        }
    else:
        feedback = {
            "below": f"{context}时重心向右腿转移不足。请先稳定右腿承重，再完成手法或步法。"
        }
    return [
        check(
            pose_id,
            f"weight_{side}",
            "pelvis_lateral_support_ratio",
            **feedback,
        )
    ]


def progress(
    pose_id: str,
    front_side: str,
    target: str,
    context: str,
) -> list[dict]:
    metric_id = f"{front_side}_front_support_progress"
    if target == "front":
        feedback = {
            "below": f"{context}时重心前移不足。请逐渐移向{front_side_cn(front_side)}腿并保持稳定。"
        }
    else:
        feedback = {
            "above": f"{context}时重心后移不足。请将重心稳稳移回后腿，前脚保持轻灵。"
        }
    return [
        check(
            pose_id,
            f"weight_{target}",
            metric_id,
            **feedback,
        )
    ]


def front_side_cn(side: str) -> str:
    return "左" if side == "left" else "右"


def wrist_height_metric(side: str, level: str) -> str:
    if level == "head":
        return f"{side}_wrist_head_height_offset"
    if level == "shoulder":
        return f"{side}_wrist_shoulder_height_offset"
    if level == "pelvis":
        return f"{side}_wrist_pelvis_height_offset"
    raise ValueError(level)


def hand_height(
    pose_id: str,
    side: str,
    level: str,
    action: str,
) -> list[dict]:
    side_cn = front_side_cn(side)
    level_cn = {"head": "眼额附近", "shoulder": "肩部附近", "pelvis": "腰胯附近"}[level]
    return [
        check(
            pose_id,
            f"{side}_hand_height",
            wrist_height_metric(side, level),
            below=f"{side_cn}手{action}高度不足。请调整到{level_cn}。",
            above=f"{side_cn}手{action}过高。请沉肩坠肘，调整到{level_cn}。",
        )
    ]


def hold_ball(pose_id: str, upper_side: str, support: str) -> list[dict]:
    upper_cn = front_side_cn(upper_side)
    lower_side = "right" if upper_side == "left" else "left"
    signed_metric = (
        "left_minus_right_wrist_height"
        if upper_side == "left"
        else "right_minus_left_wrist_height"
    )
    return unique_checks(
        support_side(pose_id, support, "抱球收脚")
        + [
            check(
                pose_id,
                "ball_size",
                "wrist_distance_shoulder_ratio",
                below="抱球时双手距离过近。请保持两臂圆撑，留出自然空间。",
                above="抱球时双手距离过远。请适当收拢双手，保持手臂弧形。",
            ),
            check(
                pose_id,
                f"{upper_side}_hand_above",
                signed_metric,
                below=f"抱球手上下位置不清楚。请保持{upper_cn}手在上、{front_side_cn(lower_side)}手在下。",
            ),
        ]
        + relaxed_elbows(pose_id)
        + trunk(pose_id, "抱球收脚")
    )


def step_pose(pose_id: str, side: str) -> list[dict]:
    side_cn = front_side_cn(side)
    back_side = "right" if side == "left" else "left"
    return unique_checks(
        [
            check(
                pose_id,
                f"{side}_step_length",
                f"{side}_step_forward_ratio",
                below=f"{side_cn}脚迈步不足。请在保持{front_side_cn(back_side)}腿承重的前提下适当迈出。",
                above=f"{side_cn}脚迈步过大。请缩短步幅，避免上体被迫前探。",
            ),
            check(
                pose_id,
                f"{side}_heel_pose",
                f"{side}_ankle_angle",
                below=f"{side_cn}脚踝勾起不足。请放松迈步，让脚跟先轻轻着地。",
                above=f"{side_cn}脚踝伸展过多。请避免脚尖抢先落地。",
            ),
        ]
        + progress(pose_id, side, "back", "迈步落脚")
        + trunk(pose_id, "迈步")
    )


def sit_back(pose_id: str, front_side: str) -> list[dict]:
    side_cn = front_side_cn(front_side)
    return unique_checks(
        progress(pose_id, front_side, "back", "后坐")
        + [
            check(
                pose_id,
                f"{front_side}_toe_raised",
                f"{front_side}_ankle_angle",
                below=f"{side_cn}脚尖翘起不足。请放松前脚并自然勾起脚尖。",
                above=f"{side_cn}脚踝勾得过紧。请适当放松。",
            )
        ]
        + trunk(pose_id, "后坐")
    )


def bow_stance(
    pose_id: str,
    front_side: str,
    *,
    high_hand: str | None = None,
    low_hand: str | None = None,
    high_level: str = "head",
) -> list[dict]:
    front_cn = front_side_cn(front_side)
    back_side = "right" if front_side == "left" else "left"
    back_cn = front_side_cn(back_side)
    checks = (
        progress(pose_id, front_side, "front", "弓步")
        + [
            check(
                pose_id,
                f"{front_side}_knee",
                f"{front_side}_knee_angle",
                below=f"{front_cn}弓步屈膝过深。请适当抬高重心。",
                above=f"{front_cn}弓步屈膝不足。请增加前腿承重和弓步幅度。",
            ),
            check(
                pose_id,
                f"{front_side}_knee_toe",
                f"{front_side}_knee_to_toe_forward_offset",
                below=f"{front_cn}膝前移不足。请自然屈膝完成弓步。",
                above=f"{front_cn}膝前移过多，可能超过脚尖。请稍向后沉胯。",
            ),
            check(
                pose_id,
                f"{back_side}_leg_extension",
                f"{back_side}_knee_angle",
                below=f"{back_cn}后腿弯曲过多。请自然蹬伸，但不要锁死膝关节。",
                above=f"{back_cn}膝过度锁直。请保持后腿自然有弹性。",
            ),
            check(
                pose_id,
                "stance_width",
                "stance_width_shoulder_ratio",
                below="弓步横向距离过窄，稳定性不足。请适当分开两脚。",
                above="弓步横向距离过宽。请适当收窄站距。",
            ),
        ]
    )
    if high_hand:
        checks += hand_height(pose_id, high_hand, high_level, "推出或分掌")
    if low_hand:
        low_cn = front_side_cn(low_hand)
        checks += [
            check(
                pose_id,
                f"{low_hand}_hand_at_hip",
                f"{low_hand}_wrist_to_hip_distance",
                above=f"{low_cn}手没有落到胯旁。请沉肩坠肘，让手自然落于胯侧。",
            )
        ]
    return unique_checks(checks + relaxed_elbows(pose_id) + trunk(pose_id, "弓步"))


def empty_stance(
    pose_id: str,
    front_side: str,
    *,
    high_hand: str | None = None,
    low_hand: str | None = None,
) -> list[dict]:
    front_cn = front_side_cn(front_side)
    checks = (
        progress(pose_id, front_side, "back", "虚步")
        + [
            check(
                pose_id,
                f"{front_side}_knee_slight_flexion",
                f"{front_side}_knee_angle",
                below=f"{front_cn}膝弯曲过多，虚步显得沉重。请减轻前腿承重。",
                above=f"{front_cn}腿过直。请保持膝部自然微屈。",
            ),
            check(
                pose_id,
                "stance_length",
                "stance_length_leg_ratio",
                below="虚步前后距离过小。请保持自然稳定的步幅。",
                above="虚步前后距离过大。请适当收近前脚，使步法轻灵。",
            ),
        ]
    )
    if high_hand:
        checks += hand_height(pose_id, high_hand, "head", "位置")
    if low_hand:
        low_cn = front_side_cn(low_hand)
        checks += [
            check(
                pose_id,
                f"{low_hand}_hand_at_hip",
                f"{low_hand}_wrist_to_hip_distance",
                above=f"{low_cn}手离胯部过远。请放松肩肘，将手收落到胯旁。",
            )
        ]
    return unique_checks(checks + relaxed_elbows(pose_id) + trunk(pose_id, "虚步"))


def cross_hands(pose_id: str, context: str) -> list[dict]:
    return unique_checks(
        [
            check(
                pose_id,
                "hands_cross_distance",
                "wrist_distance_shoulder_ratio",
                below=f"{context}时双腕挤得过紧。请保持手臂圆活。",
                above=f"{context}时双手间距过大。请将双手自然合抱于胸前。",
            ),
            check(
                pose_id,
                "hands_chest_height",
                "wrist_midpoint_chest_height_offset",
                below=f"{context}位置过低。请将双手调整到胸前。",
                above=f"{context}位置过高。请沉肩并将双手降低到胸前。",
            ),
        ]
        + relaxed_elbows(pose_id)
        + trunk(pose_id, context)
    )


def knee_lift(pose_id: str, lifted_side: str) -> list[dict]:
    support = "right" if lifted_side == "left" else "left"
    lifted_cn = front_side_cn(lifted_side)
    support_cn = front_side_cn(support)
    return unique_checks(
        [
            check(
                pose_id,
                f"{lifted_side}_knee_height",
                f"{lifted_side}_knee_height_above_hip",
                below=f"{lifted_cn}膝提起不足。请在保持平衡的前提下适当提高膝部。",
                above=f"{lifted_cn}膝提得过高。请保持动作自然，不要耸胯。",
            ),
            check(
                pose_id,
                f"{lifted_side}_knee_flexion",
                f"{lifted_side}_knee_angle",
                below=f"{lifted_cn}腿收得过紧。请放松髋膝。",
                above=f"{lifted_cn}膝弯曲不足。请先屈膝提腿，再完成后续动作。",
            ),
            check(
                pose_id,
                f"{support}_support_balance",
                f"{support}_support_balance_offset",
                above=f"单腿支撑时骨盆偏离{support_cn}脚过多。请稳住支撑腿并保持上体中正。",
            ),
        ]
        + trunk(pose_id, "提膝")
    )


def kick_pose(pose_id: str, kick_side: str) -> list[dict]:
    support = "right" if kick_side == "left" else "left"
    kick_cn = front_side_cn(kick_side)
    support_cn = front_side_cn(support)
    return unique_checks(
        [
            check(
                pose_id,
                f"{kick_side}_knee_extension",
                f"{kick_side}_knee_angle",
                below=f"{kick_cn}蹬脚伸展不足。请由屈膝逐渐向脚跟方向蹬出。",
                above=f"{kick_cn}膝过度锁直。请保持腿部伸展但不要僵硬。",
            ),
            check(
                pose_id,
                f"{kick_side}_ankle_hook",
                f"{kick_side}_ankle_angle",
                below=f"{kick_cn}脚尖回勾不足。请将劲点送到脚跟。",
                above=f"{kick_cn}脚踝勾得过紧。请保持脚踝有控制地伸展。",
            ),
            check(
                pose_id,
                f"{support}_support_balance",
                f"{support}_support_balance_offset",
                above=f"蹬脚时骨盆偏离{support_cn}脚过多。请稳住支撑腿，避免身体摇晃。",
            ),
            check(
                pose_id,
                "hands_shoulder_height",
                "mean_wrist_shoulder_height_offset",
                below="蹬脚时两手分开高度不足。请将双腕保持在肩部附近。",
                above="蹬脚时两手举得过高。请沉肩并适当降低。",
            ),
            check(
                pose_id,
                "hands_width",
                "wrist_width_shoulder_ratio",
                below="蹬脚时双臂展开不足。请随蹬腿自然向两侧分手。",
                above="蹬脚时双臂展开过大。请保持自然舒展。",
            ),
        ]
        + relaxed_elbows(pose_id)
        + trunk(pose_id, "蹬脚")
    )


def low_stance(pose_id: str, extended_side: str) -> list[dict]:
    support = "right" if extended_side == "left" else "left"
    ext_cn = front_side_cn(extended_side)
    support_cn = front_side_cn(support)
    return unique_checks(
        [
            check(
                pose_id,
                f"{support}_support_knee",
                f"{support}_knee_angle",
                below=f"{support_cn}支撑腿下蹲过深。请在可控范围内下沉。",
                above=f"{support_cn}支撑腿下蹲不足。请松腰屈膝，降低重心。",
            ),
            check(
                pose_id,
                f"{extended_side}_leg_extension",
                f"{extended_side}_knee_angle",
                below=f"{ext_cn}仆步腿弯曲过多。请向侧后方自然伸展。",
                above=f"{ext_cn}腿过度锁直。请保持伸展但不要僵硬。",
            ),
            check(
                pose_id,
                "stance_length",
                "stance_length_leg_ratio",
                below="仆步展开不足。请适当伸出侧腿。",
                above="仆步跨度过大。请缩小步幅，保证能够稳定起身。",
            ),
        ]
        + trunk(pose_id, "仆步下势")
    )


def single_leg(pose_id: str, lifted_side: str, high_hand: str) -> list[dict]:
    return unique_checks(
        knee_lift(pose_id, lifted_side)
        + hand_height(pose_id, high_hand, "shoulder", "挑起")
        + relaxed_elbows(pose_id)
    )


def pose(
    move_id: int,
    order: int,
    frame: int | None,
    keypose_dir: str | None,
    stage_name: str,
    technique: str,
    checks: Iterable[dict],
    unsupported_items: Iterable[dict] = (),
) -> dict:
    pose_id = f"{move_id}.{order}"
    result = {
        "pose_id": pose_id,
        "frame": frame,
        "image": (
            f"{keypose_dir}/{frame:05d}.jpg"
            if frame is not None and keypose_dir is not None
            else None
        ),
        "stage_name": stage_name,
        "technique": technique,
        "checks": unique_checks(checks),
        "unsupported_observations": list(unsupported_items),
    }
    if frame is None:
        result["binding_status"] = "missing_keypose_image_and_frame"
    return result


def move(
    move_id: int,
    move_name: str,
    display_name: str,
    keypose_dir: str | None,
    keyposes: list[dict],
) -> dict:
    return {
        "move_id": move_id,
        "move_name": move_name,
        "display_name": display_name,
        "keypose_dir": keypose_dir,
        "keypose_status": (
            "bound_to_images" if keypose_dir is not None else "missing_keypose_images"
        ),
        "keyposes": keyposes,
    }


def prepare_push(
    pose_id: str,
    high_side: str,
    support: str,
    context: str,
) -> list[dict]:
    return unique_checks(
        support_side(pose_id, support, context)
        + hand_height(pose_id, high_side, "head", "举至耳侧")
        + relaxed_elbows(pose_id)
        + trunk(pose_id, context)
    )


def brush_knee_push(
    pose_id: str,
    front_side: str,
    push_side: str,
) -> list[dict]:
    brush_side = front_side
    return unique_checks(
        bow_stance(
            pose_id,
            front_side,
            high_hand=push_side,
            low_hand=brush_side,
            high_level="head",
        )
        + [
            check(
                pose_id,
                f"{push_side}_forward_reach",
                f"{push_side}_arm_forward_reach",
                below=f"{front_side_cn(push_side)}掌前推不足。请随转腰自然向前推出。",
                above=f"{front_side_cn(push_side)}掌前伸过度。请保持肘部微屈，避免探肩。",
            )
        ]
    )


def cloud_pose(pose_id: str, upper_side: str, support: str) -> list[dict]:
    lower_side = "right" if upper_side == "left" else "left"
    upper_cn = front_side_cn(upper_side)
    lower_cn = front_side_cn(lower_side)
    return unique_checks(
        support_side(pose_id, support, "云手转体")
        + hand_height(pose_id, upper_side, "head", "云转")
        + [
            check(
                pose_id,
                f"{lower_side}_hand_lower",
                f"{lower_side}_wrist_pelvis_height_offset",
                below=f"{lower_cn}手运行过低。请让手臂沿腹前自然划弧。",
                above=f"{lower_cn}手没有下沉到腹前。请放松肩肘，形成上下相随的云手。",
            ),
            check(
                pose_id,
                f"{upper_side}_lateral_reach",
                f"{upper_side}_arm_lateral_reach",
                below=f"{upper_cn}手横向运转不足。请以腰带动手臂向侧方划弧。",
                above=f"{upper_cn}手横向伸展过度。请避免探肩和直臂硬拉。",
            ),
            check(
                pose_id,
                "stance_width",
                "stance_width_shoulder_ratio",
                below="云手站距过窄。请保持稳定的小开立步。",
                above="云手站距过宽。请缩小横向步幅，便于平稳移重心。",
            ),
        ]
        + relaxed_elbows(pose_id)
        + trunk(pose_id, "云手")
    )


def peng_pose(pose_id: str, front_side: str, active_side: str) -> list[dict]:
    active_cn = front_side_cn(active_side)
    return unique_checks(
        bow_stance(pose_id, front_side)
        + hand_height(pose_id, active_side, "shoulder", "掤起")
        + [
            check(
                pose_id,
                f"{active_side}_elbow_arc",
                f"{active_side}_elbow_angle",
                below=f"{active_cn}掤臂弯曲过紧。请向外圆撑。",
                above=f"{active_cn}掤臂过直。请保持前臂半圆，不要锁肘。",
            )
        ]
    )


def lu_pose(pose_id: str, front_side: str) -> list[dict]:
    return unique_checks(
        sit_back(pose_id, front_side)
        + [
            check(
                pose_id,
                "waist_turn",
                "torso_yaw_from_move_start",
                below="下捋时转腰不足。请以腰带动两臂走弧线。",
                above="下捋时转腰幅度过大。请保持转体自然连贯。",
            ),
            check(
                pose_id,
                "hands_spacing",
                "wrist_distance_shoulder_ratio",
                below="下捋时双手挤得过近。请保持两臂圆活。",
                above="下捋时双手分得过开。请保持双手协同运行。",
            ),
        ]
        + relaxed_elbows(pose_id)
    )


def ji_pose(pose_id: str, front_side: str) -> list[dict]:
    return unique_checks(
        progress(pose_id, front_side, "front", "前挤")
        + [
            check(
                pose_id,
                "hands_close",
                "wrist_distance_shoulder_ratio",
                below="挤式双手收得过紧。请保持前臂圆撑。",
                above="挤式双手间距过大。请让后手自然附于前腕附近。",
            ),
            check(
                pose_id,
                "hands_chest_height",
                "wrist_midpoint_chest_height_offset",
                below="挤式双手位置过低。请在胸前向前挤出。",
                above="挤式双手位置过高。请沉肩坠肘并适当降低。",
            ),
        ]
        + relaxed_elbows(pose_id)
        + trunk(pose_id, "前挤")
    )


def press_pose(pose_id: str, front_side: str, *, retract: bool = False) -> list[dict]:
    checks = (
        progress(pose_id, front_side, "back" if retract else "front", "后坐收手" if retract else "前按")
        + [
            check(
                pose_id,
                "hands_width",
                "wrist_width_shoulder_ratio",
                below="双手间距过窄。请保持双掌接近肩宽。",
                above="双手分得过宽。请将双掌适当向内收。",
            ),
            check(
                pose_id,
                "hands_height",
                "mean_wrist_shoulder_height_offset" if not retract else "mean_wrist_pelvis_height_offset",
                below="双手位置过低。请按该阶段的标准高度调整。",
                above="双手位置过高。请沉肩坠肘并适当降低。",
            ),
        ]
        + relaxed_elbows(pose_id)
        + trunk(pose_id, "后坐收手" if retract else "前按")
    )
    return unique_checks(checks)


def single_whip_final(pose_id: str) -> list[dict]:
    return unique_checks(
        bow_stance(pose_id, "left", high_hand="left", high_level="head")
        + [
            check(
                pose_id,
                "right_arm_lateral",
                "right_arm_lateral_reach",
                below="右勾手横向展开不足。请随转腰向右侧自然伸展。",
                above="右臂横向伸展过度。请避免探肩和锁肘。",
            ),
            check(
                pose_id,
                "left_elbow_knee_alignment",
                "left_arm_forward_reach",
                below="左掌前推不足。请让左掌与左弓步协调向前。",
                above="左掌前伸过度。请保持左肘微屈并沉肩。",
            ),
        ]
    )


def pipa_pose(pose_id: str) -> list[dict]:
    return unique_checks(
        empty_stance(pose_id, "left", high_hand="left")
        + [
            check(
                pose_id,
                "right_hand_near_left_elbow",
                "right_wrist_to_left_elbow_distance",
                above="右手离左肘里侧过远。请将右手自然收至左肘附近。",
            ),
            check(
                pose_id,
                "hands_spacing",
                "wrist_distance_shoulder_ratio",
                below="手挥琵琶两手靠得过近。请保持双臂圆活。",
                above="手挥琵琶两手分得过开。请将双手自然合于体前。",
            ),
        ]
    )


def punch_pose(pose_id: str) -> list[dict]:
    return unique_checks(
        bow_stance(pose_id, "left")
        + hand_height(pose_id, "right", "shoulder", "出拳")
        + [
            check(
                pose_id,
                "right_punch_reach",
                "right_arm_forward_reach",
                below="右拳前打不足。请沉肩垂肘，将拳自然送至胸前。",
                above="右拳伸得过远。请保持右臂微屈，避免探肩。",
            ),
            check(
                pose_id,
                "left_hand_support",
                "left_wrist_to_right_elbow_distance",
                above="左手离右前臂过远。请将左手自然附于右前臂内侧。",
            ),
        ]
    )


def shuttle_final(pose_id: str, front_side: str) -> list[dict]:
    upper_side = front_side
    push_side = "left" if front_side == "right" else "right"
    upper_cn = front_side_cn(upper_side)
    push_cn = front_side_cn(push_side)
    return unique_checks(
        bow_stance(pose_id, front_side)
        + hand_height(pose_id, upper_side, "head", "上架")
        + hand_height(pose_id, push_side, "head", "前推")
        + [
            check(
                pose_id,
                f"{upper_side}_arm_arc",
                f"{upper_side}_elbow_angle",
                below=f"{upper_cn}架掌手臂收得过紧。请保持额前圆撑。",
                above=f"{upper_cn}架掌手臂过直。请放松肘部并避免耸肩。",
            ),
            check(
                pose_id,
                f"{push_side}_forward_reach",
                f"{push_side}_arm_forward_reach",
                below=f"{push_cn}掌前推不足。请随弓步自然推出。",
                above=f"{push_cn}掌前伸过度。请保持肘部微屈。",
            ),
        ]
    )


def build_move_4() -> dict:
    keypose_dir = "4_louxi_aobu"
    stages = [
        (
            16,
            "右手举至耳侧",
            "上体转动，右手沿弧线举至右肩耳侧，左手收于胸前，重心保持在右腿。",
            lambda p: prepare_push(p, "right", "right", "举手准备"),
        ),
        (
            18,
            "收左脚准备左搂膝",
            "左脚收至右脚内侧，右手保持耳侧准备位置，两臂自然成弧形。",
            lambda p: prepare_push(p, "right", "right", "收脚准备"),
        ),
        (
            21,
            "左脚上步",
            "左脚向左前方迈出，脚跟先轻轻着地，重心暂留右腿。",
            lambda p: step_pose(p, "left"),
        ),
        (
            26,
            "左弓步搂膝推掌",
            "左腿前弓，左手搂过左膝落于胯旁，右掌由耳侧向前推出。",
            lambda p: brush_knee_push(p, "left", "right"),
        ),
        (
            28,
            "后坐左脚尖翘起",
            "重心后移至右腿，左脚尖翘起，上体保持中正。",
            lambda p: sit_back(p, "left"),
        ),
        (
            32,
            "左转举手收右脚",
            "左脚踏实承重，右脚收近；左手举至左肩耳侧，准备右搂膝拗步。",
            lambda p: prepare_push(p, "left", "left", "换式准备"),
        ),
        (
            35,
            "右脚上步",
            "右脚向右前方迈出，脚跟先轻轻着地，重心暂留左腿。",
            lambda p: step_pose(p, "right"),
        ),
        (
            39,
            "右弓步搂膝推掌",
            "右腿前弓，右手搂过右膝落于胯旁，左掌由耳侧向前推出。",
            lambda p: brush_knee_push(p, "right", "left"),
        ),
        (
            42,
            "后坐右脚尖翘起",
            "重心后移至左腿，右脚尖翘起，上体保持中正。",
            lambda p: sit_back(p, "right"),
        ),
        (
            45,
            "右转举手收左脚",
            "右脚踏实承重，左脚收近；右手举至右肩耳侧，准备再次左搂膝。",
            lambda p: prepare_push(p, "right", "right", "换式准备"),
        ),
        (
            48,
            "左脚再次上步",
            "左脚再次向左前方迈出，脚跟先轻轻着地。",
            lambda p: step_pose(p, "left"),
        ),
        (
            52,
            "左弓步再次搂膝推掌",
            "左腿前弓，左手搂膝落胯，右掌向前推出，完成搂膝拗步。",
            lambda p: brush_knee_push(p, "left", "right"),
        ),
    ]
    keyposes = []
    for order, (frame, name, technique, factory) in enumerate(stages, 1):
        keyposes.append(
            pose(
                4,
                order,
                frame,
                keypose_dir,
                name,
                technique,
                factory(f"4.{order}"),
                [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
            )
        )
    return move(4, "louxi_aobu", "左右搂膝拗步", keypose_dir, keyposes)


def build_move_5() -> dict:
    keypose_dir = "5_shouhui_pipa"
    stages = [
        (
            10,
            "右脚跟步",
            "右脚跟进半步，上体保持中正，准备将重心后移至右腿。",
            lambda p: unique_checks(
                support_side(p, "right", "跟步")
                + [
                    check(
                        p,
                        "stance_length",
                        "stance_length_leg_ratio",
                        below="跟步过近。请保留自然稳定的前后距离。",
                        above="右脚跟进不足。请适当缩短前后站距。",
                    )
                ]
                + trunk(p, "跟步")
            ),
        ),
        (
            13,
            "后坐起手",
            "重心后坐至右腿，左脚保持轻灵；左手向前上方挑起，右手回收。",
            lambda p: unique_checks(
                progress(p, "left", "back", "后坐起手")
                + hand_height(p, "left", "head", "挑起")
                + relaxed_elbows(p)
                + trunk(p, "后坐起手")
            ),
        ),
        (
            17,
            "左虚步合手",
            "左脚稍向前移成左虚步；左手高与鼻尖附近，右手置于左肘里侧。",
            pipa_pose,
        ),
    ]
    keyposes = [
        pose(
            5,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"5.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(5, "shouhui_pipa", "手挥琵琶", keypose_dir, keyposes)


def reverse_push_checks(
    pose_id: str,
    *,
    push_side: str,
    retreat_side: str,
    phase: str,
) -> list[dict]:
    front_side = "right" if retreat_side == "left" else "left"
    if phase == "turn":
        return unique_checks(
            support_side(pose_id, front_side, "转体举手")
            + hand_height(pose_id, push_side, "head", "回举")
            + relaxed_elbows(pose_id)
            + trunk(pose_id, "转体举手")
        )
    if phase == "retreat":
        return unique_checks(
            [
                check(
                    pose_id,
                    f"{retreat_side}_retreat_length",
                    "stance_length_leg_ratio",
                    below=f"{front_side_cn(retreat_side)}脚退步不足。请向斜后方自然退步。",
                    above=f"{front_side_cn(retreat_side)}脚退步过大。请缩短步幅并保持稳定。",
                )
            ]
            + support_side(pose_id, retreat_side, "退步")
            + trunk(pose_id, "退步")
        )
    low_side = "left" if push_side == "right" else "right"
    return unique_checks(
        empty_stance(pose_id, front_side, high_hand=push_side, low_hand=low_side)
        + [
            check(
                pose_id,
                f"{push_side}_forward_reach",
                f"{push_side}_arm_forward_reach",
                below=f"{front_side_cn(push_side)}掌前推不足。请随退步自然向前推出。",
                above=f"{front_side_cn(push_side)}掌前伸过度。请保持肘部微屈。",
            )
        ]
    )


def build_move_6() -> dict:
    keypose_dir = "6_daojuangong"
    frames = [9, 12, 15, 19, 22, 25, 29, 32, 36, 39, 43, 46]
    cycle_specs = [
        ("right", "left"),
        ("left", "right"),
        ("right", "left"),
        ("left", "right"),
    ]
    phase_names = [
        ("转体举手", "上体转动，后手沿弧线回举，双臂保持圆活。"),
        ("斜后退步", "一脚向斜后方退步，重心逐渐转移，上体保持中正。"),
        ("虚步卷肱推掌", "退步踏实形成虚步，前手推出，另一手回收至肋侧。"),
    ]
    keyposes = []
    for index, frame in enumerate(frames):
        cycle = index // 3
        phase_index = index % 3
        push_side, retreat_side = cycle_specs[cycle]
        phase = ("turn", "retreat", "push")[phase_index]
        name, technique = phase_names[phase_index]
        pose_id = f"6.{index + 1}"
        keyposes.append(
            pose(
                6,
                index + 1,
                frame,
                keypose_dir,
                f"第{cycle + 1}次{name}",
                technique,
                reverse_push_checks(
                    pose_id,
                    push_side=push_side,
                    retreat_side=retreat_side,
                    phase=phase,
                ),
                [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
            )
        )
    return move(6, "daojuangong", "左右倒卷肱", keypose_dir, keyposes)


def build_lanquewei(
    move_id: int,
    move_name: str,
    display_name: str,
    keypose_dir: str,
    frames: list[int],
    side: str,
) -> dict:
    support = "right" if side == "left" else "left"
    upper_side = support
    if len(frames) == 10:
        stage_specs = [
            ("转体过渡", lambda p: unique_checks(trunk(p, "转体") + relaxed_elbows(p))),
            ("抱球收脚", lambda p: hold_ball(p, upper_side, support)),
            ("上步落脚", lambda p: step_pose(p, side)),
            ("弓步掤臂", lambda p: peng_pose(p, side, side)),
            ("两手搭接准备下捋", lambda p: unique_checks(relaxed_elbows(p) + trunk(p, "搭手"))),
            ("转身下捋", lambda p: lu_pose(p, side)),
            ("搭腕准备前挤", lambda p: unique_checks(relaxed_elbows(p) + trunk(p, "搭腕"))),
            ("弓步前挤", lambda p: ji_pose(p, side)),
            ("后坐收手", lambda p: press_pose(p, side, retract=True)),
            ("弓步前按", lambda p: press_pose(p, side)),
        ]
    else:
        stage_specs = [
            ("转体换重心", lambda p: unique_checks(support_side(p, support, "换向") + trunk(p, "换向"))),
            ("抱球收脚", lambda p: hold_ball(p, upper_side, support)),
            ("上步落脚", lambda p: step_pose(p, side)),
            ("弓步掤臂", lambda p: peng_pose(p, side, side)),
            ("两手搭接准备下捋", lambda p: unique_checks(relaxed_elbows(p) + trunk(p, "搭手"))),
            ("转身下捋", lambda p: lu_pose(p, side)),
            ("搭腕准备前挤", lambda p: unique_checks(relaxed_elbows(p) + trunk(p, "搭腕"))),
            ("弓步前挤", lambda p: ji_pose(p, side)),
            ("分手准备后坐", lambda p: unique_checks(relaxed_elbows(p) + trunk(p, "分手"))),
            ("后坐收手", lambda p: press_pose(p, side, retract=True)),
            ("弓步前按", lambda p: press_pose(p, side)),
        ]
    keyposes = []
    for order, (frame, (name, factory)) in enumerate(zip(frames, stage_specs), 1):
        keyposes.append(
            pose(
                move_id,
                order,
                frame,
                keypose_dir,
                name,
                f"完成{name}阶段，保持两臂圆活、上体中正和重心转换清楚。",
                factory(f"{move_id}.{order}"),
                [PALM_GAZE, TEMPORAL_COORDINATION],
            )
        )
    return move(move_id, move_name, display_name, keypose_dir, keyposes)


def build_move_9() -> dict:
    keypose_dir = "9_danbian"
    stages = [
        (
            13,
            "后坐转体",
            "重心后坐并随上体转动，双手沿弧线运转。",
            lambda p: unique_checks(sit_back(p, "right") + relaxed_elbows(p)),
        ),
        (
            19,
            "左转运臂",
            "重心逐渐移向左腿，两手左高右低随腰向左运转。",
            lambda p: unique_checks(
                support_side(p, "left", "转体运臂")
                + hand_height(p, "left", "shoulder", "平举")
                + relaxed_elbows(p)
                + trunk(p, "转体运臂")
            ),
        ),
        (
            25,
            "勾手收脚",
            "重心移至右腿，右臂向右展开成勾手，左脚收近。",
            lambda p: unique_checks(
                support_side(p, "right", "勾手收脚")
                + hand_height(p, "right", "shoulder", "横向展开")
                + relaxed_elbows(p)
                + trunk(p, "勾手收脚")
            ),
        ),
        (
            29,
            "左脚上步",
            "左脚向左前侧方迈出，重心暂留右腿。",
            lambda p: step_pose(p, "left"),
        ),
        (
            33,
            "左弓步推掌",
            "右脚后蹬成左弓步，左掌随转体向前推出，右勾手向侧后方展开。",
            single_whip_final,
        ),
    ]
    keyposes = [
        pose(
            9,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"9.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, HAND_SHAPE],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(9, "danbian", "单鞭", keypose_dir, keyposes)


def build_move_10() -> dict:
    keypose_dir = "10_yunshou"
    frames = [17, 23, 27, 33, 38, 43]
    states = [
        ("left", "right"),
        ("right", "left"),
        ("left", "right"),
        ("right", "left"),
        ("left", "right"),
        ("right", "left"),
    ]
    keyposes = []
    for order, (frame, (upper, support)) in enumerate(zip(frames, states), 1):
        keyposes.append(
            pose(
                10,
                order,
                frame,
                keypose_dir,
                f"第{order}个云手换侧姿势",
                f"{front_side_cn(upper)}手运行至上方，另一手经腹前划弧，重心移向{front_side_cn(support)}腿。",
                cloud_pose(f"10.{order}", upper, support),
                [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
            )
        )
    return move(10, "yunshou", "云手", keypose_dir, keyposes)


def build_move_11() -> dict:
    keypose_dir = "11_danbian"
    stages = [
        (
            12,
            "转体勾手收脚",
            "重心落在右腿，右臂向右展开成勾手，左脚收至右脚内侧。",
            lambda p: unique_checks(
                support_side(p, "right", "勾手收脚")
                + hand_height(p, "right", "shoulder", "横向展开")
                + relaxed_elbows(p)
                + trunk(p, "勾手收脚")
            ),
        ),
        (
            16,
            "左脚上步",
            "左脚向左前侧方迈出，右腿保持稳定承重。",
            lambda p: step_pose(p, "left"),
        ),
        (
            20,
            "左弓步推掌",
            "左腿前弓，左掌向前推出，右勾手保持侧后方展开。",
            single_whip_final,
        ),
    ]
    keyposes = [
        pose(
            11,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"11.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, HAND_SHAPE],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(11, "danbian_2", "单鞭（第二次）", keypose_dir, keyposes)


def build_move_12() -> dict:
    keypose_dir = "12_gaotanma"
    stages = [
        (
            14,
            "右脚跟步后坐",
            "右脚跟进半步，重心逐渐后移至右腿，两肘保持微屈。",
            lambda p: unique_checks(
                support_side(p, "right", "跟步后坐")
                + relaxed_elbows(p)
                + trunk(p, "跟步后坐")
            ),
        ),
        (
            19,
            "右掌经耳侧前推",
            "上体转向前方，右掌经右耳旁向前推出，左手收至左腰前。",
            lambda p: unique_checks(
                hand_height(p, "right", "head", "前推")
                + [
                    check(
                        p,
                        "right_forward_reach",
                        "right_arm_forward_reach",
                        below="右掌前推不足。请从耳侧沿弧线自然推出。",
                        above="右掌伸得过远。请保持右肘微屈并沉肩。",
                    ),
                    check(
                        p,
                        "left_hand_at_hip",
                        "left_wrist_to_hip_distance",
                        above="左手没有收至腰胯附近。请放松左肩并自然回收。",
                    ),
                ]
                + relaxed_elbows(p)
                + trunk(p, "前推")
            ),
        ),
        (
            24,
            "左虚步推掌",
            "左脚微向前移，脚尖点地成左虚步；右掌保持在眼部附近，左手收于腰前。",
            lambda p: unique_checks(
                empty_stance(p, "left", high_hand="right", low_hand="left")
                + [
                    check(
                        p,
                        "right_forward_reach",
                        "right_arm_forward_reach",
                        below="右掌前推不足。请保持右掌自然向前。",
                        above="右掌伸得过远。请避免探肩和锁肘。",
                    )
                ]
            ),
        ),
    ]
    keyposes = [
        pose(
            12,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"12.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(12, "gaotanma", "高探马", keypose_dir, keyposes)


def split_hands_bow(pose_id: str, front_side: str) -> list[dict]:
    return unique_checks(
        bow_stance(pose_id, front_side)
        + [
            check(
                pose_id,
                "hands_shoulder_height",
                "mean_wrist_shoulder_height_offset",
                below="分手高度不足。请将双腕沿弧线分至肩部附近。",
                above="分手位置过高。请沉肩并适当降低双手。",
            ),
            check(
                pose_id,
                "hands_width",
                "wrist_width_shoulder_ratio",
                below="两臂分开不足。请自然向两侧展开。",
                above="两臂展开过大。请避免探肩和直臂硬拉。",
            ),
        ]
    )


def build_move_13() -> dict:
    keypose_dir = "13_you_dengtui"
    stages = [
        (
            12,
            "左弓步分手",
            "左脚向左前侧方进步成左弓步，两手交叉后向两侧分开。",
            lambda p: split_hands_bow(p, "left"),
        ),
        (
            16,
            "抱手收右脚",
            "重心保持在左腿，右脚收近，两手交叉合抱于胸前。",
            lambda p: unique_checks(
                support_side(p, "left", "抱手收脚") + cross_hands(p, "合抱")
            ),
        ),
        (
            24,
            "右腿屈膝提起",
            "右腿屈膝提起，左腿稳定支撑，两手准备向两侧分开。",
            lambda p: unique_checks(knee_lift(p, "right") + cross_hands(p, "提膝合手")),
        ),
        (
            27,
            "右蹬脚分手",
            "右脚向右前方蹬出，脚尖回勾；两臂向两侧分开，身体保持稳定。",
            lambda p: kick_pose(p, "right"),
        ),
    ]
    keyposes = [
        pose(
            13,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"13.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(13, "you_dengtui", "右蹬脚", keypose_dir, keyposes)


def fists_at_ears(pose_id: str) -> list[dict]:
    return unique_checks(
        bow_stance(pose_id, "right")
        + [
            check(
                pose_id,
                "fists_width",
                "wrist_distance_shoulder_ratio",
                below="两拳距离过近。请保持约一拳宽的自然间距。",
                above="两拳分得过开。请将两拳沿弧线贯至耳前。",
            ),
            check(
                pose_id,
                "fists_height",
                "mean_wrist_shoulder_height_offset",
                below="贯拳高度不足。请将两拳提高到耳部附近。",
                above="贯拳位置过高。请沉肩垂肘并适当降低。",
            ),
        ]
        + relaxed_elbows(pose_id)
    )


def build_move_14() -> dict:
    stages = [
        (
            "右腿收回屈膝落手",
            "右腿收回屈膝平举，两手向下划弧分落于右膝两侧，上体保持中正。",
            lambda p: unique_checks(
                knee_lift(p, "right")
                + [
                    check(
                        p,
                        "hands_near_knee",
                        "mean_wrist_pelvis_height_offset",
                        below="两手下落过低。请保持在提起的右膝两侧。",
                        above="两手下落不足。请随收腿自然落到右膝两侧。",
                    )
                ]
            ),
        ),
        (
            "右弓步双峰贯耳",
            "右脚落地成右弓步，两手变拳向上向前贯至耳前，两臂保持弧形。",
            fists_at_ears,
        ),
    ]
    keyposes = [
        pose(
            14,
            order,
            None,
            None,
            name,
            technique,
            factory(f"14.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT, HAND_SHAPE],
        )
        for order, (name, technique, factory) in enumerate(stages, 1)
    ]
    return move(14, "shuangfeng_guaner", "双峰贯耳", None, keyposes)


def build_move_15() -> dict:
    keypose_dir = "15_zhuanshen_zuo_dengtui"
    stages = [
        (
            13,
            "后坐左转分掌",
            "重心移至左腿，上体左转，右脚尖内扣，两臂向左右分开。",
            lambda p: unique_checks(
                support_side(p, "left", "转身分掌")
                + [
                    check(
                        p,
                        "waist_turn",
                        "torso_yaw_from_move_start",
                        below="转身幅度不足。请先移重心扣脚，再以腰带动转身。",
                        above="转身幅度过大。请保持转身自然稳定。",
                    )
                ]
                + relaxed_elbows(p)
                + trunk(p, "转身")
            ),
        ),
        (
            18,
            "抱手收左脚",
            "重心移至右腿，左脚收近，两手交叉合抱于胸前。",
            lambda p: unique_checks(
                support_side(p, "right", "抱手收脚") + cross_hands(p, "合抱")
            ),
        ),
        (
            23,
            "左腿屈膝提起",
            "左腿屈膝提起，右腿稳定支撑，两手准备向两侧分开。",
            lambda p: unique_checks(knee_lift(p, "left") + cross_hands(p, "提膝合手")),
        ),
        (
            27,
            "左蹬脚分手",
            "左脚向左前方蹬出，脚尖回勾；两臂向两侧分开。",
            lambda p: kick_pose(p, "left"),
        ),
    ]
    keyposes = [
        pose(
            15,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"15.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(15, "zhuanshen_zuo_dengtui", "转身左蹬脚", keypose_dir, keyposes)


def build_move_16() -> dict:
    stages = [
        (
            "收左腿勾手",
            "左腿收回平屈，上体右转，右手成勾手，左掌落于右肩前。",
            lambda p: unique_checks(
                knee_lift(p, "left")
                + hand_height(p, "left", "shoulder", "收至肩前")
                + trunk(p, "收腿转体")
            ),
        ),
        (
            "左仆步穿掌",
            "右腿屈膝下蹲，左腿向左侧偏后伸出成左仆步，左掌顺左腿内侧穿出。",
            lambda p: unique_checks(
                low_stance(p, "left")
                + [
                    check(
                        p,
                        "left_hand_low",
                        "left_wrist_pelvis_height_offset",
                        below="左穿掌位置过低。请沿左腿内侧自然向前穿出。",
                        above="左掌下穿不足。请随下势降低手位。",
                    )
                ]
            ),
        ),
        (
            "左弓腿起身",
            "重心前移，左腿前弓，右腿后蹬，上体起身，左掌继续向前伸出。",
            lambda p: peng_pose(p, "left", "left"),
        ),
        (
            "左独立挑掌",
            "左腿稳定支撑，右腿屈膝提起；右掌挑至右膝上方，左手落于左胯旁。",
            lambda p: unique_checks(
                single_leg(p, "right", "right")
                + [
                    check(
                        p,
                        "left_hand_at_hip",
                        "left_wrist_to_hip_distance",
                        above="左手离左胯过远。请将左手自然落于胯旁。",
                    )
                ]
            ),
        ),
    ]
    keyposes = [
        pose(
            16,
            order,
            None,
            None,
            name,
            technique,
            factory(f"16.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT, HAND_SHAPE],
        )
        for order, (name, technique, factory) in enumerate(stages, 1)
    ]
    return move(16, "zuo_xiashi_duli", "左下势独立", None, keyposes)


def build_move_17() -> dict:
    keypose_dir = "17_you_xiashi_duli"
    stages = [
        (
            13,
            "右脚落地转体",
            "右脚落于左脚前，身体随脚步向左转，上体保持中正。",
            lambda p: unique_checks(
                support_side(p, "left", "落脚转体")
                + trunk(p, "落脚转体")
            ),
        ),
        (
            17,
            "左勾手右掌收肩",
            "左手向后平举，右掌随转体收至左肩前，两臂保持圆活。",
            lambda p: unique_checks(
                hand_height(p, "right", "shoulder", "收至肩前")
                + relaxed_elbows(p)
                + trunk(p, "转体收手")
            ),
        ),
        (
            22,
            "右腿向侧后伸出",
            "重心保持在左腿，右腿向右侧偏后伸出，为右仆步做准备。",
            lambda p: unique_checks(
                support_side(p, "left", "伸腿准备")
                + [
                    check(
                        p,
                        "stance_length",
                        "stance_length_leg_ratio",
                        below="右腿向侧后伸出不足。请适当展开步幅。",
                        above="右腿伸出过远。请保持可控的下势范围。",
                    )
                ]
                + trunk(p, "伸腿准备")
            ),
        ),
        (
            26,
            "右仆步穿掌",
            "左腿屈膝下蹲，右腿向右侧伸出成右仆步，右掌顺右腿内侧穿出。",
            lambda p: unique_checks(
                low_stance(p, "right")
                + [
                    check(
                        p,
                        "right_hand_low",
                        "right_wrist_pelvis_height_offset",
                        below="右穿掌位置过低。请沿右腿内侧自然穿出。",
                        above="右掌下穿不足。请随下势降低手位。",
                    )
                ]
            ),
        ),
        (
            31,
            "右弓腿起身",
            "重心前移，右腿前弓起身，右臂继续向前伸出。",
            lambda p: peng_pose(p, "right", "right"),
        ),
        (
            37,
            "右独立挑掌",
            "右腿稳定支撑，左腿屈膝提起；左掌向前上挑，右手落于右胯旁。",
            lambda p: unique_checks(
                single_leg(p, "left", "left")
                + [
                    check(
                        p,
                        "right_hand_at_hip",
                        "right_wrist_to_hip_distance",
                        above="右手离右胯过远。请将右手自然落于胯旁。",
                    )
                ]
            ),
        ),
    ]
    keyposes = [
        pose(
            17,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"17.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT, HAND_SHAPE],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(17, "you_xiashi_duli", "右下势独立", keypose_dir, keyposes)


def build_move_18() -> dict:
    keypose_dir = "18_chuansuo"
    stages = [
        (12, "左侧抱球收右脚", "左手在上、右手在下抱球，重心在左腿。", lambda p: hold_ball(p, "left", "left")),
        (16, "右脚上步", "右脚向右前方迈出，重心暂留左腿。", lambda p: step_pose(p, "right")),
        (
            20,
            "右架左推准备",
            "右手向额前上架，左手向前准备推出，重心开始前移。",
            lambda p: unique_checks(
                progress(p, "right", "front", "架推准备")
                + hand_height(p, "right", "head", "上架")
                + hand_height(p, "left", "head", "前推")
                + relaxed_elbows(p)
                + trunk(p, "架推准备")
            ),
        ),
        (24, "右弓步穿梭", "右弓步完成，右掌架于额前，左掌向前推出。", lambda p: shuttle_final(p, "right")),
        (28, "右侧抱球收左脚", "右手在上、左手在下抱球，重心在右腿。", lambda p: hold_ball(p, "right", "right")),
        (32, "左脚上步", "左脚向左前方迈出，重心暂留右腿。", lambda p: step_pose(p, "left")),
        (
            36,
            "左架右推准备",
            "左手向额前上架，右手向前准备推出，重心开始前移。",
            lambda p: unique_checks(
                progress(p, "left", "front", "架推准备")
                + hand_height(p, "left", "head", "上架")
                + hand_height(p, "right", "head", "前推")
                + relaxed_elbows(p)
                + trunk(p, "架推准备")
            ),
        ),
        (40, "左弓步穿梭", "左弓步完成，左掌架于额前，右掌向前推出。", lambda p: shuttle_final(p, "left")),
    ]
    keyposes = [
        pose(
            18,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"18.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(18, "chuansuo", "左右穿梭", keypose_dir, keyposes)


def build_move_19() -> dict:
    keypose_dir = "19_haidi_zhen"
    stages = [
        (
            15,
            "右脚跟步提手",
            "右脚跟进半步，重心移至右腿，右手向后上方提抽。",
            lambda p: unique_checks(
                support_side(p, "right", "跟步提手")
                + hand_height(p, "right", "head", "提至耳旁")
                + relaxed_elbows(p)
                + trunk(p, "跟步提手")
            ),
        ),
        (
            20,
            "左虚步蓄势",
            "左脚稍向前移成左虚步，右手提至右耳旁，左手下落。",
            lambda p: empty_stance(p, "left", high_hand="right", low_hand="left"),
        ),
        (
            26,
            "虚步斜下插掌",
            "左虚步保持稳定，右手由耳旁斜向前下方插出，左手落于左胯旁。",
            lambda p: unique_checks(
                empty_stance(p, "left", low_hand="left")
                + [
                    check(
                        p,
                        "right_hand_low",
                        "right_wrist_pelvis_height_offset",
                        below="右插掌位置过低。请沿斜前下方自然插出。",
                        above="右插掌下落不足。请随松腰适当降低手位。",
                    ),
                    check(
                        p,
                        "right_forward_reach",
                        "right_arm_forward_reach",
                        below="右掌斜插前伸不足。请沿弧线送向前下方。",
                        above="右掌伸得过远。请避免探肩和过度前俯。",
                    ),
                ]
            ),
        ),
    ]
    keyposes = [
        pose(
            19,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"19.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(19, "haidi_zhen", "海底针", keypose_dir, keyposes)


def build_move_20() -> dict:
    keypose_dir = "20_shan_tongbei"
    stages = [
        (
            9,
            "提手收左脚",
            "上体微转，左脚收近，双手由体前上提，右手准备上架。",
            lambda p: unique_checks(
                support_side(p, "right", "提手收脚")
                + hand_height(p, "right", "head", "上提")
                + relaxed_elbows(p)
                + trunk(p, "提手收脚")
            ),
        ),
        (
            13,
            "左脚上步右掌上架",
            "左脚向前上步，右掌架于右额前，左手准备向前推出。",
            lambda p: unique_checks(
                step_pose(p, "left")
                + hand_height(p, "right", "head", "上架")
                + relaxed_elbows(p)
            ),
        ),
        (
            17,
            "左弓步架推",
            "重心前移成左弓步，右掌架于额前，左掌向前推出。",
            lambda p: unique_checks(
                bow_stance(p, "left")
                + hand_height(p, "right", "head", "上架")
                + hand_height(p, "left", "head", "前推")
                + [
                    check(
                        p,
                        "left_forward_reach",
                        "left_arm_forward_reach",
                        below="左掌前推不足。请随左弓步自然向前推出。",
                        above="左掌前伸过度。请保持左肘微屈，避免探肩。",
                    )
                ]
                + relaxed_elbows(p)
            ),
        ),
    ]
    keyposes = [
        pose(
            20,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"20.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(20, "shan_tongbei", "闪通臂", keypose_dir, keyposes)


def build_move_21() -> dict:
    keypose_dir = "21_zhuanshen_banlanchui"
    stages = [
        (
            14,
            "后坐扣脚转身",
            "重心后坐，左脚尖内扣，身体向右后转，上体保持中正。",
            lambda p: unique_checks(sit_back(p, "left") + trunk(p, "转身")),
        ),
        (
            19,
            "右手握拳收肋",
            "重心移向左腿，右手随转体变拳收至左肋旁，左掌上举。",
            lambda p: unique_checks(
                support_side(p, "left", "转身握拳")
                + hand_height(p, "left", "head", "上举")
                + relaxed_elbows(p)
                + trunk(p, "转身握拳")
            ),
        ),
        (
            23,
            "右拳搬出",
            "向右转体，右拳经胸前向前翻转搬出，左手落于胯旁。",
            lambda p: unique_checks(
                hand_height(p, "right", "shoulder", "搬出")
                + [
                    check(
                        p,
                        "right_forward_reach",
                        "right_arm_forward_reach",
                        below="右拳搬出不足。请随转腰沿弧线向前。",
                        above="右拳伸得过远。请保持肘部微屈。",
                    ),
                    check(
                        p,
                        "left_hand_at_hip",
                        "left_wrist_to_hip_distance",
                        above="左手没有落到左胯旁。请放松肩肘并自然下落。",
                    ),
                ]
                + trunk(p, "搬拳")
            ),
        ),
        (
            29,
            "右脚上步",
            "右脚向前迈出并外撇，重心逐渐移向右腿。",
            lambda p: step_pose(p, "right"),
        ),
        (
            35,
            "左脚上步拦掌",
            "左脚向前迈步，左掌向前上方拦出，右拳收至右腰旁。",
            lambda p: unique_checks(
                step_pose(p, "left")
                + hand_height(p, "left", "shoulder", "拦出")
                + [
                    check(
                        p,
                        "right_hand_at_hip",
                        "right_wrist_to_hip_distance",
                        above="右拳没有收回腰旁。请随拦掌自然回收。",
                    )
                ]
            ),
        ),
        (
            39,
            "左弓步打拳",
            "左腿前弓，右拳向前打出，高与胸平，左手附于右前臂内侧。",
            punch_pose,
        ),
    ]
    keyposes = [
        pose(
            21,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"21.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT, HAND_SHAPE],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(21, "zhuanshen_banlanchui", "转身搬拦捶", keypose_dir, keyposes)


def build_move_22() -> dict:
    keypose_dir = "22_rufeng_sibi"
    stages = [
        (
            13,
            "两掌分开回收",
            "右拳变掌，两掌翻转向上并向身体方向回收，两肘保持松沉。",
            lambda p: unique_checks(
                [
                    check(
                        p,
                        "hands_width",
                        "wrist_width_shoulder_ratio",
                        below="两掌回收时靠得过近。请保持肩肘松开。",
                        above="两掌分得过宽。请保持接近肩宽。",
                    ),
                    check(
                        p,
                        "hands_chest_height",
                        "wrist_midpoint_chest_height_offset",
                        below="两掌回收位置过低。请保持在胸前。",
                        above="两掌位置过高。请沉肩并适当降低。",
                    ),
                ]
                + relaxed_elbows(p)
                + trunk(p, "收掌")
            ),
        ),
        (
            17,
            "后坐收掌",
            "身体后坐，重心移至右腿，左脚尖翘起，两掌回收。",
            lambda p: unique_checks(sit_back(p, "left") + press_pose(p, "left", retract=True)),
        ),
        (
            24,
            "左弓步前推",
            "重心前移成左弓步，两掌由腹前向上向前推出，宽度不超过两肩。",
            lambda p: press_pose(p, "left"),
        ),
    ]
    keyposes = [
        pose(
            22,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"22.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(22, "rufeng_sibi", "如封似闭", keypose_dir, keyposes)


def build_move_23() -> dict:
    keypose_dir = "23_shizishou"
    stages = [
        (
            12,
            "后坐扣脚转体",
            "重心向左腿移动，左脚尖内扣，身体向右转。",
            lambda p: unique_checks(
                support_side(p, "left", "转体")
                + trunk(p, "转体")
            ),
        ),
        (
            17,
            "右侧弓步分手",
            "随转体形成右侧弓步，两臂向左右侧平举，肘部微屈。",
            lambda p: unique_checks(
                bow_stance(p, "right")
                + [
                    check(
                        p,
                        "hands_shoulder_height",
                        "mean_wrist_shoulder_height_offset",
                        below="两臂侧平举高度不足。请抬至肩部附近。",
                        above="两臂举得过高。请沉肩并适当降低。",
                    ),
                    check(
                        p,
                        "hands_width",
                        "wrist_width_shoulder_ratio",
                        below="两臂分开不足。请向两侧自然展开。",
                        above="两臂展开过度。请避免探肩和锁肘。",
                    ),
                ]
            ),
        ),
        (
            21,
            "收右脚合手",
            "重心移至左腿，右脚向左收回，两手经腹前向内合拢。",
            lambda p: unique_checks(
                support_side(p, "left", "收脚合手")
                + [
                    check(
                        p,
                        "stance_width",
                        "stance_width_shoulder_ratio",
                        below="右脚收得过近。请保持接近肩宽的开立步。",
                        above="右脚收回不足。请将站距调整到接近肩宽。",
                    )
                ]
                + trunk(p, "收脚")
            ),
        ),
        (
            27,
            "开立步十字手",
            "两脚开立与肩同宽，身体自然直立；两手交叉合抱于胸前，两臂撑圆。",
            lambda p: unique_checks(
                cross_hands(p, "十字手")
                + [
                    check(
                        p,
                        "stance_width",
                        "stance_width_shoulder_ratio",
                        below="开立步站距过窄。请保持与肩同宽。",
                        above="开立步站距过宽。请适当向内收脚。",
                    )
                ]
            ),
        ),
    ]
    keyposes = [
        pose(
            23,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"23.{order}"),
            [PALM_GAZE, TEMPORAL_COORDINATION, FOOT_CONTACT],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(23, "shizishou", "十字手", keypose_dir, keyposes)


def build_move_24() -> dict:
    keypose_dir = "24_shoushi"
    stages = [
        (
            10,
            "翻掌分手下落",
            "两手向外翻转并从胸前向两侧下落，全身保持放松和中正。",
            lambda p: unique_checks(
                [
                    check(
                        p,
                        "hands_width",
                        "wrist_width_shoulder_ratio",
                        below="两手分开不足。请放松肩臂并自然向两侧下落。",
                        above="两手分得过开。请保持动作自然，不要横向硬拉。",
                    ),
                    check(
                        p,
                        "hands_height",
                        "mean_wrist_pelvis_height_offset",
                        below="双手下落过快或过低。请保持均匀缓慢的收势。",
                        above="双手下落不足。请继续放松肩肘向下。",
                    ),
                ]
                + relaxed_elbows(p)
                + trunk(p, "收势分手")
            ),
        ),
        (
            16,
            "开立步垂手",
            "两腿逐渐蹬直，两手下落至大腿外侧，身体自然直立。",
            lambda p: unique_checks(
                [
                    check(
                        p,
                        "stance_width",
                        "stance_width_shoulder_ratio",
                        below="收脚前站距过窄。请保持自然开立。",
                        above="站距过宽。请准备平稳收回左脚。",
                    ),
                    check(
                        p,
                        "hands_near_thighs",
                        "mean_wrist_to_thigh_distance",
                        above="双手离大腿外侧较远。请放松肩臂，让双手自然垂落。",
                    ),
                ]
                + trunk(p, "收势直立")
            ),
        ),
        (
            22,
            "并步直立",
            "左脚收回右脚旁，双脚并拢，双手自然垂于大腿外侧，头颈正直。",
            lambda p: unique_checks(
                [
                    check(
                        p,
                        "feet_together",
                        "stance_width_shoulder_ratio",
                        above="收势后双脚没有并拢。请平稳收回左脚，恢复并步直立。",
                    ),
                    check(
                        p,
                        "hands_near_thighs",
                        "mean_wrist_to_thigh_distance",
                        above="收势后双手没有自然垂落。请放松肩肘，让双手贴近大腿外侧。",
                    ),
                    check(
                        p,
                        "neck_upright",
                        "neck_spine_angle",
                        below="收势时头颈屈曲过多。请保持头颈正直。",
                        above="收势时头颈后仰。请下颏自然微收。",
                    ),
                ]
                + trunk(p, "并步直立")
            ),
        ),
    ]
    keyposes = [
        pose(
            24,
            order,
            frame,
            keypose_dir,
            name,
            technique,
            factory(f"24.{order}"),
            [
                PALM_GAZE,
                TEMPORAL_COORDINATION,
                unsupported("呼吸与气息下沉", "SMPL运动数据不包含呼吸信号。"),
            ],
        )
        for order, (frame, name, technique, factory) in enumerate(stages, 1)
    ]
    return move(24, "shoushi", "收势", keypose_dir, keyposes)


def build_rules() -> dict:
    rules = json.loads(FIRST_THREE_PATH.read_text(encoding="utf-8"))
    rules = deepcopy(rules)
    rules["title"] = "24-form Tai Chi technique-metric-feedback rules"
    rules["status"] = "moves_1_3_reviewed_moves_4_24_draft_for_manual_review"
    rules["scope"] = {
        "move_ids": list(range(1, 25)),
        "move_names": [
            "qishi",
            "yemafenzong",
            "baiheliangchi",
            "louxi_aobu",
            "shouhui_pipa",
            "daojuangong",
            "zuo_lanquewei",
            "you_lanquewei",
            "danbian",
            "yunshou",
            "danbian_2",
            "gaotanma",
            "you_dengtui",
            "shuangfeng_guaner",
            "zhuanshen_zuo_dengtui",
            "zuo_xiashi_duli",
            "you_xiashi_duli",
            "chuansuo",
            "haidi_zhen",
            "shan_tongbei",
            "zhuanshen_banlanchui",
            "rufeng_sibi",
            "shizishou",
            "shoushi",
        ],
        "pose_count": 133,
        "image_bound_pose_count": 127,
        "unbound_pose_count": 6,
        "joint_model": "SMPL-24",
    }
    rules["sources"]["generator"] = str(Path(__file__).resolve())
    rules["body_frame"]["move_start_forward_axis"] = (
        "The body forward axis at the first valid pose of each move; used as the "
        "zero heading for signed torso-yaw comparisons."
    )
    rules["metric_definitions"].update(deepcopy(ADDITIONAL_METRICS))
    rules["moves"].extend(
        [
            build_move_4(),
            build_move_5(),
            build_move_6(),
            build_lanquewei(
                7,
                "zuo_lanquewei",
                "左揽雀尾",
                "7_zuo_lanquewei",
                [12, 15, 18, 22, 26, 31, 34, 38, 43, 49],
                "left",
            ),
            build_lanquewei(
                8,
                "you_lanquewei",
                "右揽雀尾",
                "8_you_lanquewei",
                [11, 16, 20, 24, 27, 31, 36, 39, 43, 49, 55],
                "right",
            ),
            build_move_9(),
            build_move_10(),
            build_move_11(),
            build_move_12(),
            build_move_13(),
            build_move_14(),
            build_move_15(),
            build_move_16(),
            build_move_17(),
            build_move_18(),
            build_move_19(),
            build_move_20(),
            build_move_21(),
            build_move_22(),
            build_move_23(),
            build_move_24(),
        ]
    )
    return rules


def main() -> None:
    rules = build_rules()
    OUTPUT_PATH.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
