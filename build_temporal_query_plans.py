#!/usr/bin/env python
"""Export authored move 2/3 temporal queries using a reference manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from aqa3d.temporal_keyposes import validate_temporal_plan


PROJECT = Path(__file__).resolve().parent
KEYPOSE_ROOT = Path('/home/sqw/VisualSearch/aqa/FrameData/KeyPose')


def technique_steps(text: str, move_id: int) -> list[str]:
    section = re.search(
        rf'^## 第{move_id}式[^\n]*\n(.*?)(?=^## 第|\Z)', text, re.M | re.S,
    )
    if section is None:
        raise ValueError(f'Missing technique for move {move_id}')
    actions = section[1].split('### 动作要领', 1)[1].split('### ', 1)[0]
    steps = re.findall(r'^\d+\.\s*(.+)$', actions, re.M)
    if not steps:
        raise ValueError(f'No action steps for move {move_id}')
    return steps


def measurement_catalog():
    measurements, metrics = [], []

    def add(name, operation, points, normalizer, components, description):
        query = dict(query_id=name, operation=operation, points=points, normalizer=normalizer)
        if operation == 'relative_position':
            query['coordinate_frame'] = 'torso_v1'
        measurements.append(query)
        for component in components:
            metrics.append(dict(
                metric_id=name+'_'+component,
                sources=[dict(query_id=name, component=component)],
                reduction='mean', description=description+' / '+component,
            ))

    for side in ('L', 'R'):
        add(side+'_wrist', 'relative_position', [side+'_Wrist', 'Pelvis'],
            'torso_length', ['up', 'forward'], '手腕相对骨盆，按演示者自身左右侧命名')
        add(side+'_knee', 'joint_angle', [side+'_Hip',side+'_Knee',side+'_Ankle'],
            None, ['value'], '膝关节几何夹角，减小表示屈曲增加，不代表真实承重')
    for top, bottom in [('R','L'), ('L','R')]:
        add(top+'_above_'+bottom, 'relative_position', [top+'_Wrist',bottom+'_Wrist'],
            'torso_length', ['up'], f'{top}腕减{bottom}腕的躯干向上分量，正值表示前者更高')
    add('ankle_distance', 'distance', ['L_Ankle','R_Ankle'], 'average_leg_length',
        ['value'], '双踝距离/平均腿长，用于收脚和开步的几何代理，不代表足底接触')
    add('wrist_distance', 'distance', ['L_Wrist','R_Wrist'], 'torso_length',
        ['value'], '双腕距离/躯干长，用于手部合拢或分离的辅助线索')
    return measurements, metrics


def condition(name, metric, operator, threshold, *, signal='value', role='required',
              duration=.15, weight=1.0):
    return dict(condition_id=name, metric_id=metric, signal=signal, aggregation='instant',
                operator=operator, role=role, min_duration_seconds=duration,
                score_threshold=threshold, weight=weight)


def event(name, label, technique, region, conditions, rationale, *, after=None,
          terminal=False, scope='stable_window', strategy='first_match', image=None,
          representative='max_score'):
    return dict(
        event_id=name, label=label, technique=technique,
        query=dict(
            schema_version='0.1', scope=scope,
            search_region=dict(unit='move_progress', start=region[0], end=region[1]),
            conditions=conditions, temporal_relations=[],
            selection=dict(target='window', strategy=strategy,
                           representative_frame='end' if terminal else representative),
            rationale=rationale,
        ),
        after_event=(dict(event_id=after, anchor='representative', offset_seconds=.2)
                     if after else None),
        terminal_extension='move_end' if terminal else 'none',
        auxiliary_reference_image=str(KEYPOSE_ROOT / image) if image else '',
    )


def build_plan(manifest, text, move_id):
    endpoint = next(e for e in manifest['endpoint_keyposes'] if e['move_id'] == move_id)
    steps = technique_steps(text, move_id)
    if endpoint['final_technique_step'] != steps[-1]:
        raise ValueError('TechPoint and reference manifest endpoint technique differ')
    previous = next(e for e in manifest['endpoint_keyposes'] if e['move_id'] == move_id-1)
    measurements, metrics = measurement_catalog()
    plan = dict(
        schema_version='1.0', plan_id=endpoint['move_name_pinyin']+'_temporal_keyposes_v1',
        video_id=manifest['reference']['video_id'],
        move=dict(move_id=move_id, move_name=endpoint['move_name_zh'],
                  start_seconds=previous['boundary_time_seconds'],
                  end_seconds=endpoint['boundary_time_seconds'], technique='\n'.join(steps)),
        execution=dict(smoothing_seconds=.2, derivative_window_seconds=.35,
                       bridge_gap_seconds=.15, minimum_valid_ratio=.7,
                       sensitivity_smoothing_seconds=[.15,.2,.25],
                       sensitivity_threshold_multipliers=[.9,1.,1.1]),
        measurements=measurements, metrics=metrics, events=[],
        notes=[
            '由 Codex 根据完整动作要领编写；本次不调用外部 LLM。',
            '参考视频已知招式边界限定搜索范围；末次完成事件允许延伸至已知结束定势。',
            '招式内的宽进度窗口和事件先后关系属于显式先验，不是预测时间戳。',
            '左右按演示者自身解剖方向定义，保留单侧信号，不取双侧平均。',
            'KeyPose 图片仅供展示，不作为内部事件 Gold 或评分输入。',
            '定位阶段不验证掌心朝向、视线、肌肉放松、足底接触或真实承重。',
            '在此参考视频上开发并检查；参数扰动反映敏感性，不是独立准确率评估。',
        ],
    )
    if move_id == 2:
        # Three alternating repetitions share the same anatomical criteria.
        for repeat, side, region, bow_region in [
            (1,'L',(0,.4),(0,.5)), (2,'R',(.2,.8),(.2,.85)),
            (3,'L',(.55,1),(.55,1)),
        ]:
            top = 'R' if side == 'L' else 'L'
            hold_id, bow_id = f'hold_ball_{repeat}', f'part_mane_{repeat}_{side}'
            hold_text = steps[0] if repeat == 1 else steps[2] if repeat == 2 else (
                '按左右交替的动作顺序再次抱球收脚，右手在上、左手在下，左脚收至右脚内侧。'
            )
            plan['events'].append(event(
                hold_id, f'第{repeat}次抱球收脚（{top}手在上）', hold_text, region,
                [condition('feet_close','ankle_distance_value','local_minimum',.82),
                 condition('upper_hand_order',top+'_above_'+side+'_up','positive',.5),
                 condition('upper_hand_forward',top+'_wrist_forward','local_maximum',.4,
                           role='supporting',weight=.3),
                 condition('hands_gathering','wrist_distance_value','local_minimum',.4,
                           role='supporting',weight=.2)],
                '以收脚时双踝距离低谷确定阶段，并要求上、下手的解剖侧次序正确；'
                '手部前向位置和双腕距离为辅助线索。取满足条件的首个区间。',
                after=None if repeat == 1 else f'part_mane_{repeat-1}_{top}',
            ))
            plan['events'].append(event(
                bow_id, f'第{repeat}次弓步分掌（'+('左' if side=='L' else '右')+'势）',
                steps[1] if repeat == 1 else steps[3] if repeat == 2 else steps[4], bow_region,
                [condition('feet_apart','ankle_distance_value','local_maximum',.65),
                 condition('leading_hand_raised',side+'_wrist_up','local_maximum',.7),
                 condition('front_knee_bent',side+'_knee_value','local_minimum',.55),
                 condition('parted_hand_order',side+'_above_'+top+'_up','positive',.5),
                 condition('hands_apart','wrist_distance_value','local_maximum',.5,
                           role='supporting',weight=.25)],
                '在对应抱球后，以开步、前腿屈膝和同侧手举高、异侧手较低共同定位分掌完成附近；'
                '末次事件延伸至已知招式结束边界，代表帧取边界。',
                after=hold_id, terminal=repeat==3,
            ))
    elif move_id == 3:
        plan['events'].append(event(
            'gather_hands', '转体抱手阶段', steps[0], (0,.65),
            [condition('hands_gathering','wrist_distance_value','local_minimum',.7),
             condition('left_hand_above','L_above_R_up','positive',.5),
             condition('following_step','ankle_distance_value','local_minimum',.5,
                       role='supporting',weight=.3)],
            '在招式前段寻找双腕合拢且左腕仍高于右腕的阶段，脚间距仅辅助识别跟步。',
            strategy='best_score', representative='center',
        ))
        plan['events'].append(event(
            'separate_hands_to_endpoint', '右手上提、左手下按至定势', steps[1], (.2,1),
            [condition('hands_separate_vertically','R_above_L_up','increasing',.15,
                       signal='velocity',duration=.3),
             condition('right_hand_rising','R_wrist_up','increasing',.1,
                       signal='velocity',role='supporting',duration=.3,weight=.4),
             condition('left_hand_descending','L_wrist_up','decreasing',.1,
                       signal='velocity',role='supporting',duration=.3,weight=.4)],
            '抱手之后以右腕减左腕的高度差持续增加定位上下分掌过程；'
            '允许两手有少量先后，不要求同一帧同时达到速度峰值，区间延至已知结束定势。',
            after='gather_hands', terminal=True, scope='transition_interval',
            strategy='longest_match',
        ))
    else:
        raise ValueError('Authored plans currently support moves 2 and 3')
    validate_temporal_plan(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=PROJECT/'reference_data/BV1WE411W7JB/reference_manifest.json')
    parser.add_argument('--technique', type=Path, default=PROJECT/'TechPoint.md')
    parser.add_argument('--output-dir', type=Path, default=PROJECT/'temporal_query_specs/first3_v2')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    text = args.technique.read_text(encoding='utf-8')
    plans = [build_plan(manifest,text,move_id) for move_id in (2,3)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir/(f"{p['move']['move_id']:02d}_"+p['plan_id']+'.json') for p in plans]
    if any(p.exists() for p in paths):
        parser.error('Refusing to overwrite existing query plans; choose a new output directory')
    for path, plan in zip(paths,plans):
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(path)


if __name__ == '__main__':
    main()
