#!/usr/bin/env python
"""Redraw existing temporal events with visually matched auxiliary KeyPose images."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from PIL import Image

from run_temporal_keypose_localization import (
    file_record, load_json, save_overview, write_json,
)


PROJECT = Path(__file__).resolve().parent
KEYPOSE_ROOT = Path('/home/sqw/VisualSearch/aqa/FrameData/KeyPose')
AUXILIARY_KEYPOSES = {
    2: {
        'hold_ball_1': ('2_yemafenzong/00013.jpg', '对应代表帧'),
        'part_mane_1_L': ('2_yemafenzong/00019.jpg', '对应代表帧'),
        'hold_ball_2': ('2_yemafenzong/00025.jpg', '对应代表帧'),
        'part_mane_2_R': ('2_yemafenzong/00032.jpg', '对应代表帧'),
        'hold_ball_3': ('2_yemafenzong/00038.jpg', '对应代表帧'),
        'part_mane_3_L': ('2_yemafenzong/00045.jpg', '对应结束定势'),
    },
    3: {
        'gather_hands': ('3_baiheliangchi/00014.jpg', '对应区间起点'),
        'separate_hands_to_endpoint': ('3_baiheliangchi/00026.jpg', '对应结束定势'),
    },
}
FOLDERS = {2: '02_yemafenzong', 3: '03_baiheliangchi'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path,
                        default=PROJECT/'temporal_keypose_results/first3_v2/BV1WE411W7JB')
    parser.add_argument('--keypose-root', type=Path, default=KEYPOSE_ROOT)
    parser.add_argument('--move', type=int, choices=(2, 3), nargs='+', default=[2, 3])
    args = parser.parse_args()
    jobs = []
    for move_id in dict.fromkeys(args.move):
        folder = args.results_root/FOLDERS[move_id]
        plan_path = folder/'temporal_queries.json'
        result_path = folder/'keypose_events.json'
        summary_path = folder/'summary.json'
        plan, result, summary = [load_json(p) for p in (plan_path,result_path,summary_path)]
        if plan['video_id'] != 'BV1WE411W7JB' or plan['move']['move_id'] != move_id:
            parser.error(f'Auxiliary correspondences do not match {folder}')
        mapping = AUXILIARY_KEYPOSES[move_id]
        if set(mapping) != {e['event_id'] for e in plan['events']}:
            parser.error(f'Unexpected event set in {plan_path}')
        if [e['event_id'] for e in plan['events']] != [e['event_id'] for e in result['events']]:
            parser.error(f'Query/result event order differs in {folder}')
        video = Path(summary['inputs']['video']['path'])
        if not video.is_file():
            parser.error(f'Missing video: {video}')
        overlay, records, correspondences = deepcopy(plan), [], {}
        for event in overlay['events']:
            relative_path, label = mapping[event['event_id']]
            image_path = args.keypose_root/relative_path
            with Image.open(image_path) as image:
                image.verify()
            event['auxiliary_reference_image'] = str(image_path)
            correspondences[event['event_id']] = label
            records.append(dict(event_id=event['event_id'], correspondence=label,
                                image=file_record(image_path,include_hash=True)))
        jobs.append((folder,overlay,result,summary,video,correspondences,records))

    for folder,plan,result,summary,video,correspondences,records in jobs:
        output = folder/'keypose_overview.jpg'
        temporary = folder/'.keypose_overview_with_auxiliary.tmp.jpg'
        backup = folder/'keypose_overview_without_auxiliary.jpg'
        input_records = {name: file_record(folder/name,include_hash=True) for name in (
            'temporal_queries.json','keypose_events.json','keypose_events.csv',
            'motion_signals.npz','sensitivity_runs.csv','summary.json')}
        save_overview(temporary,video,plan,result['events'],summary['video']['source_fps'],
                      auxiliary_correspondences=correspondences)
        with Image.open(temporary) as image:
            image.verify()
        if output.exists() and not backup.exists():
            shutil.copy2(output,backup)
        temporary.replace(output)
        for name,record in input_records.items():
            if file_record(folder/name,include_hash=True)['sha256'] != record['sha256']:
                raise RuntimeError(f'Source results changed unexpectedly: {folder/name}')
        write_json(folder/'auxiliary_keypose_mapping.json',dict(
            schema_version='1.0', video_id=plan['video_id'], move_id=plan['move']['move_id'],
            purpose='display-only auxiliary pose correspondence',
            selection_method='Codex visual inspection; gather_hands start follows user review',
            used_for_detection=False, source_results=input_records, auxiliary_keyposes=records,
            output=file_record(output,include_hash=True),
            previous_overview_path=str(backup),
            rendering_source=file_record(PROJECT/'run_temporal_keypose_localization.py',include_hash=True),
        ))
        print(f'Redrawn {len(records)} events: {output}')


if __name__ == '__main__':
    main()
