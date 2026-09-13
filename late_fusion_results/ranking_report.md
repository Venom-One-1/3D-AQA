# Learnable late rank-fusion experiment

## Aggregate results

| Method | Protocol | Macro Spearman | Mean rank error | Exact ranks |
|---|---|---:|---:|---:|
| metric_pose_score | no_training | 0.700 | 0.800 | 5/15 |
| metric_reviewed_pause_count | no_training | 0.338 | 1.333 | 2/15 |
| metric_weighted_center_excess | no_training | 0.867 | 0.400 | 10/15 |
| equal_weight_borda | no_training | 0.846 | 0.533 | 7/15 |
| learned_rank_fusion_oof | leave_one_move_out | 0.867 | 0.400 | 10/15 |
| learned_continuous_fusion_oof | leave_one_move_out | 0.833 | 0.533 | 8/15 |
| rule_based_in_sample_reference | in_sample_tuned_reference | 0.933 | 0.267 | 11/15 |

## Complete rankings

| Method | Move | Complete order | Spearman | Mean rank error |
|---|---|---|---:|---:|
| metric_pose_score | qishi | 4 > 10 > 2 > 1 > 3 | 0.400 | 1.200 |
| metric_pose_score | yemafenzong | 10 > 2 > 1 > 3 > 4 | 0.900 | 0.400 |
| metric_pose_score | baiheliangchi | 10 > 2 > 1 > 4 > 3 | 0.800 | 0.800 |
| metric_reviewed_pause_count | qishi | 2 = 3 = 10 > 1 > 4 | 0.335 | 1.200 |
| metric_reviewed_pause_count | yemafenzong | 1 > 3 = 10 > 2 > 4 | 0.103 | 1.800 |
| metric_reviewed_pause_count | baiheliangchi | 1 = 3 = 10 > 2 = 4 | 0.577 | 1.000 |
| metric_weighted_center_excess | qishi | 10 > 4 > 2 > 1 > 3 | 0.700 | 0.800 |
| metric_weighted_center_excess | yemafenzong | 10 > 2 > 1 > 3 > 4 | 0.900 | 0.400 |
| metric_weighted_center_excess | baiheliangchi | 10 > 1 > 2 > 3 > 4 | 1.000 | 0.000 |
| equal_weight_borda | qishi | 10 > 2 = 4 > 1 = 3 | 0.738 | 0.800 |
| equal_weight_borda | yemafenzong | 10 > 1 > 2 > 3 > 4 | 0.800 | 0.800 |
| equal_weight_borda | baiheliangchi | 10 > 1 > 2 > 3 > 4 | 1.000 | 0.000 |
| learned_rank_fusion_oof | qishi | 10 > 4 > 2 > 1 > 3 | 0.700 | 0.800 |
| learned_rank_fusion_oof | yemafenzong | 10 > 2 > 1 > 3 > 4 | 0.900 | 0.400 |
| learned_rank_fusion_oof | baiheliangchi | 10 > 1 > 2 > 3 > 4 | 1.000 | 0.000 |
| learned_continuous_fusion_oof | qishi | 10 > 4 > 2 > 1 > 3 | 0.700 | 0.800 |
| learned_continuous_fusion_oof | yemafenzong | 10 > 1 > 2 > 3 > 4 | 0.800 | 0.800 |
| learned_continuous_fusion_oof | baiheliangchi | 10 > 1 > 2 > 3 > 4 | 1.000 | 0.000 |
| rule_based_in_sample_reference | qishi | 10 > 2 > 1 > 4 > 3 | 1.000 | 0.000 |
| rule_based_in_sample_reference | yemafenzong | 10 > 1 > 2 > 3 > 4 | 0.800 | 0.800 |
| rule_based_in_sample_reference | baiheliangchi | 10 > 1 > 2 > 3 > 4 | 1.000 | 0.000 |

## Default cross-validation weights

| Input | Held-out move | Pose | Pause | Center |
|---|---|---:|---:|---:|
| rank | qishi | 0.440 | 0.000 | 0.560 |
| rank | yemafenzong | 0.267 | 0.198 | 0.535 |
| rank | baiheliangchi | 0.367 | 0.087 | 0.546 |
| continuous | qishi | 0.479 | 0.046 | 0.475 |
| continuous | yemafenzong | 0.323 | 0.257 | 0.420 |
| continuous | baiheliangchi | 0.436 | 0.076 | 0.488 |

## Cross-validation weight stability

| Input | Metric | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|
| rank | pose_score | 0.358 | 0.071 | 0.267 | 0.440 |
| rank | reviewed_pause_count | 0.095 | 0.081 | 0.000 | 0.198 |
| rank | weighted_center_excess | 0.547 | 0.010 | 0.535 | 0.560 |
| continuous | pose_score | 0.413 | 0.066 | 0.323 | 0.479 |
| continuous | reviewed_pause_count | 0.126 | 0.093 | 0.046 | 0.257 |
| continuous | weighted_center_excess | 0.461 | 0.029 | 0.420 | 0.488 |

## Exploratory full-data weights

| Input | Pose | Pause | Center |
|---|---:|---:|---:|
| rank | 0.356 | 0.096 | 0.548 |
| continuous | 0.410 | 0.130 | 0.460 |

The learned results are relative rankings for this five-student cohort. Only the leave-one-move-out rows are out-of-fold evaluations.
Unreviewed pause events are provisionally treated as valid, so the learned pause weight must remain exploratory.
