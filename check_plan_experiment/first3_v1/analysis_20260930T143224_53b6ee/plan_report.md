# 动作要领到检查计划：开发实验

前三式各三次独立调用。输入不含学生值、教师区间或旧规则。

程序只验证格式、可执行性和逐条引用；语义正确性仍未人工确认。重复一致不代表正确，区间不是动作合格阈值。本轮不输出诊断或训练反馈。

## M01_R01 · 起势

API: completed；格式: True；语义: 未复核

| 原文 | 支持类型 | 几何查询 / 依据 | 局限 |
|---|---|---|---|
| 上体保持正直 | proxy | joint_angle(Spine1, Spine2, Spine3) [value]; 测量中段脊柱（Spine1-Spine2-Spine3）的共线程度，角度接近180°表示脊柱中段较直。; 角度应接近180°，表示脊柱中段无明显弯曲或侧倾。；joint_angle(Spine2, Spine3, Neck) [value]; 测量上段脊柱至颈部（Spine2-Spine3-Neck）的共线程度，角度接近180°表示上体延伸方向一致。; 角度应接近180°，表示上段脊柱与颈部保持连贯伸直。 | torso_v1的up轴沿躯干方向而非重力方向，无法判断躯干是否在世界坐标中竖直。；仅能评估脊柱节段间的共线性，不能区分前倾、后仰或侧倾。；'正'（竖直）的判定需要重力参考，当前工具不支持。 |
| 两腿屈膝松腰下蹲 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle) [value]; 测量左膝（L_Hip-L_Knee-L_Ankle）的屈曲程度，角度明显小于180°表示膝关节弯曲。; 角度应明显小于180°，表示左膝处于屈曲状态。；joint_angle(R_Hip, R_Knee, R_Ankle) [value]; 测量右膝（R_Hip-R_Knee-R_Ankle）的屈曲程度，角度明显小于180°表示膝关节弯曲。; 角度应明显小于180°，表示右膝处于屈曲状态。 | '松腰'为肌肉放松状态，属于unsupported类别，无法从关节位置推断。；'下蹲'的绝对深度需要地面参考（ground_clearance），当前工具不支持。；膝关节屈曲角度不能推断哪条腿承重或实际重心位置。 |
| 同时两掌轻轻下按至腹前 | proxy | relative_position(L_Wrist, Pelvis) [forward, up]; 测量左腕相对骨盆在躯干坐标系中的前后和上下位置，判断左手是否位于腹前区域。; forward分量应为正值（在身体前方），up分量应接近零或略正（约在骨盆至下腹高度）。；relative_position(R_Wrist, Pelvis) [forward, up]; 测量右腕相对骨盆在躯干坐标系中的前后和上下位置，判断右手是否位于腹前区域。; forward分量应为正值（在身体前方），up分量应接近零或略正（约在骨盆至下腹高度）。 | Wrist关节点非掌心中心，位置存在偏差。；'轻轻'为力度/品质描述，无法从关节几何推断。；'下按'为运动方向/过程，端点仅能反映最终位置。；掌心朝向（palm_orientation）属于unsupported类别。 |
| 两肘松垂 | proxy | relative_position(L_Elbow, L_Shoulder) [up, right]; 测量左肘相对左肩在躯干坐标系中的上下和内外位置，判断左肘是否自然下垂且未外展。; up分量应为负值（肘低于肩），right分量绝对值应较小（肘未明显外展）。；relative_position(R_Elbow, R_Shoulder) [up, right]; 测量右肘相对右肩在躯干坐标系中的上下和内外位置，判断右肘是否自然下垂且未外展。; up分量应为负值（肘低于肩），right分量绝对值应较小（肘未明显外展）。 | '松'为肌肉放松状态，属于unsupported类别，无法从关节位置区分放松下垂与紧张持握。；仅能判断肘部空间位置，不能推断肘关节的肌张力或柔软度。 |
| 与两膝上下相对 | direct | relative_position(L_Elbow, L_Knee) [up, right, forward]; 测量左肘相对左膝在躯干坐标系中的三维偏移，验证肘是否在膝的正上方且内外、前后对齐。; up分量应为正值（肘在膝上方），right和forward分量应接近零（肘与膝在内外和前后方向对齐）。；relative_position(R_Elbow, R_Knee) [up, right, forward]; 测量右肘相对右膝在躯干坐标系中的三维偏移，验证肘是否在膝的正上方且内外、前后对齐。; up分量应为正值（肘在膝上方），right和forward分量应接近零（肘与膝在内外和前后方向对齐）。 | torso_v1的up轴沿躯干方向而非重力方向，'上下'为近似而非严格铅垂对齐。；若躯干本身有倾斜，对齐判断会受躯干姿态影响。 |
| 眼平视前方 | unsupported | 无法从SMPL-24关节位置推断眼睛注视方向。Head关节点非眼睛/前额位置，且无重力参考系判断'平视'，无朝向参考判断'前方'。 | Head关节点不代表眼睛或前额位置，无法推断视线方向。；'平视'需要重力参考以判断视线是否水平，当前工具不支持gravity_frame。；'前方'需要知道身体朝向，当前无全局朝向信息。；注视方向（gaze）从关节位置根本不可推断。 |

[完整计划与坐标/归一化定义](M01_R01/plan.json)

## M01_R02 · 起势

API: completed；格式: False；语义: 未复核

Expecting ',' delimiter: line 88 column 58 (char 2408)

## M01_R03 · 起势

API: completed；格式: True；语义: 未复核

| 原文 | 支持类型 | 几何查询 / 依据 | 局限 |
|---|---|---|---|
| 上体保持正直 | proxy | joint_angle(Pelvis, Spine1, Spine2) [value]; 检测下段脊柱（骨盆-Spine1-Spine2）是否接近伸直，反映上体是否前屈或侧弯; 角度应接近180°，表示该段脊柱无明显弯曲；joint_angle(Spine1, Spine2, Spine3) [value]; 检测上段脊柱（Spine1-Spine2-Spine3）是否接近伸直，反映上体是否前屈或侧弯; 角度应接近180°，表示该段脊柱无明显弯曲 | 无重力参考系，无法判断躯干是否真正竖直于地面；仅检测脊柱弯曲程度，不能区分前屈、后仰或侧弯方向；torso_v1的up轴随躯干方向变化，不能代替世界竖直方向 |
| 两腿屈膝松腰下蹲 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle) [value]; 检测左膝屈曲程度，反映下蹲端点时左腿弯曲状态; 角度应明显小于180°，表示左膝处于屈曲状态；joint_angle(R_Hip, R_Knee, R_Ankle) [value]; 检测右膝屈曲程度，反映下蹲端点时右腿弯曲状态; 角度应明显小于180°，表示右膝处于屈曲状态 | '松腰'为肌肉放松状态，无法从关节位置推断；'下蹲'为动态过程，端点仅能观察屈膝结果，不能验证下蹲动作本身；屈膝角度不能等同于实际承重或重心转移 |
| 同时两掌轻轻下按至腹前 | proxy | relative_position(L_Wrist, Pelvis) [forward, up]; 检测左腕相对骨盆的前方和上下位置，判断是否处于腹前区域; forward分量为正值（腕在身体前方），up分量接近零或略正（约在骨盆/腹部高度）；relative_position(R_Wrist, Pelvis) [forward, up]; 检测右腕相对骨盆的前方和上下位置，判断是否处于腹前区域; forward分量为正值（腕在身体前方），up分量接近零或略正（约在骨盆/腹部高度） | Wrist关节非掌心中心，不能精确代表手掌位置；'同时'为时序要求，端点无法验证；'轻轻'为力度/质感，无法从几何推断；'下按'为动作过程，端点仅能观察最终位置；无法推断手掌朝向（palm_orientation不支持） |
| 两肘松垂 | proxy | relative_position(L_Elbow, L_Shoulder) [up]; 检测左肘相对左肩的上下位置，判断肘部是否自然下垂; up分量为负值，表示左肘位于左肩下方（自然垂落）；relative_position(R_Elbow, R_Shoulder) [up]; 检测右肘相对右肩的上下位置，判断肘部是否自然下垂; up分量为负值，表示右肘位于右肩下方（自然垂落） | '松'为肌肉放松状态，无法从关节位置推断；仅能确认肘低于肩，不能区分主动下垂与被动放松；torso_v1的up轴随躯干倾斜变化，若躯干前倾则'下方'含义偏移 |
| 与两膝上下相对 | proxy | relative_position(L_Elbow, L_Knee) [up, forward]; 检测左肘相对左膝的上下和前后位置，判断是否上下对齐; up分量为正值（肘在膝上方），forward分量接近零（前后方向对齐）；relative_position(R_Elbow, R_Knee) [up, forward]; 检测右肘相对右膝的上下和前后位置，判断是否上下对齐; up分量为正值（肘在膝上方），forward分量接近零（前后方向对齐） | torso_v1的up轴非世界重力方向，若躯干倾斜则'上下'含义偏移；无法验证左右方向的对齐（侧向偏移需额外分析）；该对齐关系依赖于C01（上体正直）成立时torso_v1近似世界坐标 |
| 眼平视前方 | unsupported | 无法从SMPL关节位置推断视线方向。Head关节不代表眼睛/前额位置，无重力参考系无法判断'平视'，无朝向信息无法判断'前方'。 | Head关节位置不等于眼睛位置，无法推断视线方向；无重力参考系，无法判断视线是否水平；无全局朝向信息，无法判断是否面向前方；gaze/视线方向不在SMPL-24关节模型的表达能力范围内 |

[完整计划与坐标/归一化定义](M01_R03/plan.json)

## M02_R01 · 左右野马分鬃

API: completed；格式: False；语义: 未复核

Evidence must name legal query components

## M02_R02 · 左右野马分鬃

API: completed；格式: False；语义: 未复核

Expecting value: line 1 column 1 (char 0)

## M02_R03 · 左右野马分鬃

API: completed；格式: False；语义: 未复核

Duplicate query ID or equivalent query

## M03_R01 · 白鹤亮翅

API: completed；格式: False；语义: 未复核

Expecting value: line 1 column 1 (char 0)

## M03_R02 · 白鹤亮翅

API: completed；格式: False；语义: 未复核

Unknown or duplicate evidence query

## M03_R03 · 白鹤亮翅

API: completed；格式: False；语义: 未复核

Unused queries are not allowed
