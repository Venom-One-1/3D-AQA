# 修复后检查计划对照

本表中的动作解释仍是模型输出，不是经过专家确认的规则。修复只恢复序列化、接口引用和明确坐标含义；不按这些文字执行学生纠错。

## M01_R01 · 起势

[修复后 JSON](M01_R01/plan.json) | [逐项修改日志](M01_R01/repair_log.json)


| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |
|---|---|---|---|---|
| 上体保持正直 | proxy | joint_angle(Spine1, Spine2, Spine3); None; None; ['value']；joint_angle(Spine2, Spine3, Neck); None; None; ['value'] | 测量中段脊柱（Spine1-Spine2-Spine3）的共线程度，角度接近180°表示脊柱中段较直。；角度应接近180°，表示脊柱中段无明显弯曲或侧倾。；测量上段脊柱至颈部（Spine2-Spine3-Neck）的共线程度，角度接近180°表示上体延伸方向一致。；角度应接近180°，表示上段脊柱与颈部保持连贯伸直。 | torso_v1的up轴沿躯干方向而非重力方向，无法判断躯干是否在世界坐标中竖直。；仅能评估脊柱节段间的共线性，不能区分前倾、后仰或侧倾。；'正'（竖直）的判定需要重力参考，当前工具不支持。 |
| 两腿屈膝松腰下蹲 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value']；joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value'] | 测量左膝（L_Hip-L_Knee-L_Ankle）的屈曲程度，角度明显小于180°表示膝关节弯曲。；角度应明显小于180°，表示左膝处于屈曲状态。；测量右膝（R_Hip-R_Knee-R_Ankle）的屈曲程度，角度明显小于180°表示膝关节弯曲。；角度应明显小于180°，表示右膝处于屈曲状态。 | '松腰'为肌肉放松状态，属于unsupported类别，无法从关节位置推断。；'下蹲'的绝对深度需要地面参考（ground_clearance），当前工具不支持。；膝关节屈曲角度不能推断哪条腿承重或实际重心位置。 |
| 同时两掌轻轻下按至腹前 | proxy | relative_position(L_Wrist, Pelvis); torso_length; torso_v1; ['forward', 'up']；relative_position(R_Wrist, Pelvis); torso_length; torso_v1; ['forward', 'up'] | 测量左腕相对骨盆在躯干坐标系中的前后和上下位置，判断左手是否位于腹前区域。；forward分量应为正值（在身体前方），up分量应接近零或略正（约在骨盆至下腹高度）。；测量右腕相对骨盆在躯干坐标系中的前后和上下位置，判断右手是否位于腹前区域。；forward分量应为正值（在身体前方），up分量应接近零或略正（约在骨盆至下腹高度）。 | Wrist关节点非掌心中心，位置存在偏差。；'轻轻'为力度/品质描述，无法从关节几何推断。；'下按'为运动方向/过程，端点仅能反映最终位置。；掌心朝向（palm_orientation）属于unsupported类别。 |
| 两肘松垂 | proxy | relative_position(L_Elbow, L_Shoulder); torso_length; torso_v1; ['up', 'right']；relative_position(R_Elbow, R_Shoulder); torso_length; torso_v1; ['up', 'right'] | 测量左肘相对左肩在躯干坐标系中的上下和内外位置，判断左肘是否自然下垂且未外展。；up分量应为负值（肘低于肩），right分量绝对值应较小（肘未明显外展）。；测量右肘相对右肩在躯干坐标系中的上下和内外位置，判断右肘是否自然下垂且未外展。；up分量应为负值（肘低于肩），right分量绝对值应较小（肘未明显外展）。 | '松'为肌肉放松状态，属于unsupported类别，无法从关节位置区分放松下垂与紧张持握。；仅能判断肘部空间位置，不能推断肘关节的肌张力或柔软度。 |
| 与两膝上下相对 | direct | relative_position(L_Elbow, L_Knee); torso_length; torso_v1; ['up', 'right', 'forward']；relative_position(R_Elbow, R_Knee); torso_length; torso_v1; ['up', 'right', 'forward'] | 测量左肘相对左膝在躯干坐标系中的三维偏移，验证肘是否在膝的正上方且内外、前后对齐。；up分量应为正值（肘在膝上方），right和forward分量应接近零（肘与膝在内外和前后方向对齐）。；测量右肘相对右膝在躯干坐标系中的三维偏移，验证肘是否在膝的正上方且内外、前后对齐。；up分量应为正值（肘在膝上方），right和forward分量应接近零（肘与膝在内外和前后方向对齐）。 | torso_v1的up轴沿躯干方向而非重力方向，'上下'为近似而非严格铅垂对齐。；若躯干本身有倾斜，对齐判断会受躯干姿态影响。 |
| 眼平视前方 | unsupported |  | 无法从SMPL-24关节位置推断眼睛注视方向。Head关节点非眼睛/前额位置，且无重力参考系判断'平视'，无朝向参考判断'前方'。 | Head关节点不代表眼睛或前额位置，无法推断视线方向。；'平视'需要重力参考以判断视线是否水平，当前工具不支持gravity_frame。；'前方'需要知道身体朝向，当前无全局朝向信息。；注视方向（gaze）从关节位置根本不可推断。 |

## M01_R02 · 起势

[修复后 JSON](M01_R02/plan.json) | [逐项修改日志](M01_R02/repair_log.json)

- 待语义复核：仅修复文字序列化；脊柱共线是否能代理上体正直、Hand 与掌心的对应关系未得到语义确认。

| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |
|---|---|---|---|---|
| 上体保持正直 | proxy | joint_angle(Pelvis, Spine1, Spine2); None; None; ['value']；joint_angle(Spine1, Spine2, Spine3); None; None; ['value']；joint_angle(Spine2, Spine3, Neck); None; None; ['value'] | 下段脊柱（骨盆-Spine1-Spine2）的伸直程度，反映躯干是否弯曲；接近180°，表示该段脊柱无明显弯曲；中段脊柱（Spine1-Spine2-Spine3）的伸直程度；接近180°，表示该段脊柱无明显弯曲；上段脊柱至颈部（Spine2-Spine3-Neck）的伸直程度；接近180°，表示上体至颈部无明显前倾或后仰弯曲 | torso_v1的up轴随躯干方向定义，无法判断躯干是否相对重力竖直；无重力参考系，不能区分躯干直立与整体前倾/后倾；仅能检测脊柱弯曲，不能检测整体倾斜 |
| 两腿屈膝松腰下蹲 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value']；joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value'] | 左膝屈曲程度，反映左腿是否弯曲下蹲；明显小于180°，表示左膝处于屈曲状态；右膝屈曲程度，反映右腿是否弯曲下蹲；明显小于180°，表示右膝处于屈曲状态 | 膝关节屈曲角度不能等同于实际下蹲深度（无地面/重力参考）；"松腰"涉及肌肉放松状态，无法从关节几何推断；膝弯曲不能推断哪条腿承重或重心转移 |
| 同时两掌轻轻下按至腹前 | proxy | relative_position(L_Hand, HipMidpoint); torso_length; torso_v1; ['forward', 'up']；relative_position(R_Hand, HipMidpoint); torso_length; torso_v1; ['forward', 'up'] | 左手相对髋部中点的前方和上方分量，判断是否位于腹前；forward分量为正（在身体前方），up分量适中（约在腹部高度，非过高或过低）；右手相对髋部中点的前方和上方分量，判断是否位于腹前；forward分量为正（在身体前方），up分量适中（约在腹部高度，非过高或过低） | Hand关节点非掌心中心，位置存在偏差；"轻轻"为力度描述，无法从几何测量；"下按"为动作方向，端点姿态无法验证运动方向；torso_v1坐标系随躯干倾斜变化，若躯干前倾则forward并非真正的前方 |
| 两肘松垂 | proxy | relative_position(L_Elbow, L_Shoulder); torso_length; torso_v1; ['up']；relative_position(R_Elbow, R_Shoulder); torso_length; torso_v1; ['up'] | 左肘相对左肩的垂直分量，判断肘是否自然下垂；up分量为负值，表示左肘位于左肩下方；右肘相对右肩的垂直分量，判断肘是否自然下垂；up分量为负值，表示右肘位于右肩下方 | "松"为肌肉张力状态，无法从关节位置推断；仅能确认肘低于肩，不能区分主动下垂与被动悬挂；torso_v1的up轴随躯干方向，若躯干倾斜则"下方"含义改变 |
| 与两膝上下相对 | direct | relative_position(L_Elbow, L_Knee); torso_length; torso_v1; ['up', 'forward', 'right']；relative_position(R_Elbow, R_Knee); torso_length; torso_v1; ['up', 'forward', 'right'] | 左肘相对左膝的位置，验证肘在膝正上方且前后左右偏移小；up分量为正（肘在膝上方），forward和right分量接近零（上下对齐）；右肘相对右膝的位置，验证肘在膝正上方且前后左右偏移小；up分量为正（肘在膝上方），forward和right分量接近零（上下对齐） | torso_v1坐标系非重力系，若躯干倾斜则"上下"非真正竖直方向；对齐精度受SMPL关节定位误差影响 |
| 眼平视前方 | unsupported |  | 无法验证：Head关节非眼睛/前额位置，不能推断视线方向；无重力参考系不能判断"平"（水平）；无世界坐标前方参考不能判断"前方"。 | Head关节点位于颅底附近，非眼睛位置，不能推断注视方向；无重力参考系，无法判断视线是否水平；无世界坐标前方定义，无法判断是否朝前；即使头颈与脊柱对齐，也不能等同于平视前方 |

## M01_R03 · 起势

[修复后 JSON](M01_R03/plan.json) | [逐项修改日志](M01_R03/repair_log.json)


| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |
|---|---|---|---|---|
| 上体保持正直 | proxy | joint_angle(Pelvis, Spine1, Spine2); None; None; ['value']；joint_angle(Spine1, Spine2, Spine3); None; None; ['value'] | 检测下段脊柱（骨盆-Spine1-Spine2）是否接近伸直，反映上体是否前屈或侧弯；角度应接近180°，表示该段脊柱无明显弯曲；检测上段脊柱（Spine1-Spine2-Spine3）是否接近伸直，反映上体是否前屈或侧弯；角度应接近180°，表示该段脊柱无明显弯曲 | 无重力参考系，无法判断躯干是否真正竖直于地面；仅检测脊柱弯曲程度，不能区分前屈、后仰或侧弯方向；torso_v1的up轴随躯干方向变化，不能代替世界竖直方向 |
| 两腿屈膝松腰下蹲 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value']；joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value'] | 检测左膝屈曲程度，反映下蹲端点时左腿弯曲状态；角度应明显小于180°，表示左膝处于屈曲状态；检测右膝屈曲程度，反映下蹲端点时右腿弯曲状态；角度应明显小于180°，表示右膝处于屈曲状态 | '松腰'为肌肉放松状态，无法从关节位置推断；'下蹲'为动态过程，端点仅能观察屈膝结果，不能验证下蹲动作本身；屈膝角度不能等同于实际承重或重心转移 |
| 同时两掌轻轻下按至腹前 | proxy | relative_position(L_Wrist, Pelvis); torso_length; torso_v1; ['forward', 'up']；relative_position(R_Wrist, Pelvis); torso_length; torso_v1; ['forward', 'up'] | 检测左腕相对骨盆的前方和上下位置，判断是否处于腹前区域；forward分量为正值（腕在身体前方），up分量接近零或略正（约在骨盆/腹部高度）；检测右腕相对骨盆的前方和上下位置，判断是否处于腹前区域；forward分量为正值（腕在身体前方），up分量接近零或略正（约在骨盆/腹部高度） | Wrist关节非掌心中心，不能精确代表手掌位置；'同时'为时序要求，端点无法验证；'轻轻'为力度/质感，无法从几何推断；'下按'为动作过程，端点仅能观察最终位置；无法推断手掌朝向（palm_orientation不支持） |
| 两肘松垂 | proxy | relative_position(L_Elbow, L_Shoulder); torso_length; torso_v1; ['up']；relative_position(R_Elbow, R_Shoulder); torso_length; torso_v1; ['up'] | 检测左肘相对左肩的上下位置，判断肘部是否自然下垂；up分量为负值，表示左肘位于左肩下方（自然垂落）；检测右肘相对右肩的上下位置，判断肘部是否自然下垂；up分量为负值，表示右肘位于右肩下方（自然垂落） | '松'为肌肉放松状态，无法从关节位置推断；仅能确认肘低于肩，不能区分主动下垂与被动放松；torso_v1的up轴随躯干倾斜变化，若躯干前倾则'下方'含义偏移 |
| 与两膝上下相对 | proxy | relative_position(L_Elbow, L_Knee); torso_length; torso_v1; ['up', 'forward']；relative_position(R_Elbow, R_Knee); torso_length; torso_v1; ['up', 'forward'] | 检测左肘相对左膝的上下和前后位置，判断是否上下对齐；up分量为正值（肘在膝上方），forward分量接近零（前后方向对齐）；检测右肘相对右膝的上下和前后位置，判断是否上下对齐；up分量为正值（肘在膝上方），forward分量接近零（前后方向对齐） | torso_v1的up轴非世界重力方向，若躯干倾斜则'上下'含义偏移；无法验证左右方向的对齐（侧向偏移需额外分析）；该对齐关系依赖于C01（上体正直）成立时torso_v1近似世界坐标 |
| 眼平视前方 | unsupported |  | 无法从SMPL关节位置推断视线方向。Head关节不代表眼睛/前额位置，无重力参考系无法判断'平视'，无朝向信息无法判断'前方'。 | Head关节位置不等于眼睛位置，无法推断视线方向；无重力参考系，无法判断视线是否水平；无全局朝向信息，无法判断是否面向前方；gaze/视线方向不在SMPL-24关节模型的表达能力范围内 |

## M02_R01 · 左右野马分鬃

[修复后 JSON](M02_R01/plan.json) | [逐项修改日志](M02_R01/repair_log.json)

- 待语义复核：只统一坐标分量表达，不确认手高、步幅、自然蹬直等动作解释或代理指标的有效性。

| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |
|---|---|---|---|---|
| 上体微向左转 | proxy | vector_angle(L_Hip, R_Hip, L_Shoulder, R_Shoulder); None; None; ['value'] | 肩轴与髋轴之间的夹角可反映躯干是否存在相对于骨盆的水平旋转；夹角应略小于180度（即肩轴与髋轴不完全平行），表明存在一定程度的扭转 | 无法区分主动转体与步法带动的被动旋转；vector_angle为无符号角度，无法判断旋转方向是向左还是向右；微转幅度小，测量精度可能不足以可靠检测 |
| 左脚向左前方迈出 | proxy | relative_position(L_Ankle, HipMidpoint); average_leg_length; torso_v1; ['right', 'forward'] | 左踝相对髋中点在torso_v1坐标系中的right和forward分量可描述左前方的位置关系；向左对应right为负。；right分量为负值（左脚在身体左侧），forward分量为正值（左脚在身体前方） | Ankle不是脚底接触点，无法精确反映脚的实际落点；torso_v1坐标系随躯干方向变化，若躯干已旋转则左前方含义有偏差；迈出是动态过程，终点位置仅为结果近似 |
| 脚跟先着地 | unsupported |  | 脚跟先着地是着地瞬间的时序事件，属于过渡过程，终点静态姿态无法验证着地顺序 | 着地顺序是时间序列事件，非终点几何属性；SMPL关节点无法提供地面接触信息；Ankle不代表脚跟或脚底接触点 |
| 随即脚掌踏实 | unsupported |  | 脚掌踏实涉及脚与地面的接触状态和压力分布，终点几何无法验证 | 踏实是接触力和压力分布的概念，非关节几何可测；SMPL不提供地面接触或压力信息；无法从关节位置推断实际承重状态 |
| 右腿自然蹬直 | direct | joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value'] | 右髋-右膝-右踝的夹角直接反映右腿伸直程度；夹角应接近180度，表明右腿基本伸直 | 自然蹬直中的'自然'暗示非完全锁死，但无法从几何区分自然伸直与过度伸直；joint_angle为无符号夹角，不区分过伸与屈曲方向 |
| 成左弓步 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value']；joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value']；distance(L_Ankle, R_Ankle); average_leg_length; None; ['value'] | 左膝弯曲是弓步前腿的核心特征；左膝夹角应明显小于180度，表明前腿弯曲；右腿伸直是弓步后腿的核心特征；右膝夹角应接近180度，表明后腿蹬直；两踝之间的距离反映弓步的前后跨度；两踝间距应较大，反映弓步的前后开立 | 弓步还要求重心分配和胯部朝向，这些无法从关节几何完全验证；无法验证重心是否偏向前腿；踝间距不能区分前后方向与左右方向的分量 |
| 两手随转体分别向左上、右下分开 | proxy | relative_position(L_Wrist, ShoulderMidpoint); torso_length; torso_v1; ['right', 'up']；relative_position(R_Wrist, HipMidpoint); torso_length; torso_v1; ['right', 'up']；distance(L_Wrist, R_Wrist); torso_length; None; ['value'] | 左腕相对肩中点的位置可反映左手是否向左上方分开；right分量为负值（左手在左侧），up分量为正值（左手在肩上方或附近）；右腕相对髋中点的位置可反映右手是否落于右下方；right分量为正值（右手在右侧），up分量为负值或接近零（右手在髋旁或略低）；两腕间距反映两手分开的幅度；两腕间距应较大，反映两手已充分分开 | 随转体分开是动态过程，终点仅能验证最终位置；Wrist不代表手掌中心，无法精确反映手的空间方位；无法验证分开路径是否随转体完成 |
| 左手高与眼平 | proxy | relative_position(L_Wrist, Head); torso_length; torso_v1; ['up'] | 左腕相对Head的up分量可反映左手是否接近头部（眼）高度；up分量应接近零或略为负值，表明左腕高度与头部（近似眼高）接近 | Head关节点不等于眼睛位置，存在系统性偏差；Wrist不等于手掌中心，实际手掌高度可能有偏移；眼平高度因人而异，Head作为参考有近似误差 |
| 掌心斜向上 | unsupported |  | 掌心斜向上涉及手掌朝向，SMPL-24关节点无法提供手掌法线方向 | palm_orientation在unsupported列表中明确标注不支持；Wrist和Hand关节点无法推断掌心朝向；需要手掌法线或前臂旋前/旋后角度，当前工具无法提供 |
| 右手落于右胯旁 | proxy | relative_position(R_Wrist, HipMidpoint); torso_length; torso_v1; ['right', 'up', 'forward'] | 右腕相对髋中点的位置可反映右手是否在右胯旁；right分量为正值（右手在右侧），up分量接近零或略负（与髋同高或略低），forward分量接近零（在胯旁而非前方） | Wrist不等于手掌，实际手的位置可能有偏移；胯旁是模糊区域，精确位置难以定义；无法验证手是否自然放松 |
| 掌心向下 | unsupported |  | 掌心向下涉及手掌朝向，SMPL-24关节点无法提供手掌法线方向 | palm_orientation在unsupported列表中明确标注不支持；无法从关节位置推断掌心朝向 |
| 指尖向前 | unsupported |  | 指尖向前涉及手指朝向，SMPL-24关节点无法提供指尖方向 | SMPL-24无手指关节点，无法推断指尖方向；Hand关节点位置不能代表指尖朝向；需要手部精细模型才能验证 |
| 眼看左手 | proxy | vector_angle(Neck, Head, Neck, L_Wrist); None; None; ['value'] | Neck-Head向量与Neck-L_Wrist向量的夹角可近似反映头部朝向是否指向左手方向；夹角应较小，表明头部朝向与左手方向大致一致 | Head关节点不等于眼睛位置，无法确定实际视线方向；头部朝向不等于注视方向，眼球转动无法从关节位置推断；Neck-Head向量仅近似反映头部朝向，精度有限 |

## M02_R02 · 左右野马分鬃

[修复后 JSON](M02_R02/plan.json) | [逐项修改日志](M02_R02/repair_log.json)


| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |
|---|---|---|---|---|
| 上体微向左转 | unsupported |  | 「上体微向左转」描述的是一个旋转过程。torso_v1坐标系本身随躯干旋转，无法在端点测量躯干相对于世界或下肢的绝对转向。没有可用的世界参考系或重力系来量化水平面内的躯干旋转角度。 | torso_v1的left轴由髋和肩混合定义，无法分离躯干相对于骨盆的水平旋转；无重力参考系或全局朝向数据，无法测量绝对转向；「微转」是过程描述，端点几何无法还原旋转量 |
| 左脚向左前方迈出 | proxy | relative_position(L_Ankle, R_Ankle); average_leg_length; torso_v1; ['forward', 'right'] | 观察端点时左踝相对右踝在躯干坐标系中的前后和左右偏移，间接反映左脚向左前方迈出的结果；forward分量为正值（左脚在右脚前方），right分量为负值（左脚在右脚左侧，即身体左方） | 「迈出」是动态过程，端点仅反映最终位置而非运动轨迹；无法确认迈出方向是否严格为「左前方」而非其他方向；torso_v1坐标系随躯干变化，若躯干已旋转则前后左右含义随之改变 |
| 脚跟先着地 | unsupported |  | 「脚跟先着地」描述的是着地顺序（脚跟先于脚掌），这是一个时间序列上的接触过程，端点静态几何无法测量着地顺序或接触点。 | SMPL关节点不包含足底接触信息；着地顺序是时间过程，端点无法还原；Ankle关节点不代表脚跟或脚掌接触点 |
| 随即脚掌踏实 | unsupported |  | 「脚掌踏实」描述的是脚掌与地面的接触状态和承重过程，端点几何无法测量地面接触或实际承重。 | 无地面接触力或接触面积数据；「踏实」涉及承重和肌肉控制，超出关节几何测量范围；actual_weight_bearing在工具中标记为不支持 |
| 右腿自然蹬直 | direct | joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value'] | 测量右腿在端点时的膝关节夹角，蹬直时该角度应接近180度（完全伸直）；右膝夹角接近180度，表示右腿基本伸直 | 「自然」包含肌肉放松程度，几何无法测量；「蹬直」中的「蹬」暗示主动发力过程，端点仅能观察伸直结果；joint_angle是无符号几何夹角，不等同于临床膝关节伸展角度 |
| 成左弓步 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value']；joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value']；relative_position(L_Ankle, R_Ankle); average_leg_length; torso_v1; ['forward']；distance(L_Ankle, R_Ankle); average_leg_length; None; ['value'] | 测量左膝夹角，弓步时前腿（左腿）应明显弯曲，角度显著小于180度；左膝夹角明显小于180度，表示左腿屈膝；测量右膝夹角，弓步时后腿（右腿）应接近伸直；右膝夹角接近180度，表示右腿伸直；测量左踝相对右踝的前后距离，弓步时前脚应在后脚前方；forward分量为正值，表示左脚在右脚前方；测量两踝之间的总距离，弓步时两脚间应有较大的前后跨度；两踝距离较大，反映弓步的步幅 | 弓步要求前腿承重、后腿蹬地，但actual_weight_bearing不可测量；无法验证重心是否落在前腿；torso_v1中的前后方向随躯干朝向变化，若躯干旋转则方向含义改变；无法验证步幅是否符合该动作的标准比例 |
| 两手随转体分别向左上、右下分开 | proxy | relative_position(L_Wrist, ShoulderMidpoint); torso_length; torso_v1; ['right', 'up']；relative_position(R_Wrist, ShoulderMidpoint); torso_length; torso_v1; ['right', 'up']；relative_position(L_Wrist, R_Wrist); torso_length; torso_v1; ['right', 'up'] | 测量左腕相对肩中点的位置，验证左手是否在身体左上方；right分量为负值（左手在身体左侧），up分量为正值（左手在肩上方）；测量右腕相对肩中点的位置，验证右手是否在身体右下方；right分量为正值（右手在身体右侧），up分量为负值（右手在肩下方）；测量两腕之间的相对位置差，验证两手是否形成左上右下的对角分离；right分量为负值（左腕在右腕左侧），up分量为正值（左腕在右腕上方） | 「随转体」是过程描述，端点无法验证分手是否由转体带动；torso_v1坐标系随躯干旋转，若躯干已左转则左右含义随之改变；无法验证分手的轨迹和时序 |
| 左手高与眼平 | proxy | relative_position(L_Wrist, Head); torso_length; torso_v1; ['up'] | 测量左腕相对头部的上下偏移，若左手与眼平则左腕应略低于Head（因Head在头顶，眼在Head下方）；up分量为较小的负值或接近零，表示左腕高度接近头部（眼睛略低于Head点） | Head关节点不代表眼睛位置，实际眼睛在Head下方，因此存在系统性偏差；Wrist关节点不代表手掌中心，手掌展开时指尖可能高于腕关节；无法精确量化「与眼平」的容差范围 |
| 掌心斜向上 | unsupported |  | 「掌心斜向上」描述的是手掌朝向，属于掌面方向信息。SMPL-24关节点不包含手掌法向量或掌面朝向数据，palm_orientation在工具中标记为不支持。 | palm_orientation在工具不支持列表中；SMPL-24无手掌法向量或手指展开信息；无法从腕关节位置推断掌心朝向 |
| 右手落于右胯旁 | proxy | relative_position(R_Wrist, R_Hip); torso_length; torso_v1; ['right', 'up', 'forward'] | 测量右腕相对右髋的位置偏移，验证右手是否在右髋附近；各分量绝对值均较小，表示右腕在右髋附近；right分量可能为小正值（略偏外侧），up分量接近零或略负 | 「胯旁」是模糊区域，无精确定义；Wrist关节点不代表手掌中心，实际手掌位置可能有偏移；无法区分「落于」的过程与端点位置 |
| 掌心向下 | unsupported |  | 「掌心向下」描述的是手掌朝向，属于掌面方向信息。SMPL-24关节点不包含手掌法向量数据，palm_orientation在工具中标记为不支持。 | palm_orientation在工具不支持列表中；SMPL-24无手掌法向量信息；无法从腕关节位置推断掌心朝向 |
| 指尖向前 | unsupported |  | 「指尖向前」描述的是手指指向，SMPL-24关节点中无指尖或手指方向数据，无法测量。 | SMPL-24无指尖关节点；无法从腕关节位置推断手指指向；手指方向需要更精细的手部模型 |
| 眼看左手 | unsupported |  | 「眼看左手」描述的是视线方向（gaze），SMPL-24关节点不包含眼睛位置或视线方向数据。Head关节点位于头顶附近，不能代表眼睛位置或视线方向。 | SMPL-24无眼睛或视线方向数据；Head关节点不代表眼睛位置，不能推断注视方向；即使头部朝向左手方向，也不能确认视线是否看向左手 |

## M02_R03 · 左右野马分鬃

[修复后 JSON](M02_R03/plan.json) | [逐项修改日志](M02_R03/repair_log.json)


| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |
|---|---|---|---|---|
| 上体微向左转 | proxy | relative_position(L_Wrist, R_Wrist); torso_length; torso_v1; ['right']；relative_position(ShoulderMidpoint, HipMidpoint); torso_length; torso_v1; ['right'] | 左手相对右手在躯干坐标系左右方向的分量可间接反映转体后两臂的左右分布；左手相对右手在躯干坐标系中偏左（right分量为负值）；肩中点相对髋中点在躯干坐标系中的左右偏移可间接反映上体是否有轻微侧转；肩中点相对髋中点有轻微左右偏移，但幅度很小 | torso_v1坐标系随躯干旋转，无法测量躯干相对世界坐标的绝对转向；微转幅度极小，SMPL精度可能不足以区分；转体是动态过程，端点仅能反映结果姿态的间接线索 |
| 左脚向左前方迈出 | proxy | relative_position(L_Ankle, R_Ankle); average_leg_length; torso_v1; ['forward', 'right']；relative_position(L_Ankle, HipMidpoint); average_leg_length; torso_v1; ['forward', 'right'] | 左踝相对右踝在躯干坐标系中的前后和左右分量反映迈步后左脚的位置；左踝相对右踝在forward方向为正（前方），right方向为负（左侧）；左踝相对髋中点的位置反映左脚在身体前方的程度；左踝在髋中点前方偏左 | 迈出是动态过程，端点仅反映最终站位；无法确认迈步方向是否为左前方而非其他方向；torso_v1坐标系随躯干方向变化，非世界坐标 |
| 脚跟先着地 | unsupported |  | 脚跟先着地是着地顺序的时间过程描述，端点静态姿态无法区分着地顺序 | 着地顺序是时间序列信息，静态端点无法验证；SMPL关节点无法区分脚跟与脚掌的接触时序；无地面接触力或地面高度信息 |
| 随即脚掌踏实 | unsupported |  | 脚掌踏实涉及脚与地面的接触状态和压力分布，端点静态姿态无法验证 | 踏实涉及地面接触力和压力分布，SMPL关节位置无法测量；无地面平面或接触力信息；Ankle关节点不代表脚掌接触面 |
| 右腿自然蹬直 | direct | joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value'] | 右腿髋-膝-踝关节角反映右腿伸直程度；右腿膝关节角度接近180度，表示腿部基本伸直 | 自然蹬直中的'自然'涉及肌肉放松程度，无法从关节角度判断；关节角度为无符号几何角度，非临床屈曲角；无法区分主动蹬直与被动伸直 |
| 成左弓步 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value']；joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value']；relative_position(L_Ankle, R_Ankle); average_leg_length; torso_v1; ['forward'] | 左腿髋-膝-踝关节角反映前腿弯曲程度，弓步前腿应明显弯曲；左腿膝关节角度明显小于180度，表示前腿弯曲；右腿髋-膝-踝关节角反映后腿伸直程度，弓步后腿应接近伸直；右腿膝关节角度接近180度；左右踝前后距离反映弓步的前后跨度；左踝在右踝前方，前后分量较大 | 弓步还涉及重心分配和髋部高度，仅凭关节角和踝间距不能完全确认；无法验证重心是否偏前腿；torso_v1坐标系非世界坐标，无法测量绝对高度变化 |
| 两手随转体分别向左上、右下分开 | proxy | relative_position(L_Wrist, R_Wrist); torso_length; torso_v1; ['right', 'up']；distance(L_Wrist, R_Wrist); torso_length; None; ['value'] | 左手相对右手在躯干坐标系中的左右和上下分量反映两臂分开的方向和幅度；左手相对右手偏左（right分量为负）且偏上（up分量为正）；左右腕间距反映两臂分开的总体幅度；左右腕间距较大，表示两臂已充分分开 | 随转体分开是动态过程，端点仅反映最终位置；无法确认分开路径是否随转体完成；掌心朝向无法从腕关节位置推断 |
| 左手高与眼平 | proxy | relative_position(L_Wrist, Head); torso_length; torso_v1; ['up'] | 左腕相对头部在躯干坐标系中的上下分量反映左手与头部的相对高度；左腕与头部在up方向上接近等高，up分量接近零 | Head关节点位于头部中心附近，非眼睛或前额位置；Wrist关节点非手掌中心，实际手掌高度可能有偏差；与眼平是相对世界水平面的描述，torso_v1坐标系随躯干倾斜而变化 |
| 掌心斜向上 | unsupported |  | 掌心斜向上涉及手掌朝向，SMPL-24关节点无法测量掌心方向 | palm_orientation在工具不支持列表中；Wrist和Hand关节点无法确定掌心法线方向；需要手部详细模型或传感器才能测量掌心朝向 |
| 右手落于右胯旁 | direct | relative_position(R_Wrist, R_Hip); torso_length; torso_v1; ['right', 'up', 'forward']；relative_position(R_Wrist, HipMidpoint); torso_length; torso_v1; ['right', 'up'] | 右腕相对右髋在躯干坐标系中的三维分量反映右手是否位于右胯旁；右腕与右髋在right、up、forward方向上均接近，各分量绝对值较小；右腕相对髋中点的位置作为补充，确认右手在髋部附近偏右；右腕相对髋中点偏右（right分量为正），上下接近 | 胯旁是模糊区域描述，精确位置因人而异；Wrist关节点非手掌中心，实际手掌位置可能有偏差 |
| 掌心向下 | unsupported |  | 掌心向下涉及手掌朝向，SMPL-24关节点无法测量掌心方向 | palm_orientation在工具不支持列表中；Wrist和Hand关节点无法确定掌心法线方向；需要手部详细模型或传感器才能测量掌心朝向 |
| 指尖向前 | unsupported |  | 指尖向前涉及手指指向，SMPL-24关节点无手指末端信息 | SMPL-24模型无指尖关节点；无法从腕或手掌关节推断指尖方向；需要手部详细模型才能测量指尖朝向 |
| 眼看左手 | proxy | relative_position(Head, L_Wrist); torso_length; torso_v1; ['right', 'up', 'forward'] | 头部相对左腕在躯干坐标系中的位置关系可间接反映头部是否朝向左手方向；头部相对左腕在forward方向为正（头在左手后方），right分量接近零或略正，表示头部大致面向左手方向 | Head关节点非眼睛位置，无法确定实际视线方向； gaze方向无法从关节位置推断；头部朝向与视线方向可能不一致 |

## M03_R01 · 白鹤亮翅

[修复后 JSON](M03_R01/plan.json) | [逐项修改日志](M03_R01/repair_log.json)

- 待语义复核：C03 用骨盆偏移、膝角推断重心或承重，不能视为已证实的受力结论；本轮仅执行几何测量。
- 待语义复核：C15 对微屈引入了大于90度的范围，未获得教师数据或专家验证，不应执行为质量阈值。
- 待语义复核：C15 用手腕相对肩的高度解释肘部下坠，证据并未直接测量肘部位置。
- 待语义复核：Wrist/Head/Hand 等代理点的支持等级和要领关联仍待语义复核。

| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |
|---|---|---|---|---|
| 右脚跟进半步 | unsupported |  | 右脚跟进半步是过渡动作，描述的是移动过程而非定势静态几何关系。端点无法验证脚是否'跟进'了半步。 | 过渡过程，端点无法验证移动距离和方向；无地面参考系，无法测量绝对位移 |
| 上体后坐 | unsupported |  | 上体后坐是过渡过程描述，表示重心后移的动态过程。端点几何无法验证'后坐'这一动作方向。 | 后坐是过程描述，非端点静态属性；无重力参考系，无法判断躯干相对地面的后倾 |
| 重心移至右腿 | proxy | relative_position(Pelvis, AnkleMidpoint); average_leg_length; torso_v1; ['right', 'forward']；joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value']；joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value'] | 骨盆相对双踝中点的水平偏移可间接反映重心偏向哪一侧；骨盆在right分量上偏向右脚方向（即right分量为正值，因为骨盆偏向右脚意味着骨盆在踝中点的右侧），表明重心偏右；右膝弯曲程度可间接反映右腿承重屈膝状态；右膝关节角小于180度，呈明显屈曲状态，与承重腿一致；左膝相对伸直可间接反映左腿为虚步非承重；左膝关节角接近或大于右膝关节角，表明左腿相对伸展 | 无法直接测量实际负重分配；骨盆偏移是间接推断，不等于重心精确位置；膝关节弯曲不等于承重，仅为间接线索 |
| 上体先向右转 | unsupported |  | 上体先向右转是过渡过程中的旋转方向，端点无法验证旋转的先后顺序。 | 旋转顺序是过程属性，端点无法验证；无全局朝向参考，无法测量绝对转向 |
| 再微向左转 | unsupported |  | 再微向左转是过渡过程中的旋转方向，端点无法验证旋转的先后顺序和最终朝向的绝对角度。 | 旋转顺序是过程属性，端点无法验证；无全局朝向参考，无法测量绝对转向角度 |
| 左脚稍向前移 | unsupported |  | 左脚稍向前移是过渡动作，描述移动过程。端点无法验证脚是否移动了以及移动了多少。 | 移动过程无法从端点验证；无地面参考系和初始位置信息 |
| 脚尖点地成左虚步 | proxy | relative_position(L_Ankle, R_Ankle); average_leg_length; torso_v1; ['forward']；joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value']；relative_position(L_Foot, L_Ankle); average_leg_length; torso_v1; ['up'] | 左踝相对右踝的前方偏移可反映虚步中前脚的位置；左踝在forward分量上为正值，表明左脚位于右脚前方；左膝相对伸直是虚步前腿的特征之一；左膝关节角较大，接近伸直状态；左脚相对左踝的垂直分量可间接提示脚尖点地姿态（脚尖低于踝关节）；up分量为负值或接近零，提示脚尖向下或接近地面 | 无法验证实际地面接触；Foot关节不等于脚尖精确位置；无法区分脚尖点地与全脚着地；虚步的'虚'即非承重无法直接测量 |
| 两手随转体向右上、左下分开 | proxy | relative_position(R_Wrist, L_Wrist); torso_length; torso_v1; ['right', 'up'] | 右手腕相对左手腕的偏移可验证两手在右上、左下方向分开的端点结果；right分量为正值（右手在左手右侧），up分量为正值（右手在左手上方） | '随转体'是过程描述，端点无法验证；仅能验证分开的端点结果，不能验证分开路径 |
| 定势时 | unsupported |  | '定势时'是时间标记，表示后续描述适用于定势时刻，本身不包含可测量的几何要求。 | 时间标记，非几何要求 |
| 右手位于头部右前方 | direct | relative_position(R_Wrist, Head); torso_length; torso_v1; ['right', 'forward'] | 右手腕相对头部的右侧和前方偏移直接描述'头部右前方'的位置关系；right分量为正值（右手在头部右侧），forward分量为正值（右手在头部前方） | Wrist关节不等于手掌中心，但作为手臂末端位置是合理近似 |
| 掌心中心大致与额头同高或略高 | proxy | relative_position(R_Wrist, Head); torso_length; torso_v1; ['up'] | 右手腕相对头部的垂直分量可近似反映手掌与头部的相对高度；up分量接近零或为小正值，表明手腕与头部大致同高或略高 | Wrist不是掌心中心，存在解剖偏差；Head不是额头，存在解剖偏差；两个近似叠加，精度有限 |
| 指尖可自然高于头顶 | proxy | relative_position(R_Hand, R_Wrist); torso_length; torso_v1; ['up']；relative_position(R_Wrist, Head); torso_length; torso_v1; ['up'] | R_Hand相对R_Wrist的垂直分量可辅助判断手部远端是否高于手腕；up分量为正值，表明手部远端高于手腕，间接支持指尖高于头顶；结合手腕与头部的高度关系，若手腕已接近头部高度且手部远端更高，则指尖可能高于头顶；up分量接近零或正值，配合手部远端向上延伸 | R_Hand不是指尖精确位置；Head不是头顶最高点；'可'字表示允许而非必须，端点无法判断是否'自然' |
| 掌心斜向左后方 | unsupported |  | 掌心斜向左后方是手掌朝向要求，SMPL-24关节点无法推断掌心方向。 | palm_orientation在工具不支持列表中；关节位置无法推断手掌法线方向 |
| 右肩松沉 | proxy | relative_position(R_Shoulder, Head); torso_length; torso_v1; ['up'] | 右肩相对头部的垂直距离可间接反映肩部是否下沉（未耸肩）；up分量为明显负值，表明右肩明显低于头部，肩部处于下沉位置 | '松'是肌肉放松状态，无法从关节位置测量；仅能间接推断肩部位置偏低，不能证明'松沉'的质感 |
| 肘部自然下坠并保持微屈 | direct | joint_angle(R_Shoulder, R_Elbow, R_Wrist); None; None; ['value']；relative_position(R_Wrist, R_Shoulder); torso_length; torso_v1; ['up'] | 右肘关节角直接反映肘部屈曲程度；关节角小于180度但大于90度，表明肘部微屈而非完全伸直或大幅弯曲；右手腕相对右肩的垂直位置可辅助判断肘部是否处于下坠状态（手腕高于肩则肘部可能未下坠）；up分量为正值但不过大，表明手腕高于肩部但肘部仍低于手腕，符合肘部下坠微屈 | '自然下坠'的质感无法精确量化；关节角是无符号几何角度，非临床屈曲角 |
| 右臂呈舒展的弧形 | proxy | joint_angle(R_Shoulder, R_Elbow, R_Wrist); None; None; ['value']；joint_angle(R_Collar, R_Shoulder, R_Elbow); None; None; ['value'] | 肘部微屈是弧形臂的必要条件；关节角明显小于180度，表明手臂非完全伸直；肩-肘角度可辅助判断上臂与躯干的关系，配合肘角共同描述弧形；关节角小于180度，表明上臂相对躯干有展开角度，配合肘部微屈形成弧形 | '舒展'是主观质感描述，无法精确量化；两个关节角只能近似描述弧形，无法验证曲线的连续性 |
| 左手落于左胯前 | direct | relative_position(L_Wrist, L_Hip); torso_length; torso_v1; ['forward', 'up', 'right'] | 左手腕相对左髋的位置直接描述'左胯前'的空间关系；forward分量为正值（左手在左髋前方），up分量接近零或略负（与胯部大致同高或略低），right分量接近零（在左髋正前方而非偏右） | Wrist不是手掌中心，但作为手臂末端位置是合理近似；'胯'的精确解剖位置与Hip关节有偏差 |
| 掌心向下 | unsupported |  | 掌心向下是手掌朝向要求，SMPL-24关节点无法推断掌心方向。 | palm_orientation在工具不支持列表中；关节位置无法推断手掌法线方向 |
| 指尖向前 | unsupported |  | 指尖向前是手指朝向要求，SMPL-24关节点无法推断指尖方向。 | 手指方向无法从关节位置推断；L_Hand可近似手部远端方向但精度不足以判断'指尖向前' |
| 眼平视前方 | unsupported |  | 眼平视前方涉及视线方向，Head关节不等于眼睛位置，且无全局参考系判断'前方'。 | Head不是眼睛/额头位置；无全局参考系，无法判断绝对'前方'；视线方向无法从关节位置推断 |

## M03_R02 · 白鹤亮翅

[修复后 JSON](M03_R02/plan.json) | [逐项修改日志](M03_R02/repair_log.json)

- 待语义复核：修复引用并移除未使用查询，不代表肩部高度可以可靠判定松沉，也不补写被遗漏的检查。

| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |
|---|---|---|---|---|
| 右脚跟进半步 | unsupported |  | 「跟进半步」是一个过渡动作，描述右脚从某处移动到另一处的过程。定态无法验证移动过程本身。虽然终点时右脚位置可测，但无法判断它是否经历了「跟进」这一动作。 | 过渡动作，定态无法验证移动过程；无法确定起点位置，因此无法判断是否移动了半步 |
| 上体后坐 | unsupported |  | 「上体后坐」描述的是一个过渡过程（上体从前倾或直立变为后坐），定态只能观察到最终躯干姿态，无法验证是否经历了后坐过程。 | 后坐是过渡动作，定态无法验证过程；无法从终点推断起始姿态 |
| 重心移至右腿 | unsupported |  | 「重心移至右腿」涉及实际负重分配，SMPL关节位置无法直接测量重心或负重比例。虽然可从腿部弯曲角度间接推测，但这属于不可靠推断。 | actual_weight_bearing属于不支持的测量；关节角度不能直接推断负重分配；无法从几何确定重心位置 |
| 上体先向右转 | unsupported |  | 「上体先向右转」描述的是过渡过程中的旋转方向和顺序，定态无法验证旋转方向或先后顺序。 | 过渡动作，定态无法验证旋转方向；无法验证动作顺序（先右转再左转） |
| 再微向左转 | unsupported |  | 「再微向左转」描述的是过渡过程中的旋转方向和顺序，定态无法验证。 | 过渡动作，定态无法验证旋转方向；无法验证动作顺序 |
| 左脚稍向前移 | unsupported |  | 「左脚稍向前移」描述的是过渡动作（左脚从某处向前移动），定态无法验证移动过程。 | 过渡动作，定态无法验证移动过程；无法确定起始位置 |
| 脚尖点地成左虚步 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value']；relative_position(L_Ankle, R_Ankle); average_leg_length; torso_v1; ['forward'] | 虚步时左腿通常较为伸直或微屈，膝关节角度可反映左腿的弯曲程度；左膝关节角度应接近伸直（接近180度）或仅微屈，明显大于右膝弯曲角度；虚步时左脚应在右脚前方，forward分量可反映前后相对位置；forward分量应为正值，表示左脚在右脚前方 | 无法验证脚尖是否真正点地接触地面；无法区分脚尖点地与全脚掌着地；Ankle不是脚尖位置，存在近似误差 |
| 两手随转体向右上、左下分开 | proxy | relative_position(R_Wrist, Head); torso_length; torso_v1; ['up', 'right']；relative_position(L_Wrist, Pelvis); torso_length; torso_v1; ['up', 'right'] | 右手应位于头部附近偏上偏右，验证右上分开的结果；up分量应为正值或接近零（右手在头部高度或略高），right分量应为正值（右手在头部右侧）；左手应位于身体下方偏左，验证左下分开的结果；up分量应为负值（左手在骨盆下方或同高），right分量应为负值（左手在身体左侧） | 「随转体」是过渡过程，定态无法验证；只能验证分开的结果位置，不能验证分开的路径 |
| 定势时 | unsupported |  | 「定势时」是时间标记，表示后续描述适用于定态。它本身不包含可测量的身体几何要求。 | 纯时间标记，无独立可测内容 |
| 右手位于头部右前方 | direct | relative_position(R_Wrist, Head); torso_length; torso_v1; ['right', 'forward', 'up'] | 直接测量右手腕相对于头部在躯干坐标系中的三维位置关系；right分量为正值（右手在头部右侧），forward分量为正值（右手在头部前方），up分量接近零或略正（与头部同高或略高） | Wrist是腕关节而非手掌中心，存在一定近似；Head是头顶中心而非额头，高度参考有偏差 |
| 掌心中心大致与额头同高或略高 | proxy | relative_position(R_Wrist, Head); torso_length; torso_v1; ['up'] | 用腕关节与头顶的高度差近似掌心与额头的高度关系；up分量应接近零或略为负值（腕关节约在头顶高度或略低，因为掌心在腕关节远端，而额头低于头顶） | Wrist不是掌心中心，存在位置偏差；Head不是额头，头顶高于额头；两个近似叠加，精度有限 |
| 指尖可自然高于头顶 | proxy | relative_position(R_Hand, R_Wrist); torso_length; torso_v1; ['up']；relative_position(R_Wrist, Head); torso_length; torso_v1; ['up'] | R_Hand相对于R_Wrist的up分量可反映手掌延伸方向，间接推断指尖是否高于头顶；up分量应为正值，表示手掌向上传展，指尖方向朝上；腕关节与头顶的高度关系，结合手掌方向可推断指尖高度；up分量接近零或略负，配合手掌向上延伸，指尖可高于头顶 | R_Hand不是指尖，是手掌代表点；无法精确确定指尖位置；「可」字表示允许而非必须，定态无法区分自然与否 |
| 掌心斜向左后方 | unsupported |  | 「掌心斜向左后方」涉及掌心朝向，属于palm_orientation，在工具不支持列表中。SMPL关节位置无法推断手掌法线方向。 | palm_orientation属于不支持的测量；无法从关节位置推断掌心朝向 |
| 右肩松沉 | proxy | relative_position(R_Shoulder, Pelvis); torso_length; torso_v1; ['up']；relative_position(R_Shoulder, R_Hip); torso_length; torso_v1; ['up'] | 右肩相对于骨盆在躯干坐标系中的高度，可反映肩部是否下沉（非耸肩）；up分量应为正值但不过大，表示右肩在正常解剖位置而非耸起；右肩相对于右髋的高度，可辅助判断肩部是否下沉；up分量为正值，反映肩髋之间的正常距离 | 「松」涉及肌肉放松，属于muscle_relaxation，不可测；只能测肩部位置高度，不能区分主动下沉与被动位置；无法判断是否「松」 |
| 肘部自然下坠并保持微屈 | proxy | joint_angle(R_Shoulder, R_Elbow, R_Wrist); None; None; ['value']；relative_position(R_Wrist, R_Elbow); torso_length; torso_v1; ['up'] | 右肘关节角度直接反映肘部弯曲程度，微屈对应接近但小于180度的角度；角度应明显小于180度（非完全伸直）但大于90度（非大幅弯曲），体现微屈状态；腕关节相对于肘关节的高度，若肘部下坠则腕应高于肘或接近同高；up分量应为正值或接近零，表示腕在肘上方或同高，反映肘部低于腕的下坠状态 | 「自然」涉及肌肉状态，不可测；肘关节角度是无符号几何角度，非临床屈曲角；无法区分主动下坠与被动位置 |
| 右臂呈舒展的弧形 | proxy | joint_angle(R_Shoulder, R_Elbow, R_Wrist); None; None; ['value']；relative_position(R_Wrist, R_Shoulder); torso_length; torso_v1; ['right', 'up', 'forward'] | 肘关节角度反映臂部弯曲程度，弧形要求非完全伸直；角度应小于180度但大于约120度，体现舒展的微弯弧形；腕相对于肩的三维位置可反映臂部展开的方向和幅度；right和up分量均为正值，表示右臂向右上方舒展展开 | 「弧形」是连续曲线特征，仅凭三个关节点无法完整描述曲线形状；「舒展」含主观审美判断，几何只能近似；无法测量臂部肌肉张力或柔软度 |
| 左手落于左胯前 | direct | relative_position(L_Wrist, L_Hip); torso_length; torso_v1; ['right', 'up', 'forward']；relative_position(L_Wrist, Pelvis); torso_length; torso_v1; ['right', 'up', 'forward'] | 直接测量左手腕相对于左髋的位置关系；forward分量为正值（左手在左髋前方），up分量接近零或略负（左手在髋部高度或略低），right分量接近零或略正（左手在左髋内侧偏中）；补充验证左手相对于骨盆的位置，确认在左胯前方；forward分量为正值，right分量为负值（偏左），up分量接近零或略负 | Wrist是腕关节而非手掌中心，「落于」的精确位置有偏差；「胯前」的「前」在躯干坐标系中可能因躯干倾斜而有偏差 |
| 掌心向下 | unsupported |  | 「掌心向下」涉及掌心朝向，属于palm_orientation，在工具不支持列表中。 | palm_orientation属于不支持的测量；无法从关节位置推断掌心朝向 |
| 指尖向前 | unsupported |  | 「指尖向前」涉及指尖方向，SMPL无指尖关节点，且手指方向无法从腕或手掌点可靠推断。 | 无指尖关节点；无法从L_Hand或L_Wrist可靠推断指尖方向；手指姿态不在SMPL-24模型中 |
| 眼平视前方 | unsupported |  | 「眼平视前方」涉及视线方向和眼睛位置，SMPL的Head关节是头顶中心而非眼睛，无法推断视线方向。 | Head不是眼睛或额头位置；无法从Head关节推断视线方向；「平视」需要重力参考系，torso_v1不提供重力方向 |

## M03_R03 · 白鹤亮翅

[修复后 JSON](M03_R03/plan.json) | [逐项修改日志](M03_R03/repair_log.json)


| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |
|---|---|---|---|---|
| 右脚跟进半步 | proxy | relative_position(R_Foot, L_Foot); average_leg_length; torso_v1; ['forward', 'right'] | 观察端点时右脚相对左脚的前后和左右位置，间接反映跟步后的站位；右脚相对左脚在forward方向上接近或略偏后，right方向上有一定偏移，表明两脚距离缩短 | '跟进'是动态过程，端点无法验证移动方向和步幅；无法确认是否恰好为半步 |
| 上体后坐 | proxy | relative_position(Pelvis, AnkleMidpoint); average_leg_length; torso_v1; ['forward', 'up'] | 骨盆相对踝中点的前后位置可间接反映后坐程度；骨盆相对踝中点在forward方向上偏负（偏后），表明上体后坐 | '后坐'是动态过程，端点仅反映最终位置；无法区分后坐与初始站位差异 |
| 重心移至右腿 | proxy | relative_position(Pelvis, R_Ankle); average_leg_length; torso_v1; ['right', 'forward']；relative_position(Pelvis, L_Ankle); average_leg_length; torso_v1; ['right', 'forward']；joint_angle(R_Hip, R_Knee, R_Ankle); None; None; ['value']；joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value'] | 骨盆相对右踝的位置反映骨盆是否偏向右腿上方；骨盆相对右踝在right方向上接近零或略偏左，表明骨盆位于右踝上方附近；骨盆相对左踝的位置反映骨盆是否远离左腿；骨盆相对左踝在right方向上偏负（偏右），表明骨盆远离左踝；右膝弯曲程度可间接反映右腿承重姿态；右膝关节角明显小于180度，表明右膝弯曲呈支撑状态；左膝弯曲程度与右膝对比，辅助判断虚实；左膝关节角相对较大（较伸直），与右膝形成对比 | 重心是力学量，SMPL关节位置无法直接测量；膝弯曲不等于承重，仅为间接推断；无法测量实际地面反作用力 |
| 上体先向右转 | unsupported |  | 上体先向右转是过渡过程中的旋转方向，端点无法验证旋转的先后顺序 | '先向右转'是时序性描述，静态端点无法验证旋转方向和顺序 |
| 再微向左转 | unsupported |  | 再微向左转是过渡过程中的旋转方向，端点无法验证旋转的先后顺序和幅度 | '再微向左转'是时序性描述，静态端点无法验证旋转方向和顺序 |
| 左脚稍向前移 | proxy | relative_position(L_Foot, R_Foot); average_leg_length; torso_v1; ['forward'] | 左脚相对右脚的前后位置反映左脚是否前移；左脚相对右脚在forward方向上为正值（偏前），表明左脚位于右脚前方 | '稍向前移'是动态过程，端点仅反映最终相对位置；无法确认移动距离是否为'稍' |
| 脚尖点地成左虚步 | proxy | joint_angle(L_Hip, L_Knee, L_Ankle); None; None; ['value']；relative_position(L_Foot, R_Foot); average_leg_length; torso_v1; ['forward'] | 左膝较伸直是虚步的典型特征；左膝关节角较大（接近伸直），与右膝弯曲形成对比；左脚在右脚前方是虚步的站位特征；左脚相对右脚在forward方向上为正值 | '脚尖点地'涉及脚与地面接触方式，SMPL无法测量；Ankle不是脚尖，Foot不是脚底接触点；无法区分脚尖点地与全脚着地 |
| 两手随转体向右上、左下分开 | proxy | relative_position(R_Hand, L_Hand); torso_length; torso_v1; ['right', 'up'] | 右手相对左手的位置反映两手分开的方向和程度；右手相对左手在right方向上为正值（偏右），在up方向上为正值（偏高），表明右上左下分开 | '随转体'是动态过程，端点无法验证分开路径；无法验证分开是否随转体完成 |
| 定势时 | unsupported |  | '定势时'是时间标记，不是可测量的身体姿态要求 | '定势时'仅标示时间点，无独立的可测量内容 |
| 右手位于头部右前方 | direct | relative_position(R_Wrist, Head); torso_length; torso_v1; ['right', 'forward'] | 右腕相对头部的位置反映右手是否在头部右前方；右腕相对头部在right方向上为正值（偏右），在forward方向上为正值（偏前） | Wrist不是掌心中心，但作为手部代表点是最佳可用近似 |
| 掌心中心大致与额头同高或略高 | proxy | relative_position(R_Hand, Head); torso_length; torso_v1; ['up'] | 右手相对头部的高度差反映手掌与头部的相对高度；右手相对头部在up方向上接近零或略为正值，表明手掌高度与头部大致相同或略高 | Head不是额头位置，是头部中心；R_Hand不是掌心中心；无法精确对应'额头'高度 |
| 指尖可自然高于头顶 | proxy | relative_position(R_Hand, Head); torso_length; torso_v1; ['up']；relative_position(R_Hand, R_Wrist); torso_length; torso_v1; ['up'] | 右手相对头部的高度反映指尖是否高于头顶；右手相对头部在up方向上为正值或接近零，表明手部高度不低于头顶；手相对腕的高度差反映手指延伸方向；R_Hand相对R_Wrist在up方向上为正值，表明手指朝上延伸 | R_Hand不是指尖，是手部代表点；'可自然高于'是允许性描述，非严格要求 |
| 掌心斜向左后方 | unsupported |  | 掌心朝向是手掌朝向，SMPL关节位置无法推断掌心方向 | palm_orientation在unsupported列表中；无法从关节位置推断掌心朝向 |
| 右肩松沉 | proxy | relative_position(R_Shoulder, Neck); torso_length; torso_v1; ['up'] | 右肩相对颈部的高度差反映肩部是否下沉（不耸肩）；右肩相对颈部在up方向上为负值，且幅度适中，表明肩部自然下沉而非耸起 | '松'涉及肌肉放松状态，无法从关节位置测量；仅能反映肩部高度，不能区分主动下沉与被动放松 |
| 肘部自然下坠并保持微屈 | proxy | relative_position(R_Elbow, R_Shoulder); torso_length; torso_v1; ['up']；joint_angle(R_Shoulder, R_Elbow, R_Wrist); None; None; ['value'] | 肘相对肩的高度反映肘部是否下坠；右肘相对右肩在up方向上为负值，表明肘部低于肩部；肘关节角反映肘部弯曲程度；肘关节角小于180度但大于90度，表明微屈而非完全伸直或过度弯曲 | '自然下坠'涉及重力和放松状态，仅能通过几何位置近似；无法区分主动下坠与被动位置 |
| 右臂呈舒展的弧形 | proxy | joint_angle(R_Shoulder, R_Elbow, R_Wrist); None; None; ['value'] | 肘关节角反映手臂弯曲程度，弧形要求手臂不完全伸直；肘关节角明显小于180度，表明手臂有弯曲形成弧形 | '舒展'是主观描述，无法量化；关节角仅反映弯曲程度，不能验证弧形的流畅性 |
| 左手落于左胯前 | direct | relative_position(L_Wrist, L_Hip); torso_length; torso_v1; ['forward', 'up', 'right'] | 左腕相对左髋的位置反映左手是否在左胯前方；左腕相对左髋在forward方向上为正值（偏前），在up方向上接近零或略为正值，在right方向上接近零 | Wrist不是掌心，但作为手部代表点是最佳可用近似 |
| 掌心向下 | unsupported |  | 掌心向下是手掌朝向，SMPL关节位置无法推断掌心方向 | palm_orientation在unsupported列表中；无法从关节位置推断掌心朝向 |
| 指尖向前 | proxy | relative_position(L_Hand, L_Wrist); torso_length; torso_v1; ['forward'] | 左手相对左腕的前后方向反映手指延伸方向；L_Hand相对L_Wrist在forward方向上为正值，表明手指朝前延伸 | L_Hand不是指尖，是手部代表点；无法精确验证指尖方向，仅为近似推断 |
| 眼平视前方 | proxy | relative_position(Head, Neck); torso_length; torso_v1; ['forward', 'up'] | 头部相对颈部的位置反映头部是否前倾或后仰；头部相对颈部在up方向上为正值（头在颈上方），在forward方向上接近零或略为正值，表明头部大致正直不低头不仰头 | Head不是眼睛/额头，无法确定视线方向；无法区分平视与略微俯视/仰视；无法验证是否看向'前方'而非其他方向 |
