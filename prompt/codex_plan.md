# Codex Plan：实现 SMPL 24-Joints 角度指标计算脚本

## 目标

实现一个通用的角度指标计算模块，用于从 SMPL 回归得到的 24 个 3D joints 中提取一批后续可用于太极拳动作质量评估的候选指标。

该脚本不直接生成训练反馈，也不绑定具体招式。它只负责把 3D joints 转换成标准化的角度指标和少量辅助几何指标。后续的“动作要领—指标—反馈模板”表会从这些指标中选择当前招式需要关注的部分。

整体流程为：

```text
SMPL 3D Joints
    ↓
Angle Metric Extractor
    ↓
通用角度指标表 / JSON / CSV
    ↓
动作要领—指标—反馈模板
    ↓
训练反馈生成
```

---

## 1. 关节点编号定义

请在代码中固定一份 SMPL 24-joint mapping。当前使用如下编号：

```python
SMPL_24_JOINTS = {
    "Pelvis": 0,
    "L_Hip": 1,
    "R_Hip": 2,
    "Spine1": 3,
    "L_Knee": 4,
    "R_Knee": 5,
    "Spine2": 6,
    "L_Ankle": 7,
    "R_Ankle": 8,
    "Spine3": 9,
    "L_Foot": 10,
    "R_Foot": 11,
    "Neck": 12,
    "L_Collar": 13,
    "R_Collar": 14,
    "Head": 15,
    "L_Shoulder": 16,
    "R_Shoulder": 17,
    "L_Elbow": 18,
    "R_Elbow": 19,
    "L_Wrist": 20,
    "R_Wrist": 21,
    "L_Hand": 22,
    "R_Hand": 23,
}
```

注意：不要假设左/右和图像左右一致。这里的 L/R 是人体自身的左/右。

---

## 2. 输入输出设计

### 2.1 输入格式

核心函数接收 Torch Tensor：

```python
joints: torch.Tensor
```

支持以下两种形状：

```python
(T, 24, 3)
```

或者单帧：

```python
(24, 3)
```

其中：

* `T` 表示帧数或关键帧数量；
* `24` 表示 SMPL 24 个 joints；
* `3` 表示三维坐标。

如果输入是 `(24, 3)`，内部自动扩展成 `(1, 24, 3)`。

### 2.2 输出格式

核心函数输出一个 `pandas.DataFrame`，每一行对应某一帧的一个指标：

```text
frame_id, metric_id, value, unit, status
```

示例：

```text
frame_id, metric_id, value, unit, status
0, left_knee_angle, 145.23, degree, valid
0, right_knee_angle, 146.71, degree, valid
0, left_elbow_angle, 162.40, degree, valid
0, shoulder_height_diff, 0.031, normalized_length, valid
```

其中：

* `metric_id` 是指标名称；
* `value` 是指标值；
* `unit` 可以是 `degree`、`ratio`、`normalized_length`；
* `status` 用于标记是否计算成功，例如 `valid`、`invalid_zero_length`、`missing_joint`。

---


## 3. 第一版候选指标库

不要暴力计算所有三点组合。只计算后续可能用于太极拳反馈的指标。

### 3.1 下肢指标

#### left_knee_angle

```python
angle(L_Hip, L_Knee, L_Ankle)
```

解释：左膝屈曲程度。

#### right_knee_angle

```python
angle(R_Hip, R_Knee, R_Ankle)
```

解释：右膝屈曲程度。

#### left_ankle_angle

```python
angle(L_Knee, L_Ankle, L_Foot)
```

解释：左踝姿态。注意 foot 关键点可能不稳定，但仍然保留。

#### right_ankle_angle

```python
angle(R_Knee, R_Ankle, R_Foot)
```

解释：右踝姿态。

#### left_hip_angle

```python
angle(Spine1, L_Hip, L_Knee)
```

解释：左髋屈曲/大腿相对躯干的角度。

#### right_hip_angle

```python
angle(Spine1, R_Hip, R_Knee)
```

解释：右髋屈曲/大腿相对躯干的角度。

#### left_leg_opening_angle

```python
angle(R_Hip, L_Hip, L_Knee)
```

解释：左腿相对于骨盆横向的展开程度。

#### right_leg_opening_angle

```python
angle(L_Hip, R_Hip, R_Knee)
```

解释：右腿相对于骨盆横向的展开程度。

---

### 3.2 上肢指标

#### left_elbow_angle

```python
angle(L_Shoulder, L_Elbow, L_Wrist)
```

解释：左肘屈曲程度。

#### right_elbow_angle

```python
angle(R_Shoulder, R_Elbow, R_Wrist)
```

解释：右肘屈曲程度。

#### left_shoulder_angle_neck

```python
angle(Neck, L_Shoulder, L_Elbow)
```

解释：左上臂相对于肩颈方向的打开程度。

#### right_shoulder_angle_neck

```python
angle(Neck, R_Shoulder, R_Elbow)
```

解释：右上臂相对于肩颈方向的打开程度。

#### left_shoulder_angle_spine

```python
angle(Spine3, L_Shoulder, L_Elbow)
```

解释：左上臂相对于躯干方向的夹角。

#### right_shoulder_angle_spine

```python
angle(Spine3, R_Shoulder, R_Elbow)
```

解释：右上臂相对于躯干方向的夹角。

#### left_wrist_arm_angle

```python
angle(L_Elbow, L_Wrist, L_Hand)
```

解释：左腕/手部相对于前臂方向的角度。SMPL hand joint 较粗糙，保留但后续谨慎使用。

#### right_wrist_arm_angle

```python
angle(R_Elbow, R_Wrist, R_Hand)
```

解释：右腕/手部相对于前臂方向的角度。SMPL hand joint 较粗糙，保留但后续谨慎使用。

---

### 3.3 躯干与身体整体指标

#### spine_bend_angle_1

```python
angle(Pelvis, Spine1, Spine2)
```

解释：低位脊柱弯曲程度。

#### spine_bend_angle_2

```python
angle(Spine1, Spine2, Spine3)
```

解释：中位脊柱弯曲程度。

#### neck_spine_angle

```python
angle(Spine3, Neck, Head)
```

解释：头颈姿态。

#### shoulder_hip_twist_angle

计算肩线和髋线的夹角：

```python
shoulder_vec = L_Shoulder - R_Shoulder
hip_vec = L_Hip - R_Hip
angle_vectors(shoulder_vec, hip_vec)
```

解释：肩线与髋线之间的相对旋转，可作为躯干转体程度的粗略指标。

注意：这个指标没有方向，只表示夹角大小。第一版先输出绝对夹角即可。

#### trunk_vector_angle_with_spine

```python
angle(Pelvis, Spine3, Neck)
```

解释：躯干上段姿态，可作为躯干是否过度弯曲的粗略指标。

---

### 3.4 辅助非角度指标

虽然该脚本重点是角度，但建议同时输出少量辅助指标，因为后续反馈很有用。

#### shoulder_height_diff

```python
L_Shoulder[y] - R_Shoulder[y]
```

解释：左右肩高度差。

需要支持配置 `vertical_axis`，默认 `1`，即默认 y 轴是竖直方向。

#### hip_height_diff

```python
L_Hip[y] - R_Hip[y]
```

解释：左右髋高度差。

#### hand_height_diff

```python
L_Wrist[y] - R_Wrist[y]
```

解释：左右手高度差。

#### pelvis_height

```python
Pelvis[y]
```

解释：骨盆高度。后续可以和标准动作对比，用于判断重心下沉程度。

#### normalized_pelvis_height

建议用身体尺度归一化：

```python
body_scale = mean_bone_length or distance(Pelvis, Neck)
normalized_pelvis_height = Pelvis[y] / body_scale
```

如果坐标系原点不固定，该指标可能不稳定。第一版可以保留，但在文档中注明需要结合统一坐标系或标准对齐后使用。

---


## 关键设计原则

实现时请遵守以下原则：

1. **通用指标计算与太极拳语义解释解耦**。
   脚本只计算指标，不判断动作对错。

2. **只计算有解剖学意义的候选指标**。
   不要暴力枚举所有三点夹角。

3. **所有角度默认输出 degree**。
   便于人工检查和后续反馈模板编写。

4. **必须处理异常情况**。
   包括零长度骨骼、输入维度错误、缺少 npz key 等。

5. **指标命名稳定且可读**。
   后续“动作要领—指标—反馈模板”表会依赖这些 `metric_id`。

6. **保持可扩展性**。
   后续会增加标准动作对比、阈值判断、front/back leg 映射和自然语言反馈生成。
