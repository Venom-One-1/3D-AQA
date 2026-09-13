# 3D-AQA 动作质量评估与 Late Fusion 技术文档

> 文档状态：依据 2026-09-09 项目代码与现有实验结果整理。  
> 项目路径：`/home/sqw/Projects/3D-AQA`  
> 当前性质：研究原型、相对排名实验，不是已经标定的绝对动作评分系统。

## 1. 文档目的

当前 3D-AQA 质量评估由三类顶层指标组成：

1. `pose_score`：通过 SMPL 局部关节旋转的 Geodesic Distance 衡量姿态标准程度。
2. `reviewed_pause_count`：统计教师活动阶段内学生出现的异常停顿事件。
3. `weighted_center_excess`：衡量学生骨盆/重心代理轨迹相对教师群体的异常程度。

早期实验使用预定义扣分公式将三类指标组合为一个分数。最新实验改用 Late Fusion：
先让每项指标独立产生学生相对排名，再通过等权 Borda 或可学习权重融合排名。

本文记录当前代码真正实现的计算过程、默认参数、数据文件、实验协议和限制。后续若
代码与本文不一致，应以代码和新实验的 `summary.json` 为准，并同步更新本文。

## 2. 当前实验范围

当前 Late Fusion 实验只覆盖以下数据：

- 学生：`1, 2, 3, 4, 10`。
- 招式：`qishi`、`yemafenzong`、`baiheliangchi`。
- 样本数：每式 5 名学生，共 15 个“学生-招式”样本。
- 人工 Ground Truth：`human_rankings.csv` 中每式一张 1 到 5 的完整排名表。
- 默认参考教学视频：`QxVvRcRn2TA`。

因此，统计上真正独立的监督单元更接近 3 张招式排名表，而不是 15 个相互独立的
训练样本。所有权重和相关性都应当解释为小样本探索性结果。

## 3. 总体数据流

```text
PHALP / 4D-Humans tracking
        |
        +--> 5 FPS SMPL local pose --> Geodesic cost matrix --> DTW
        |                                      |
        |                                      +--> keyframe distance
        |                                      +--> DTW path mean distance
        |                                                   |
        |                                      teacher robust calibration
        |                                                   |
        |                                              pose_score
        |
        +--> 原始帧率 SMPL local pose --> SO(3) 平滑 --> 关节角速度
        |                                                   |
        |                              DTW 教师阶段映射 + 教师活动模板
        |                                                   |
        |                                      reviewed_pause_count
        |
        +--> 原始帧率 SMPL-24 joints + camera translation
                                               |
                              身体坐标系、骨盆轨迹、进度重采样
                                               |
                               10 教师模板与留一教师稳健标定
                                               |
                                  weighted_center_excess
                                               |
              pose / pause / center 各自转为每式相对排名
                                               |
                      Borda 或可学习 Pairwise Late Fusion
                                               |
              完整排序 + Spearman + 平均排名误差 + 命中数
```

需要特别区分两种时间轴：

- DTW 和姿态距离使用 5 FPS 均匀采样序列。
- 停顿和骨盆轨迹尽量在 tracking 覆盖的原始视频帧率上计算。

DTW 用于建立动作阶段对应关系。停顿时长仍在学生原始时间轴上统计，避免 DTW
将学生的长停顿压缩成教师的一帧后掩盖问题。

## 4. Geodesic Distance 姿态分

主要实现：

- `aqa3d/geodesic.py`
- `aqa3d/smpl_dtw.py`
- `aqa3d/pose_score.py`
- `run_pose_score_experiment.py`

### 4.1 姿态表示

每帧姿态为：

```text
(23, 3, 3)
```

即 SMPL `body_pose` 的 23 个局部关节旋转矩阵。Pelvis 的 `global_orient` 不参与
比较，因此评分不直接惩罚学生与教师的整体朝向差异。当前 23 个关节默认等权。

### 4.2 单关节 Geodesic Distance

给定学生与教师同一局部关节的旋转矩阵 \(R_s\) 和 \(R_t\)：

\[
R_{rel}=R_sR_t^T
\]

\[
d(R_s,R_t)=\arccos\left(
\operatorname{clip}\left(\frac{\operatorname{tr}(R_{rel})-1}{2},-1,1\right)
\right)
\]

结果首先以弧度计算。一个学生帧 \(i\) 与教师帧 \(j\) 的 DTW local cost 是 23 个
局部关节距离的算术平均：

\[
C_{ij}=\frac{1}{23}\sum_{k=1}^{23}d(R^s_{i,k},R^t_{j,k})
\]

### 4.3 5 FPS SMPL-Geodesic DTW

学生和教师视频都均匀采样到 5 FPS，再从 tracking 结果取对应帧的 SMPL pose。
默认 DTW 对角步系数 `coefficient=1.0`，累计代价递推为：

\[
D_{ij}=\min\begin{cases}
C_{ij}+D_{i-1,j}\\
C_{ij}+D_{i,j-1}\\
\alpha C_{ij}+D_{i-1,j-1}
\end{cases}
\]

其中当前 \(\alpha=1.0\)。从累计矩阵右下角回溯获得完整单调 DTW 路径 \(P\)。

全过程姿态距离不是最终累计代价，而是回溯路径上 local cost 的平均值：

\[
d_{path}=\frac{1}{|P|}\sum_{(i,j)\in P}C_{ij}
\]

### 4.4 教师关键帧与关键帧距离

教师关键帧由教学视频原始图像的运动峰值提取：

1. 将相邻帧转换到 LUV 色彩空间。
2. 计算相邻帧平均绝对像素差。
3. 使用默认 25 帧滑动核平滑差分序列。
4. 使用 `argrelextrema(..., order=15)` 提取局部峰值。
5. 根据与端点的时间间隔补充动作开始或结束帧。
6. 将原视频关键帧映射到最近的 5 FPS 采样帧。

对每个教师关键帧 \(j\)，DTW 路径可能给出多个学生候选帧。代码选择 local
Geodesic cost 最小的学生帧，而不是选择累计 DTW cost 最小的帧：

\[
i^*(j)=\arg\min_{i:(i,j)\in P} C_{ij}
\]

关键帧平均距离为：

\[
d_{key}=\frac{1}{K}\sum_{j\in keyframes}C_{i^*(j),j}
\]

学生评分阶段固定使用 `QxVvRcRn2TA` 作为参考教师。

### 4.5 教师距离的稳健标定

当前姿态分标定使用 5 个教师：

```text
QxVvRcRn2TA
BV1iE411c7Ni_p03
BV1tk4y1r7Yr_p27
an5qNCspzUw
i8kMrJmAfjU
```

每一式对 5 个教师进行全部有向两两比较，共 `5 * 4 = 20` 个教师配对。关键帧
距离和路径距离分别建立标定分布：

\[
m=\operatorname{median}(d)
\]

\[
MAD=\operatorname{median}(|d-m|)
\]

\[
s=\max(1.4826\times MAD,1.0\text{ degree})
\]

学生距离转换为稳健标准化值：

\[
z_{key}=\frac{d_{key}-m_{key}}{s_{key}},\qquad
z_{path}=\frac{d_{path}-m_{path}}{s_{path}}
\]

### 4.6 `pose_score` 公式

默认关键帧权重为 0.4，DTW 路径权重为 0.6：

\[
z_{pose}=0.4z_{key}+0.6z_{path}
\]

教师中位水平被映射到 95 分，每偏离一个稳健尺度扣 10 分：

\[
S_{pose}=\operatorname{clip}(95-10z_{pose},0,100)
\]

代码还分别保存：

\[
S_{key}=\operatorname{clip}(95-10z_{key},0,100)
\]

\[
S_{path}=\operatorname{clip}(95-10z_{path},0,100)
\]

这里的 95、10、0.4 和 0.6 是当前预定义参数，不是由 Late Fusion 学习得到的。

## 5. 异常停顿次数

主要实现：

- `aqa3d/motion_quality.py`
- `aqa3d/velocity_quality.py`
- `run_pose_score_experiment.py`
- `run_uncapped_pause_penalty_experiment.py`

### 5.1 逐帧运动强度

停顿检测使用原始帧率的 23 个 SMPL 局部旋转序列。首先在 SO(3) 上进行默认
0.20 秒中心滑动平滑：窗口内对旋转矩阵求均值，再通过 SVD 投影回合法旋转矩阵。

相邻帧关节角速度为：

\[
v_{t,k}=d(R_{t,k},R_{t-1,k})\times FPS
\]

单位为 degree/s。身体区域运动强度采用区域关节角速度的 RMS。当前区域为：

| 区域 | 关节 |
|---|---|
| left_arm | shoulder, elbow, wrist |
| right_arm | shoulder, elbow, wrist |
| left_leg | hip, knee, ankle, foot |
| right_leg | hip, knee, ankle, foot |
| trunk | Spine1, Spine2, Spine3, Neck |
| whole_body | 上述区域核心关节的并集 |

停顿评分实际使用 `whole_body`。Collars、Head 和 Hands 不进入首版聚合。tracking
ID 切换点附近的帧被标记无效，不参与速度和停顿统计。

### 5.2 学生静止阈值

学生运动强度序列记为 \(E_s(t)\)。噪声底从最低 25% 强度帧估计：

\[
n_s=\operatorname{median}(L)+3\times1.4826\times MAD(L)
\]

其中 \(L\) 是不高于第 25 百分位数的帧。如果序列接近恒定运动，最低四分位数不再
像噪声簇，代码会将噪声底置为 0，避免把匀速运动全部识别成停顿。

从教师运动模型取得该式全身速度 P90，学生停顿阈值为：

\[
\theta_{pause}=\max(n_s,0.05\times P90_{teacher})
\]

满足以下条件的连续低速区间构成学生候选停顿：

- \(E_s(t)\leq\theta_{pause}\)。
- 连接不超过 0.10 秒的短暂非停顿缺口。
- 删除短于 0.40 秒的停顿片段。

### 5.3 教师活动阶段映射

教师活动模板来自 `TeacherMotionModel`。当前模型使用前述 5 个教师、24式、5 FPS
教师对齐结果建立，进度网格为 101 点。停顿实验取：

```text
active_probability >= 0.60
```

作为教师活动阶段，并在教师时间轴前后扩张 0.40 秒以容忍轻微对齐误差。

学生原始帧到教师阶段的映射过程是：

1. 在 5 FPS DTW 路径中，一个学生采样帧若对应多个教师帧，取教师索引中位数。
2. 对教师索引执行单调累积修正。
3. 将 5 FPS 映射线性插值到学生原始 tracking 帧。
4. 根据映射后的教师进度查询教师活动掩码。

一个学生停顿事件只有在其至少 70% 的帧落入教师活动阶段时，才被定义为异常停顿：

\[
\frac{\#\{t\in event:teacher\_active(t)\}}{|event|}\geq0.70
\]

由此得到 `unexpected_pause_count`。

### 5.4 人工复核后的计数

Late Fusion 使用 `reviewed_pause_count`，由：

```text
unexpected_pause_events.csv
+ pause_event_reviews.csv
```

合并获得。若 `(case_id, event_index)` 出现在复核文件中，则采用人工 `is_valid`；
未出现在复核文件中的事件暂时默认有效。

当前只有 Student 2 的 `yemafenzong` 5 个事件逐项复核，其中第 4 个事件被标记为
误判。因此现有 30 个候选事件中，5 个经过人工复核，另外 25 个仍按自动结果处理。
这也是停顿权重只能作探索性解释的重要原因。

## 6. 身体重心/骨盆轨迹指标

主要实现：

- `aqa3d/pelvis_quality.py`
- `run_pelvis_quality_analysis.py`
- `aqa3d/center_fusion.py`
- `run_center_fusion_experiment.py`

### 6.1 “重心”的实际含义

当前实现没有计算基于身体各分段质量的真实 Center of Mass。它将 SMPL Pelvis
关节点和 PHALP camera translation 作为身体重心运动的代理。因此更准确的名称是
“骨盆/根节点轨迹指标”。

其中：

- `support_*`：Pelvis 相对双踝中点的轨迹，是主要证据。
- `root_*`：PHALP camera translation 的变化，是辅助证据。

单目 camera translation 可能受检测框尺度、相机运动和深度漂移影响，尤其不能将
`root_forward`、`root_lateral` 当作可靠的真实世界位移。

### 6.2 身体坐标系

4D-Humans camera-space SMPL 输出以负 Y 为向上方向：

\[
up=(0,-1,0)
\]

身体左右轴由髋连线和肩连线在地面上的投影共同确定：

\[
lateral=normalize(0.7\,lateral_{hip}+0.3\,lateral_{shoulder})
\]

若肩轴与髋轴方向相反，先翻转肩轴。身体前后轴为：

\[
forward=normalize(up\times lateral)
\]

这样不同视频里人物朝向不同时，水平重心转移仍在各自身体坐标系中比较，而不是
直接比较相机坐标 X/Z。

身体尺度用于归一化距离：

\[
scale=|Pelvis-Neck|+\frac{L_{left\ leg}+L_{right\ leg}}{2}
\]

最终取有效帧尺度的中位数。

### 6.3 七路基础信号

定义双踝中点：

\[
A=\frac{L_{ankle}+R_{ankle}}{2}
\]

以及支撑相对向量：

\[
p_{support}=Pelvis-A
\]

投影并除以身体尺度后得到：

- `support_height`
- `support_lateral`
- `support_forward`

camera translation 以最初 0.5 秒有效帧中位数为原点，得到：

- `root_vertical`
- `root_lateral`
- `root_forward`

身体左右轴的 yaw 相对第一帧变化形成 `body_yaw_change_degrees`。

信号默认平滑 0.20 秒。另用至少 1.00 秒窗口估计竖直动作意图，原信号减去慢趋势
得到竖直残差，供起伏诊断使用。tracking ID 切换点前后一帧被排除。

### 6.4 教师模板和动作进度

重心实验使用 10 个教师，比姿态分和停顿教师模型多 5 个新教学视频。10 个教师
都通过 5 FPS SMPL-Geodesic DTW 映射到参考视频 `QxVvRcRn2TA` 的动作进度。

每式七路信号重采样到 201 个进度点：

```text
0.000, 0.005, ..., 1.000
```

教师模板保存每个进度点的 median、P10 和 P90。学生三段动作同样通过 DTW 获得
参考进度并重采样到该网格。

### 6.5 当前进入 Late Fusion 的重心指标

`weighted_center_excess` 并非直接使用全部骨盆诊断指标，而是按招式选择以下分项：

| 招式 | 分项 | 内部权重 | 最小稳健尺度 |
|---|---|---:|---:|
| qishi | root_vertical_lowering_error | 0.35 | 0.010 |
| qishi | horizontal_transfer_rmse | 0.30 | 0.005 |
| qishi | support_endpoint_error | 0.20 | 0.005 |
| qishi | vertical_trajectory_rmse | 0.15 | 0.005 |
| yemafenzong | horizontal_transfer_rmse | 0.50 | 0.005 |
| yemafenzong | support_endpoint_error | 0.30 | 0.005 |
| yemafenzong | vertical_trajectory_rmse | 0.20 | 0.005 |
| baiheliangchi | vertical_range_error | 0.50 | 0.050 |
| baiheliangchi | support_endpoint_error | 0.30 | 0.005 |
| baiheliangchi | forward_transfer_rmse | 0.20 | 0.005 |

主要分项定义如下：

- `horizontal_transfer_rmse`：完整进度上 `(support_lateral, support_forward)`
  与教师中位模板的二维 RMSE。
- `support_endpoint_error`：动作结束点二维支撑相对坐标与教师模板的欧氏距离。
- `vertical_trajectory_rmse`：`support_height` 全进度 RMSE。
- `forward_transfer_rmse`：`support_forward` 全进度 RMSE。
- `vertical_range_error`：

  \[
  \left|\log\frac{range_{student}}{range_{teacher}}\right|
  \]

- `root_vertical_lowering_error`：比较前 10% 与后 10% 进度的 root vertical 中位值，
  下降量定义为开始高度减结束高度，然后取学生与教师下降量的绝对差。

起势最初尝试过 Pelvis 相对脚踝的下降量，但学生长裤等情况下腿部 SMPL 重建偏差
会让双踝和膝关节位置联动，掩盖真实下蹲。因此当前起势改用 camera-space root Y
下降作为代理。它更符合现有视频的人工观察，但仍可能受到相机和尺度变化影响。

### 6.6 留一教师标定与综合异常量

对每名教师：

1. 用其余 9 名教师的中位轨迹建立模板。
2. 计算被留出教师的每个重心分项误差。
3. 收集每式、每分项的 10 个 leave-one-teacher-out 误差。

对每个分项建立稳健标定：

\[
s_m=\max(1.4826\times MAD_m,minimum\_scale_m)
\]

学生分项误差的标准化值与超额异常为：

\[
z_m=\frac{error_m-median_m}{s_m}
\]

\[
e_m=\max(0,z_m-1)
\]

即教师留一误差中位数以上的第一个稳健尺度被视为容忍区，不产生异常量。招式级
重心异常为预定义内部权重的加权和：

\[
E_{center}=\sum_m w_m e_m
\]

该值就是 Late Fusion 使用的 `weighted_center_excess`，数值越小越好。这里的招式
内部分项选择和权重仍是规则设计，不是 Late Fusion 自动学习出来的。

## 7. 旧版规则融合基线

最新 Late Fusion 仍保留旧版规则融合结果作为对照：

\[
S_{rule}=S_{pose}-p_{pause}N_{pause}
-\min(p_{center}E_{center},15)
\]

现有敏感性实验扫描：

- `p_pause in {2, 4, 5, 6, 8}`，停顿总扣分不设上限。
- `p_center in {0, 2, 3, 5}`，重心扣分上限为 15。

在当前 15 个样本上观测到的最优配置为：

```text
p_pause = 6
p_center = 2
```

但参数选择和评估使用了同一批人工排名，所以这是 `in_sample_tuned_reference`，
不能作为泛化性能与 Late Fusion 的 OOF 结果直接比较。

## 8. Late Fusion

主要实现：

- `aqa3d/rank_fusion.py`
- `run_late_fusion_experiment.py`

### 8.1 三个顶层输入

| metric_id | 来源 | 方向 |
|---|---|---|
| pose_score | `student_pose_scores.csv` | 越大越好 |
| reviewed_pause_count | 停顿事件与人工复核合并 | 越小越好 |
| weighted_center_excess | `student_center_summary.csv` | 越小越好 |

脚本首先导出通用 long-format 特征表：

```text
case_id, move_name, student_id, metric_id, raw_value, higher_is_better
```

每一式必须有完整的 5 名学生和 3 项指标。缺失、重复、非有限值或人工排名不完整时
直接报错，不静默填补。

### 8.2 单指标相对排名

每个指标都在“同一招式的 5 名学生”内部排名。平局使用平均名次。将名次转换成
统一的质量方向：

\[
q_{i,m}=1-\frac{rank_{i,m}-1}{N-1}
\]

其中第一名为 1，最后一名为 0，\(N=5\)。这使 pose、pause 和 center 三种不同
单位能够在排名层面融合。

重要含义：同一学生加入不同的比较群体后，\(q\) 可能改变。因此它是
cohort-relative 量，不是可脱离其他学生单独解释的绝对分数。

### 8.3 连续值对照

代码同时保留一个连续输入对照。先将所有指标方向调整为“越大越好”，再在每一式
内部进行 min-max 归一化：

\[
x'_{i,m}=\frac{x_{i,m}-\min_i x_{i,m}}
{\max_i x_{i,m}-\min_i x_{i,m}}
\]

如果某指标在该式对所有学生完全相同，则统一赋值 0.5。连续值对照保留了学生间
差异幅度，但仍是当前 cohort 内的相对归一化，不是绝对评分。

### 8.4 等权 Borda 基线

三个归一化排名质量值等权平均：

\[
Q_i^{Borda}=\frac{1}{3}\sum_{m=1}^{3}q_{i,m}
\]

\(Q\) 越大，综合排名越高。该方法不使用人工排名学习参数。


### 8.5 可学习 Rank Fusion

对于同一招式中的任意两名学生，如果人工排名表明学生 \(i\) 优于学生 \(j\)，
就构造一个两两偏序训练样本：

\[
\Delta q_{ij}=q_i-q_j
\]

权重通过以下带正则项的 pairwise logistic loss 学习：

\[
L(w)=\frac{1}{P}\sum_{(i,j)}
\log\left(1+\exp(-w^T\Delta q_{ij})\right)
+\lambda\left\|w-\frac{1}{M}\mathbf{1}\right\|_2^2
\]

约束为：

\[
w_m\geq 0,\qquad \sum_m w_m=1
\]

这里 \(M=3\)，分别对应 pose、pause 和 center。非负约束保证某一指标变好不会
直接降低融合质量；和为 1 使权重便于解释。正则项把小样本下的权重拉向等权，
降低某一折出现极端权重的风险。

默认配置：

- 优化器：SLSQP。
- 初始权重：\((1/3,1/3,1/3)\)。
- 正则系数：\(\lambda=0.1\)。
- 无截距项。
- `maxiter=1000`，`ftol=1e-12`。
- 每个训练招式有 5 名学生，因此产生 \(C_5^2=10\) 个偏序对；每折用两个招式，
  共 20 个训练偏序对。

学习后，学生 \(i\) 的融合质量为：

\[
Q_i=w^Tq_i
\]

\(Q_i\) 从大到小生成最终名次。该优化学习的是“各指标对人工偏序的重要性”，
不是从特征直接回归一个绝对动作质量分数。

### 8.6 Leave-One-Move-Out 验证

当前只有三式人工排名，因此采用三折 leave-one-move-out（LOMO）：

| 测试招式 | 训练招式 |
|---|---|
| qishi | yemafenzong, baiheliangchi |
| yemafenzong | qishi, baiheliangchi |
| baiheliangchi | qishi, yemafenzong |

每次只用两个训练招式学习一套全局权重，再预测被留出的招式。三个测试折拼接形成
15 个 student-move 样本的 out-of-fold（OOF）结果。这样至少保证被评价招式的
人工排名没有参与本折权重学习。

代码还会在全部三式上拟合 `full_data_weights.csv`，仅用于观察数据偏好，
不能当作泛化结果。正则敏感性实验固定测试
\(\lambda\in\{0,0.01,0.1,1,10\}\)，不能根据测试招式表现反向挑选最优值。

### 8.7 连续指标融合对照

除排名质量 \(q\) 外，程序还将连续归一化值输入同一个可学习模型，形成
`learned_continuous_fusion` 对照。它用于回答“保留指标差值大小是否比只保留顺序
更有效”，但仍然只具有当前比较群体内的相对含义。


## 9. 最新 Late Fusion 实验结果

结果来源：`late_fusion_results/ranking_report.md` 和
`late_fusion_results/evaluation_summary.csv`。

### 9.1 人工排名

当前 Ground Truth 为：

| 招式 | 人工完整排序 |
|---|---|
| qishi | 10 > 2 > 1 > 4 > 3 |
| yemafenzong | 10 > 2 > 1 > 4 > 3 |
| baiheliangchi | 10 > 1 > 2 > 3 > 4 |

人工排名不仅考虑关键姿态，还综合考虑连贯性、动作幅度等整体表现。

### 9.2 汇总性能

| 方法 | 协议 | Spearman 宏平均 | 总体平均排名误差 | 精确名次 |
|---|---|---:|---:|---:|
| 单指标：pose | no training | 0.700 | 0.800 | 5/15 |
| 单指标：pause | no training | 0.338 | 1.333 | 2/15 |
| 单指标：center | no training | 0.867 | 0.400 | 10/15 |
| 等权 Borda | no training | 0.846 | 0.533 | 7/15 |
| 可学习 rank fusion | LOMO OOF | 0.867 | 0.400 | 10/15 |
| 可学习 continuous fusion | LOMO OOF | 0.833 | 0.533 | 8/15 |
| 旧规则融合 | in-sample tuned | 0.933 | 0.267 | 11/15 |

“Spearman 宏平均”是三个招式 Spearman 的算术平均；“总体平均排名误差”是 15 个
student-move 样本的绝对名次误差平均；“精确名次”是预测名次与人工名次完全相同
的样本数量。

### 9.3 OOF Rank Fusion 完整排序

| 招式 | 预测排序 | Spearman | 平均排名误差 |
|---|---|---:|---:|
| qishi | 10 > 4 > 2 > 1 > 3 | 0.700 | 0.800 |
| yemafenzong | 10 > 2 > 1 > 3 > 4 | 0.900 | 0.400 |
| baiheliangchi | 10 > 1 > 2 > 3 > 4 | 1.000 | 0.000 |

其主要错误集中在 `qishi`：Student 4 的预测位置仍然偏高。这说明即使加入异常
停顿和重心代理，现有三项顶层指标仍未充分刻画起势中的动作僵硬、幅度不足或
其他整体质量缺陷。

### 9.4 学得权重

默认 \(\lambda=0.1\) 的 rank 输入折间权重如下：

| 留出招式 | pose | pause | center |
|---|---:|---:|---:|
| qishi | 0.440 | 0.000 | 0.560 |
| yemafenzong | 0.267 | 0.198 | 0.535 |
| baiheliangchi | 0.367 | 0.087 | 0.546 |
| 均值 | 0.358 | 0.095 | 0.547 |
| 标准差 | 0.071 | 0.081 | 0.010 |

连续值输入的折间均值为 pose 0.413、pause 0.126、center 0.461。全数据探索性 rank
权重为 pose 0.356、pause 0.096、center 0.548。

当前数据中 center 权重最高且折间最稳定，pause 权重最低且变化较大。但不能据此
得出“重心天然最重要、停顿天然不重要”的一般性结论，原因包括：

1. 当前只监督三个招式，训练信号非常少。
2. 三项指标可能相关，权重会在相关特征之间重新分配。
3. 停顿标签尚未全部人工复核，含有测量误差。
4. center 单指标在这 15 个样本上本身就与人工排名高度一致。
5. 人工排名是综合判断，却没有对姿态、停顿、重心分别标注子分数。

### 9.5 当前能支持的结论

1. Late Fusion 的代码路径可行，能够统一不同方向、不同单位的指标，并自然扩展
   新的 `metric_id`。
2. 在当前小样本上，可学习 rank fusion 的 OOF 表现优于等权 Borda，但没有超过
   center 单指标；因此“学习融合一定优于最佳单指标”尚未得到证明。
3. 连续值融合没有优于 rank fusion，说明当前数据下保留数值间距暂未带来收益；
   这也可能是各指标量纲和标定尚不稳定造成的。
4. 旧规则融合的数值最好，但其惩罚参数是在同一组 15 个排名上调出的，只能作为
   in-sample 上限参考，不能与 OOF 方法作公平的泛化比较。
5. 现阶段系统适合输出当前五名学生中的相对排序，不适合给单个新学生输出具有
   固定含义的绝对分数。
6. 下一步最重要的不是继续微调三个权重，而是增加独立招式、学生和完整人工复核，
   再进行按视频或按受试者划分的外部验证。


## 10. 输入、输出与复现

### 10.1 默认输入

`run_late_fusion_experiment.py` 默认读取：

| 文件 | 用途 |
|---|---|
| `pose_score_experiment_results/student_pose_scores.csv` | 姿态基础分和相关距离 |
| `pose_score_experiment_results/unexpected_pause_events.csv` | 自动检测的异常停顿事件 |
| `pause_event_reviews.csv` | 人工事件复核 |
| `center_fusion_experiment_results/student_center_summary.csv` | 重心融合汇总指标 |
| `human_rankings.csv` | 每式人工相对排名 |

组装后的通用特征表采用 long format：

~~~text
case_id, move_name, student_id, metric_id, raw_value, higher_is_better
~~~

代码要求每一式恰好存在预期数量的学生，三项指标和人工排名均完整。缺失、重复、
非有限值、学生集合不一致都会明确报错，不会静默填值。

### 10.2 运行命令

在项目根目录执行：

~~~bash
cd /home/sqw/Projects/3D-AQA
conda run -n 4d-humans python run_late_fusion_experiment.py
~~~

覆盖输入或输出路径时可使用：

~~~bash
conda run -n 4d-humans python run_late_fusion_experiment.py \
  --pose-scores <student_pose_scores.csv> \
  --pause-events <unexpected_pause_events.csv> \
  --pause-reviews <pause_event_reviews.csv> \
  --center-summary <student_center_summary.csv> \
  --human-rankings <human_rankings.csv> \
  --output-root <output_directory>
~~~

默认正则为 0.1。可通过 `--regularization` 修改主实验正则，通过
`--regularization-sensitivity` 提供敏感性列表。

### 10.3 输出文件

默认输出目录为 `late_fusion_results/`：

| 文件 | 内容 |
|---|---|
| `fusion_features.csv` | 三项原始指标组成的 long-format 特征表 |
| `metric_rankings.csv` | 原始值、单指标名次、rank quality 和连续归一化值 |
| `fold_weights.csv` | 每个 LOMO 折的权重、优化状态及权重稳定性汇总 |
| `oof_rankings.csv` | 所有方法逐学生、逐招式的分数和预测名次 |
| `evaluation_summary.csv` | 每式及总体 Spearman、排名误差、命中数和完整排序 |
| `regularization_sensitivity.csv` | 不同正则系数下的 OOF 指标和权重 |
| `full_data_weights.csv` | 全三式探索性拟合权重 |
| `ranking_report.md` | 便于人工阅读的结果总览 |
| `summary.json` | 实验规模、默认参数、复核状态和限制声明 |

留档和论文报告时应优先引用 `evaluation_summary.csv` 的
`leave_one_move_out` 行，并同时展示完整排序。不能只报告相关系数，也不能把
`full_data_weights.csv` 或 `in_sample_tuned_reference` 当成测试结果。

### 10.4 相关测试

Late Fusion 的直接测试为：

~~~bash
conda run -n 4d-humans python -m pytest tests/test_rank_fusion.py
~~~

与三个上游指标相关的重点测试包括：

~~~text
tests/test_geodesic.py
tests/test_smpl_dtw.py
tests/test_pose_score.py
tests/test_motion_quality.py
tests/test_velocity_quality.py
tests/test_uncapped_pause_penalty.py
tests/test_pelvis_quality.py
tests/test_center_fusion.py
~~~

当修改公共 tracking、DTW 或 SMPL 读取代码时，应再运行完整测试集，避免动作分割
与现有评分行为发生无意变化。

## 11. 如何加入新指标

例如后续加入动作节奏指标 `tempo_ratio_error`：

1. 在上游程序中为每个 `case_id` 计算一个固定语义的标量。
2. 明确方向：误差型指标通常 `higher_is_better=False`。
3. 将其追加到 `fusion_features.csv` 的 long-format 结构。
4. 在 `METRIC_SPECS` 中注册，而不是把新指标硬编码进某个总分公式。
5. 保证每式每名学生都有该指标；缺失值应在上游解决。
6. 重新进行按招式、学生或视频隔离的交叉验证。
7. 同时报告新指标单独排名、等权融合和可学习融合，判断它是否提供增量信息。
8. 检查折间权重稳定性、与已有指标的相关性，以及移除该指标后的消融结果。

如果数据规模扩大，应优先使用 group-aware 的验证方式，例如按学生留出或按完整
视频留出，确保同一练习者或同源视频不会同时进入训练集和测试集。


## 12. 已知限制与解释边界

### 12.1 姿态分

- 4D-Humans 的遮挡、衣着和视角误差会进入 SMPL 局部旋转。
- 23 个关节当前等权，未体现不同招式的关键关节。
- DTW 路径均值受对齐路径选择影响，不能完全分离姿态误差与节奏差异。
- 关键帧只评价若干姿势，可能被“关键姿势正确但全过程僵硬”的样本绕过。
- 95 分基准和每尺度 10 分属于人为标定，并非绝对动作质量量尺。

### 12.2 停顿

- 自动停顿依赖阈值、平滑、最短持续时间和教师活动模板。
- 目前只有 5/30 个事件人工复核，其余事件默认有效。
- `reviewed_pause_count` 只统计次数，不区分事件持续时间、涉及身体区域和严重程度。
- 合理的太极定势不能被简单视为错误，必须依赖教师活动阶段过滤。

### 12.3 重心

- Pelvis 是身体重心代理，不是真实 Center of Mass。
- 单目 camera translation 不是可靠的世界坐标轨迹。
- 地面、相机运动和脚部接触尚未完整建模。
- 当前每一式使用不同分项规则，包含领域先验和人工设计。
- 起势的竖直下降使用 camera root Y，是为绕开 SMPL 腿部弯曲偏差采取的折衷，
  在移动相机视频上尤其需要谨慎。

### 12.4 Late Fusion

- 当前只有 3 张独立招式排名表，权重方差估计和泛化结论都不稳定。
- 排名融合丢弃了绝对差距；连续融合又依赖当前 cohort 的 min-max 范围。
- 权重描述预测贡献，不等于动作教学中的生理或技术重要性。
- 单个新学生没有比较群体时，当前模型不能生成稳定的独立排名。
- 人工 Ground Truth 本身是综合主观判断，尚无多标注者一致性分析。
- 新增高度相关的指标会发生“重复计票”，需要相关性和消融检查。

## 13. 建议的后续优先级

1. 完成人工停顿事件复核，并保存复核者、判定和备注，先降低标签噪声。
2. 扩展到更多招式、学生和完整视频；训练/测试按学生或视频来源分组。
3. 为姿态、连贯性、幅度、重心和节奏分别采集子项人工等级，使每个指标可以独立
   验证，而不是只拟合一个综合排名。
4. 加入动作节奏和幅度指标，但先做单指标有效性与消融，再进入融合。
5. 评估教师间分布和标注者间一致性，报告 bootstrap 置信区间。
6. 若最终需要单人绝对分数，另行建立基于教师分布的固定标尺或有监督 ordinal
   model；不要直接把当前 cohort-relative rank quality 当成绝对分数。
7. 保留规则系统用于生成可解释反馈，保留 Late Fusion 用于相对排序。两者可以共享
   底层指标，但不必承担同一个功能。

## 14. 代码索引

| 主题 | 首要入口 |
|---|---|
| Late Fusion 批处理 | `run_late_fusion_experiment.py` |
| 排名、偏序损失、LOMO | `aqa3d/rank_fusion.py` |
| 姿态分实验 | `run_pose_score_experiment.py` |
| 姿态稳健标定 | `aqa3d/pose_score.py` |
| Geodesic Distance | `aqa3d/geodesic.py` |
| SMPL-Geodesic DTW | `aqa3d/smpl_dtw.py` |
| 原始帧运动与停顿 | `aqa3d/motion_quality.py` |
| 教师速度/活动模板 | `aqa3d/velocity_quality.py` |
| 停顿复核计数与扣分实验 | `run_uncapped_pause_penalty_experiment.py` |
| 骨盆轨迹特征 | `aqa3d/pelvis_quality.py` |
| 重心指标选择与稳健超额 | `aqa3d/center_fusion.py` |
| 重心融合实验 | `run_center_fusion_experiment.py` |
| 人工排名 | `human_rankings.csv` |
| 当前可读实验报告 | `late_fusion_results/ranking_report.md` |

## 15. 一页式记忆

- 姿态：5 FPS、23 个 SMPL 局部旋转、关键帧距离 40% + DTW 路径均值 60%，再由
  5 教师稳健分布映射为 `pose_score`。
- 停顿：原始帧率局部旋转角速度低于自适应阈值至少 0.40 秒，并且事件至少 70%
  落在教师活动阶段，才记为异常停顿。
- 重心：Pelvis/根节点代理，不是真实 COM；10 教师模板，按招式选择分项并汇总为
  `weighted_center_excess`。
- 融合：每式内部先排名，转为 0 到 1 的 rank quality，再用非负、和为 1 的权重
  做 pairwise logistic rank fusion。
- 验证：三折 leave-one-move-out；默认 \(\lambda=0.1\)。
- 当前 OOF：Spearman 宏平均 0.867，平均名次误差 0.400，精确名次 10/15。
- 最重要的边界：这是五名学生当前 cohort 内的相对排名模型，不是绝对评分器。
