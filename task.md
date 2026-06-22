# Codex Plan: 基于 SMPL 局部关节旋转的 Geodesic Distance 动作相似度计算

## 目标

实现一个模块，用于定量衡量学生太极拳动作与标准教学动作之间的姿态相似度。输入是两段已经时序对齐的 SMPL pose 序列，输出包括：

1. 每帧、每个关节的旋转误差；
2. 整段动作的加权 geodesic distance；
3. 基于距离转换得到的动作标准程度分数。

核心思想是：SMPL 的局部关节旋转位于三维旋转空间 (SO(3))，因此不直接比较 rotation matrix 或 axis-angle 的欧氏距离，而是使用 geodesic distance 衡量两个旋转之间的最短角距离。

---

## 数学定义

对于学生动作和标准动作中第 t 帧、第 j 个关节的旋转矩阵：

$
R^{stu}_{t,j}, R^{std}_{t,j} \in SO(3)
$

先计算相对旋转：

$
\Delta R_{t,j} = R^{stu}_{t,j} (R^{std}_{t,j})^T
$

再计算 geodesic distance：

$
d_{t,j} =
\arccos
\left(
\frac{\mathrm{tr}(\Delta R_{t,j}) - 1}{2}
\right)
$

其中 $d_{t,j}$ 的单位是弧度，取值范围为 $[0, \pi]$。

整段动作的加权距离为：

$
D =
\frac{
\sum_{t=1}^{T}\sum_{j=1}^{J}
\alpha_t w_j d_{t,j}
}{
\sum_{t=1}^{T}\sum_{j=1}^{J}
\alpha_t w_j
}
$

其中：

* $T$：关键帧数量；
* $J$：参与比较的 SMPL 局部关节数量，默认 23；
* $\alpha_t$：帧权重；
* $w_j$：关节权重。

最后将距离转换为百分制分数：

$
Score = 100 \cdot \exp(-D / \tau)
$

其中 $\tau$ 是温度系数，用来控制距离到分数的映射尺度。

---

## 重要设计原则

1. 默认只比较 SMPL 的 body pose，即 23 个局部关节旋转，不比较 global orientation。
2. 不直接比较原始 3D joints，也不直接比较 axis-angle 的欧氏距离。
3. 输入可以支持 rotation matrix 或 axis-angle 两种格式。
4. 所有计算应支持 batch 维度，方便后续训练和评估。
5. 计算 arccos 前必须进行 clamp，避免数值误差导致 NaN。
6. 保留每帧每关节的 error map，方便后续可视化和反馈生成。

---
