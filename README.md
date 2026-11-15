# volumetric-fog-integrator
Python 游戏体积雾积分库，处理非均匀密度、透射率、散射光和多段介质

## 用法

```python
from volfog import Medium, DirectionalLight, PointLight, integrate_ray, henyey_greenstein

fog = Medium(min_corner=(-1, -1, -1), max_corner=(1, 1, 1),
             sigma_a=(0.1, 0.1, 0.1), sigma_s=(0.4, 0.5, 0.6),
             emission=(0.0, 0.0, 0.0), phase=henyey_greenstein(0.7))
sun = DirectionalLight(direction=(0, 0, -1), radiance=(1, 1, 1),
                       visibility=((4.5, 5.5),))  # 部分遮挡
result = integrate_ray((0, 0, -5), (0, 0, 1), [fog], [sun], t_max=10.0,
                       tol=1e-6, max_depth=20)
# result.radiance / result.transmittance / result.error / result.subdivisions
```

## 积分方法

1. **分段**：把相机射线裁剪到 `[0, t_max]`，在每个介质包围盒（AABB，slab 求交）
   的进出点和每个光源可见性区间端点处切分，得到一组基本段。每个基本段内
   重叠介质集合和光照可见状态都是常量，因此密度突变、体积边界、阴影边缘
   都落在段边界上，积分结果对这些位置连续（相机在介质内部时入射点被裁剪
   到 `t=0`，自然支持）。
2. **重叠合并**：同一段内多个介质的 RGB 吸收/散射系数和发光项**求和一次**
   （物理上等于混合介质），相函数按 σ_s 加权混合。不会重复累加，也不会
   静默采用最后写入的密度；`overlap="raise"` 可改为直接报错。
3. **解析积分**：段内消光系数为常数，透射率用 Beer–Lambert `exp(-σ_t·L)`，
   发光项和平行光内散射用闭式 `σ_s·p·L·(1-exp(-σ_t·L))/σ_t`（含 σ_t→0
   的极限保护），这些部分零数值误差。
4. **自适应求积**：点光源的 `1/r²` 衰减和沿线变化的相角没有闭式解，用
   自适应 Simpson 求积，返回 Richardson 外推的误差估计并计入
   `result.error`。

## 最大细分预算

`max_depth`（默认 20）是自适应 Simpson 的最大递归深度，每次求积最多使用
`2**(max_depth+1)` 个子区间；预算耗尽时不再细分，残余误差仍计入
`result.error`。`result.subdivisions` 报告整条射线实际使用的子区间总数，
`tol` 控制每次求积的绝对误差容差。积分过程只依赖确定性浮点运算和固定
遍历顺序，同输入结果逐位一致。

## 输出

每条 RGB 光线返回：`radiance`（到达相机的内散射辐射）、`transmittance`
（整段透射率，用于乘背景/表面辐射）、`error`（RGB 求积误差估计）、
`subdivisions`（实际细分次数）。

## 与真实多重散射的差距

本库只计算**单次散射**：光从光源到介质点再到相机，忽略介质内部的二次及
更高阶散射。因此在光学厚度大（σ_t·L ≳ 1）或强散射介质中会偏暗——真实
多次散射会把能量重新扩散回视线方向，并柔化阴影边界。此外不含：介质内
的光源衰减（光源到散射点之间假定为真空，仅支持可见性区间的硬遮挡）、
多次散射引起的颜色 bleeding、以及时间/空间蓝噪声抖动等实时近似。需要
更高保真度时，应把本库的单次散射结果作为下界，或换用路径追踪/光子
映射类求解器。

## 测试

```
python -m unittest
```

覆盖：恒定密度 Beer–Lambert、解析指数衰减、两个相邻密度段及其边界
连续性、部分遮挡及遮挡边缘连续性、强前向散射（HG g=0.9）、重叠介质
不重复累加、容差收紧后的收敛性、细分预算上限和同输入确定性。
