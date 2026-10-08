这个项目专注于通过分析不同模型的动态过程，理解计算神经科学中的动力学行为。

分析的数学技巧：相平面方法。

目前只专注于二维动力学系统：

$$
\frac{dx}{dt}=F(x,y),\qquad \frac{dy}{dt}=G(x,y)
$$

二维系统可以通过相平面直观展示系统的动态行为。

## 当前功能

- 绘制二维系统的向量场和流线
- 绘制 $F(x,y)=0$ 与 $G(x,y)=0$ 两条 nullcline
- 数值寻找 fixed points
- 通过 Jacobian 的特征值分析 fixed point 的局部稳定性
- 支持输入非线性的 $F$ 和 $G$

## 基本使用

```python
from dynamics import PhasePlane
import matplotlib.pyplot as plt
import numpy as np

phase_plane = PhasePlane(
	function_F=lambda x, y: np.sin(x) - y,
	function_G=lambda x, y: x - x**3 / 3 - y,
	interval_length=4,
	grid_size=50,
)

figure, axis, fixed_points, analysis = phase_plane.plot_plane()
print(fixed_points)
print(analysis)
plt.show()
```

其中 `function_F` 和 `function_G` 分别表示 $dx/dt$ 和 $dy/dt$。

## 当前限制

- 目前只支持二维系统
- fixed point 通过数值方法寻找，复杂系统可能漏检
- 稳定性分析是 fixed point 附近的局部分析