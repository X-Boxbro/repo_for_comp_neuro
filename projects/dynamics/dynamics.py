import numpy as np
import matplotlib.pyplot as plt

class PhasePlane:
    def __init__(self, function_F=None, function_G=None, dimension=2,
                 step=1e-2, interval_length=2, grid_size=25):
        if dimension != 2:
            raise ValueError("PhasePlane only supports two-dimensional systems")

        self.dim=dimension
        #将幂函数用次数递减的两个系数列表来表示，先忽略其他非线性函数
        self.function_F=function_F
        self.function_G=function_G
        #用于画图的step
        self.step=step
        self.interval_lenth=interval_length
        self.grid_size=grid_size
        #每隔arrow_interval标记一下这个点的arrow并绘图绘制出箭头
        self.arrow_interval=max(1, int(self.grid_size / 5))
        #用于记录x,y的nullcline点的曲线
        self.curve_x=np.empty((0, 2))
        self.curve_y=np.empty((0, 2))
        #用于记录fixed point坐标
        self.fp=np.empty((0, 2))

    def _check_functions(self):
        if not callable(self.function_F) or not callable(self.function_G):
            raise ValueError("function_F and function_G must be callable")

    def _evaluate(self, x, y):
        self._check_functions()
        return float(self.function_F(x, y)), float(self.function_G(x, y))

    def _jacobian(self, x, y):
        delta = max(self.step, 1e-6)
        f_x1, g_x1 = self._evaluate(x + delta, y)
        f_x0, g_x0 = self._evaluate(x - delta, y)
        f_y1, g_y1 = self._evaluate(x, y + delta)
        f_y0, g_y0 = self._evaluate(x, y - delta)
        return np.array([
            [(f_x1 - f_x0) / (2 * delta), (f_y1 - f_y0) / (2 * delta)],
            [(g_x1 - g_x0) / (2 * delta), (g_y1 - g_y0) / (2 * delta)],
        ])

    def _newton_refine(self, point, tolerance=1e-8, max_iterations=30):
        point = np.asarray(point, dtype=float).copy()
        for _ in range(max_iterations):
            value = np.asarray(self._evaluate(*point))
            if np.linalg.norm(value) < tolerance:
                return point
            try:
                correction = np.linalg.solve(self._jacobian(*point), value)
            except np.linalg.LinAlgError:
                return None
            point -= correction
            if np.any(np.abs(point) > self.interval_lenth / 2):
                return None
        return point if np.linalg.norm(self._evaluate(*point)) < tolerance else None

    def find_fixed_points(self, tolerance=1e-6):
        """Find and return fixed points inside the plotting square."""
        self._check_functions()
        half_length = self.interval_lenth / 2
        coordinates = np.linspace(-half_length, half_length, self.grid_size)
        candidates = []
        values = np.empty((self.grid_size, self.grid_size, 2))

        for row, y in enumerate(coordinates):
            for column, x in enumerate(coordinates):
                values[row, column] = self._evaluate(x, y)

        for row in range(self.grid_size - 1):
            for column in range(self.grid_size - 1):
                cell = values[row:row + 2, column:column + 2]
                if np.max(np.linalg.norm(cell, axis=2)) == 0:
                    candidates.append((coordinates[column], coordinates[row]))
                elif np.min(cell[:, :, 0]) <= 0 <= np.max(cell[:, :, 0]) and \
                        np.min(cell[:, :, 1]) <= 0 <= np.max(cell[:, :, 1]):
                    candidates.append((
                        (coordinates[column] + coordinates[column + 1]) / 2,
                        (coordinates[row] + coordinates[row + 1]) / 2,
                    ))

        fixed_points = []
        for candidate in candidates:
            point = self._newton_refine(candidate, tolerance=tolerance)
            if point is None or np.linalg.norm(self._evaluate(*point)) > tolerance:
                continue
            duplicate_tolerance = max(1e-4, 2 * self.step)
            if not any(
                np.linalg.norm(point - old_point) < duplicate_tolerance
                for old_point in fixed_points
            ):
                fixed_points.append(point)

        self.fp = np.array(fixed_points) if fixed_points else np.empty((0, 2))
        return self.fp

    def analyze_fixed_points(self, fixed_points=None, tolerance=1e-7):
        if fixed_points is None:
            fixed_points = self.find_fixed_points()

        analysis = []
        for point in np.asarray(fixed_points):
            jacobian = self._jacobian(*point)
            eigenvalues = np.linalg.eigvals(jacobian)
            real_parts = np.real(eigenvalues)
            if real_parts.max() < -tolerance:
                classification = "stable"
            elif real_parts.min() > tolerance:
                classification = "unstable"
            elif real_parts.min() < -tolerance < real_parts.max():
                classification = "saddle"
            elif np.any(np.abs(real_parts) <= tolerance):
                classification = "non-hyperbolic"
            else:
                classification = "center/focus"

            analysis.append({
                "point": np.asarray(point),
                "jacobian": jacobian,
                "eigenvalues": eigenvalues,
                "classification": classification,
            })
        return analysis


    def plot_plane(self):
        #解函数方程F,G让他们等于0,F=0得到x'关于(x,y)的轨迹，G=0得到y'关于(x,y)的轨迹，填入self.curve中

        #将self.curve的交点记录到self.fp中

        #绘图，对于x属于[-interval_length/2,+interval_length/2],y属于[-interval_length/2,+interval_length/2]的方形区间，每个点进行求x'和y'，绘制一个小arrow指向梯度方向。对于fixedpoint要通过正负dx,正负dy的变化看x'和y‘变化以画四个arrow，用eignvalue进行稳定性分析
        self._check_functions()
        half_length = self.interval_lenth / 2
        coordinates = np.linspace(-half_length, half_length, self.grid_size)
        x_grid, y_grid = np.meshgrid(coordinates, coordinates)
        f_grid = np.empty_like(x_grid)
        g_grid = np.empty_like(y_grid)

        for row in range(self.grid_size):
            for column in range(self.grid_size):
                f_grid[row, column], g_grid[row, column] = self._evaluate(
                    x_grid[row, column], y_grid[row, column]
                )

        self.curve_x = np.column_stack((x_grid.ravel(), f_grid.ravel()))
        self.curve_y = np.column_stack((y_grid.ravel(), g_grid.ravel()))
        fixed_points = self.find_fixed_points()
        analysis = self.analyze_fixed_points(fixed_points)

        speed = np.hypot(f_grid, g_grid)
        speed[speed == 0] = 1
        fig, axis = plt.subplots(figsize=(8, 8))
        axis.streamplot(
            x_grid, y_grid, f_grid / speed, g_grid / speed,
            density=1.2, color=np.log1p(speed), cmap="viridis"
        )
        axis.contour(x_grid, y_grid, f_grid, levels=[0], colors="tab:blue")
        axis.contour(x_grid, y_grid, g_grid, levels=[0], colors="tab:orange")

        for item in analysis:
            x, y = item["point"]
            axis.scatter(x, y, color="black", zorder=3)
            axis.annotate(item["classification"], (x, y), xytext=(5, 5),
                          textcoords="offset points")

        axis.set_xlim(-half_length, half_length)
        axis.set_ylim(-half_length, half_length)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title("Two-dimensional phase plane")
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig, axis, fixed_points, analysis


# Keep the original class spelling available for existing code.
Phase_plane = PhasePlane


if __name__ == "__main__":
    phase_plane = PhasePlane(
    function_F=lambda x, y: np.sin(x) - y,
    function_G=lambda v, w: v - v**3 / 3 - w,
    interval_length=4,
    grid_size=50
    )
    figure, axis, fixed_points, analysis = phase_plane.plot_plane()
    plt.show()