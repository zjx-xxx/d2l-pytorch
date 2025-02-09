## 常用函数

### `d2l.set_figsize()`
功能: 用于设置 Matplotlib 图形的默认尺寸（宽度和高度）。这是 `d2l` 库的一个辅助函数，可以方便地调整图表的大小。

### `d2l.plt`
`d2l.plt` 是 `matplotlib.pyplot` 的快捷别名（别名通常是 `import matplotlib.pyplot as plt`），在 `d2l` 中已经将其封装为 `plt`，方便用户直接调用作图功能。
#### `d2l.plt.subplots(num_rows, num_cols, figsize=figsize)`
- `subplots()`:
	- 是 `matplotlib.pyplot` 中的一个函数，用于创建一个包含多个子图的图形。你可以指定行数和列数，以便在同一个图形中显示多个图表。
	- 它返回一个图形对象（`fig`）和一个包含多个子图（axes）的数组（`axs`）。
- - **`num_rows, num_cols`**:
    - 这些是子图的行数和列数，决定了最终图形中子图的排列方式。比如 `num_rows=2, num_cols=3` 会创建一个 2 行 3 列的子图网格。
- **`figsize=figsize`**:
    - `figsize` 是图形的大小，通常是一个元组 `(宽度, 高度)`，单位是英寸。比如 `figsize=(10, 8)` 会创建一个宽 10 英寸，高 8 英寸的图形。
    - 这个参数是可选的，默认大小会根据需要设置，但你可以指定它来调整图形的大小。