# metrics.py 使用说明

## 环境

使用项目自带的虚拟环境 `my312`，不要用系统 Python：

```bash
my312/bin/python metrics.py <reference_image> <test_image>
```

## 基本用法

```bash
# 单次评估
my312/bin/python metrics.py to.png my.png

# 屏蔽 torchvision 的 pretrained 弃用警告（推荐）
my312/bin/python metrics.py to.png my.png 2>/dev/null

# 连续评估多张图
my312/bin/python metrics.py to.png new_my_no_border.png 2>/dev/null && \
my312/bin/python metrics.py to.png new_my_border.png 2>/dev/null && \
my312/bin/python metrics.py to.png my.png 2>/dev/null
```

## 注意事项

### 图像对齐
- reference 和 test 图片必须**严格对齐**（内容位置一致），否则 SSIM 和色彩保真度会严重失真
- 尺寸不同时脚本会自动 resize test 图到 reference 尺寸，但这不等于内容对齐
- A 组图像：`to.png` / `my.png`（1920×1186，严格对齐）
- B 组图像：`1-to.png` / `1-my.png`（1476×890，严格对齐）

### 参数说明
- 第一个参数：**reference（参考图）**，即论文中的原画 `to.png`
- 第二个参数：**test（待评估图）**，即自己生成的结果

### 各指标说明

| 指标 | 说明 |
|------|------|
| LPIPS | VGG11 各 ReLU 层特征的余弦相似度均值，范围 [0,1]，越高越相似 |
| SSIM | 5×5 窗口的结构相似度，范围 [0,1] |
| 色彩保真度 | BGR 像素平均绝对差 / 255，结果 = 1 - mean_diff/255 |
| 笔触纹理 | LBP（radius=3, n_points=24, uniform）直方图交集 |
| 表面纹理 | min(拉普拉斯方差_ref, 拉普拉斯方差_test) / max(...) |
| 光泽度 | HSV-V 直方图 Bhattacharyya（权重0.55）+ 高亮像素比例差（权重0.45） |

### 论文目标值参考

| 指标 | A:Ours | B:秋山问道图 |
|------|--------|-------------|
| LPIPS | 0.55 | 0.29 |
| SSIM | 0.74 | 0.33 |
| 色彩保真度 | 0.96 | 0.81 |
| 笔触纹理 | 0.98 | 0.81 |
| 表面纹理 | 0.78 | 0.16 |
| 光泽度 | 0.99 | 0.94 |

### 依赖安装

```bash
pip install opencv-python scikit-image torch torchvision pillow numpy
```
