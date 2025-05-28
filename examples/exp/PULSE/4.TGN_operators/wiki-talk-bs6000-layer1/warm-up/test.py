import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 加载两个 PNG 图像
img1 = mpimg.imread('original.png')
img2 = mpimg.imread('optimized.png')

# 确保两个图像高度相同，如果不同则调整
if img1.shape[0] != img2.shape[0]:
    img2 = img2[:img1.shape[0], :, :]

# 创建一个新的画布
fig, ax = plt.subplots(figsize=(12, 6))  # 调整画布大小
ax.imshow(np.hstack((img1, img2)))
ax.axis('off')  # 关闭坐标轴

# 保存为 PDF，设置更高的DPI
plt.savefig('output.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
plt.close()

print("PDF 文件已生成：output.pdf")