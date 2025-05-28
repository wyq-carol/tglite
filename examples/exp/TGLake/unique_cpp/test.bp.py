import torch
import custom_unique

# 示例：获取唯一值
# tensor = torch.tensor([1, 2, 2, 3, 4, 4, 4])
tensor = torch.tensor([2, 2, 1, 3, 4, 4, 4])
unique_values, inverse_indices, counts = custom_unique.unique(tensor, return_inverse=True, return_counts=True)

print("Unique values:", unique_values)  # 输出：tensor([1, 2, 3, 4])
print("Inverse indices:", inverse_indices)  # 输出：tensor([0, 1, 1, 2, 3, 3, 3])
print("Counts:", counts)  # 输出：tensor([1, 2, 1, 3])