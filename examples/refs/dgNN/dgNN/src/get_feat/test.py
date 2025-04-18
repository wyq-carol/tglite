import torch
import get_feat
import time
if __name__ == "__main__":
# @triton.jit
# def gather_memory_kernel(
#     indices_ptr,           # [N]
#     data_table_ptr,        # [total_data_num, 2]
#     memory_ptr,            # [memory_size, dim]
#     output_ptr,            # [N, dim]
#     num_slots: tl.constexpr,
#     dim: tl.constexpr,
#     stride_dt_row: tl.constexpr,
#     stride_dt_col: tl.constexpr,
#     stride_mem_row: tl.constexpr,
#     stride_out_row: tl.constexpr,
#     BLOCK_DIM: tl.constexpr,
# ):
#     pid_n = tl.program_id(0)  # 当前行索引
#     pid_d = tl.program_id(1)  # 特征维度块索引

#     d = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
#     d_mask = d < dim

#     index = tl.load(indices_ptr + pid_n)

#     # 取出 block_id 和 slot_id
#     block_id = tl.load(data_table_ptr + index * stride_dt_row + 0 * stride_dt_col)
#     slot_id  = tl.load(data_table_ptr + index * stride_dt_row + 1 * stride_dt_col)
#     block_id = block_id.to(tl.int64)
#     pos = block_id * num_slots + slot_id

#     mem_offset = pos * stride_mem_row + d
#     out_offset = pid_n * stride_out_row + d

#     val = tl.load(memory_ptr + mem_offset, mask=d_mask, other=0)
#     tl.store(output_ptr + out_offset, val, mask=d_mask)

    indices = torch.load("indices.pt") # torch.Size([5888])
    data_table = torch.load("data_table.pt") # torch.Size([7833140, 2])
    memory = torch.load("memory.pt") # torch.Size([46820125, 172])
    num_slots = torch.load("num_slots.pt") # 25
    # import pdb;pdb.set_trace()

    start = time.time()
    torch.cuda.synchronize()
    pos = data_table[:, 0][indices] * num_slots + data_table[:, 1][indices]
    output_std = memory[pos]
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time taken 0: {end - start} seconds")

    start = time.time()
    torch.cuda.synchronize()
    output = get_feat.get_feat(num_slots, indices, data_table, memory)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time taken 1: {end - start} seconds")
    print(output.shape)

    assert torch.allclose(output_std, output)
    # print(output)
