import triton
import triton.language as tl

# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_DIM": 32}, num_warps=2),
#         triton.Config({"BLOCK_DIM": 64}, num_warps=2),
#         triton.Config({"BLOCK_DIM": 128}, num_warps=4),
#     ],
#     key=["dim"]
# )
@triton.jit
def get_mem_data_kernel(
    indices_ptr,            # [N]
    data_status_ptr,        # [total_data_num] int32
    data_table_ptr,         # [total_data_num, 2]
    memory_ptr,             # [memory_size, dim]
    output_ptr,             # [N, dim]
    num_slots_per_block: tl.constexpr,
    dim: tl.constexpr,
    total_data_num: tl.constexpr,
    total_mem_row: tl.constexpr,
    stride_dt_row: tl.constexpr,  # data_table.stride(0)
    stride_dt_col: tl.constexpr,  # data_table.stride(1)
    stride_mem_row: tl.constexpr,
    stride_out_row: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)  # 当前行
    pid_d = tl.program_id(1)  # 特征维度块索引

    d = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    d_mask = d < dim

    index = tl.load(indices_ptr + pid_n)
    if index == -1 or index >= total_data_num:
        return

    valid = tl.load(data_status_ptr + index).to(tl.int1)
    if not valid:
        return

    block_id = tl.load(data_table_ptr + index * stride_dt_row + 0 * stride_dt_col)
    slot_id  = tl.load(data_table_ptr + index * stride_dt_row + 1 * stride_dt_col)
    block_id = block_id.to(tl.int64)

    pos = block_id * num_slots_per_block + slot_id
    if pos >= total_mem_row:
        return

    mem_offset = pos * stride_mem_row + d
    out_offset = pid_n * stride_out_row + d

    val = tl.load(memory_ptr + mem_offset, mask=d_mask, other=0)
    tl.store(output_ptr + out_offset, val, mask=d_mask)
