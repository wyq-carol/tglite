import torch
import math
import time
import nvtx
import triton
import triton.language as tl

import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, '../../examples/refs/dgNN/dgNN/src/get_feat'))
sys.path.append(os.path.join(this_dir, '../../examples/refs/dgNN/dgNN/src/concat_mail'))
import get_feat
import concat_mail

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_DIM": 32}, num_warps=2),
        triton.Config({"BLOCK_DIM": 64}, num_warps=2),
        triton.Config({"BLOCK_DIM": 128}, num_warps=4),
    ],
    key=["dim"]
)
@triton.jit
def get_mem_data_kernel(
    indices_ptr,            # [N]
    data_status_ptr,        # [total_data_num] int32
    data_table_ptr,         # [total_data_num, 2]
    memory_ptr,             # [memory_size, dim]
    output_ptr,             # [N, dim]
    num_slots_per_block: tl.constexpr,
    dim: tl.constexpr,
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
    if index < 0:
        return

    valid = tl.load(data_status_ptr + index).to(tl.int1)

    block_id = tl.load(data_table_ptr + index * stride_dt_row + 0 * stride_dt_col)
    slot_id  = tl.load(data_table_ptr + index * stride_dt_row + 1 * stride_dt_col)

    block_id = block_id.to(tl.int64)
    pos = block_id * num_slots_per_block + slot_id
    mem_offset = pos * stride_mem_row + d
    out_offset = pid_n * stride_out_row + d

    if pos < 0 or not valid:
        val = d * 0.0
    else:
        val = tl.load(memory_ptr + mem_offset, mask=d_mask, other=0)
    tl.store(output_ptr + out_offset, val, mask=d_mask)

import triton
import triton.language as tl

@triton.jit
def gather_memory_kernel(
    indices_ptr,           # [N]
    data_table_ptr,        # [total_data_num, 2]
    memory_ptr,            # [memory_size, dim]
    output_ptr,            # [N, dim]
    num_slots: tl.constexpr,
    dim: tl.constexpr,
    stride_dt_row: tl.constexpr,
    stride_dt_col: tl.constexpr,
    stride_mem_row: tl.constexpr,
    stride_out_row: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)  # 当前行索引
    pid_d = tl.program_id(1)  # 特征维度块索引

    d = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    d_mask = d < dim

    index = tl.load(indices_ptr + pid_n)

    # 取出 block_id 和 slot_id
    block_id = tl.load(data_table_ptr + index * stride_dt_row + 0 * stride_dt_col)
    slot_id  = tl.load(data_table_ptr + index * stride_dt_row + 1 * stride_dt_col)
    block_id = block_id.to(tl.int64)
    pos = block_id * num_slots + slot_id

    mem_offset = pos * stride_mem_row + d
    out_offset = pid_n * stride_out_row + d

    val = tl.load(memory_ptr + mem_offset, mask=d_mask, other=0)
    tl.store(output_ptr + out_offset, val, mask=d_mask)


class BlockPool:
    """ Unity BlockPool """
    def __init__(self, total_mem_gb: int = 20, block_elements: int = 4300, device: str = "cuda"):
        self.device = torch.device(device)
        self.block_elements = block_elements 
        self.element_size = torch.finfo(torch.float32).bits // 8  # 4 bytes
        
        bytes_per_block = block_elements * self.element_size
        self.total_blocks = int(total_mem_gb * 1024**3) // bytes_per_block
        
        self.memory = torch.empty(
            self.total_blocks * block_elements,
            dtype=torch.float32,
            device=self.device
        )
        
        self.block_status = torch.zeros(self.total_blocks, dtype=torch.bool, device=self.device)

    def allocate_block(self, num_needed) -> torch.Tensor:
        """ return block view and block_id """
        free_blocks = torch.where(self.block_status == False)[0]
        if (free_blocks.size(0) < num_needed):
            raise ValueError(f"ERROR: NO FREE SPACES : free blocks: {free_blocks.size(0)}, need: {num_needed}")
            return None
        free_blocks = free_blocks[:num_needed]
        self.block_status[free_blocks] = True     
        return free_blocks 
    
    
    def free_block(self, block_id: int):
        """ free block by block_id """
        pass
            
    def print_status(self):
        print(f"BlockPool: ")
        print(f"  Total Blocks: {self.total_blocks}")
        print(f"  Block Elements: {self.block_elements}")
        print(f"  Element Size: {self.element_size} bytes")
        print(f"  Total Memory: {self.total_blocks * self.block_elements * self.element_size / 1024**3:.2f} GB")
        print(f"  Free Blocks: {self.total_blocks - torch.sum(self.block_status)}")
        print(f"  Used Blocks: {torch.sum(self.block_status)}")
        print(f"  Memory Usage: {torch.sum(self.block_status) * self.block_elements * self.element_size / 1024**3:.2f} GB")
        
class BlockManager:
    def __init__(self, pool: BlockPool, source_tensor: torch.Tensor, feature_size: int, max_idx: int, init_blocks = 4000, DEBUG = False):
        self.DEBUG = DEBUG
        if pool.block_elements % feature_size != 0:
            raise ValueError(f"Feature size {feature_size} must divide block elements {pool.block_elements}")
        self.pool = pool
        self.feature_size = feature_size
        self.num_slots = pool.block_elements // feature_size
        self.num_blocks = init_blocks
        self.max_idx = max_idx
        self.source_tensor = source_tensor
        
        # a data table (N , 2), True mains valid
        self.data_table = torch.zeros((self.max_idx, 2), dtype=torch.int32, device=pool.device) 
        self.data_status = torch.zeros(self.max_idx, dtype=torch.bool, device=pool.device)
        
        # a space table (block num , slot num), True mains used        
        free_blocks = pool.allocate_block(self.num_blocks)
        blockid = free_blocks.repeat_interleave(self.num_slots).to(torch.int32).to(self.pool.device) 
        slotid = torch.tile(torch.arange(self.num_slots, dtype=torch.int32, device=self.pool.device), (self.num_blocks,)) 
        self.space_table = torch.stack((blockid, slotid), dim=1).to(self.pool.device)
        self.space_status = torch.zeros((self.num_blocks, self.num_slots), dtype=torch.bool, device=self.pool.device)
        
        self.memory = self.pool.memory.reshape(-1, self.feature_size)

    def next_power_size(self, new_need : int):
        used_blocks = int(torch.ceil(self.space_status.sum() / self.num_slots).item())
        return (1 << (math.ceil(math.log2(new_need / self.num_slots + used_blocks)))) 
        
    
    def resize (self, need_slots: int):
        """ resize block manager """
        alloc_blocks = (self.next_power_size(need_slots) - self.num_blocks)
        free_blocks = self.pool.allocate_block(alloc_blocks)
        blockid = free_blocks.repeat_interleave(self.num_slots).to(torch.int32).to(self.pool.device) 
        slotid = torch.tile(torch.arange(self.num_slots, dtype=torch.int32, device=self.pool.device), (alloc_blocks, )) 
        
        new_table = torch.stack((blockid, slotid), dim=1).to(self.pool.device)
        self.space_table = torch.cat((self.space_table, new_table), dim=0)
        
        new_status = torch.zeros((free_blocks.size(0), self.num_slots), dtype=torch.bool, device=self.pool.device)
        self.space_status = torch.cat((self.space_status, new_status), dim=0)

    @torch.compile
    def get_data_batch_ori(self, indices: torch.Tensor) -> torch.Tensor:
        """ if there is possiblity to load data not in buffer, use sorce_tensor to load data """
        # 找出 -1 的位置
        mask_neg1 = (indices == -1)
        # 找出不是 -1 的有效索引
        valid_indices = indices[~mask_neg1]
        # 检查这些有效索引中有没有尚未加载的（即 data_status 为 False）
        invalid_indices = valid_indices[~self.data_status[valid_indices]]
        if invalid_indices.size(0) > 0:
            if self.source_tensor is not None:
                self.copy_from_cpu_batch(invalid_indices)
            else:
                print(f"STRANGE: no source tensor, cannot load data")
                return None
        
        # 从 data_table 和 memory 中提取
        info = torch.zeros((indices.size(0), 2), dtype=torch.int32, device=indices.device)
        info[~mask_neg1] = self.data_table[valid_indices]

        pos = info[:, 0] * self.num_slots + info[:, 1]

        data_copy = torch.zeros((indices.size(0), self.memory.size(1)), dtype=self.memory.dtype, device=self.memory.device)
        data_copy[~mask_neg1] = self.memory[pos[~mask_neg1]]
        return data_copy
    
    # def get_data_batch(self, indices: torch.Tensor) -> torch.Tensor:
    #     """ if there is possiblity to load data not in buffer, use sorce_tensor to load data """
    #     # 找出 -1 的位置
    #     mask_neg1 = (indices == -1)
    #     # 找出不是 -1 的有效索引
    #     valid_indices = indices[~mask_neg1]
    #     # 从 data_table 和 memory 中提取
    #     info = torch.zeros((indices.size(0), 2), dtype=torch.int32, device=indices.device)
    #     info[~mask_neg1] = self.data_table[valid_indices]

    #     pos = info[:, 0] * self.num_slots + info[:, 1]

    #     data_copy = torch.zeros((indices.size(0), self.memory.size(1)), dtype=self.memory.dtype, device=self.memory.device)
    #     data_copy[~mask_neg1] = self.memory[pos[~mask_neg1]]
    #     return data_copy
    
    def get_data_batch(self, indices: torch.Tensor) -> torch.Tensor:
        return get_feat.get_feat(self.num_slots, indices, self.data_table, self.memory)
    
    # @torch.compile
    #TODO: nfeat efeat
    # def get_data_batch(self, indices: torch.Tensor) -> torch.Tensor:
    #     """ if there is possiblity to load data not in buffer, use sorce_tensor to load data """
    #     # info = self.data_table[indices]
    #     # pos = info[:, 0] * self.num_slots + info[:, 1]
    #     # data_copy = self.memory[pos]
    #     with nvtx.annotate("get_data_batch inner"):
    #         BLOCK_DIM = 64
    #         N, dim = indices.shape[0], self.memory.shape[1]

    #         grid = lambda meta: (N, (dim + meta['BLOCK_DIM'] - 1) // meta['BLOCK_DIM'])

    #         # if self.feature_size == 172:
    #         #     torch.save(self.data_table, "/home/tglite/examples/refs/dgNN/dgNN/src/get_feat/data_table.pt")
    #         #     torch.save(indices, "/home/tglite/examples/refs/dgNN/dgNN/src/get_feat/indices.pt")
    #         #     torch.save(self.memory, "/home/tglite/examples/refs/dgNN/dgNN/src/get_feat/memory.pt")
    #         #     torch.save(self.num_slots, "/home/tglite/examples/refs/dgNN/dgNN/src/get_feat/num_slots.pt")
    #         #     exit()

    #         output_tensor = torch.empty((indices.shape[0], dim), device=self.memory.device)
    #         with nvtx.annotate("get_data_batch kernal"):
    #             gather_memory_kernel[grid](
    #                 indices,
    #                 self.data_table,
    #                 self.memory,
    #                 output_tensor,  # torch.empty((N, dim), device=...)
    #                 num_slots=self.num_slots,
    #                 dim=dim,
    #                 stride_dt_row=self.data_table.stride(0),
    #                 stride_dt_col=self.data_table.stride(1),
    #                 stride_mem_row=self.memory.stride(0),
    #                 stride_out_row=output_tensor.stride(0),
    #                 BLOCK_DIM=BLOCK_DIM
    #             )
    #     return output_tensor
    

    def update_data_batch(self, indices: torch.Tensor, data: torch.Tensor):
        """ set data in batch """
        invalid_indices = self.check_data_valid(indices)
        if (invalid_indices.size(0) > 0):
            # print(f"invalid indices: {invalid_indices}")
            self.alloc_for_batch(invalid_indices)
            self.data_status[invalid_indices] = True
            
        info = self.data_table[indices]
        pos = info[:, 0] * self.num_slots + info[:, 1]
        self.memory[pos] = data.to(self.pool.device)
        
    def check_data_valid(self, indices: torch.Tensor) -> torch.Tensor:
        """ check if indices are valid , return invalid indices """
        valid = self.data_status[indices]
        invalid = indices[~valid]
        return invalid
    
    def alloc_for_batch(self, indices: torch.Tensor):
        """ waring : this is a !!!INSIDE API!!!, please make sure all the indices in data_table are valid, u can use check_data_valid to flit it """
        # Step 1 : check fot valid space
        free_spaces = self.free_spaces()
        if (free_spaces.size(0) < indices.size(0)):
            # if (self.DEBUG):
            # print(f"free spaces: {free_spaces.size(0)}, indices: {indices.size(0)}")
            self.resize(indices.size(0))
            free_spaces = self.free_spaces()
            
            # return
        # Step 2 : alloc for indices
        alloc = free_spaces[:indices.size(0)]
        self.space_status[alloc[:, 0], alloc[:, 1]] = True
        
        self.data_table[indices] = self.space_table[alloc[:, 0] * self.num_slots +  alloc[:, 1]]
        
        free_spaces = self.free_spaces()
        # print(f"free spaces: {free_spaces.size(0)}, indices: {indices.size(0)}")
        
    def free_spaces(self) -> torch.Tensor:
        """ free spaces, its blockid """
        free_space = (~self.space_status).nonzero().to(torch.int32)
        return free_space
        


        
    def copy_from_cpu_batch(self, indices: torch.Tensor):
        """ copy data with index in indices from cpu to gpu """
        if indices.is_cpu:
            indices_gpu = indices.to(device='cuda')
        else:
            indices_gpu = indices
            indices = indices_gpu.to(device='cpu')
        need_to_load = self.check_data_valid(indices_gpu)
        if (need_to_load.size(0) == 0):
            # print(f"all have loaded, pass load phase") 
            return 
        self.alloc_for_batch(need_to_load)
        # target_places = self.get_data_batch_forced(indices_gpu)
        
        info = self.data_table[indices_gpu]
        pos = info[:, 0] * self.num_slots + info[:, 1]
        self.memory[pos] = self.source_tensor[indices].to('cuda')
        self.data_status[indices_gpu] = True
        
    def print_status(self):
        print(f"BlockManager: ")
        print(f"  Total Blocks: {self.num_blocks}")
        print(f"  Block Elements: {self.num_slots}")
        print(f"  Element Size: {self.feature_size} bytes")
        print(f"  Total Memory: {self.num_blocks * self.num_slots * self.feature_size / 1024**3:.2f} GB")
        print(f"  Free Blocks: {self.num_blocks - torch.sum(self.space_status)}")
        print(f"  Used Blocks: {torch.sum(self.space_status)}")
        print(f"  Memory Usage: {torch.sum(self.space_status) * self.num_slots * self.feature_size / 1024**3:.2f} GB")

class MemMailManager:
    def __init__(self, pool: BlockPool, feature_size: int, max_idx: int, efeat_manager: BlockManager,init_blocks = 4000, blocks_for_cache = 1000, DEBUG = False):
        self.DEBUG = DEBUG
        if pool.block_elements % feature_size != 0:
            raise ValueError(f"Feature size {feature_size} must divide block elements {pool.block_elements}")
        self.pool = pool
        self.feature_size = feature_size
        self.num_slots_per_block = pool.block_elements // feature_size
        self.num_blocks_got = init_blocks
        self.max_idx = max_idx
        self.cache_table_len = blocks_for_cache * self.num_slots_per_block

        
        # latest data table (N , 2), True mains valid
        self.data_table = torch.zeros((self.max_idx, 2), dtype=torch.int32, device=pool.device) 
        self.data_status = torch.zeros(self.max_idx, dtype=torch.bool, device=pool.device) # prepare for future feature, now not used
        self.data_ref = torch.zeros(self.max_idx, dtype=torch.int32, device=pool.device)
        
        # cache table , True mains valid
        self.cache_table = torch.zeros((self.cache_table_len, 2), dtype=torch.int32, device=pool.device)
        # self.cache_status = torch.zeros(self.cache_table_len, dtype=torch.bool, device=pool.device)
        self.cache_ref = torch.zeros(self.cache_table_len, dtype=torch.int32, device=pool.device)
        
        # mailbox table, should be putted on gpu all, so no status
        self.mailbox_table = torch.full((self.max_idx, 2), -1, dtype=torch.int32, device=pool.device)
        # self.cache_mask = torch.zeros((self.max_idx, 2), dtype=torch.bool, device=pool.device)
        # self.mailbox_status = torch.zeros(self.max_idx, dtype=torch.bool, device=pool.device)
        
        # a space table (block num , slot num per block), True mains used        
        free_blocks = pool.allocate_block(self.num_blocks_got)
        blockid = free_blocks.repeat_interleave(self.num_slots_per_block).to(torch.int32).to(self.pool.device) 
        slotid = torch.tile(torch.arange(self.num_slots_per_block, dtype=torch.int32, device=self.pool.device), (self.num_blocks_got,)) 
        self.space_table = torch.stack((blockid, slotid), dim=1).to(self.pool.device)
        self.space_status = torch.zeros((self.num_blocks_got, self.num_slots_per_block), dtype=torch.bool, device=self.pool.device)
        self.memory = self.pool.memory.reshape(-1, self.feature_size)
        
        self.mem_time = torch.zeros(self.max_idx, dtype=torch.float32, device=pool.device)
        self.mail_time = torch.zeros(self.max_idx, dtype=torch.float32, device=pool.device)
        self.mail_efeat_table = torch.full((self.max_idx,), -1, dtype=torch.int32, device=pool.device)
        
        self.efeat_manager = efeat_manager
        self.counter = 0

    def reset(self):
        """ reset mem manager """
        self.data_table = torch.zeros((self.max_idx, 2), dtype=torch.int32, device=self.pool.device)
        self.data_status = torch.zeros(self.max_idx, dtype=torch.bool, device=self.pool.device)
        self.data_ref = torch.zeros(self.max_idx, dtype=torch.int32, device=self.pool.device)
        
        self.cache_table = torch.zeros((self.cache_table_len, 2), dtype=torch.int32, device=self.pool.device)
        self.cache_ref = torch.zeros(self.cache_table_len, dtype=torch.int32, device=self.pool.device)
        
        self.mailbox_table = torch.full((self.max_idx, 2), -1, dtype=torch.int32, device=self.pool.device)
        self.mail_time = torch.zeros(self.max_idx, dtype=torch.float32, device=self.pool.device)
        self.mail_efeat_table = torch.full((self.max_idx,), -1, dtype=torch.int32, device=self.pool.device)

        self.mem_time = torch.zeros(self.max_idx, dtype=torch.float32, device=self.pool.device)
        self.space_status = torch.zeros((self.num_blocks_got, self.num_slots_per_block), dtype=torch.bool, device=self.pool.device)
        
             
        

    def next_power_size(self, new_need : int):
        used_blocks = int(torch.ceil(self.space_status.sum() / self.num_slots_per_block).item())
        return (1 << (math.ceil(math.log2(new_need / self.num_slots_per_block + used_blocks)))) 
    
    def next_cache_size(self, new_need : int):
        used_blocks = int(torch.ceil(self.space_status.sum() / self.num_slots_per_block).item())
        return (1 << (math.ceil(math.log2(new_need / self.num_slots_per_block + used_blocks)))) 
        
    
    def resize (self, need_slots: int):
        """ resize block manager """
        alloc_blocks = self.next_power_size(need_slots) - self.num_blocks_got
        free_blocks = self.pool.allocate_block(alloc_blocks)
        blockid = free_blocks.repeat_interleave(self.num_slots_per_block).to(torch.int32).to(self.pool.device) 
        slotid = torch.tile(torch.arange(self.num_slots_per_block, dtype=torch.int32, device=self.pool.device), (alloc_blocks, )) 
        
        new_table = torch.stack((blockid, slotid), dim=1).to(self.pool.device)
        self.space_table = torch.cat((self.space_table, new_table), dim=0)
        
        new_status = torch.zeros((free_blocks.size(0), self.num_slots_per_block), dtype=torch.bool, device=self.pool.device)
        self.space_status = torch.cat((self.space_status, new_status), dim=0)


    def get_mem_data_ori(self, indices: torch.Tensor) -> torch.Tensor:
        """ 现在没有实现CPU交换, 请确保数据都在GPU上且有效(第0轮更新前的初始数据被全部视为无效) """
        # 找出 -1 的位置
        mask_neg1 = (indices == -1)
        # 找出不是 -1 的有效索引
        valid_indices = indices[~mask_neg1]
        # 检查这些有效索引中有没有尚未加载的（即 data_status 为 False）
        invalid_indices = valid_indices[~self.data_status[valid_indices]]
        if invalid_indices.size(0) > 0:
            # print(f"STRANGE : invalid indices: {invalid_indices}")  
            return None
        # 从 data_table 和 memory 中提取
        info = torch.zeros((indices.size(0), 2), dtype=torch.int32, device=indices.device)
        info[~mask_neg1] = self.data_table[valid_indices]
        pos = info[:, 0] * self.num_slots_per_block + info[:, 1]
        data_copy = torch.zeros((indices.size(0), self.memory.size(1)), dtype=self.memory.dtype, device=self.memory.device)
        data_copy[~mask_neg1] = self.memory[pos[~mask_neg1]]

        return data_copy
    
    
    def get_mem_data(self, indices: torch.Tensor) -> torch.Tensor:
        with nvtx.annotate("get mem", color="purple"):
            assert indices.device.type == "cuda"
            N = indices.shape[0]
            dim = self.memory.shape[1]

            output = torch.empty((N, dim), dtype=self.memory.dtype, device=indices.device)

            grid = lambda meta: (
                N,
                triton.cdiv(dim, meta["BLOCK_DIM"]),
            )

            
            get_mem_data_kernel[grid](
                indices_ptr=indices,  # shape [N]
                data_status_ptr=self.data_status.to(torch.int8),  # shape [total_data_num]
                data_table_ptr=self.data_table,  # shape [total_data_num, 2]
                memory_ptr=self.memory,          # shape [memory_size, dim]
                output_ptr=output,               # shape [N, dim]
                num_slots_per_block=self.num_slots_per_block,
                dim=dim,
                stride_dt_row=2,
                stride_dt_col=1,  
                stride_mem_row=self.memory.stride(0),
                stride_out_row=self.memory.stride(0),
            )
                
            
        return output

    def update_mem_batch(self, indices: torch.Tensor, data: torch.Tensor, time:torch.Tensor,  up_mailbox_uniq: torch.Tensor = None, up_mailbox_nbr: torch.Tensor = None):
        """ update mem, if u dont tell me mailbox update info, i will conservatively dump to cache """
        # not stored yet, no need to dump to cache
        invalid_indices, valid_indices = self.check_data_valid(indices)
        if (invalid_indices.size(0) > 0):
            # print(f"invalid indices: {invalid_indices}")
            self.alloc_for_batch(invalid_indices)
            self.data_status[invalid_indices] = True

        # check for dump to cache
        if (up_mailbox_uniq == None):
            dump_indices = valid_indices[self.data_ref[valid_indices] > 0]
        else:            
            drop_lines_mailbox = self.mailbox_table[up_mailbox_uniq]
            mask = drop_lines_mailbox >= 0
            filtered = drop_lines_mailbox[mask]
            unique, counts = torch.unique(filtered, return_counts=True)
            
            mask_valid = unique < self.max_idx
            non_cached_indices = unique[mask_valid]
            non_cached_counts = counts[mask_valid]

            cached_indices = unique[~mask_valid] - self.max_idx
            cached_counts = counts[~mask_valid]
            
            self.data_ref[non_cached_indices] -= non_cached_counts
            self.cache_ref[cached_indices] -= cached_counts
            
            
            
            dump_indices = valid_indices[self.data_ref[valid_indices] > 0]
            unique, counts = torch.unique(torch.cat((up_mailbox_nbr, up_mailbox_uniq)), return_counts=True)        
            self.data_ref[unique] += counts

        self.dump_to_cache(dump_indices)            

        invalid_indices, _ = self.check_data_valid(indices)
        if (invalid_indices.size(0) > 0):
            # print(f"invalid indices: {invalid_indices}")
            self.alloc_for_batch(invalid_indices)
            self.data_status[invalid_indices] = True
            
        info = self.data_table[indices]
        pos = info[:, 0] * self.num_slots_per_block + info[:, 1]
        self.memory[pos] = data.to(self.pool.device)
        self.mem_time[indices] = time
        
    def check_data_valid(self, indices: torch.Tensor) -> torch.Tensor:
        """ check if indices are valid , return invalid indices """
        valid = self.data_status[indices]
        invalid = indices[~valid]
        valid = indices[valid]
        return invalid, valid
    
    def alloc_for_batch(self, indices: torch.Tensor):
        """ waring : this is a !!!INSIDE API!!!, please make sure all the indices in data_table are valid, u can use check_data_valid to flit it """
        # Step 1 : check fot valid space
        free_spaces = self.free_spaces()
        if (free_spaces.size(0) < indices.size(0)):
            # if (self.DEBUG):
            # print(f"free spaces: {free_spaces.size(0)}, indices: {indices.size(0)}")
            self.resize(indices.size(0))
            free_spaces = self.free_spaces()
            
            # return
        # Step 2 : alloc for indices
        alloc = free_spaces[:indices.size(0)]
        self.space_status[alloc[:, 0], alloc[:, 1]] = True
        
        self.data_table[indices] = self.space_table[alloc[:, 0] * self.num_slots_per_block +  alloc[:, 1]]
        
        free_spaces = self.free_spaces()
        # print(f"free spaces: {free_spaces.size(0)}, indices: {indices.size(0)}")
        
    def resize_cache(self, need_slots: int):
        """ resize cache """
        # free_spaces = self.free_spaces()
        # if (free_spaces.size(0) < need_slots):
        #     print(f"free slots: {free_spaces.size(0)}, need: {need_slots}")
        #     return
        # alloc = free_spaces[:need_slots]
        # self.space_status[alloc[:, 0], alloc[:, 1]] = True
        self.cache_table = torch.cat((self.cache_table, torch.zeros((need_slots, 2), dtype=torch.int32, device=self.pool.device)))
        self.cache_ref = torch.cat((self.cache_ref, torch.zeros(need_slots, dtype=torch.int32, device=self.pool.device)), dim=0)
         
        
        
    def dump_to_cache(self, dump_indices:torch.Tensor)->torch.Tensor:
        """ waring : this is a !!!INSIDE API!!! """
        # check for cache empty spaces
        free_cache_slots = (self.cache_ref == 0).nonzero(as_tuple=True)[0].to(torch.int32)
        if (free_cache_slots.size(0) < dump_indices.size(0)):
            # print(f"free cache slots: {free_cache_slots.size(0)}, dump indices: {dump_indices.size(0)}")
            self.resize_cache(dump_indices.size(0))
            free_cache_slots = (self.cache_ref == 0).nonzero(as_tuple=True)[0].to(torch.int32).nonzero(as_tuple=True)[0].to(torch.int32)
        
        # alloc for cache
        alloc = free_cache_slots[:dump_indices.size(0)]
        # self.cache_status[alloc] = True
        
        # update cache table
        # print("indices_shape", dump_indices.shape)
        # breakpoint()
        self.cache_table[alloc] = self.data_table[dump_indices]
        self.data_status[dump_indices] = False
        self.cache_ref[alloc] = self.data_ref[dump_indices]
        self.data_ref[dump_indices] = 0
        
        # 查表替换
        flat = self.mailbox_table.flatten()
        idx = torch.bucketize(flat, dump_indices, right=False)
        valid_idx = idx < dump_indices.size(0)
        matched = torch.zeros_like(flat, dtype=torch.bool)
        matched[valid_idx] = flat[valid_idx] == dump_indices[idx[valid_idx]]
        flat[matched] = alloc[idx[matched]] + self.max_idx
        table_replaced = flat.view_as(self.mailbox_table)
        self.mailbox_table = table_replaced
        
    import torch

    def resolve_indices(self, P: torch.Tensor, tensorA: torch.Tensor, tensorB: torch.Tensor) -> torch.Tensor:
        """
        解析 P 中的 index，对应到 tensorA 或 tensorB 的行。

        参数:
            P:       [N, 2] 的整数张量，每行包含两个 index。
            tensorA: [num_A, F] 的特征张量。
            tensorB: [num_B, F] 的特征张量。

        返回:
            [N, 2, F] 的张量，其中每个 index 被映射到对应的行特征。
        """
        N, F = P.size(0), tensorA.size(1)

        is_B = P >= self.max_idx
        P_a_idx = torch.where(is_B, torch.tensor(-1, device=P.device), P)
        P_b_idx = torch.where(is_B, P - self.max_idx, torch.tensor(-1, device=P.device))

        # result = torch.empty((N, 2, F), dtype=tensorA.dtype, device=tensorA.device)

        # for i in range(2):
        #     a_mask = P_a_idx[:, i] != -1
        #     b_mask = P_b_idx[:, i] != -1
        #     result[a_mask, i] = tensorA[P_a_idx[a_mask, i]]
        #     result[b_mask, i] = tensorB[P_b_idx[b_mask, i]]
            # 初始化结果张量，默认填充为 (0, -1)
        result = torch.empty((N, 2, F), dtype=tensorA.dtype, device=tensorA.device)
        result[:, :, 0] = 0
        result[:, :, 1] = -1

        # tensorA 部分
        flat_a_mask = P_a_idx != -1
        flat_a_idx = P_a_idx[flat_a_mask]
        result[flat_a_mask] = tensorA[flat_a_idx]

        # tensorB 部分
        flat_b_mask = P_b_idx != -1
        flat_b_idx = P_b_idx[flat_b_mask]
        result[flat_b_mask] = tensorB[flat_b_idx]

        return result

    def get_mailbox_data(self, uniq: torch.Tensor)->torch.Tensor:
        """ get mailbox data """
        # resolved = self.resolve_indices(self.mailbox_table[uniq], self.data_table, self.cache_table)
        
        # blockids = resolved[:, :, 0]  # shape: (N, 2)
        # slots = resolved[:, :, 1]  # shape: (N, 2)

        # # 计算 a * M + b
        # result = blockids * self.num_slots_per_block + slots  # shape: (N, 2)
        # mask0 = result[:, 0] != -1
        # uniq_data = torch.zeros((result.size(0), self.memory.size(1)), device=self.memory.device, dtype=self.memory.dtype)
        # uniq_data[mask0] = self.memory[result[mask0, 0]]

        # # 处理 nbr_data（对应 result[:, 1]）
        # mask1 = result[:, 1] != -1
        # nbr_data = torch.zeros((result.size(0), self.memory.size(1)), device=self.memory.device, dtype=self.memory.dtype)
        # nbr_data[mask1] = self.memory[result[mask1, 1]]
        mem_data = concat_mail.get_mailbox_all_data(self.num_slots_per_block, 
                                            self.efeat_manager.num_slots,
                                            uniq,
                                            self.mailbox_table,
                                            self.data_table,
                                            self.cache_table,
                                            self.memory,
                                            
                                            self.mail_efeat_table,
                                            self.efeat_manager.data_table,
                                            self.efeat_manager.memory,
                                            )
        
        # efeat_data = self.efeat_manager.get_data_batch_ori(self.mail_efeat_table[uniq])
        # res = torch.cat((mem_data, efeat_data), dim=1)
        # res = torch.cat((uniq_data, nbr_data, efeat_data), dim=1)
        res = mem_data
        return res 
    
    # def get_mailbox_data(self, uniq: torch.Tensor)->torch.Tensor:
    #     """ get mailbox data """
    #     resolved = self.resolve_indices(self.mailbox_table[uniq], self.data_table, self.cache_table)
        
    #     blockids = resolved[:, :, 0]  # shape: (N, 2)
    #     slots = resolved[:, :, 1]  # shape: (N, 2)

    #     # 计算 a * M + b
    #     result = blockids * self.num_slots_per_block + slots  # shape: (N, 2)
        
    #     # uniq_data = self.memory[result[:, 0]]
    #     # nbr_data = self.memory[result[:, 1]]
    #     # 初始化：设置为 zeros，shape 与 memory 中每条数据一致
    #     # zero_vec = torch.zeros_like(self.memory[0])

    #     # 处理 uniq_data（对应 result[:, 0]）
    #     mask0 = result[:, 0] != -1
    #     uniq_data = torch.zeros((result.size(0), self.memory.size(1)), device=self.memory.device, dtype=self.memory.dtype)
    #     uniq_data[mask0] = self.memory[result[mask0, 0]]

    #     # 处理 nbr_data（对应 result[:, 1]）
    #     mask1 = result[:, 1] != -1
    #     nbr_data = torch.zeros((result.size(0), self.memory.size(1)), device=self.memory.device, dtype=self.memory.dtype)
    #     nbr_data[mask1] = self.memory[result[mask1, 1]]
    #     efeat_data = self.efeat_manager.get_data_batch_ori(self.mail_efeat_table[uniq])
    #     res = torch.cat((uniq_data, nbr_data, efeat_data), dim=1)
    #     return res
    
    def get_mailbox_time(self, indices: torch.Tensor):
        return self.mail_time[indices]
    
    def get_mem_time(self, indices: torch.Tensor):
        return self.mem_time[indices]
    
        
    def free_spaces(self) -> torch.Tensor:
        """ get free spaces """
        free_space = (~self.space_status).nonzero().to(torch.int32)
        return free_space
        

    def update_mailbox(self, uniq: torch.Tensor, nbr: torch.Tensor, efeat: torch.Tensor, time: torch.Tensor):
        """ update mailbox """
        self.mailbox_table[uniq] = torch.stack((uniq, nbr), dim=1)
        self.mail_time[uniq] = time
        self.mail_efeat_table[uniq] = efeat
        
    def print_status(self):
        print(f"BlockManager: ")
        print(f"  Total Blocks: {self.num_blocks_got}")
        print(f"  Block Elements: {self.num_slots_per_block}")
        print(f"  Element Size: {self.feature_size} bytes")
        print(f"  Total Memory: {self.num_blocks_got * self.num_slots_per_block * self.feature_size / 1024**3:.2f} GB")
        print(f"  Free Blocks: {self.num_blocks_got - torch.sum(self.space_status)}")
        print(f"  Used Blocks: {torch.sum(self.space_status)}")
        print(f"  Memory Usage: {torch.sum(self.space_status) * self.num_slots_per_block * self.feature_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    shared_pool = BlockPool(total_mem_gb=16, block_elements=10)
    
    data_scale = 10
    
    efeat = torch.randn(data_scale, 10, dtype=torch.float32, device='cuda')
    efeat_manager = BlockManager(shared_pool, efeat, 10, data_scale, 10, True)
    efeat_manager.copy_from_cpu_batch(indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32, device='cuda'))
    
    manager = MemMailManager(shared_pool, 5, data_scale,efeat_manager,  15, 10, DEBUG=True)
    manager.update_mem_batch(indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32, device='cuda'), 
                             data=torch.randn(10, 5, dtype=torch.float32, device='cuda'),
                             time=torch.tensor([0,1,2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float32, device='cuda'),
                             up_mailbox_uniq=torch.tensor([1,5, 8], dtype=torch.int32, device='cuda'),
                             up_mailbox_nbr=torch.tensor([2,8,9], dtype=torch.int32, device='cuda')) 
    manager.update_mailbox(torch.tensor([1,5,8], dtype=torch.int32, device='cuda'),
                           torch.tensor([2,8,9], dtype=torch.int32, device='cuda'),
                           torch.tensor([0,1,2], dtype=torch.int32, device='cuda'),
                           time=torch.tensor([0,1,2], dtype=torch.float32, device='cuda'),)
    manager.update_mem_batch(indices=torch.tensor([0, 2, 8, 9], dtype=torch.int32, device='cuda'), 
                             data=torch.randn(4, 5, dtype=torch.float32, device='cuda'),
                             time=torch.tensor([0,1,2, 3], dtype=torch.float32, device='cuda'),
                             up_mailbox_uniq=torch.tensor([1, 2], dtype=torch.int32, device='cuda'),
                             up_mailbox_nbr=torch.tensor([5, 6], dtype=torch.int32, device='cuda')) 
    manager.update_mailbox(torch.tensor([1, 2], dtype=torch.int32, device='cuda'), torch.tensor([5, 6], dtype=torch.int32, device='cuda'), 
                           torch.tensor([0,1], dtype=torch.int32, device='cuda'),
                           time=torch.tensor([0,1], dtype=torch.float32, device='cuda'),)
    manager.get_mailbox_data(torch.tensor([1, 2, 5, 8], dtype=torch.int32, device='cuda'))
    manager.update_mem_batch(indices=torch.tensor([], dtype=torch.int32, device='cuda'), 
                             data=torch.randn(0, 5, dtype=torch.float32, device='cuda'),
                             time=torch.tensor([], dtype=torch.float32, device='cuda'),
                             up_mailbox_uniq=torch.tensor([8], dtype=torch.int32, device='cuda'),
                             up_mailbox_nbr=torch.tensor([9], dtype=torch.int32, device='cuda')) 
    manager.update_mailbox(torch.tensor([8], dtype=torch.int32, device='cuda'), torch.tensor([1], dtype=torch.int32, device='cuda'),
                           torch.tensor([9], dtype=torch.int32, device='cuda'),
                           time=torch.tensor([0], dtype=torch.float32, device='cuda'),)
    manager.get_mailbox_data(torch.tensor([1, 2, 5, 8], dtype=torch.int32, device='cuda'))
    print(manager.get_mailbox_data(torch.tensor([3], dtype=torch.int32, device='cuda')))