#######################################################
#
#  UPDATE LOG:
#  version5 edition0 feat    : triton kernels added
#  version5 edition1 bug fix : fixed ref bugs 
#  version6 edition0 feat    : turned to cuda kernels
#  version6 edition1 tidy    : cleanup codes
#  version6 edition2 feat    : add update mem kernels
#  version6 edition3 feat    : add alloc kernels
#  version6 edition4 bug fix : alloc kernel bug
#  version7 edition0 feat    : pre alloc for mem manager
#  version7 edition1 feat    : update mem kernel fine tune
#  version7 edition2 feat    : update mem offline
#  version7 edition3 feat    : final tune for unique
#  version8 : phase 1 end, this is a stable version

#  version9 : phase 2 start
#  version9 edition0 feat    : add edge manager demo , running ok
########################################################                             


import torch
import math
import nvtx
import triton.language as tl
import tglite.our_kernels.mem_manager_kernels.mem_manager as mem_manager_kernels



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
    
    def get_data_batch(self, indices: torch.Tensor) -> torch.Tensor:
        return mem_manager_kernels.get_feat_data(self.num_slots, indices, self.data_table, self.memory)
    
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

class EdgeManager:
    def __init__(self, pool: BlockPool, source_tensor: torch.Tensor, feature_size: int, max_idx: int, window_size = 6000 * 5, DEBUG = False):
        self.DEBUG = DEBUG
        if pool.block_elements % feature_size != 0:
            raise ValueError(f"Feature size {feature_size} must divide block elements {pool.block_elements}")
        self.pool = pool
        self.feature_size = feature_size
        self.num_slots = pool.block_elements // feature_size
        self.num_blocks = math.ceil(window_size / self.num_slots)
        
        self.max_idx = max_idx
        self.source_tensor = source_tensor
        
        self.window_size = window_size
        
        self.win_start = 0
        self.win_end = window_size
        
        # a data table (N , 2), True mains valid
        self.data_table = torch.zeros((self.window_size, 2), dtype=torch.int32, device=pool.device) 
        
        # a space table (block num , slot num), True mains used        
        free_blocks = pool.allocate_block(self.num_blocks)
        blockid = free_blocks.repeat_interleave(self.num_slots).to(torch.int32).to(self.pool.device) 
        slotid = torch.tile(torch.arange(self.num_slots, dtype=torch.int32, device=self.pool.device), (self.num_blocks,)) 
        self.space_table = torch.stack((blockid, slotid), dim=1).to(self.pool.device)
        self.space_status = torch.zeros((self.num_blocks, self.num_slots), dtype=torch.bool, device=self.pool.device)
        
        self.memory = self.pool.memory.reshape(-1, self.feature_size)
        
        self.init_window_load()
        
    def init_window_load (self):
        """ init window load """
        self.alloc_for_batch(torch.arange(self.window_size, dtype=torch.int32, device=self.pool.device))
        info = self.data_table
        pos = info[:, 0] * self.num_slots + info[:, 1]
        self.memory[pos] = self.source_tensor[self.win_start:self.win_end].to('cuda')
        
    def update_window (self, stride):
        """ update window, stride is the num edges updated  """
        self.win_start = self.win_start + stride
        self.win_end = min(self.win_end + stride, self.max_idx)
        self.window_size = self.win_end - self.win_start
        self.data_table = torch.cat((self.data_table[stride:, :], self.data_table[:stride, :]), dim=0)
        
        info = self.data_table[self.window_size - stride:self.window_size]
        pos = info[:, 0] * self.num_slots + info[:, 1]
        self.memory[pos] = self.source_tensor[self.win_end - stride : self.win_end].to('cuda')
        
    def get_edata(self, indices: torch.Tensor) -> torch.Tensor:
        """ get edata with index in indices """
        res = torch.empty((indices.shape[0], self.feature_size), dtype=torch.float32, device=self.pool.device)
        valid_mask = (indices >= self.win_start) & (indices < self.win_end)
        valid_indices_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_indices = indices[valid_mask]
        invalid_indices_idx = torch.nonzero(~valid_mask, as_tuple=True)[0]
        invalid_indices = indices[~valid_mask].cpu()
        block_ids = self.data_table[valid_indices - self.win_start, 0]
        slot_ids = self.data_table[valid_indices - self.win_start, 1]
        res[valid_indices_idx] = self.memory[block_ids * self.num_slots + slot_ids]
        res[invalid_indices_idx] = self.source_tensor[invalid_indices].to('cuda')
        
        print("missed: ", invalid_indices)
        return res

        
        

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
    
    def get_data_batch(self, indices: torch.Tensor) -> torch.Tensor:
        return mem_manager_kernels.get_feat_data(self.num_slots, indices, self.data_table, self.memory)
    
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
        # Step 2 : alloc for indices
        alloc = free_spaces[:indices.size(0)]
        self.space_status[alloc[:, 0], alloc[:, 1]] = True
        self.data_table[indices] = self.space_table[alloc[:, 0] * self.num_slots +  alloc[:, 1]]

        
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
        # self.cache_table_len = blocks_for_cache * self.num_slots_per_block
        self.cache_table_len = 2*max_idx

        
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
        
        # a space table (block num , slot num per block), True mains used        
        free_blocks = pool.allocate_block(self.num_blocks_got)
        blockid = free_blocks.repeat_interleave(self.num_slots_per_block).to(torch.int32).to(self.pool.device) 
        slotid = torch.tile(torch.arange(self.num_slots_per_block, dtype=torch.int32, device=self.pool.device), (self.num_blocks_got,)) 
        self.space_table = torch.stack((blockid, slotid), dim=1).to(self.pool.device)
        self.space_status = torch.ones((self.num_blocks_got, self.num_slots_per_block), dtype=torch.int32, device=self.pool.device)
        self.memory = self.pool.memory.reshape(-1, self.feature_size)
        
        self.mem_time = torch.zeros(self.max_idx, dtype=torch.float32, device=pool.device)
        self.mail_time = torch.zeros(self.max_idx, dtype=torch.float32, device=pool.device)
        self.mail_efeat_table = torch.full((self.max_idx,), -1, dtype=torch.int32, device=pool.device)
        
        self.efeat_manager = efeat_manager
        self.counter = 0
        
        self.pre_alloc()

    def pre_alloc(self):
        """ pre alloc for mailbox """
        self.alloc_for_batch(torch.arange(self.max_idx, dtype=torch.int32, device=self.pool.device))
        # self.data_status = torch.ones(self.max_idx, dtype=torch.bool, device=self.pool.device)
        # warning for data status

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
        self.space_status = torch.ones((self.num_blocks_got * self.num_slots_per_block), dtype=torch.int32, device=self.pool.device)
        
             
        

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
        
        with nvtx.annotate("table manage", color = 'red'):
            blockid = free_blocks.repeat_interleave(self.num_slots_per_block).to(torch.int32).to(self.pool.device) 
            slotid = torch.tile(torch.arange(self.num_slots_per_block, dtype=torch.int32, device=self.pool.device), (alloc_blocks, )) 
            
            new_table = torch.stack((blockid, slotid), dim=1).to(self.pool.device)
            self.space_table = torch.cat((self.space_table, new_table), dim=0)
            
            new_status = torch.ones((free_blocks.size(0) * self.num_slots_per_block), dtype=torch.int32, device=self.pool.device)
            self.space_status = torch.cat((self.space_status, new_status), dim=0)
    
    
    def get_mem_data(self, indices: torch.Tensor) -> torch.Tensor:
        with nvtx.annotate("get mem", color="purple"):
            return mem_manager_kernels.get_mem_data(
                self.num_slots_per_block,
                indices,
                self.data_table,
                self.data_status,
                self.memory,
            ) 
            
        return output

    def update_mem_batch(self, unique, counts, indices: torch.Tensor, data: torch.Tensor, time:torch.Tensor,  up_mailbox_uniq: torch.Tensor = None, up_mailbox_nbr: torch.Tensor = None):
        """ update mem, if u dont tell me mailbox update info, i will conservatively dump to cache """
        # check for dump to cache
        with nvtx.annotate("check dump", color='orange'): 
            dump_indices = mem_manager_kernels.dump_launcher(
                self.max_idx,
                self.mailbox_table,
                up_mailbox_uniq,
                up_mailbox_nbr,
                self.cache_ref,
                self.data_ref,
                indices,
                unique,
                counts
            )

        with nvtx.annotate("dumping", color='orange'):
            if (dump_indices.shape[0] > 0):
                mem_manager_kernels.cache_dumper(
                    dump_indices,
                    self.data_table,
                    self.data_ref,
                    self.data_status,
                    self.cache_table,
                    self.cache_ref,
                    self.mailbox_table
                )        
                with nvtx.annotate("alloc", color='purple'):
                    mem_manager_kernels.launch_allocate_space_kernel(
                dump_indices,
                self.space_status, 
                self.data_status,
                self.space_table, 
                self.data_table, 
                self.num_slots_per_block
            )
            
        with nvtx.annotate("update", color='orange'):
            mem_manager_kernels.write_data_to_memory(
                self.data_table,
                indices,
                data,
                self.memory,
                self.num_slots_per_block
            )
        with nvtx.annotate("update time", color='orange'):
            self.mem_time[indices] = time
        
    def check_data_valid(self, indices: torch.Tensor) -> torch.Tensor:
        """ check if indices are valid , return invalid indices """
        # valid = self.data_status[indices]
        # invalid = indices[~valid]
        # valid = indices[valid]
        invalid, valid = mem_manager_kernels.check_data_valid(
            indices,
            self.data_status
        )
        return invalid, valid
    
    def alloc_for_batch(self, indices: torch.Tensor):
        """ waring : this is a !!!INSIDE API!!!, please make sure all the indices in data_table are valid, u can use check_data_valid to flit it """
        with nvtx.annotate("alloc", color='orange'):
            mem_manager_kernels.launch_allocate_space_kernel(
            indices,
            self.space_status, 
            self.data_status,
            self.space_table, 
            self.data_table, 
            self.num_slots_per_block
        )
         
        
        
    def dump_to_cache(self, dump_indices:torch.Tensor)->torch.Tensor:
        """ waring : this is a !!!INSIDE API!!! """
        # check for cache empty spaces
        if (dump_indices.shape[0] == 0):
            return
        # free_cache_slots = (self.cache_ref == 0).nonzero(as_tuple=True)[0].to(torch.int32)
        # alloc = free_cache_slots[:dump_indices.size(0)]
        mem_manager_kernels.cache_dumper(
            # (self.cache_ref == 0).to(torch.int32),
            dump_indices,
            self.data_table,
            self.data_ref,
            self.data_status,
            self.cache_table,
            self.cache_ref,
            self.mailbox_table
        )
            
            # 查表替换
        # with nvtx.annotate("main body", color = "yellow"):
        #     flat = self.mailbox_table.flatten()
        #     idx = torch.bucketize(flat, dump_indices, right=False)
        #     valid_idx = idx < dump_indices.size(0)
        #     matched = torch.zeros_like(flat, dtype=torch.bool)
        #     matched[valid_idx] = flat[valid_idx] == dump_indices[idx[valid_idx]]
        #     flat[matched] = alloc[idx[matched]] + self.max_idx
        #     table_replaced = flat.view_as(self.mailbox_table)
        #     self.mailbox_table = table_replaced
        

    def get_mailbox_data(self, uniq: torch.Tensor)->torch.Tensor:
        """ get mailbox data """
        return mem_manager_kernels.get_mailbox_data(self.num_slots_per_block, 
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
    
    
    def get_mailbox_time(self, indices: torch.Tensor):
        return self.mail_time[indices]
    
    def get_mem_time(self, indices: torch.Tensor):
        return self.mem_time[indices]
    
        
    def free_spaces(self) -> torch.Tensor:
        """ get free spaces """
        free_space = self.space_status.nonzero().to(torch.int32)
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
    shared_pool = BlockPool(total_mem_gb=1, block_elements=4)
    
    edata = torch.randn(20, 2)
    print(edata)
    efeat_manager = EdgeManager(shared_pool, edata, 2, 20, 9)
    gotdata = efeat_manager.get_edata(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).cpu()
    assert torch.allclose(gotdata, edata[:10])
    
    idx = torch.tensor([4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 13, 11], dtype=torch.int32, device='cuda')
    gotdata = efeat_manager.get_edata(idx).cpu()
    assert torch.allclose(gotdata, edata[idx.cpu()])
    
    efeat_manager.update_window(3)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.int32, device='cuda')
    gotdata = efeat_manager.get_edata(idx).cpu()
    assert torch.allclose(gotdata, edata[idx.cpu()])
    
    efeat_manager.update_window(3)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.int32, device='cuda')
    gotdata = efeat_manager.get_edata(idx).cpu()
    assert torch.allclose(gotdata, edata[idx.cpu()])

    efeat_manager.update_window(3)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int32, device='cuda')
    gotdata = efeat_manager.get_edata(idx).cpu()
    assert torch.allclose(gotdata, edata[idx.cpu()])

    efeat_manager.update_window(3)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int32, device='cuda')
    gotdata = efeat_manager.get_edata(idx).cpu()
    assert torch.allclose(gotdata, edata[idx.cpu()])