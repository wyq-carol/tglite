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
#  version8 : phase 1 end

###  version10 : this is a refacted version  ###
# version10 alpha1 : this is an early preview version
# version10 alpha2 : fixed many bugs, can stably work
# version10 beta1  : new algorithm applied
########################################################                             


import torch
import math
import nvtx

class BlockPool:
    """ Unity BlockPool """
    def __init__(self, total_mem_gb: int = 20, block_elements: int = 4300, device: str = "cuda"):
        print("USING TORCH VERSION 10 beta2")
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
        
    def generate_mem_map(self, num_slots_per_block : int) -> torch.Tensor:
        return torch.full((self.total_blocks * num_slots_per_block, ), fill_value=-1, device=self.device, dtype=torch.int32)

    def allocate_block(self, num_needed : int) -> torch.Tensor:
        """ return block view and block_id """
        free_blocks = torch.where(self.block_status == False)[0]
        if (free_blocks.size(0) < num_needed):
            raise ValueError(f"ERROR: NO FREE SPACES : free blocks: {free_blocks.size(0)}, need: {num_needed}")
        free_blocks = free_blocks[:num_needed]
        self.block_status[free_blocks] = True     
        return free_blocks 
    
    
    def free_block(self, blockids: torch.Tensor):
        """ free block by block_id """
        self.block_status[blockids] = False
            
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
    def __init__(self, pool: BlockPool, source_tensor: torch.Tensor, feature_size: int, max_idx: int, max_blocks = 4000):
        if pool.block_elements % feature_size != 0:
            raise ValueError(f"Feature size {feature_size} must divide block elements {pool.block_elements}")
        self.pool = pool
        self.feature_size = feature_size
        self.num_slots_per_block = pool.block_elements // feature_size
        self.max_idx = max_idx + 1
        self.source_tensor = source_tensor
        
        # a data table
        self.data_table = torch.full((self.max_idx, ), -1,  dtype=torch.int32, device=pool.device) 
        
        # a space table 
        self.space_table = torch.full((max_blocks * self.num_slots_per_block, ), -1,  dtype=torch.int32, device=pool.device)   
        self.space_status = torch.full((max_blocks * self.num_slots_per_block, ), -1,  dtype=torch.int32, device=pool.device)      
        free_blocks = pool.allocate_block(max_blocks)
        
        block_ids = free_blocks.view(-1, 1)  # shape: [num_blocks, 1]
        slot_offsets = torch.arange(self.num_slots_per_block, device=self.pool.device, dtype=torch.int32).view(1, -1)  # shape: [1, slots]
        indices = block_ids * self.num_slots_per_block + slot_offsets  # shape: [num_blocks, slots]

        indices = indices.reshape(-1)  # shape: [num_blocks * slots]
        idx = torch.arange(max_blocks * self.num_slots_per_block, device=self.pool.device, dtype=torch.int32)
        self.space_table[idx] = indices.to(torch.int32)  
        self.space_status[idx] = 0       
        
        self.memory = self.pool.memory.reshape(-1, self.feature_size)
        
    def resize (self, need_slots: int):
        """ resize block manager """
        pass
    
    def get_data_batch(self, indices: torch.Tensor) -> torch.Tensor:
        pos_ind = self.data_table[indices]
        pos = self.space_table[pos_ind]
        return self.memory[pos]
    
    def update_data_batch(self, indices: torch.Tensor, data: torch.Tensor):
        """ set data in batch """
        invalid_indices = self.check_data_valid(indices)
        if (invalid_indices.size(0) > 0):
            self.alloc_for_batch(invalid_indices)
            
        pos_ind = self.data_table[indices]
        pos = self.space_table[pos_ind]
        self.memory[pos] = data.to(self.pool.device)
        
    def check_data_valid(self, indices: torch.Tensor) -> torch.Tensor:
        """ check if indices are valid , return invalid indices """
        invalid = self.data_table[indices] < 0
        return indices[invalid]
    
    def alloc_for_batch(self, indices: torch.Tensor):
        """ waring : this is a !!!INSIDE API!!! """
        # Step 1 : check fot valid space
        free_spaces = self.free_spaces()
        if (free_spaces.size(0) < indices.size(0)):
            raise Exception("Out of manager mem, this is a unreailzed error")
        # Step 2 : alloc for indices
        alloc = free_spaces[:indices.size(0)]
        self.space_status[alloc] = 1
        self.data_table[indices] = alloc
        
    def free_spaces(self) -> torch.Tensor:
        """ free spaces, its blockid """
        free_space = torch.nonzero(self.space_status == 0, as_tuple=False).squeeze().to(torch.int32)
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
            return 
        self.alloc_for_batch(need_to_load)
        pos_ind = self.data_table[indices_gpu]
        # TODO : mem abuse
        pos = self.space_table[pos_ind]
        self.memory[pos] = self.source_tensor[indices].to('cuda')
        
    def unload_data(self, indices : torch.Tensor):
        pass
        
    def print_status(self):
        print(f"BlockManager: ")
        used_blocks = torch.nonzero(self.space_status == 1, as_tuple=False).squeeze().to(torch.int32).size(0)
        free_blocks = torch.nonzero(self.space_status == 0, as_tuple=False).squeeze().to(torch.int32).size(0)
        print(f"  Total Blocks: {used_blocks + free_blocks}")
        print(f"  Free Blocks: {free_blocks}")
        # print(f"  Cache Usage: {}")

class MemMailManager:
    def __init__(self, pool: BlockPool, feature_size: int, max_idx: int, efeat_manager: BlockManager, max_blocks = 4000, DEBUG = True):
        self.DEBUG = DEBUG
        
        print("USING TORCH VERSION 10 beta2")
        print(f'maxidx: {max_idx}')
        if pool.block_elements % feature_size != 0:
            raise ValueError(f"Feature size {feature_size} must divide block elements {pool.block_elements}")
        self.pool = pool
        self.feature_size = feature_size
        self.num_slots_per_block = pool.block_elements // feature_size
        self.max_idx = max_idx + 1
        self.cache_table_len = 2 * max_idx

        # latest data table 
        self.data_table = torch.full((self.max_idx, ), -1,  dtype=torch.int32, device=pool.device) 
        
        # mailbox table
        self.mailbox_table = torch.full((self.max_idx, 2), -1, dtype=torch.int32, device=pool.device)
        
        # a space table (block num , slot num per block), True mains used        
        self.space_table = torch.full((max_blocks * self.num_slots_per_block, ), -1,  dtype=torch.int32, device=pool.device)   
        self.space_status = torch.full((max_blocks * self.num_slots_per_block, ), -1,  dtype=torch.int32, device=pool.device)      
        free_blocks = pool.allocate_block(max_blocks)
        
        block_ids = free_blocks.view(-1, 1)  # shape: [num_blocks, 1]
        slot_offsets = torch.arange(self.num_slots_per_block, device=self.pool.device, dtype=torch.int32).view(1, -1)  # shape: [1, slots]
        indices = block_ids * self.num_slots_per_block + slot_offsets  # shape: [num_blocks, slots]

        indices = indices.reshape(-1)  # shape: [num_blocks * slots]
        idx = torch.arange(max_blocks * self.num_slots_per_block, device=self.pool.device, dtype=torch.int32)
        self.space_table[idx] = indices.to(torch.int32)  
        self.space_status[idx] = 0       
        self.memory = self.pool.memory.reshape(-1, self.feature_size)
        
        self.mem_time = torch.zeros(self.max_idx, dtype=torch.float32, device=pool.device)
        self.mail_time = torch.zeros(self.max_idx, dtype=torch.float32, device=pool.device)
        self.mail_efeat_table = torch.full((self.max_idx,), -1, dtype=torch.int32, device=pool.device)
        
        self.efeat_manager = efeat_manager
        self.round = 1
        
        if self.DEBUG:
            self.uniq_tester = torch.zeros((self.max_idx, self.feature_size), dtype=torch.float32, device="cuda")
            self.nbr_tester = torch.zeros((self.max_idx, self.feature_size), dtype=torch.float32, device="cuda")


    def reset(self):
        """ reset mem manager """
        self.data_table = torch.full((self.max_idx, ), -1,  dtype=torch.int32, device=self.pool.device)
        
        self.mailbox_table = torch.full((self.max_idx, 2), -1, dtype=torch.int32, device=self.pool.device)
        self.mail_time = torch.zeros(self.max_idx, dtype=torch.float32, device=self.pool.device)
        self.mail_efeat_table = torch.full((self.max_idx,), -1, dtype=torch.int32, device=self.pool.device)

        self.mem_time = torch.zeros(self.max_idx, dtype=torch.float32, device=self.pool.device)
        self.space_status[self.space_status > 0] = 0 
        
        if self.DEBUG:
            self.uniq_tester = torch.zeros((self.max_idx, self.feature_size), dtype=torch.float32, device="cuda")
            self.nbr_tester = torch.zeros((self.max_idx, self.feature_size), dtype=torch.float32, device="cuda")

        
    
    def resize (self, need_slots: int):
        """ resize block manager """
        pass
    
    
    def get_mem_data(self, indices: torch.Tensor) -> torch.Tensor:
        pos_ind = self.data_table[indices] 
        mem_data = torch.zeros((len(pos_ind), self.memory.size(1)), device=self.memory.device, dtype=self.memory.dtype)
        valid_mask = pos_ind != -1
        valid_pos = self.space_table[pos_ind[valid_mask]]
        mem_data[valid_mask] = self.memory[valid_pos]
        return mem_data

    def update_mem_batch(self, unique, counts, indices: torch.Tensor, data: torch.Tensor, time:torch.Tensor,  up_mailbox_uniq: torch.Tensor = None, up_mailbox_nbr: torch.Tensor = None):
        """ update mem """
        
        slots = self.data_table[indices]
        valid_slots = slots >= 0
        self.space_status[slots[valid_slots]] -= 1

        drop_lines_mailbox = self.mailbox_table[up_mailbox_uniq]
        filtered = drop_lines_mailbox[drop_lines_mailbox >= 0]
        unique, counts = torch.unique(filtered, return_counts=True)
        self.space_status[unique] -= counts

        has_ref = self.space_status[slots] > 0
        tobe_replaced_indices = indices[has_ref]
        self.data_table[tobe_replaced_indices] = -1

        slots = self.data_table[indices]
        valid_slots = slots >= 0
        self.space_status[slots[valid_slots]] += 1

        

        with nvtx.annotate("check", color='purple'):
            invalid_indices = indices[self.data_table[indices] < 0]
        with nvtx.annotate("alloc", color='purple'):
            if (invalid_indices.size(0) > 0):
                self.alloc_for_batch(invalid_indices)
                
        with nvtx.annotate("unique", color='yellow'):
            unique, counts = torch.unique(torch.cat((up_mailbox_uniq, up_mailbox_nbr)), return_counts=True)
        slot_idx = self.data_table[unique]
        mask = slot_idx >= 0
        self.space_status[slot_idx[mask]] += counts[mask]
            
        with nvtx.annotate("update", color='orange'):
            pos_ind = self.data_table[indices]
            pos = self.space_table[pos_ind]
            self.memory[pos] = data.to(self.pool.device)
            self.mem_time[indices] = time
            
            
        if self.DEBUG:
            uniq = up_mailbox_uniq
            nbr = up_mailbox_nbr
            uniq_pos_ind = self.data_table[uniq]
            nbr_pos_ind  = self.data_table[nbr]
            self.mailbox_table[uniq] = torch.stack((uniq_pos_ind, nbr_pos_ind), dim=1)

            self.uniq_tester[uniq] = self.get_mem_data(uniq)
            self.nbr_tester[uniq] = self.get_mem_data(nbr)
                
            uniq_data = torch.zeros((uniq.size(0), self.feature_size), device=self.pool.device, dtype=torch.float32)
            nbr_data = torch.zeros((uniq.size(0), self.feature_size), device=self.pool.device, dtype=torch.float32)
            uniq_data[uniq_pos_ind >= 0] = self.memory[self.space_table[uniq_pos_ind[uniq_pos_ind >= 0]]]
            nbr_data[nbr_pos_ind >= 0] = self.memory[self.space_table[nbr_pos_ind[nbr_pos_ind >= 0]]]
            
            if not torch.allclose(uniq_data, self.uniq_tester[uniq], rtol = 1e-5):
                raise Exception(f"wrong at {self.round}")
            if not torch.allclose(nbr_data, self.nbr_tester[uniq], rtol = 1e-5):
                raise Exception(f"wrong at {self.round}")
        
        self.round += 1
        
        
        
    def check_data_valid(self, indices: torch.Tensor) -> torch.Tensor:
        """ check if indices are valid , return invalid indices """
        valid = self.data_table[indices] >= 0
        invalid = indices[~valid]
        valid = indices[valid]
        return invalid, valid
    
    def alloc_for_batch(self, indices: torch.Tensor):
        """ waring : this is a !!!INSIDE API!!! """
        # Step 1 : check fot valid space
        free_spaces = self.free_spaces()
        if (free_spaces.size(0) < indices.size(0)):
            print(f"free spaces: {free_spaces.size(0)}, indices: {indices.size(0)}")
            raise Exception("manager out of mem")
        # Step 2 : alloc for indices
        alloc = free_spaces[:indices.size(0)]
        self.space_status[alloc] = 1
        self.data_table[indices] = alloc
        

    def get_mailbox_data(self, uniq: torch.Tensor)->torch.Tensor:
        """ get mailbox data """
        uniq_pos_ind = self.mailbox_table[uniq][:, 0]
        nbr_pos_ind  = self.mailbox_table[uniq][:, 1]
        
        uniq_data = torch.zeros((uniq.size(0), self.feature_size), device=self.pool.device, dtype=torch.float32)
        nbr_data = torch.zeros((uniq.size(0), self.feature_size), device=self.pool.device, dtype=torch.float32)
        uniq_data[uniq_pos_ind >= 0] = self.memory[self.space_table[uniq_pos_ind[uniq_pos_ind >= 0]]]
        nbr_data[nbr_pos_ind >= 0] = self.memory[self.space_table[nbr_pos_ind[nbr_pos_ind >= 0]]]
        
        if self.DEBUG:
            if not torch.allclose(uniq_data, self.uniq_tester[uniq], rtol = 1e-5):
                raise Exception(f"wrong at {self.round}")
            if not torch.allclose(nbr_data, self.nbr_tester[uniq], rtol = 1e-5):
                raise Exception(f"wrong at {self.round}")
        
        efeat = self.efeat_manager.get_data_batch(self.mail_efeat_table[uniq])
        out = torch.cat((uniq_data, nbr_data, efeat), dim=1)
        
        return out
    
    
    def get_mailbox_time(self, indices: torch.Tensor):
        return self.mail_time[indices]
    
    def get_mem_time(self, indices: torch.Tensor):
        return self.mem_time[indices]
    
        
    def free_spaces(self) -> torch.Tensor:
        """ get free spaces """
        free_space = torch.nonzero(self.space_status == 0, as_tuple=False).squeeze().to(torch.int32)
        return free_space
        

    def update_mailbox(self, uniq: torch.Tensor, nbr: torch.Tensor, efeat: torch.Tensor, time: torch.Tensor):
        """ update mailbox """
        uniq_pos = self.data_table[uniq]
        nbr_pos  = self.data_table[nbr]
        self.mailbox_table[uniq] = torch.stack((uniq_pos, nbr_pos), dim=1)
        self.mail_time[uniq] = time
        self.mail_efeat_table[uniq] = efeat
        

    def print_status(self):
        print(f"BlockManager: ")
        used_blocks = torch.nonzero(self.space_status == 1, as_tuple=False).squeeze().to(torch.int32).size(0)
        free_blocks = torch.nonzero(self.space_status == 0, as_tuple=False).squeeze().to(torch.int32).size(0)
        print(f"  Total Blocks: {used_blocks + free_blocks}")
        print(f"  Free Blocks: {free_blocks}")
        print(f"  Used Blocks: {used_blocks}")


if __name__ == "__main__":
    shared_pool = BlockPool(total_mem_gb=16, block_elements=10)
    
    data_scale = 10
    
    efeat = torch.randn(data_scale, 10, dtype=torch.float32, device='cuda')
    efeat_manager = BlockManager(shared_pool, efeat, 10, data_scale - 1, 10)
    efeat_manager.copy_from_cpu_batch(indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32, device='cuda'))
    
    manager = MemMailManager(shared_pool, 5, data_scale - 1,efeat_manager,  15)
    manager.update_mem_batch(indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device='cuda'), 
                             data=torch.randn(5, 5, dtype=torch.float32, device='cuda'),
                             time=torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device='cuda'),
                             up_mailbox_uniq=torch.tensor([0, 2, 8], dtype=torch.int32, device='cuda'),
                             up_mailbox_nbr=torch.tensor([2, 4, 9], dtype=torch.int32, device='cuda'), unique=None, counts=None) 
    manager.update_mailbox(torch.tensor([0, 2, 8], dtype=torch.int32, device='cuda'),
                           torch.tensor([2, 4, 9], dtype=torch.int32, device='cuda'),
                           torch.tensor([0, 1, 2], dtype=torch.int32, device='cuda'),
                           time=torch.tensor([0, 1, 2], dtype=torch.float32, device='cuda'),)
    manager.update_mem_batch(indices=torch.tensor([0, 2, 3, 4, 5, 6], dtype=torch.int32, device='cuda'), 
                             data=torch.randn(6, 5, dtype=torch.float32, device='cuda'),
                             time=torch.tensor([0, 2, 3, 4, 5, 6], dtype=torch.float32, device='cuda'),
                             up_mailbox_uniq=torch.tensor([2, 8], dtype=torch.int32, device='cuda'),
                             up_mailbox_nbr=torch.tensor([5, 6], dtype=torch.int32, device='cuda'), unique=None, counts=None) 
    manager.update_mailbox(torch.tensor([2, 8], dtype=torch.int32, device='cuda'), torch.tensor([5, 6], dtype=torch.int32, device='cuda'), 
                           torch.tensor([5, 6], dtype=torch.int32, device='cuda'),
                           time=torch.tensor([2, 8], dtype=torch.float32, device='cuda'),)
    manager.update_mem_batch(indices=torch.tensor([0, 2], dtype=torch.int32, device='cuda'), 
                             data=torch.randn(2, 5, dtype=torch.float32, device='cuda'),
                             time=torch.tensor([0, 2], dtype=torch.float32, device='cuda'),
                             up_mailbox_uniq=torch.tensor([0], dtype=torch.int32, device='cuda'),
                             up_mailbox_nbr=torch.tensor([2], dtype=torch.int32, device='cuda'), unique=None, counts=None) 
    manager.update_mailbox(torch.tensor([0], dtype=torch.int32, device='cuda'), torch.tensor([2], dtype=torch.int32, device='cuda'), 
                           torch.tensor([2], dtype=torch.int32, device='cuda'),
                           time=torch.tensor([0], dtype=torch.float32, device='cuda'),)
    manager.get_mailbox_data(torch.tensor([3], dtype=torch.int32, device='cuda'))