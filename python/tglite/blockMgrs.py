import torch
from collections import deque
from typing import Dict, Tuple, List, Deque
import time
import concurrent.futures

class BlockPool:
    """ Unity BlockPool """
    def __init__(self, total_mem_gb: int = 20, block_elements: int = 4096, device: str = "cuda"):
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
        
        self.free_blocks: Deque[int] = deque(range(self.total_blocks))
        self.used_blocks = set()

    def allocate_blocks(self, num_blocks: int) -> List[Tuple[torch.Tensor, int]]:
        """ return block view and block_id """
        if not self.free_blocks:
            raise RuntimeError("Block pool exhausted")
        
        assert num_blocks <= len(self.free_blocks), "Not enough free blocks"
        block_ids = [self.free_blocks.popleft() for _ in range(num_blocks)]
        self.used_blocks.update(block_ids)
        
        return block_ids

    def allocate_block_id(self) -> int:
        """ return block_id """
        if not self.free_blocks:
            raise RuntimeError("Block pool exhausted")
        
        block_id = self.free_blocks.popleft()
        self.used_blocks.add(block_id)
        
        return block_id

    def allocate_block(self) -> Tuple[torch.Tensor, int]:
        """ return block view and block_id """
        if not self.free_blocks:
            raise RuntimeError("Block pool exhausted")
        
        block_id = self.free_blocks.popleft()
        self.used_blocks.add(block_id)
        
        start = block_id * self.block_elements
        end = start + self.block_elements
        return self.memory[start:end], block_id

    def free_block(self, block_id: int):
        """ free block by block_id """
        if block_id in self.used_blocks:
            self.used_blocks.remove(block_id)
            self.free_blocks.append(block_id)
            
    def print_status(self):
        print(f"BlockPool: ")
        print(f"Total Blocks={self.total_blocks}")
        print(f"Free Blocks={len(self.free_blocks)}, Used Blocks={len(self.used_blocks)}")
        print(f"block size: {self.block_elements} elements, {self.block_elements * self.element_size} bytes")
        print(f"Usage Percent: {len(self.used_blocks) / self.total_blocks * 100} %")

class Block:
    def __init__(self, block_id: int, block_manager_id: int, block_view: torch.Tensor, feature_size: int, have_free_slots: bool = True):
        self.block_id = block_id
        self.block_manager_id = block_manager_id
        self.feature_size = feature_size
        self.slots_per_block = len(block_view) // feature_size
        
        # reshape block view
        self.data = block_view.view(-1, feature_size)
        
        if have_free_slots:
            self.free_slots = deque(range(self.slots_per_block))
            self.used_slots = set()
        else:
            self.free_slots = deque()
            self.used_slots = set(range(self.slots_per_block))

    @property
    def is_full(self):
        return len(self.free_slots) == 0

    @property
    def is_empty(self):
        return len(self.used_slots) == 0

    def allocate(self) -> int:
        if self.free_slots:
            slot = self.free_slots.popleft()
            self.used_slots.add(slot)
            return slot
        return -1

    def free(self, slot: int):
        if slot in self.used_slots:
            self.used_slots.remove(slot)
            self.free_slots.append(slot)
            # self.data[slot].zero_() invilidate

class BlockManager:
    def __init__(self, pool: BlockPool, feature_size: int, max_manager_index: int):
        if pool.block_elements % feature_size != 0:
            raise ValueError(f"Feature size {feature_size} must divide block elements {pool.block_elements}")
        self.pool = pool
        self.num_slots = self.pool.block_elements // feature_size
        self.feature_size = feature_size
        self.blocks: List[Block] = []
        self.free_blocks = deque()
        self.index2managerBlk:  torch.Tensor = torch.ones(max_manager_index, dtype=torch.int32, device="cuda") * -1
        self.index2managerSlot: torch.Tensor = torch.ones(max_manager_index, dtype=torch.int32, device="cuda") * -1
        self.index2blk: torch.Tensor = torch.ones(max_manager_index, dtype=torch.int32, device="cuda") * -1

    def init(self, indices: torch.Tensor, cpu_tensor: torch.Tensor):
        ### allocate(self, indices: torch.Tensor): # indices: 预分配的nid/eid
        start = time.time()
        element_multiplier = self.pool.block_elements
        memory = self.pool.memory
        feature_size = self.feature_size
        num_slots = self.num_slots
        def create_block(block_id, have_free_slots):
            start_index = block_id * element_multiplier
            end_index = start_index + element_multiplier
            return Block(block_id, len(self.blocks), memory[start_index:end_index], feature_size, have_free_slots=have_free_slots)
        
        num_blocks = indices.shape[0] // self.num_slots # 需要多少个无空slot 的blocks
        last_block_slots = indices.shape[0] % self.num_slots # 最后一个block 的slots 数
        blocks_ids = self.pool.allocate_blocks(num_blocks)
        # 使用并行处理初始化blocks
        
        # 维护block 使用slot 的情况
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        #     self.blocks = list(executor.map(create_block, blocks_ids, [False] * num_blocks))
        self.blocks = list(map(create_block, blocks_ids, [False] * num_blocks))
        if last_block_slots > 0:
            last_block_id = self.pool.allocate_block_id()
            blocks_ids.append(last_block_id)
            last_block = create_block(last_block_id, True)
            assert last_block_slots <= self.num_slots, "last block slots must be less than num_slots"
            _ = [last_block.free_slots.pop() for _ in range(last_block_slots)]
            self.blocks.append(last_block)
        
        # 1000 = num_blocks(62) * num_slots(16) + last_block_slots(8)
        # 维护索引
        if last_block_slots == 0:
            self.index2blk[indices] = torch.tensor(blocks_ids, dtype=torch.int32, device="cuda").repeat_interleave(num_slots)
            self.index2managerBlk[indices] = torch.arange(num_blocks, dtype=torch.int32, device="cuda").repeat_interleave(num_slots)
            self.index2managerSlot[indices] = torch.arange(num_slots, dtype=torch.int32, device="cuda").repeat(num_blocks)
        else: # if last_block_slots > 0
            self.index2blk[indices] = torch.cat((torch.tensor(blocks_ids[:-1], dtype=torch.int32, device="cuda").repeat_interleave(num_slots), torch.tensor(blocks_ids[-1], dtype=torch.int32, device="cuda").repeat(last_block_slots)))
            self.index2managerBlk[indices] = torch.cat((torch.arange(num_blocks, dtype=torch.int32, device="cuda").repeat_interleave(num_slots), torch.tensor(num_blocks, dtype=torch.int32, device="cuda").repeat(last_block_slots)))
            self.index2managerSlot[indices] = torch.cat((torch.arange(num_slots, dtype=torch.int32, device="cuda").repeat(num_blocks), torch.arange(last_block_slots, dtype=torch.int32, device="cuda")))
            self.free_blocks.append(blocks_ids[-1])

        ### copy_from_cpu_batch(self, indices: torch.Tensor, cpu_tensor: torch.Tensor): # 主动写入
        memory_indices = self.index2blk[indices] * num_slots + self.index2managerSlot[indices]
        self.pool.memory.reshape(-1, feature_size)[memory_indices] = cpu_tensor[indices].to(self.pool.device, non_blocking=True)
        end = time.time()
        print(f"[TIME] init time: {end - start}s")

    def allocate(self, indices: torch.Tensor): # indices: 预分配的nid/eid
        """ alloc for a index list """
        indices = indices.tolist()
        for idx in indices:
            if self.index2managerBlk[idx] != -1:
                continue

            while self.free_blocks:
                block = self.free_blocks[0]
                if (slot := block.allocate()) != -1:
                    self.index2blk[idx] = block.block_id
                    self.index2managerBlk[idx] = block.block_manager_id
                    self.index2managerSlot[idx] = slot
                    if block.is_full:
                        self.free_blocks.popleft()
                    break
                else:
                    self.free_blocks.popleft()
            else:
                block_view, block_id = self.pool.allocate_block()
                new_block = Block(block_id, len(self.blocks), block_view, self.feature_size)
                self.blocks.append(new_block)
                self.free_blocks.append(new_block)
                slot = new_block.allocate()
                self.index2blk[idx] = new_block.block_id
                self.index2managerBlk[idx] = new_block.block_manager_id
                self.index2managerSlot[idx] = slot

    def free(self, indices: torch.Tensor):
        """ free indices from block manager """
        indices = indices.tolist()
        blocks_to_check = set()
        for idx in indices:
            if self.index2managerBlk[idx] == -1:
                continue
            block = self.index2managerBlk[idx]
            slot = self.index2managerBlk[idx]
            block.free(slot)
            self.index2blk[idx] = -1
            self.index2managerBlk[idx] = -1
            self.index2managerSlot[idx] = -1
            blocks_to_check.add(block)

        for block in blocks_to_check:
            if not block.is_full and block not in self.free_blocks:
                self.free_blocks.appendleft(block)

    def compact(self):
        """ return free blocks to pool """
        new_blocks = []
        for block in self.blocks:
            if not block.is_empty:
                new_blocks.append(block)
            else:
                self.pool.free_block(block.block_id)
        self.blocks = new_blocks
        self.free_blocks = deque([b for b in self.blocks if not b.is_full])
        
    def get_data(self, index: int) -> torch.Tensor:
        """ get data of {index} from block manager """
        if self.index2managerBlk[index] != -1:
            # print(f"index: {index}")
            # print(f"self.index2managerBlk: {self.index2managerBlk}")
            manager_block = self.index2managerBlk[index]
            slot = self.index2managerSlot[index]
            return self.blocks[manager_block].data[slot]
        else:
            raise ValueError(f"Index {index} not alloced in block manager")

    def set_data(self, index: int, data: torch.Tensor):
        """ updata data of {index} from block manager"""
        if self.index2managerBlk[index] != -1:
            manager_block = self.index2managerBlk[index]
            slot = self.index2managerSlot[index]
            self.blocks[manager_block].data[slot] = data
        else:
            raise ValueError(f"Index {index} not alloced in block manager")

    def get_data_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """ waring : get !copied! data of indices """
        # start = time.time()
        indices = self.index2blk[indices] * self.num_slots + self.index2managerSlot[indices]
        data_views = self.pool.memory.reshape(-1, self.feature_size)[indices]
        # assert data_views.untyped_storage().data_ptr()== self.pool.memory.untyped_storage().data_ptr()
        # print(f"[TIME] get_data_batch time: {time.time() - start}s")
        return data_views

    def set_data_batch(self, indices: torch.Tensor, data: torch.Tensor):
        """ set data in batch """
        indices = indices.tolist()
        if len(indices) != data.shape[0]:
            raise ValueError("Indices and data batch size mismatch")
        
        for i, idx in enumerate(indices):
            if self.index2managerBlk[idx] == -1:
                raise KeyError(f"Index {idx} not allocated")
            manager_block = self.index2managerBlk[idx]
            slot = self.index2managerSlot[idx]
            self.blocks[manager_block].data[slot] = data[i]

    def __contains__(self, idx: int) -> bool:
        return self.index2managerBlk[idx] != -1

    def print_status(self):
        print(f"[PRINT STATUS]")
        print(f"Total blocks: {len(self.blocks)}")
        print(f"Free blocks: {len(self.free_blocks)}")
        print(f"Active indices: {len(self.index2managerBlk) - torch.sum(self.index2managerBlk == -1)}")

    def copy_from_cpu_batch(self, indices: torch.Tensor, cpu_tensor: torch.Tensor): # 主动写入
        # ! 改掉for 循环
        start = time.time()
        indices = indices.tolist()
        for idx in indices:
            gpu_data = self.get_data(idx)
            gpu_data.copy_(cpu_tensor[idx].to(self.pool.device, non_blocking=True))
        end = time.time()
        print(f"[TIME] copy_from_cpu_batch time: {end - start}s")

if __name__ == "__main__":
    shared_pool = BlockPool(total_mem_gb=20, block_elements=4096)
    
    manager_256 = BlockManager(shared_pool, feature_size=256, max_manager_index=1000)  # 4096/256=16 slots/block
    # cpu_data_256 = torch.arange(1000).repeat_interleave(256).reshape(1000, 256).to(torch.float32)
    cpu_data_256 = torch.randn(1000, 256)
    manager_256.init(torch.arange(1000), cpu_data_256)
    manager_256.print_status()
    batch_256 = manager_256.get_data_batch(torch.tensor([0, 1]))
    are_close = torch.allclose(cpu_data_256[0], batch_256[0].cpu(), atol=1e-5)
    assert are_close == True, "Data mismatch!"

    manager_128 = BlockManager(shared_pool, feature_size=128, max_manager_index=200)  # 4096/128=32 slots/block
    cpu_data_128 = torch.randn(200, 128)
    manager_128.init(torch.arange(200), cpu_data_128)
    manager_128.print_status()
    batch_128 = manager_128.get_data_batch(torch.tensor([0, 1]))
    are_close = torch.allclose(cpu_data_128[0], batch_128[0].cpu(), atol=1e-5)
    assert are_close == True, "Data mismatch!"
    
    shared_pool.print_status()