import torch
import tglite as tg

from torch import nn, Tensor
from tglite.nn import TemporalAttnLayer
from tglite._stats import tt
import threading

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..')) 
import support
from tglite.gpu_mem_track import *
from tglite.mymodule import *
import time
import nvtx
import tglite.config
from tglite._block import TBlock

class TGN(nn.Module):
    def __init__(self, ctx: tg.TContext,
                 dim_node: int, dim_edge: int, dim_time: int, dim_embed: int,
                 sampler: tg.TSampler, num_layers=2, num_heads=2, dropout=0.1,
                 dedup: bool = True):
        super().__init__()
        self.ctx = ctx
        self.dim_embed = dim_embed
        self.dim_edge = dim_edge
        self.dim_mail = 2 * self.dim_embed + self.dim_edge # self.dim_mail + self.dim_time == self.dim_mailbox
        self.dim_time = dim_time
        self.num_layers = num_layers
        self.nfeat_map = None if dim_node == dim_embed else nn.Linear(dim_node, dim_embed)
        self.mem_cell = GRUCell(2 * dim_embed + dim_edge + dim_time, dim_embed)
        self.mem_time_encode = tg.nn.TimeEncode(dim_time)
        self.attn = nn.ModuleList([
            TemporalAttnLayer(ctx,
                num_heads=num_heads,
                dim_node=dim_embed,
                dim_edge=dim_edge,
                dim_time=dim_time,
                dim_out=dim_embed,
                dropout=dropout)
            for i in range(num_layers)])
        self.sampler = sampler
        self.edge_predictor = support.EdgePredictor(dim=dim_embed)
        self.dedup = dedup

        self.is_train = True
        self._new_samples = None
        self.sampling_thread = None
        self.next_data = None
        # wyq add record uniq-nbrs
        self.dstnodes2latestNbrs = torch.zeros([tglite.config.num_nodes, 2], dtype=torch.long) # 可以用int # TODO **去冗余重建_mailbox**

    def forward(self, batch: tg.TBatch) -> Tensor:
        if tglite.config.PERF_CEIL and self.is_train:
            return self.forward2_perfCeil(batch) # TODO
        elif tglite.config.ON_HETER:
            return self.forward0(batch)
        elif tglite.config.OFFLINE_SAMPLE and self.is_train:
            return self.forward1_offlineSample(batch)
        else:
            return self.forward_origin(batch)



    def forward0(self, batch: tg.TBatch) -> Tensor:
        # 针对 preload + memory.update
        # 目标：时间最好不要长 观察到显存下降
        # 1. 先去重
        # 2. 实现HETER-AWARE 的存储分布
        # 3. 根据存储分布切pipeline
        with nvtx.annotate("forward", color="purple"):
            head = batch.block(self.ctx)

            # setup message passing
            for i in range(self.num_layers): # TODO 两层GNN有额外的问题，先处理1层
                tail = head if i == 0 \
                    else tail.next_block(include_dst=True, use_dst_times=False)
                tail = tg.op.dedup(tail) if self.dedup else tail
                with nvtx.annotate("sample", color="purple"):
                    tail = self.sampler.sample(tail)
            # TODO 这里可以暂时整体offload -> 未来加入采样动态性的考虑

            if tail.num_dst() > 0:
                with nvtx.annotate("update mem", color="purple"):
                    ## well_optimized_memUpd
                    mem = self.well_optimized_memUpd(head, tail, batch)

                    
                    ## uniqLoadFeat_memUpd
                    # mem = self.update_memory0_uniqLoadFeat_memUpd(head, tail, batch)


                    # mem = self.update_memory0_pipeline_cudagraph_memUpd(head, tail, batch)
                    # mem = self.update_memory0_pipeline_alloc_memUpd(head, tail, batch)

                    ## pipeline_memUpd **comm(mem, mailbox) comp(update mem)**
                    # mem = self.update_memory0_pipeline_memUpd(head, tail, batch)
                    ## no_redundant_memUpd
                    # mem = self.update_memory0_no_redundant_memUpd(head, tail, batch) # [TODO @ZZZ]
                    ## unique_nodes_memUpd
                    # mem = self.update_memory0_unique_nodes(head, tail, batch)
                    ## no pipeline
                    # mem = self.update_memory0(head, tail, batch)
                nfeat = tail.nfeat() if self.nfeat_map is None else self.nfeat_map(tail.nfeat())
                tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
                tail.srcdata['h'] = nfeat[tail.num_dst():] + mem[tail.num_dst():]
                del nfeat
                del mem

            # compute embeddings
            with nvtx.annotate("op aggr", color="purple"):
                embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
            del head
            del tail

            # compute scores
            with nvtx.annotate("compute score", color="purple"):
                src, dst, neg = batch.split_data(embeds)
                scores = self.edge_predictor(src, dst)
                if neg is not None:
                    scores = (scores, self.edge_predictor(src, neg))
            del embeds
            del src
            del dst
            del neg
            with nvtx.annotate("save raw msgs", color="purple"):
                self.save_raw_msgs0(batch)
                ## no_redundant_memUpd
                # self.save_raw_msgs0_no_redundant_memUpd(batch)
            return scores
        
    ## well_optimized_memUpd
    def well_optimized_memUpd(self, head: tg.TBlock, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # load data / feats
        with nvtx.annotate("preload data/feat", color="purple"):
            tg.op.preload0_uniqLoadFeat_noMailMem(head, use_pin=True)
        
        cdev = blk.g.compute_device()
        nodes = blk.allnodes()
        unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)


        # TODO cal delta： mail_delta[nxt_start_idx:nxt_end_idx] 而不是 mail_delta[unique_nodes[nxt_start_idx:nxt_end_idx]]
        unique_mail_ts = blk.g.mailbox.time[unique_nodes]
        delta = unique_mail_ts - blk.g.mem.time[unique_nodes]
        mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to(cdev)) # on GPU

        # # test accuracy
        # mail_standard = torch.cat([blk._g.mailbox.mail[unique_nodes].to(cdev), mail_delta], dim=1)
        # mem_standard = blk._g.mem._data[unique_nodes].to(cdev)
        # mem_standard = self.mem_cell(mail_standard, mem_standard)

        # TODO divide pipelines
        # stage0_index = math.floor(len(unique_nodes)/3)
        # stage_indices = [(0, stage0_index), (stage0_index, math.floor(len(unique_nodes)*2/3)), (math.floor(len(unique_nodes)*2/3), len(unique_nodes))] 
        stage0_index = math.floor(len(unique_nodes)/2)
        stage_indices = [(0, stage0_index), (stage0_index, len(unique_nodes))] 
        # pre-alloc
        # max_size = math.ceil(len(unique_nodes)/2)
        # pre_allocated_mem = torch.empty((max_size, 100), device=cdev)
        # 0阶段通信
        with nvtx.annotate(f"memUpdStage_comm_{0}", color="green"):
            unique_nodes_slice = unique_nodes[0:stage0_index]
            # all on cpu
            cur_mail_mem = blk._load_mail_mem_data_slice(unique_nodes_slice)
        # for 循环
        comp_stream_memUpd = torch.cuda.Stream()
        comm_stream_memUpd = torch.cuda.Stream()
        nxt_mail_mem = None
        mems = []
        for i, (start_idx, nxt_idx) in enumerate(stage_indices):
            with nvtx.annotate(f"Stage_{i}", color="blue"):
                # nxt阶段通信
                if i < len(stage_indices) - 1:
                    with torch.cuda.stream(comm_stream_memUpd), nvtx.annotate(f"memUpdStage_comm_{i+1}", color="green"):
                        nxt_start_idx, nxt_end_idx = stage_indices[i + 1]
                        nxt_unique_nodes_slice = unique_nodes[nxt_start_idx:nxt_end_idx]
                        # all on cpu
                        nxt_mail_mem = blk._load_mail_mem_data_slice(nxt_unique_nodes_slice)
                # cur阶段计算
                with torch.cuda.stream(comp_stream_memUpd), nvtx.annotate(f"memUpdStage_comp_{i}", color="red"):
                    cur_mail = torch.cat([cur_mail_mem[:, :self.dim_mail], mail_delta[start_idx: nxt_idx]], dim=1)
                    cur_mem = cur_mail_mem[:, self.dim_mail:]
                    mem = self.mem_cell(cur_mail, cur_mem)
                mems.append(mem)
                torch.cuda.synchronize()
                # 在nxt阶段使用通信好的张量
                if nxt_mail_mem is not None:
                    cur_mail_mem = nxt_mail_mem
                    nxt_mail_mem = None
        torch.cuda.current_stream().wait_stream(comp_stream_memUpd)
        mem = torch.cat(mems)
        # assert torch.allclose(mem_standard, mem) # accuracy test pass! \O/

        # 写回也可以pipeline TODO 不一定有必要，可以和后面的计算掩盖
        blk.g.mem.update(unique_nodes, mem, unique_mail_ts) # 目前为写回CPU

        return mem[inverse_indices]

    # uniqLoadFeat_memUpd: node feat and edge feat
    def update_memory0_uniqLoadFeat_memUpd(self, head: tg.TBlock, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # load data / feats
        with nvtx.annotate("preload data/feat", color="purple"):
            tg.op.preload0_uniqLoadFeat(head, use_pin=True)

        cdev = blk.g.compute_device()
        nodes = blk.allnodes()
        time_start_0 = tt.start()
        mail_ts = blk.g.mailbox.time[nodes] # on cpu? no new segments
        tt.tt_mail_ts_load += tt.elapsed(time_start_0)

        delta = mail_ts - blk.g.mem.time[nodes]
        delta = delta.squeeze().to(cdev)
        with nvtx.annotate("update mem-precompute_times", color="purple"):
            mail = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta) # on cpu
        mail = torch.cat([blk.mail(), mail], dim=1) # no new segments 不等同于no request

        mem = blk.mem_data() # on cpu?
        time_start_1 = tt.start()
        with nvtx.annotate("update mem-mem_cell", color="purple"):
            mem = self.mem_cell(mail, mem)
        tt.t_mem_update_gru_cell += tt.elapsed(time_start_1)
        time_start_2 = tt.start()
        blk.g.mem.update(nodes, mem, mail_ts)
        tt.t_mem_update_after += tt.elapsed(time_start_2)
        return mem

    # TODO pipeline_cudagraph_memzUpd comm(mem, mailbox) comp(update mem)
    def update_memory0_pipeline_cudagraph_memUpd(self, head: tg.TBlock, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        pass

    # TODO pipeline_alloc_memUpd comm(mem, mailbox) comp(update mem)
    # TODO alloc
    # TODO cudaEvent
    # TODO precompute在CPU上会变得很合理，这样load一次是完整的mem_mail_delta 
    def update_memory0_pipeline_alloc_memUpd(self, head: tg.TBlock, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        torch.autograd.set_detect_anomaly(True)
        # TODO 为什么切pipeline之后通信变少的和通信段数不成比例
        with nvtx.annotate("preload data/feat", color="purple"):
            tg.op.preload0_noMailMem(head, use_pin=True)

        cdev = blk.g.compute_device()
        nodes = blk.allnodes()
        unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)

        # TODO 传输过来再做计算
        # TODO cal delta： mail_delta[nxt_start_idx:nxt_end_idx] 而不是 mail_delta[unique_nodes[nxt_start_idx:nxt_end_idx]]
        unique_mail_ts = blk.g.mailbox.time[unique_nodes]
        delta = unique_mail_ts - blk.g.mem.time[unique_nodes]
        mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to(cdev)) # on GPU

        # # test accuracy
        # mail_standard = torch.cat([blk._g.mailbox.mail[unique_nodes].to(cdev), mail_delta], dim=1)
        # mem_standard = blk._g.mem._data[unique_nodes].to(cdev)
        # mem_standard = self.mem_cell(mail_standard, mem_standard)

        # TODO divide pipelines
        # stage0_index = math.floor(len(unique_nodes)/3)
        # stage_indices = [(0, stage0_index), (stage0_index, math.floor(len(unique_nodes)*2/3)), (math.floor(len(unique_nodes)*2/3), len(unique_nodes))] 
        stage0_index = math.floor(len(unique_nodes)/2)
        stage_indices = [(0, stage0_index), (stage0_index, len(unique_nodes))] 
        # pre-alloc
        pre_alloc_cur_mem_mail = torch.empty((math.ceil(len(unique_nodes)/2), 572), device=cdev)
        pre_alloc_nxt_mem_mail = torch.empty((math.ceil(len(unique_nodes)/2), 572), device=cdev)
        pre_alloc_mems = torch.empty((len(unique_nodes), 100), device=cdev)
        mems = []
        # 0阶段通信
        with nvtx.annotate(f"memUpdStage_comm_{0}", color="green"):
            unique_nodes_slice = unique_nodes[0:stage0_index]
            # all on cpu
            pre_alloc_cur_mem_mail[0:stage0_index, self.dim_mail + self.dim_embed:self.dim_mail + self.dim_embed + self.dim_time] = mail_delta[0:stage0_index]
            pre_alloc_cur_mem_mail[0:stage0_index, 0:self.dim_mail + self.dim_embed] = (blk._load_mem_mail_data_slice(unique_nodes_slice))
        # for 循环
        comp_stream_memUpd = torch.cuda.Stream()
        comm_stream_memUpd = torch.cuda.Stream()
        nxt_mem_mail = None
        for i, (start_idx, end_idx) in enumerate(stage_indices):
            with nvtx.annotate(f"Stage_{i}", color="blue"):
                # nxt阶段通信
                if i < len(stage_indices) - 1:
                    with torch.cuda.stream(comm_stream_memUpd), nvtx.annotate(f"memUpdStage_comm_{i+1}", color="green"):
                        nxt_start_idx, nxt_end_idx = stage_indices[i + 1]
                        nxt_unique_nodes_slice = unique_nodes[nxt_start_idx:nxt_end_idx]
                        # all on cpu
                        nxt_mem_mail = blk._load_mem_mail_data_slice(nxt_unique_nodes_slice)
                        pre_alloc_nxt_mem_mail[0:nxt_end_idx-nxt_start_idx, 0:self.dim_mail + self.dim_embed] = nxt_mem_mail
                        pre_alloc_nxt_mem_mail[0:nxt_end_idx-nxt_start_idx, self.dim_mail + self.dim_embed:self.dim_mail + self.dim_embed + self.dim_time] = mail_delta[nxt_start_idx:nxt_end_idx]
                # cur阶段计算
                with torch.cuda.stream(comp_stream_memUpd), nvtx.annotate(f"memUpdStage_comp_{i}", color="red"):
                    cur_mem = pre_alloc_cur_mem_mail[0:end_idx-start_idx, 0:100]
                    cur_mail = pre_alloc_cur_mem_mail[0:end_idx-start_idx, 100:]
                    mem = self.mem_cell(cur_mail, cur_mem)
                    # pre_alloc_mems[start_idx:end_idx] = mem
                mems.append(mem)
                torch.cuda.synchronize()
                # 在nxt阶段使用通信好的张量
                if nxt_mem_mail is not None:
                    nxt_start_idx, nxt_end_idx = stage_indices[i + 1]
                    pre_alloc_cur_mem_mail[0:nxt_end_idx-nxt_start_idx, :] = pre_alloc_nxt_mem_mail[0:nxt_end_idx-nxt_start_idx, :]
                    nxt_mem_mail = None
        torch.cuda.current_stream().wait_stream(comp_stream_memUpd)
        mem = torch.cat(mems)
        # assert torch.allclose(mem_standard, mem) # accuracy test pass! \O/
        print(f"pre_alloc_mems: {pre_alloc_mems.size()}")

        # 写回也可以pipeline TODO 不一定有必要，可以和后面的计算掩盖
        # blk.g.mem.update(unique_nodes, pre_alloc_mems, unique_mail_ts) # 目前为写回CPU
        blk.g.mem.update(unique_nodes, mem, unique_mail_ts) # 目前为写回CPU

        return pre_alloc_mems[inverse_indices]

    # pipeline_memUpd comm(mem, mailbox) comp(update mem)
    def update_memory0_pipeline_memUpd(self, head: tg.TBlock, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # TODO 写得没什么问题，但是通信太长计算太短，pipeline 不起来 -> 不对，没有用pin mem
        # TODO using pinned_memory
        # TODO using cuda.graph
        # load data / feats
        with nvtx.annotate("preload data/feat", color="purple"):
            tg.op.preload0_noMailMem(head, use_pin=True)

        cdev = blk.g.compute_device()
        nodes = blk.allnodes()
        unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)


        # TODO cal delta： mail_delta[nxt_start_idx:nxt_end_idx] 而不是 mail_delta[unique_nodes[nxt_start_idx:nxt_end_idx]]
        unique_mail_ts = blk.g.mailbox.time[unique_nodes]
        delta = unique_mail_ts - blk.g.mem.time[unique_nodes]
        mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to(cdev)) # on GPU

        # # test accuracy
        # mail_standard = torch.cat([blk._g.mailbox.mail[unique_nodes].to(cdev), mail_delta], dim=1)
        # mem_standard = blk._g.mem._data[unique_nodes].to(cdev)
        # mem_standard = self.mem_cell(mail_standard, mem_standard)

        # TODO divide pipelines
        # stage0_index = math.floor(len(unique_nodes)/3)
        # stage_indices = [(0, stage0_index), (stage0_index, math.floor(len(unique_nodes)*2/3)), (math.floor(len(unique_nodes)*2/3), len(unique_nodes))] 
        stage0_index = math.floor(len(unique_nodes)/2)
        stage_indices = [(0, stage0_index), (stage0_index, len(unique_nodes))] 
        # pre-alloc
        # max_size = math.ceil(len(unique_nodes)/2)
        # pre_allocated_mem = torch.empty((max_size, 100), device=cdev)
        # 0阶段通信
        with nvtx.annotate(f"memUpdStage_comm_{0}", color="green"):
            unique_nodes_slice = unique_nodes[0:stage0_index]
            # all on cpu
            cur_mail_mem = blk._load_mail_mem_data_slice(unique_nodes_slice)
            print(f"cur_mail_mem {cur_mail_mem.size()}")
        # for 循环
        comp_stream_memUpd = torch.cuda.Stream()
        comm_stream_memUpd = torch.cuda.Stream()
        nxt_mail_mem = None
        mems = []
        for i, (start_idx, nxt_idx) in enumerate(stage_indices):
            with nvtx.annotate(f"Stage_{i}", color="blue"):
                # nxt阶段通信
                if i < len(stage_indices) - 1:
                    with torch.cuda.stream(comm_stream_memUpd), nvtx.annotate(f"memUpdStage_comm_{i+1}", color="green"):
                        nxt_start_idx, nxt_end_idx = stage_indices[i + 1]
                        nxt_unique_nodes_slice = unique_nodes[nxt_start_idx:nxt_end_idx]
                        # all on cpu
                        nxt_mail_mem = blk._load_mail_mem_data_slice(nxt_unique_nodes_slice)
                        print(f"nxt_mail_mem {nxt_mail_mem.size()}")
                # cur阶段计算
                with torch.cuda.stream(comp_stream_memUpd), nvtx.annotate(f"memUpdStage_comp_{i}", color="red"):
                    cur_mail = torch.cat([cur_mail_mem[:, :self.dim_mail], mail_delta[start_idx: nxt_idx]], dim=1)
                    cur_mem = cur_mail_mem[:, self.dim_mail:]
                    mem = self.mem_cell(cur_mail, cur_mem)
                mems.append(mem)
                torch.cuda.synchronize()
                # 在nxt阶段使用通信好的张量
                if nxt_mail_mem is not None:
                    cur_mail_mem = nxt_mail_mem
                    nxt_mail_mem = None
        torch.cuda.current_stream().wait_stream(comp_stream_memUpd)
        mem = torch.cat(mems)
        # assert torch.allclose(mem_standard, mem) # accuracy test pass! \O/
        print()

        # 写回也可以pipeline TODO 不一定有必要，可以和后面的计算掩盖
        blk.g.mem.update(unique_nodes, mem, unique_mail_ts) # 目前为写回CPU

        return mem[inverse_indices]
                
    # no_redundant_memUpd
    def update_memory0_no_redundant_memUpd(self, head: tg.TBlock, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # load data / feats
        with nvtx.annotate("preload data/feat", color="purple"):
            tg.op.preload(head, use_pin=True)

        cdev = blk.g.compute_device()
        nodes = blk.allnodes()
        unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)
        unique_mail_ts = blk.g.mailbox.time[unique_nodes]
        delta = unique_mail_ts - blk.g.mem.time[unique_nodes]
        mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to(cdev))
        mem = blk._g.mem._data[unique_nodes].to(cdev)
        mail = torch.cat([blk._g.mailbox.mail[unique_nodes].to(cdev), mail_delta], dim=1)
        # TODO 不要load 完整的blk._g.mailbox.mail[unique_nodes]
        # 将blk._g.mem._data[unique_nodes] 和 blk._g.mailbox.mail[unique_nodes] 组成一个新张量TMP 和索引表 传输到GPU 上
        # 根据TMP 和索引重建mail 和 mem

        # YOUR_CODE HERE

        # # 测试正确性
        # assert torch.allclose(mail, YOUR_mail)

        # # 统计YOUR_mail 相比于mail 少load 了多少；注意控制显存峰值
        mem = self.mem_cell(mail, mem)
        blk.g.mem.update(unique_nodes, mem, unique_mail_ts)

        return mem[inverse_indices]

    # unique_nodes_memUpd
    def update_memory0_unique_nodes(self, head: tg.TBlock, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # load data / feats
        with nvtx.annotate("preload data/feat", color="purple"):
            tg.op.preload0_noMailMem(head, use_pin=True)
        
        # 1. [DONE]对update_memory0进行去重
        # 2. [TODO]将下个batch使用的特征放在GPU上，其余放在CPU上
        # 3. [TODO]将热节点放在GPU上，其余放在CPU上
        cdev = blk.g.compute_device()
        nodes = blk.allnodes()
        unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)
        unique_mail_ts = blk.g.mailbox.time[unique_nodes]
        delta = unique_mail_ts - blk.g.mem.time[unique_nodes]
        mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to(cdev)) # on GPU
        mail_mem = blk._load_mail_mem_data_slice(unique_nodes)
        mem = mail_mem[:, self.dim_mail:]
        mail = torch.cat([mail_mem[:, :self.dim_mail], mail_delta], dim=1)


        # # rebuild mailbox from mem (按unique_nodes索引mailbox):
        # TODO **去冗余重建_mailbox**
        # mailbox_uniq = blk._g.mailbox._mail[unique_nodes, :self.dim_embed].to(cdev)
        # mailbox_nbrs = blk._g.mailbox._mail[unique_nodes, self.dim_embed:2*self.dim_embed].to(cdev)
        # mailbox_edata = blk._g.mailbox._mail[unique_nodes, 2*self.dim_embed:].to(cdev)
        # assert torch.allclose(mail, my_mail)
        

        # after rebuild mailbox from mem
        mem = self.mem_cell(mail, mem)
        blk.g.mem.update(unique_nodes, mem, unique_mail_ts) # 目前为写回CPU

        return mem[inverse_indices]
    
    def update_memory0(self, head: tg.TBlock, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # load data / feats
        with nvtx.annotate("preload data/feat", color="purple"):
            tg.op.preload(head, use_pin=True)

        cdev = blk.g.compute_device()
        nodes = blk.allnodes()
        time_start_0 = tt.start()
        mail_ts = blk.g.mailbox.time[nodes] # on cpu? no new segments
        tt.tt_mail_ts_load += tt.elapsed(time_start_0)

        delta = mail_ts - blk.g.mem.time[nodes]
        delta = delta.squeeze().to(cdev)
        with nvtx.annotate("update mem-precompute_times", color="purple"):
            mail = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta) # on cpu
        mail = torch.cat([blk.mail(), mail], dim=1) # no new segments 不等同于no request

        mem = blk.mem_data() # on cpu?
        time_start_1 = tt.start()
        with nvtx.annotate("update mem-mem_cell", color="purple"):
            mem = self.mem_cell(mail, mem)
        tt.t_mem_update_gru_cell += tt.elapsed(time_start_1)
        time_start_2 = tt.start()
        blk.g.mem.update(nodes, mem, mail_ts)
        tt.t_mem_update_after += tt.elapsed(time_start_2)
        return mem

    def save_raw_msgs0_no_redundant_memUpd(self, batch: tg.TBatch):
        # 虽然写得不对 但是精度完全不会掉 能水就不要放过 WYQ TODO
        sdev = batch.g.storage_device()
        mem = batch.g.mem.data

        with nvtx.annotate("save raw msgs-block_adj", color="red"):
            # 由于new blk 我肯定load了很多没用的东西
            blk = batch.block_adj(self.ctx)
        with nvtx.annotate("save raw msgs-op.coalesce", color="red"):
            blk = tg.op.coalesce(blk, by='latest')

        with nvtx.annotate("save raw msgs-uniq nbrs", color="red"):
            # 写回数据量实际很小 μs级别
            uniq = torch.from_numpy(blk.dstnodes).long().to(sdev)
            nbrs = torch.from_numpy(blk.srcnodes).long().to(sdev)
            
            # TODO **去冗余重建_mailbox**
            # self.dstnodes2latestNbrs[uniq, 0] = nbrs
            if self.dim_edge > 0:
                eids = torch.from_numpy(blk.eid).long().to(sdev)
                # self.dstnodes2latestNbrs[uniq, 1] = eids
                mail = torch.cat([mem[uniq], mem[nbrs], batch.g.efeat[eids]], dim=1)
            else:
                mail = torch.cat([mem[uniq], mem[nbrs]], dim=1)
            mail_ts = torch.from_numpy(blk.ets).to(sdev)
        with nvtx.annotate("save raw msgs-store mail_ts", color="red"):
            batch.g.mailbox.store(uniq, mail, mail_ts)

    def save_raw_msgs0(self, batch: tg.TBatch):
        # 虽然写得不对 但是精度完全不会掉 能水就不要放过 WYQ TODO
        sdev = batch.g.storage_device()
        mem = batch.g.mem.data

        with nvtx.annotate("save raw msgs-block_adj", color="red"):
            # 由于new blk 我肯定load了很多没用的东西
            blk = batch.block_adj(self.ctx)
        with nvtx.annotate("save raw msgs-op.coalesce", color="red"):
            blk = tg.op.coalesce(blk, by='latest')

        with nvtx.annotate("save raw msgs-uniq nbrs", color="red"):
            # 写回数据量实际很小 μs级别
            uniq = torch.from_numpy(blk.dstnodes).long().to(sdev)
            nbrs = torch.from_numpy(blk.srcnodes).long().to(sdev)
            
            # TODO **去冗余重建_mailbox**
            # self.dstnodes2latestNbrs[uniq, 0] = nbrs
            if self.dim_edge > 0:
                eids = torch.from_numpy(blk.eid).long().to(sdev)
                # self.dstnodes2latestNbrs[uniq, 1] = eids
                mail = torch.cat([mem[uniq], mem[nbrs], batch.g.efeat[eids]], dim=1)
            else:
                mail = torch.cat([mem[uniq], mem[nbrs]], dim=1)
            mail_ts = torch.from_numpy(blk.ets).to(sdev)
        with nvtx.annotate("save raw msgs-store mail_ts", color="red"):
            batch.g.mailbox.store(uniq, mail, mail_ts)



    def _load_new_samples(self):
        if self._new_samples is None:
            file_path = os.path.join(tglite.config.log_dir, f"new_samples_{tglite.config.log_name}.pt")
            self._new_samples = torch.load(file_path)
            self.next_data = self._new_samples[0]

    def forward1_offlineSample(self, batch: tg.TBatch) -> Tensor:
        print(f"batch {batch._b_id}")
        with nvtx.annotate("forward", color="purple"):
        #     # setup message passing
            # with nvtx.annotate("dedup and cache", color="purple"):
            #     head = batch.block(self.ctx)
            #     for i in range(self.num_layers):
            #         tail = head if i == 0 \
            #             else tail.next_block(include_dst=True, use_dst_times=False)
            #         tail = tg.op.dedup(tail) if self.dedup else tail
            #         with nvtx.annotate("sample", color="purple"):
            #             tail = self.sampler.sample(tail)
            # # head._dsttimes = torch.randn([head._dstnodes.shape[0]]).numpy()
            # # print(f"head._dstnodes {head._dstnodes.shape}")
            # # print(f"head._dstindex {head._dstindex.shape}")

            # # offline sample
            with nvtx.annotate("offline sample", color="purple"):
                # TODO
                #  TBlock def __init__(self, ctx: 'TContext', layer: int, dstnodes: np.ndarray, dsttimes: np.ndarray,
                #  dstindex: np.ndarray = None, srcnodes: np.ndarray = None,
                #  eid: np.ndarray = None, ets: np.ndarray = None):
                _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, \
                prev_eids, next_eids, \
                prev_nodes, next_nodes, \
                unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets, \
                _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, \
                _eids_nxt, _idx_eids_nxt, \
                _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu, \
                _nids_nxt, _idx_nids_nxt = self._new_samples[batch._b_id]
            #     print(f"b_dstnodes {b_dstnodes.size()}")
            #     print(f"b_dstindex {b_dstindex.size()}")
            #     print(f"b_srcnodes {b_srcnodes.size()}")
            #     print(f"b_eids {b_eids.size()}")
            #     print(f"b_ets {b_ets.size()}")
            #     print("*****")
            #     print(f"unique_eids {unique_eids.size()}")
            #     print(f"_reverse_eids {_reverse_eids.size()}")
            #     print(f"_unique_nids {_unique_nids.size()}")
            #     print(f"_reverse_nids {_reverse_nids.size()}")
            #     print(f"_unique_ets {_unique_ets.size()}")
            #     print(f"_reverse_ets {_reverse_ets.size()}")
            #     print("*****")
            #     print(f"_eids_pre {_eids_pre.size()}")
            #     print(f"_eids_nxt {_eids_nxt.size()}")
            #     print(f"_nids_pre {_nids_pre.size()}")
            #     print(f"_nids_nxt {_nids_nxt.size()}")
            #     print()
                # TODO samples add dsttimes & TODO add two-layer offline
                head = TBlock(self.ctx, 0, b_dstnodes.numpy(), b_dsttimes.numpy(), b_dstindex.numpy(), b_srcnodes.numpy(), b_eids.numpy(), b_ets.numpy())
                for i in range(self.num_layers):
                    tail = head if i == 0 \
                    else tail.next_block(include_dst=True, use_dst_times=False) # TODO 好像不需要add two-layer offline
                    tg.op.dedup1_offlineSample(tail, _inv_idx) # TODO _reverse_eids 不对 这里是找回bs*3的映射


            # load data / feats
            with nvtx.annotate("preload data/feat", color="purple"):
                tg.op.preload(head, use_pin=True)
            if tail.num_dst() > 0:
                # t_start = tt.start()
                with nvtx.annotate("update mem", color="purple"):
                    mem = self.update_memory(tail, batch)
                # tt.t_update_memory += tt.elapsed(t_start)
                nfeat = tail.nfeat() if self.nfeat_map is None else self.nfeat_map(tail.nfeat())
                # torch.cuda.empty_cache()
                tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
                tail.srcdata['h'] = nfeat[tail.num_dst():] + mem[tail.num_dst():]
                # tt.t_mem_update += tt.elapsed(t_start)
                del nfeat
                del mem

            # compute embeddings
            with nvtx.annotate("op aggr", color="purple"):
                # torch.cuda.empty_cache() # del 的变量占用的空间?不会立即释放
                embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
            del head
            del tail

            # compute scores
            with nvtx.annotate("compute score", color="purple"):
                # torch.cuda.empty_cache()
                src, dst, neg = batch.split_data(embeds)
                scores = self.edge_predictor(src, dst) # TODO 这里有冗余吗? 16000->18000 而且src是不是算了两遍
                # torch.cuda.empty_cache()
                if neg is not None:
                    scores = (scores, self.edge_predictor(src, neg)) # 可以这里再转回来? TODO compute score总共430μs 空间有限
            del embeds
            del src
            del dst
            del neg

            # memory messages
            t_start = tt.start()
            with nvtx.annotate("save raw msgs", color="purple"):
                self.save_raw_msgs(batch)
            tt.t_post_update += tt.elapsed(t_start)

            return scores



    def save_raw_msgs2_perfCeil(self, batch: tg.TBatch):
        sdev = batch.g.storage_device()
        mem = batch.g.mem.data

        with nvtx.annotate("save raw msgs-block_adj", color="red"):
            # 由于new blk 我肯定load了很多没用的东西
            blk = batch.block_adj(self.ctx)
        with nvtx.annotate("save raw msgs-op.coalesce", color="red"):
            blk = tg.op.coalesce(blk, by='latest')

        with nvtx.annotate("save raw msgs-uniq nbrs", color="red"):
            # 写回数据量实际很小 μs级别
            uniq = torch.from_numpy(blk.dstnodes).long().to(sdev)
            nbrs = torch.from_numpy(blk.srcnodes).long().to(sdev)
            
            # TODO **去冗余重建_mailbox**
            # self.dstnodes2latestNbrs[uniq, 0] = nbrs
            if self.dim_edge > 0:
                eids = torch.from_numpy(blk.eid).long().to(sdev)
                # self.dstnodes2latestNbrs[uniq, 1] = eids
                mail = torch.cat([mem[uniq], mem[nbrs], batch.g.efeat[eids]], dim=1)
            else:
                mail = torch.cat([mem[uniq], mem[nbrs]], dim=1)
            mail_ts = torch.from_numpy(blk.ets).to(sdev)
        with nvtx.annotate("save raw msgs-store mail_ts", color="red"):
            batch.g.mailbox.store(uniq, mail, mail_ts, uniq, nbrs, batch._b_id)

    # no_redundant_memUpd + uniqLoadFeat_memUpd: node feat and edge feat
    def update_memory2_no_redundant_memUpd(self, head: tg.TBlock, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # load data / feats
        with nvtx.annotate("preload data/feat", color="purple"):
            tg.op.preload0_uniqLoadFeat_noMailMem(head, use_pin=True)

        with nvtx.annotate("preload mem/mailbox", color="purple"):
            cdev = blk.g.compute_device()
            nodes = blk.allnodes()
            unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)
            unique_mail_ts = blk.g.mailbox.time[unique_nodes]
            delta = unique_mail_ts - blk.g.mem.time[unique_nodes]
            mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to(cdev))
            mem = blk._g.mem._data[unique_nodes].to(cdev)
            mail = torch.cat([blk._g.mailbox.mail[unique_nodes].to(cdev), mail_delta], dim=1)
        # TODO 不要load 完整的blk._g.mailbox.mail[unique_nodes]
        # 将blk._g.mem._data[unique_nodes] 和 blk._g.mailbox.mail[unique_nodes] 组成一个新张量TMP 和索引表 传输到GPU 上
        # 根据TMP 和索引重建mail 和 mem

        # YOUR_CODE HERE

        # # 测试正确性
        # assert torch.allclose(mail, YOUR_mail)

        # # 统计YOUR_mail 相比于mail 少load 了多少；注意控制显存峰值
        with nvtx.annotate("nn.GRUCell", color="purple"):
            mem = self.mem_cell(mail, mem)
        with nvtx.annotate("blk.g.mem.update", color="purple"):
            blk.g.mem.update(unique_nodes, mem, unique_mail_ts)

        return mem[inverse_indices]

    def _load_new_samples2(self):
        if self._new_samples is None:
            file_path = os.path.join(tglite.config.log_dir, f"new_samples_{tglite.config.log_name}.pt")
            self._new_samples = torch.load(file_path)
            self.next_data = self._new_samples[0]
            _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, \
            prev_eids, next_eids, \
            prev_nodes, next_nodes, \
            unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets, \
            _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, \
            _eids_nxt, _idx_eids_nxt, \
            _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu, \
            _nids_nxt, _idx_nids_nxt = self.next_data
            # preload nfeat
            ## 1 layer: self.layer == 0 TODO 不过preload 一般只preload 1层就可以
            ## self.ctx._nxt_nfeat_pins = self.ctx._get_nfeat_pin(self.layer, len(_nids_cpu), self._g.nfeat.shape[1])
            self.ctx._nxt_nfeat_pins = self.ctx._get_nfeat_pin(0, len(_nids_cpu), self.ctx._g.nfeat.shape[1])
            with nvtx.annotate("index_select", color="red"):
                torch.index_select(self.ctx._g.nfeat, 0, _nids_cpu, out=self.ctx._nxt_nfeat_pins)
            # preload efeat
            self.ctx._nxt_efeat_pins = self.ctx._get_efeat_pin(0, len(_eids_cpu), self.ctx._g.efeat.shape[1])
            with nvtx.annotate("index_select", color="red"):
                torch.index_select(self.ctx._g.efeat, 0, _eids_cpu, out=self.ctx._nxt_efeat_pins)
    
    def forward2_perfCeil(self, batch: tg.TBatch) -> Tensor:
        def sampling(self, _b_id):
            # _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, \
            # prev_eids, next_eids, \
            # prev_nodes, next_nodes, \
            # unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets, \
            # _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, \
            # _eids_nxt, _idx_eids_nxt, \
            # _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu, \
            # _nids_nxt, _idx_nids_nxt = self._new_samples[_b_id]
            with nvtx.annotate("thread sampling", color="purple"):
                if _b_id > len(self._new_samples):
                    self.next_data = None
                self.next_data = self._new_samples[_b_id]

        # print(f"batch {batch._b_id}")

        # 先集成优化，再节约已经GPU上的load feat
        with nvtx.annotate("forward", color="purple"):

            # 提前取下一个batch & TODO add preload logic
            with nvtx.annotate("thread sampling nxt", color="purple"):
                _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, \
                prev_eids, next_eids, \
                prev_nodes, next_nodes, \
                unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets, \
                _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, \
                _eids_nxt, _idx_eids_nxt, \
                _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu, \
                _nids_nxt, _idx_nids_nxt = self.next_data

                # Sampling for next batch
                sampling_thread = threading.Thread(target=sampling, args=(self, batch._b_id + 1,))
                sampling_thread.start()

            # # offline sample
            with nvtx.annotate("offline sample", color="purple"):
                # TODO
                #  TBlock def __init__(self, ctx: 'TContext', layer: int, dstnodes: np.ndarray, dsttimes: np.ndarray,
                #  dstindex: np.ndarray = None, srcnodes: np.ndarray = None,
                #  eid: np.ndarray = None, ets: np.ndarray = None):
                
                '''
                # new_sample = (
                #     b_inv_idx,
                #     b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets,
                #     prev_eids, next_eids,
                #     prev_nodes, next_nodes,
                #     _unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets,
                #     _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu,
                #     _eids_nxt, _idx_eids_nxt,
                #     _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu,
                #     _nids_nxt, _idx_nids_nxt
                # )
                print(f"b_dstnodes {b_dstnodes.size()}")
                print(f"b_dstindex {b_dstindex.size()}")
                print(f"b_srcnodes {b_srcnodes.size()}")
                print(f"b_eids {b_eids.size()}")
                print(f"b_ets {b_ets.size()}")
                print("*****")
                print(f"unique_eids {unique_eids.size()}")
                print(f"_reverse_eids {_reverse_eids.size()}")
                print(f"_unique_nids {_unique_nids.size()}")
                print(f"_reverse_nids {_reverse_nids.size()}")
                print(f"_unique_ets {_unique_ets.size()}")
                print(f"_reverse_ets {_reverse_ets.size()}")
                print("*****")
                print(f"_eids_pre {_eids_pre.size()}")
                print(f"_eids_nxt {_eids_nxt.size()}")
                print(f"_nids_pre {_nids_pre.size()}")
                print(f"_nids_nxt {_nids_nxt.size()}")
                print()
                '''

                # TODO add two-layer offline
                with nvtx.annotate("offline sample-TBlock", color="purple"):
                    head = TBlock(self.ctx, 0, b_dstnodes.numpy(), b_dsttimes.numpy(), b_dstindex.numpy(), b_srcnodes.numpy(), b_eids.numpy(), b_ets.numpy())
                with nvtx.annotate("dedup1 offline sample", color="purple"):
                    for i in range(self.num_layers):
                        tail = head if i == 0 \
                        else tail.next_block(include_dst=True, use_dst_times=False) # TODO 好像不需要add two-layer offline
                        tg.op.dedup1_offlineSample(tail, _inv_idx) 

            with nvtx.annotate("update mem", color="purple"):
                with nvtx.annotate("preload data/feat", color="purple"):
                    if self.ctx.preload_thread is not None:
                        self.ctx.preload_thread.join()

                    # tg.op.preload0_uniqLoadFeat_noMailMem(head, use_pin=True)
                    # _eids_pre, _nids_pre, _eids_nxt, _nids_nxt
                    curr = head
                    while curr.next is not None:
                        curr = curr.next
                    while curr is not None:
                        if curr.num_dst() > 0:
                            if curr.has_nbrs():
                                if curr.next is None:
                                    with nvtx.annotate("preload nfeat", color="red"):
                                        curr._load_nfeat0_uniqLoadFeat_alreadyOnGPU(_unique_nids, _reverse_nids, _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu, _nids_nxt, _idx_nids_nxt, use_pin=True)
                                with nvtx.annotate("preload efeat", color="red"):
                                    # TODO refine efeat alike nfeat
                                    curr._load_efeat0_uniqLoadFeat_alreadyOnGPU(_eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, _eids_nxt, _idx_eids_nxt, use_pin=True)
                        curr = curr.prev

            if tail.num_dst() > 0:
                # 用现有方法和冗余性观察降低显存和时间的bottleneck
                # load data / feats
                with nvtx.annotate("preload data/mem,mailbox", color="purple"):
                    cdev = tail.g.compute_device()
                    nodes = tail.allnodes()
                    with nvtx.annotate("unique", color="red"):
                        unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)
                    
                    '''
                    mem_bids = tail.g.mem.get_nids_bids(unique_nodes)
                    mailbox_uniq_bids, mailbox_nbrs_bids = tail.g.mailbox.get_nids_bids(unique_nodes)
                    def calculate_total_redundancy_ratio(unique_nodes, mem_bids, mailbox_uniq_bids, mailbox_nbrs_bids):
                        # 拼接 unique_nodes 和 mem_bids
                        combined_mem_bids = torch.cat((unique_nodes.unsqueeze(1), mem_bids), dim=1)
                        # 拼接 unique_nodes 和 mailbox_uniq_bids
                        combined_mailbox_uniq_bids = torch.cat((unique_nodes.unsqueeze(1), mailbox_uniq_bids), dim=1)

                        # 合并三组张量
                        all_combined = torch.cat((combined_mem_bids, combined_mailbox_uniq_bids, mailbox_nbrs_bids), dim=0)

                        # 合并前的总行数
                        total_rows_before = combined_mem_bids.size(0) + combined_mailbox_uniq_bids.size(0) + mailbox_nbrs_bids.size(0)

                        # 对合并后的张量进行去重
                        unique_combined = torch.unique(all_combined, dim=0)
                        total_rows_after = unique_combined.size(0)

                        # 计算冗余比例
                        redundancy_ratio = (total_rows_before - total_rows_after) / total_rows_before
                        print(f"    mem/mailbox redundancy_ratio {redundancy_ratio}")
                        return redundancy_ratio
                    calculate_total_redundancy_ratio(unique_nodes, mem_bids, mailbox_uniq_bids, mailbox_nbrs_bids)
                    '''

                    unique_mail_ts = tail.g.mailbox.time[unique_nodes]
                    delta = unique_mail_ts - tail.g.mem.time[unique_nodes]
                    mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to(cdev))
                    mem = tail._g.mem._data[unique_nodes].to(cdev)
                    mail = torch.cat([tail._g.mailbox.mail[unique_nodes].to(cdev), mail_delta], dim=1)
                # TODO 不要load 完整的blk._g.mailbox.mail[unique_nodes]
                # 将blk._g.mem._data[unique_nodes] 和 blk._g.mailbox.mail[unique_nodes] 组成一个新张量TMP 和索引表 传输到GPU 上
                # 根据TMP 和索引重建mail 和 mem

                # YOUR_CODE HERE

                # # 测试正确性
                # assert torch.allclose(mail, YOUR_mail)

                # # 统计YOUR_mail 相比于mail 少load 了多少；注意控制显存峰值
                with nvtx.annotate("nn.GRUCell", color="purple"):
                    mem = self.mem_cell(mail, mem)
                with nvtx.annotate("blk.g.mem.update", color="purple"):
                    # tail.g.mem.update(unique_nodes, mem, unique_mail_ts, batch._b_id) # TODO UNDO for statistic
                    tail.g.mem.update(unique_nodes, mem, unique_mail_ts)

                mem = mem[inverse_indices]

            nfeat = tail.nfeat() if self.nfeat_map is None else self.nfeat_map(tail.nfeat())
            tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
            tail.srcdata['h'] = nfeat[tail.num_dst():] + mem[tail.num_dst():]
            del nfeat
            del mem

            # compute embeddings
            with nvtx.annotate("op aggr", color="purple"):
                embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
            del head
            del tail

            # compute scores
            with nvtx.annotate("compute score", color="purple"):
                src, dst, neg = batch.split_data(embeds)
                scores = self.edge_predictor(src, dst)
                if neg is not None:
                    scores = (scores, self.edge_predictor(src, neg))
            del embeds
            del src
            del dst
            del neg
            with nvtx.annotate("save raw msgs", color="purple"):
                # self.save_raw_msgs2_perfCeil(batch) # TODO UNDO for statistic
                self.save_raw_msgs(batch)
                ## no_redundant_memUpd
                # self.save_raw_msgs0_no_redundant_memUpd(batch)
            return scores



    def forward_origin(self, batch: tg.TBatch) -> Tensor:
        with nvtx.annotate("forward", color="purple"):
            # setup message passing
            with nvtx.annotate("dedup and cache", color="purple"):

                head = batch.block(self.ctx)

                for i in range(self.num_layers):
                    tail = head if i == 0 \
                        else tail.next_block(include_dst=True, use_dst_times=False)
                    if tglite.config.ON_STATISTIC:
                        tail, inv_idx = tg.op.dedup_statistic(tail) if self.dedup else tail
                        with nvtx.annotate("sample", color="purple"):
                            tail = self.sampler.sample(tail) # 先去重再采样
                        # 1. 统计热节点
                        node_centric_skew(tail._dstnodes, tail._srcnodes)
                        # 2. 将采样完整offload 出去 WYQ_TODO: 保存一整个sample文件
                        add_samples((inv_idx, tail._dstnodes, tail._dsttimes, tail._dstindex, tail._srcnodes, tail._eid, tail._ets))
                    else: 
                        tail = tg.op.dedup(tail) if self.dedup else tail
                        with nvtx.annotate("sample", color="purple"):
                            tail = self.sampler.sample(tail) # 先去重再采样
                        

            # load data / feats
            with nvtx.annotate("preload data/feat", color="purple"):

                tg.op.preload(head, use_pin=True)
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            if tail.num_dst() > 0:
                # t_start = tt.start()
                with nvtx.annotate("update mem", color="purple"):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    mem = self.update_memory(tail, batch)
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                # tt.t_update_memory += tt.elapsed(t_start)
                nfeat = tail.nfeat() if self.nfeat_map is None else self.nfeat_map(tail.nfeat())
                # torch.cuda.empty_cache()
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
                tail.srcdata['h'] = nfeat[tail.num_dst():] + mem[tail.num_dst():]
                # tt.t_mem_update += tt.elapsed(t_start)
                del nfeat
                del mem

            # compute embeddings
            with nvtx.annotate("op aggr", color="purple"):
                # torch.cuda.empty_cache() # del 的变量占用的空间?不会立即释放
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
            del head
            del tail

            # compute scores
            with nvtx.annotate("compute score", color="purple"):
                # torch.cuda.empty_cache()
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                src, dst, neg = batch.split_data(embeds)
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                scores = self.edge_predictor(src, dst)
                # torch.cuda.empty_cache()
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                if neg is not None:
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    scores = (scores, self.edge_predictor(src, neg))
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            del embeds
            del src
            del dst
            del neg

            # memory messages
            t_start = tt.start()
            with nvtx.annotate("save raw msgs", color="purple"):
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                self.save_raw_msgs(batch)
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            tt.t_post_update += tt.elapsed(t_start)

            return scores

    def update_memory(self, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        cdev = blk.g.compute_device()
        nodes = blk.allnodes()

        # index, reverse = torch.unique(nodes, return_inverse=True)
        # mail_ts = blk.g.mailbox.time[index][reverse]
        time_start_0 = tt.start()
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        mail_ts = blk.g.mailbox.time[nodes] # on cpu? no new segments
        # index, reverse = torch.unique(nodes, return_inverse=True)
        # mail_ts = blk.g.mailbox.time[index][reverse]
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        tt.tt_mail_ts_load += tt.elapsed(time_start_0)
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        delta = mail_ts - blk.g.mem.time[nodes]
        delta = delta.squeeze().to(cdev)
        with nvtx.annotate("update mem-precompute_times", color="purple"):
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            mail = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta) # on cpu
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        mail = torch.cat([blk.mail(), mail], dim=1) # no new segments 不等同于no request
        # torch.cuda.empty_cache() # 这里mail占用的空间?不会立即释放
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        mem = blk.mem_data() # on cpu?
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        time_start_1 = tt.start()
        with nvtx.annotate("update mem-mem_cell", color="purple"):
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            mem = self.mem_cell(mail, mem)
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        tt.t_mem_update_gru_cell += tt.elapsed(time_start_1)
        time_start_2 = tt.start()
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        blk.g.mem.update(nodes, mem, mail_ts)
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        tt.t_mem_update_after += tt.elapsed(time_start_2)
        return mem

    def save_raw_msgs(self, batch: tg.TBatch):
        # 如果成立则最后无需写回
        sdev = batch.g.storage_device()
        mem = batch.g.mem.data
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        with nvtx.annotate("save raw msgs-block_adj", color="red"):
            # 由于new blk 我肯定load了很多没用的东西
            blk = batch.block_adj(self.ctx)
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        with nvtx.annotate("save raw msgs-op.coalesce", color="red"):
            blk = tg.op.coalesce(blk, by='latest')
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        with nvtx.annotate("save raw msgs-uniq nbrs", color="red"):
            uniq = torch.from_numpy(blk.dstnodes).long().to(sdev)
            nbrs = torch.from_numpy(blk.srcnodes).long().to(sdev)
            if self.dim_edge > 0:
                eids = torch.from_numpy(blk.eid).long().to(sdev)
                mail = torch.cat([mem[uniq], mem[nbrs], batch.g.efeat[eids]], dim=1)
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            else:
                mail = torch.cat([mem[uniq], mem[nbrs]], dim=1)
            mail_ts = torch.from_numpy(blk.ets).to(sdev)
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        with nvtx.annotate("save raw msgs-store mail_ts", color="red"):
            batch.g.mailbox.store(uniq, mail, mail_ts)
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
