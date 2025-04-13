import torch
import tglite as tg

from torch import nn, Tensor
from tglite.nn import TemporalAttnLayer, TemporalAttnLayer0_2_perfCeil, TemporalAttnLayer_2_perfCeil, TemporalAttnLayer_precompute, TemporalAttnLayer_0_fusion1_testblkm, TemporalAttnLayer_1_fusion1_testblkm
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
import copy
import tglite._c as _c

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
        # self.nfeat_map = None if dim_node == dim_embed else nn.Linear(dim_node, dim_embed)
        self.nfeat_map = nn.Linear(dim_node, dim_embed)
        self.mem_cell = GRUCell(2 * dim_embed + dim_edge + dim_time, dim_embed)
        self.mem_time_encode = tg.nn.TimeEncode(dim_time)
        if tglite.config.TEST_BLKM:
            # TODO
            self.attn0 = TemporalAttnLayer_0_fusion1_testblkm(ctx,
                            num_heads=num_heads,
                            dim_node=dim_embed,
                            dim_edge=dim_edge,
                            dim_time=dim_time,
                            dim_out=dim_embed,
                            layer=0,
                            dropout=dropout)
            self.attn1 = TemporalAttnLayer_1_fusion1_testblkm(ctx,
                            num_heads=num_heads,
                            dim_node=dim_embed,
                            dim_edge=dim_edge,
                            dim_time=dim_time,
                            dim_out=dim_embed,
                            layer=1,
                            dropout=dropout)
        elif tglite.config.PERF_CEIL:
            self.attn0 = TemporalAttnLayer0_2_perfCeil(ctx,
                              num_heads=num_heads,
                              dim_node=dim_embed,
                              dim_edge=dim_edge,
                              dim_time=dim_time,
                              dim_out=dim_embed,
                              dropout=dropout)
            # self.attn1 = TemporalAttnLayer0_2_perfCeil(ctx,
            #                     num_heads=num_heads,
            #                     dim_node=dim_embed,
            #                     dim_edge=dim_edge,
            #                     dim_time=dim_time,
            #                     dim_out=dim_embed,
            #                     dropout=dropout)
            self.attn1 = TemporalAttnLayer_2_perfCeil(ctx,
                                num_heads=num_heads,
                                dim_node=dim_embed,
                                dim_edge=dim_edge,
                                dim_time=dim_time,
                                dim_out=dim_embed,
                                dropout=dropout)
            self.attn = nn.ModuleList([
                TemporalAttnLayer(ctx,
                    num_heads=num_heads,
                    dim_node=dim_embed,
                    dim_edge=dim_edge,
                    dim_time=dim_time,
                    dim_out=dim_embed,
                    dropout=dropout)
                for i in range(num_layers)])
        else:
            # self.attn = nn.ModuleList([
            #     TemporalAttnLayer(ctx,
            #         num_heads=num_heads,
            #         dim_node=dim_embed,
            #         dim_edge=dim_edge,
            #         dim_time=dim_time,
            #         dim_out=dim_embed,
            #         dropout=dropout)
            #     for i in range(num_layers)])
            self.attn = nn.ModuleList([
                TemporalAttnLayer_precompute(ctx,
                    num_heads=num_heads,
                    dim_node=dim_embed,
                    dim_edge=dim_edge,
                    dim_time=dim_time,
                    dim_out=dim_embed,
                    dropout=dropout)
                for i in range(num_layers)])
            self.attn_reversed = nn.ModuleList([  # 逆序初始化
                TemporalAttnLayer(ctx,
                                num_heads=num_heads,
                                dim_node=dim_embed,
                                dim_edge=dim_edge,
                                dim_time=dim_time,
                                dim_out=dim_embed,
                                dropout=dropout)
                for i in range(num_layers - 1, -1, -1) 
            ])
            self.attn0 = TemporalAttnLayer(ctx,
                                num_heads=num_heads,
                                dim_node=dim_embed,
                                dim_edge=dim_edge,
                                dim_time=dim_time,
                                dim_out=dim_embed,
                                dropout=dropout)
            self.attn1 = TemporalAttnLayer(ctx,
                                num_heads=num_heads,
                                dim_node=dim_embed,
                                dim_edge=dim_edge,
                                dim_time=dim_time,
                                dim_out=dim_embed,
                                dropout=dropout)

        self.sampler = sampler
        self.edge_predictor = support.EdgePredictor(dim=dim_embed)
        self.dedup = dedup

        self.is_train = None

        # data for sample
        self.sampling_thread = None
        self._next_dstnodes = None
        self._next_dsttimes = None
        self.curr_data = None
        self.next_data = None
        # data for mailboxUpd
        self._mailboxUpd_samples = None
        self.curr_mailboxUpd = None
        self.next_mailboxUpd = None

        # wyq add record uniq-nbrs
        self.dstnodes2latestNbrs = torch.zeros([tglite.config.num_nodes, 2], dtype=torch.long) # 可以用int # TODO **去冗余重建_mailbox**

    def forward(self, batch: tg.TBatch) -> Tensor:
        # print(f"    HERE forward")
        if tglite.config.TEST_BLKM:
            return self.forward_test_blkm(batch)
        elif tglite.config.PERF_CEIL: # if tglite.config.PERF_CEIL and self.is_train: # 训练推理需要调用一个东西
            return self.forward2_perfCeil(batch) # TODO
        elif tglite.config.ON_HETER:
            return self.forward0(batch)
        elif tglite.config.OFFLINE_SAMPLE and self.is_train:
            return self.forward1_offlineSample(batch)
        elif tglite.config.PERF_CEIL_BASE and self.is_train:
            return self.forward_origin_newBase(batch)
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
        # print(f"batch {batch._b_id}")
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
                _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta, \
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


    def save_raw_msgs2_perfCeil_noStatistic(self, batch: tg.TBatch):
        sdev = batch.g.storage_device()
        mem = batch.g.mem.data

        with nvtx.annotate("save raw msgs-block_adj", color="red"):
            # 由于new blk 我肯定load了很多没用的东西 # 怎么区分哪些是negs, 哪些是pos
            blk = batch.block_adj(self.ctx)
        with nvtx.annotate("save raw msgs-op.coalesce", color="red"):
            blk = tg.op.coalesce(blk, by='latest')

        with nvtx.annotate("save raw msgs-uniq nbrs", color="red"):
            # 可以直接在这里拿到uniq和nbrs
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

    def save_raw_msgs_TEST_BLKM(self, batch: tg.TBatch):
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
            # eids = torch.from_numpy(blk.eid).long().to(sdev)
            # self.dstnodes2latestNbrs[uniq, 1] = eids

            efeat = self.ctx.manager_efeat.get_data_batch(blk.eid)
            mail = torch.cat([mem[uniq], mem[nbrs], efeat.to("cpu")], dim=1)
            mail_ts = torch.from_numpy(blk.ets).to(sdev)
        with nvtx.annotate("save raw msgs-store mail_ts", color="red"):
            batch.g.mailbox.store(uniq, mail, mail_ts, uniq, nbrs, batch._b_id)

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
            # 消融一下
            torch.set_float32_matmul_precision('high') # TODO tensorcore
            self.ctx.compiled_forward_redundancy_mul = torch.compile(self.attn0.compute_z_ours, dynamic=True)

            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.overrides")
    
    def _init_samples0_TEST_BLKM(self):
        self.curr_data = self._new_samples[0]
        _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta, \
        prev_eids, next_eids, \
        prev_nodes, next_nodes, \
        unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets, \
        Q_node_idx, reindex, \
        unique_dst_nodes, inverse_dstnodes, unique_src_nodes, inverse_srcnodes, \
        _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, \
        _eids_nxt, _idx_eids_nxt, \
        _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu, \
        _nids_nxt, _idx_nids_nxt = self.curr_data    
    
    def _init_samples0_2_perfCeil(self):
            self.curr_data = self._new_samples[0]
            _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta, \
            prev_eids, next_eids, \
            prev_nodes, next_nodes, \
            unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets, \
            _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, \
            _eids_nxt, _idx_eids_nxt, \
            _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu, \
            _nids_nxt, _idx_nids_nxt = self.curr_data
            # preload nfeat
            ## 1 layer: self.layer == 0 TODO 不过preload 一般只preload 1层就可以
            ## self.ctx._nxt_nfeat_pins = self.ctx._get_nfeat_pin(self.layer, len(_nids_cpu), self._g.nfeat.shape[1])
            self.ctx._cur_nfeat_pins = self.ctx._get_nfeat_pin(0, len(_nids_cpu), self.ctx._g.dim_node)
            with nvtx.annotate("index_select", color="red"):
                torch.index_select(self.ctx._g.nfeat, 0, _nids_cpu, out=self.ctx._cur_nfeat_pins)
            # preload efeat
            self.ctx._cur_efeat_pins = self.ctx._get_efeat_pin(0, len(_eids_cpu), self.ctx._g.dim_edge)
            with nvtx.annotate("index_select", color="red"):
                torch.index_select(self.ctx._g.efeat, 0, _eids_cpu, out=self.ctx._cur_efeat_pins)
    
    def forward2_perfCeil(self, batch: tg.TBatch) -> Tensor:
        # print(f"    HERE forward2_perfCeil")
        def sampling(self, _b_id):
            # _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta, \
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
            if self.is_train:
                with nvtx.annotate("thread sampling nxt", color="purple"):
                    # print(f"    here thread sampling nxt")
                    assert self.curr_data is not None # 同步点在support.py 的preload
                    _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta, \
                    prev_eids, next_eids, \
                    prev_nodes, next_nodes, \
                    unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets, \
                    _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, \
                    _eids_nxt, _idx_eids_nxt, \
                    _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu, \
                    _nids_nxt, _idx_nids_nxt = self.curr_data
                    self.curr_data = None
                    # print(f"    self.curr_data {self.curr_data}")
                    # print(f"    here self.curr_data = None")
                    # print(f"_unique_ets {_unique_ets}")
                    # print(f"_unique_ets {_unique_ets.shape}")
                    # print(f"_reverse_ets {_reverse_ets}")
                    # print(f"_reverse_ets {_reverse_ets.shape}")
                    # print()

                    # Sampling for next batch 每次只预取一个batch
                    assert self.next_data == None
                    self.sampling_thread = threading.Thread(target=sampling, args=(self, batch._b_id + 1,))
                    self.sampling_thread.start()

                # # offline sample
                with nvtx.annotate("offline sample", color="purple"):
                    # TODO
                    #  TBlock def __init__(self, ctx: 'TContext', layer: int, dstnodes: np.ndarray, dsttimes: np.ndarray,
                    #  dstindex: np.ndarray = None, srcnodes: np.ndarray = None,
                    #  eid: np.ndarray = None, ets: np.ndarray = None):
                    
                    '''
                    # new_sample = (
                    #     b_inv_idx,
                    #     b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta,
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
                                        curr._load_efeat0_uniqLoadFeat_alreadyOnGPU(unique_eids, _reverse_eids, _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, _eids_nxt, _idx_eids_nxt, use_pin=True)
                            curr = curr.prev

                    if tail.num_dst() > 0:
                        # 用现有方法和冗余性观察降低显存和时间的bottleneck
                        '''
                        # load data / feats
                        with nvtx.annotate("preload data/mem,mailbox", color="purple"):
                            cdev = tail.g.compute_device()
                            # nodes = tail.allnodes()
                            # unique_nodes, inverse_indices = torch.unique(nodes.to("cuda"), return_inverse=True)
                            # assert torch.allclose(_unique_nids.long().to("cuda"), unique_nodes)
                            
                            
                            # mem_bids = tail.g.mem.get_nids_bids(unique_nodes)
                            # mailbox_uniq_bids, mailbox_nbrs_bids = tail.g.mailbox.get_nids_bids(unique_nodes)
                            # def calculate_total_redundancy_ratio(unique_nodes, mem_bids, mailbox_uniq_bids, mailbox_nbrs_bids):
                            #     # 拼接 unique_nodes 和 mem_bids
                            #     combined_mem_bids = torch.cat((unique_nodes.unsqueeze(1), mem_bids), dim=1)
                            #     # 拼接 unique_nodes 和 mailbox_uniq_bids
                            #     combined_mailbox_uniq_bids = torch.cat((unique_nodes.unsqueeze(1), mailbox_uniq_bids), dim=1)

                            #     # 合并三组张量
                            #     all_combined = torch.cat((combined_mem_bids, combined_mailbox_uniq_bids, mailbox_nbrs_bids), dim=0)

                            #     # 合并前的总行数
                            #     total_rows_before = combined_mem_bids.size(0) + combined_mailbox_uniq_bids.size(0) + mailbox_nbrs_bids.size(0)

                            #     # 对合并后的张量进行去重
                            #     unique_combined = torch.unique(all_combined, dim=0)
                            #     total_rows_after = unique_combined.size(0)

                            #     # 计算冗余比例
                            #     redundancy_ratio = (total_rows_before - total_rows_after) / total_rows_before
                            #     print(f"    mem/mailbox redundancy_ratio {redundancy_ratio}")
                            #     return redundancy_ratio
                            # calculate_total_redundancy_ratio(unique_nodes, mem_bids, mailbox_uniq_bids, mailbox_nbrs_bids)
                            

                            with nvtx.annotate("cal delta", color="red"):
                                unique_mail_ts = tail.g.mailbox.time[_unique_nids]
                                delta = unique_mail_ts - tail.g.mem.time[_unique_nids]
                            mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to(cdev))
                            with nvtx.annotate("mem mail to(cuda)"):
                                mem = tail._g.mem._data[_unique_nids].to(cdev)
                                mail = torch.cat([tail._g.mailbox.mail[_unique_nids].to(cdev), mail_delta], dim=1)
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
                            tail.g.mem.update(_unique_nids, mem, unique_mail_ts)

                        mem = mem[_reverse_nids]'''

                        with nvtx.annotate("cal mail_delta", color="red"): # TODO
                            unique_mail_ts = tail.g.mailbox.time[_unique_nids] # on cpu
                            delta = unique_mail_ts - tail.g.mem.time[_unique_nids]
                            mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to("cuda"))
                        # TODO divide pipelines
                        # TODO build new pipelines
                        # stage0_index = math.floor(len(unique_nodes)/3)
                        # stage_indices = [(0, stage0_index), (stage0_index, math.floor(len(unique_nodes)*2/3)), (math.floor(len(unique_nodes)*2/3), len(unique_nodes))] 
                        stage0_index = math.floor(len(_unique_nids)/2)
                        stage_indices = [(0, stage0_index), (stage0_index, len(_unique_nids))] 
                        # pre-alloc
                        # max_size = math.ceil(len(unique_nodes)/2)
                        # pre_allocated_mem = torch.empty((max_size, 100), device=cdev)
                        # 0阶段通信
                        with nvtx.annotate(f"memUpdStage_comm_{0}", color="green"):
                            unique_nodes_slice = _unique_nids[0:stage0_index]
                            # all on cpu
                            cur_mail_mem = tail._load_mail_mem_data_slice(unique_nodes_slice)
                            # print(f"cur_mail_mem {cur_mail_mem.size()}")
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
                                        nxt_unique_nodes_slice = _unique_nids[nxt_start_idx:nxt_end_idx]
                                        # all on cpu
                                        nxt_mail_mem = tail._load_mail_mem_data_slice(nxt_unique_nodes_slice)
                                        # print(f"nxt_mail_mem {nxt_mail_mem.size()}")
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
                        # print()

                        # 写回也可以pipeline TODO 不一定有必要，可以和后面的计算掩盖
                        tail.g.mem.update(_unique_nids, mem, unique_mail_ts) # 目前为写回CPU

                        # no scatter
                        # mem = mem[_reverse_nids]

                # no scatter               
                nfeat = tail.nfeat() if self.nfeat_map is None else self.nfeat_map(tail.nfeat())

                # print(f"_reverse_nids[:tail.num_dst()] {_reverse_nids[:tail.num_dst()].shape}")
                # print(f"nfeat[:tail.num_dst()] {nfeat[:tail.num_dst()].shape}") # \o/ acc success! test passed!
                # print()

                ''' 
                tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
                tail.srcdata['h'] = nfeat[tail.num_dst():] + mem[tail.num_dst():]
                '''
                # no scatter
                nodeData = nfeat + mem
                del nfeat
                del mem

                # # compute embeddings
                # with nvtx.annotate("op aggr", color="purple"):
                #     embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h') # 消融一下(对support.py的修改没有问题，这一模块rebase精度可以恢复)
                '''
                # 消融一下
                with nvtx.annotate("op.aggregate", color="green"):
                    blk = tail
                    output = None
                    while blk is not None:
                        output = blk.apply(list(reversed(self.attn))[blk.layer])
                        if blk.prev is not None and output is not None and 'h': # TODO for two layer
                            if blk._include_prev_dst:
                                print("HERE3")
                                num_dst = blk.prev.num_dst()
                                with nvtx.annotate("aggregate blk.prev.dstdata/srcdata", color="red"):
                                    blk.prev.dstdata['h'] = output[:num_dst]
                                    blk.prev.srcdata['h'] = output[num_dst:]
                            else:
                                print("HERE4")
                                with nvtx.annotate("op.aggregate blk.prev.dstdata", color="green"):
                                    blk.prev.srcdata['h'] = output
                        blk.clear_data()
                        blk.clear_hooks()
                        blk = blk.prev
                embeds = output'''

                with nvtx.annotate("op.aggregate", color="green"):
                    output = None
                    with nvtx.annotate("blk.apply", color="red"):
                        with nvtx.annotate("blk.apply fn", color="red"):
                            '''
                            node_unique, node_inverse = torch.unique(tail.srcdata['h'], dim=0, return_inverse=True)
                            efeat_unique, efeat_inverse = torch.unique(tail.efeat(), dim=0, return_inverse=True) # 172的长度可能乘起来不够高效
                            time_unique, time_inverse = torch.unique(nbrs_time_feat, dim=0, return_inverse=True) # TODO _unique_ets, _reverse_ets
                            '''
                            # input node unique _reverse_nids[:tail.num_dst()]
                            # output = self.attn0(tail, nodeData_unique, nodeData_inverse, edgeData_unique, edgeData_inverse, time_d_unique, time_d_inverse)
                            output = self.attn0(tail, nodeData, _reverse_nids.to("cuda"), tail.efeat(), _reverse_eids.to("cuda"), _unique_time_delta, _reverse_time_delta)
                            # output = self.attn0(tail)
                        with nvtx.annotate("blk.apply run_hooks", color="green"): # run hooks
                            output = tail.run_hooks(output)
                    tail.clear_data()
                    tail.clear_hooks()
                    if self.num_layers > 1:
                        num_dst = head.num_dst()
                        head.dstdata['h'] = output[:num_dst]
                        head.srcdata['h'] = output[num_dst:]
                        # head.srcdata['h'] = output
                        # output = self.attn1(head, nodeData, _reverse_nids.to("cuda"), head.efeat(), _reverse_eids.to("cuda"), _unique_time_delta, _reverse_time_delta)
                        # output = self.attn1(head, nodeData, _reverse_nids.to("cuda"), head.efeat(), _reverse_eids.to("cuda"), _unique_time_delta, _reverse_time_delta)
                        output = self.attn1(head)
                        output = head.run_hooks(output)
                        head.clear_data()
                        head.clear_hooks()
                embeds = output

                del head
                del tail

                # compute scores
                with nvtx.annotate("compute score", color="purple"):
                    src, dst, neg = batch.split_data(embeds) # 这里还可以去冗余
                    scores = self.edge_predictor(src, dst)
                    if neg is not None:
                        scores = (scores, self.edge_predictor(src, neg))
                del embeds
                del src
                del dst
                del neg
                with nvtx.annotate("save raw msgs", color="purple"):
                    # self.save_raw_msgs2_perfCeil(batch) # TODO 用于 statistic，换成初始版
                    self.save_raw_msgs2_perfCeil_noStatistic(batch)
                    # self.save_raw_msgs(batch) # TODO 还可以优化
                    ## no_redundant_memUpd
                    # self.save_raw_msgs0_no_redundant_memUpd(batch)
                return scores
            else:
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
                        nodeData = nfeat + mem
                        nodeData_unique, nodeData_inverse = torch.unique(nodeData, dim=0, return_inverse=True)
                        tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
                        tail.srcdata['h'] = nfeat[tail.num_dst():] + mem[tail.num_dst():]
                        edgeData_unique, edgeData_inverse = torch.unique(tail.efeat(), dim=0, return_inverse=True)
                        time_d_unique, time_d_inverse = torch.unique(tail.time_deltas(), dim=0, return_inverse=True)
                        # tt.t_mem_update += tt.elapsed(t_start)
                        del nfeat
                        del mem

                    # compute embeddings
                    # with nvtx.annotate("op aggr", color="purple"):
                    #     # torch.cuda.empty_cache() # del 的变量占用的空间?不会立即释放
                    #     # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    #     embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
                    # ! 训练改 推理也要改 self.xxx_along with forward train
                    with nvtx.annotate("op.aggregate", color="green"):
                        output = None
                        with nvtx.annotate("blk.apply", color="red"):
                            with nvtx.annotate("blk.apply fn", color="red"):
                                output = self.attn0(tail, nodeData_unique, nodeData_inverse, edgeData_unique, edgeData_inverse, time_d_unique, time_d_inverse)
                            with nvtx.annotate("blk.apply run_hooks", color="green"): # run hooks
                                output = tail.run_hooks(output)
                        tail.clear_data()
                        tail.clear_hooks()
                        if self.num_layers > 1:
                            # output = self.attn1(head, nodeData, _reverse_nids.to("cuda"), head.efeat(), _reverse_eids.to("cuda"), _unique_time_delta, _reverse_time_delta)
                            output = self.attn1(head)
                            output = head.run_hooks(output)
                            head.clear_data()
                            head.clear_hooks()
                    embeds = output
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


    def _load_mailboxUpd_samples_blkm(self):
        if self._mailboxUpd_samples is None:
            if self.is_train:
                file_path = os.path.join(tglite.config.log_dir, f"tglake_res_mailboxUpd/TRAIN_mailboxUpdBatchs_{tglite.config.log_name}.pt")
            else:
                file_path = os.path.join(tglite.config.log_dir, f"tglake_res_mailboxUpd/EVAL_mailboxUpdBatchs_{tglite.config.log_name}.pt")
            self._mailboxUpd_samples = torch.load(file_path)

    def _init_samples0_online_TEST_BLKM(self, batch):
        print(f"_init_samples0_online_TEST_BLKM")
        g = self.ctx.graph
        nids = g._edges[batch._beg_idx:batch._end_idx].T.reshape(-1)
        nids = np.concatenate([nids, batch._neg_nodes]).astype(np.int32) # include neg nodes
        times = np.tile(g._times[batch._beg_idx:batch._end_idx], 3).astype(np.float32)

        if self.num_layers == 1:
            _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
            unique_eids, inverse_eids, unique_nids, \
            unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
            Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst = self.sample_our(g, nids, times)
        else:
            assert self.num_layers == 2 # 两层的去重可以更极致
            # b2_nodes = torch.cat([b_dstnodes, b_srcnodes])
            # b2_times = torch.cat([b_dsttimes, b_ets])
            _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
            unique_eids, inverse_eids, unique_nids, \
            unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
            Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst, \
            b2_inv_idx, b2_dstnodes, b2_dsttimes, b2_dstindex, b2_srcnodes, b2_eids, b2_ets, b2_unique_time_delta, b2_inverse_time_delta, \
            b2_unique_eids, b2_inverse_eids, b2_unique_nids, \
            b2_unique_dstnodes, b2_inverse_dstnodes, b2_unique_srcnodes, b2_inverse_srcnodes, \
            b2_Q_node_idx, b2_reindex, b2_reduce_idx, b2_num_src, b2_num_dst = self.sample_our_layer2(g, nids, times)

        if len(self._mailboxUpd_samples) > batch._b_id + 1:
            mailbox_uniq, mailbox_nbrs, mailbox_ets, mailbox_eid = self._mailboxUpd_samples[batch._b_id + 1]

        with torch.cuda.StreamContext(torch.cuda.Stream()):
            if self.num_layers == 1:
                self.curr_data = _inv_idx.to("cuda"), b_dstnodes.to("cuda"), b_dsttimes.to("cuda"), b_dstindex.to("cuda"), b_srcnodes.to("cuda"), b_eids.to("cuda"), b_ets.to("cuda"), \
                                unique_time_delta.to("cuda"), inverse_time_delta.to("cuda"), \
                                unique_eids.to("cuda"), inverse_eids.to("cuda"), unique_nids.to("cuda"), \
                                unique_dstnodes.to("cuda"), inverse_dstnodes.to("cuda"), unique_srcnodes.to("cuda"), inverse_srcnodes.to("cuda"), \
                                Q_node_idx.to("cuda"), reindex.to("cuda"), reduce_idx.to("cuda"), b_num_src, b_num_dst
            else:
                assert self.num_layers == 2
                self.curr_data = _inv_idx.to("cuda"), b_dstnodes.to("cuda"), b_dsttimes.to("cuda"), b_dstindex.to("cuda"), b_srcnodes.to("cuda"), b_eids.to("cuda"), b_ets.to("cuda"), \
                                unique_time_delta.to("cuda"), inverse_time_delta.to("cuda"), \
                                unique_eids.to("cuda"), inverse_eids.to("cuda"), unique_nids.to("cuda"), \
                                unique_dstnodes.to("cuda"), inverse_dstnodes.to("cuda"), unique_srcnodes.to("cuda"), inverse_srcnodes.to("cuda"), \
                                Q_node_idx.to("cuda"), reindex.to("cuda"), reduce_idx.to("cuda"), b_num_src, b_num_dst, \
                                b2_inv_idx.to("cuda"), b2_dstnodes.to("cuda"), b2_dsttimes.to("cuda"), b2_dstindex.to("cuda"), b2_srcnodes.to("cuda"), b2_eids.to("cuda"), b2_ets.to("cuda"), \
                                b2_unique_time_delta.to("cuda"), b2_inverse_time_delta.to("cuda"), \
                                b2_unique_eids.to("cuda"), b2_inverse_eids.to("cuda"), b2_unique_nids.to("cuda"), \
                                b2_unique_dstnodes.to("cuda"), b2_inverse_dstnodes.to("cuda"), b2_unique_srcnodes.to("cuda"), b2_inverse_srcnodes.to("cuda"), \
                                b2_Q_node_idx.to("cuda"), b2_reindex.to("cuda"), b2_reduce_idx.to("cuda"), b2_num_src, b2_num_dst
            self.curr_mailboxUpd = mailbox_uniq.to("cuda"), mailbox_nbrs.to("cuda"), mailbox_ets.to("cuda"), mailbox_eid.to("cuda")
            TEST_BLKM_preload_sampling_event = torch.cuda.Event()
            TEST_BLKM_preload_sampling_event.record()
        TEST_BLKM_preload_sampling_event.synchronize()

    def dedup_our(self, nodes, times):
            _, nodes, times, _inv_idx = _c.dedup_targets(nodes, times)
            return nodes, times, _inv_idx

    # TODO unique_consecutive
    def sample_our(self, g, nids, times): # TODO sample 这还得花时间想清楚 是否需要重排?
        dstnodes, dsttimes, _inv_idx = self.dedup_our(nids, times)  # dedup and cache
        _inv_idx = torch.tensor(_inv_idx)
        dstindex, srcnodes, eid, ets = self.sampler.sample_no_blk(g, dstnodes, dsttimes)
        b_dstnodes = torch.tensor(dstnodes)
        b_dsttimes = torch.tensor(dsttimes)
        b_dstindex = torch.tensor(dstindex)
        b_srcnodes = torch.tensor(srcnodes)
        b_eids = torch.tensor(eid)
        b_ets = torch.tensor(ets)
        b_num_src = b_srcnodes.shape[0]
        b_num_dst = b_dstnodes.shape[0]
        time_delta = b_dsttimes[b_dstindex] - b_ets
        all_nids = torch.cat([b_dstnodes, b_srcnodes])
        # unique_time_delta, inverse_time_delta = torch.unique_consecutive(time_delta, return_inverse=True)
        # unique_nids, inverse_nids = torch.unique_consecutive(all_nids, return_inverse=True)
        # unique_eids, inverse_eids = torch.unique_consecutive(b_eids, return_inverse=True)
        # unique_dstnodes, inverse_dstnodes = torch.unique_consecutive(inverse_nids[:b_dstnodes.shape[0]], return_inverse=True)
        # unique_srcnodes, inverse_srcnodes = torch.unique_consecutive(inverse_nids[b_dstnodes.shape[0]:], return_inverse=True)
        unique_time_delta, inverse_time_delta = torch.unique(time_delta, return_inverse=True)
        unique_nids, inverse_nids = torch.unique(all_nids, return_inverse=True)
        unique_eids, inverse_eids = torch.unique(b_eids, return_inverse=True)
        unique_dstnodes, inverse_dstnodes = torch.unique(inverse_nids[:b_dstnodes.shape[0]], return_inverse=True)
        unique_srcnodes, inverse_srcnodes = torch.unique(inverse_nids[b_dstnodes.shape[0]:], return_inverse=True)
        x = torch.arange(b_srcnodes.shape[0])
        Q_node_idx = inverse_dstnodes[b_dstindex[x]]
        reindex = torch.unique(b_dstindex, return_inverse=True)[1]
        # reduce_idx = torch.tensor(b_dstindex).int()
        reduce_idx = b_dstindex.clone().detach().int()

        return _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
                unique_eids, inverse_eids, unique_nids, \
                unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
                Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst
    
    def sample_our_layer2(self, g, nids, times):
        dstnodes, dsttimes, _inv_idx = self.dedup_our(nids, times)  # dedup and cache
        _inv_idx = torch.tensor(_inv_idx)
        dstindex, srcnodes, eid, ets = self.sampler.sample_no_blk(g, dstnodes, dsttimes)
        b_dstnodes = torch.tensor(dstnodes)
        b_dsttimes = torch.tensor(dsttimes)
        b_dstindex = torch.tensor(dstindex)
        b_srcnodes = torch.tensor(srcnodes)
        b_eids = torch.tensor(eid)
        b_ets = torch.tensor(ets)
        b_num_src = b_srcnodes.shape[0]
        b_num_dst = b_dstnodes.shape[0]
        time_delta = b_dsttimes[b_dstindex] - b_ets
        unique_time_delta, inverse_time_delta = torch.unique(time_delta, return_inverse=True)
        all_nids = torch.cat([b_dstnodes, b_srcnodes])
        unique_nids, _ = torch.unique(all_nids, return_inverse=True)
        unique_eids, inverse_eids = torch.unique(b_eids, return_inverse=True)
        x = torch.arange(b_srcnodes.shape[0])
        reindex = torch.unique(b_dstindex, return_inverse=True)[1]
        reduce_idx = b_dstindex.clone().detach().int()

        b2_nodes = torch.cat([b_dstnodes, b_srcnodes])
        b2_times = torch.cat([b_dsttimes, b_ets])

        b2_dstnodes, b2_dsttimes, b2_inv_idx = self.dedup_our(b2_nodes, b2_times)  # dedup and cache
        b2_inv_idx = torch.tensor(b2_inv_idx)
        b2_dstindex, b2_srcnodes, b2_eid, b2_ets = self.sampler.sample_no_blk(g, b2_dstnodes, b2_dsttimes)
        b2_dstnodes = torch.tensor(b2_dstnodes)
        b2_dsttimes = torch.tensor(b2_dsttimes)
        b2_dstindex = torch.tensor(b2_dstindex)
        b2_srcnodes = torch.tensor(b2_srcnodes)
        b2_eids = torch.tensor(b2_eid)
        b2_ets = torch.tensor(b2_ets)

        unique_dst_inv_idx, inverse_dst_inv_idx = torch.unique(b2_inv_idx[:b_dstnodes.shape[0]], return_inverse=True)
        unique_src_inv_idx, inverse_src_inv_idx = torch.unique(b2_inv_idx[b_dstnodes.shape[0]:], return_inverse=True)
        unique_dstnodes = unique_dst_inv_idx
        inverse_dstnodes = inverse_dst_inv_idx
        unique_srcnodes = unique_src_inv_idx
        inverse_srcnodes = inverse_src_inv_idx
        Q_node_idx = inverse_dst_inv_idx[b_dstindex[x]]

        b2_num_src = b2_srcnodes.shape[0]
        b2_num_dst = b2_dstnodes.shape[0]
        b2_time_delta = b2_dsttimes[b2_dstindex] - b2_ets
        b2_all_nids = torch.cat([b2_dstnodes, b2_srcnodes])
        b2_unique_time_delta, b2_inverse_time_delta = torch.unique(b2_time_delta, return_inverse=True)
        b2_unique_nids, b2_inverse_nids = torch.unique(b2_all_nids, return_inverse=True)
        b2_unique_eids, b2_inverse_eids = torch.unique(b2_eids, return_inverse=True)
        b2_unique_dstnodes, b2_inverse_dstnodes = torch.unique(b2_inverse_nids[:b2_dstnodes.shape[0]], return_inverse=True)
        b2_unique_srcnodes, b2_inverse_srcnodes = torch.unique(b2_inverse_nids[b2_dstnodes.shape[0]:], return_inverse=True)
        x = torch.arange(b2_srcnodes.shape[0])
        b2_Q_node_idx = b2_inverse_dstnodes[b2_dstindex[x]]
        b2_reindex = torch.unique(b2_dstindex, return_inverse=True)[1]
        # reduce_idx = torch.tensor(b_dstindex).int()
        b2_reduce_idx = b2_dstindex.clone().detach().int()
        
        return _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
                unique_eids, inverse_eids, unique_nids, \
                unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
                Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst, \
                b2_inv_idx, b2_dstnodes, b2_dsttimes, b2_dstindex, b2_srcnodes, b2_eids, b2_ets, b2_unique_time_delta, b2_inverse_time_delta, \
                b2_unique_eids, b2_inverse_eids, b2_unique_nids, \
                b2_unique_dstnodes, b2_inverse_dstnodes, b2_unique_srcnodes, b2_inverse_srcnodes, \
                b2_Q_node_idx, b2_reindex, b2_reduce_idx, b2_num_src, b2_num_dst
   
    def forward_test_blkm(self, batch: tg.TBatch) -> Tensor:
        with nvtx.annotate("forward", color="purple"):
            g = self.ctx.graph
            # with nvtx.annotate("thread sampling", color="purple"):
            #     def sampling(self, batch):
            #         # sampling
            #         nids = g._edges[batch._end_idx:batch._nxt_idx].T.reshape(-1)
            #         nids = np.concatenate([nids, batch._nxt_neg_nodes]).astype(np.int32) # include neg nodes
            #         times = np.tile(g._times[batch._end_idx:batch._nxt_idx], 3).astype(np.float32)
            #         if self.num_layers == 1:
            #             _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
            #             unique_eids, inverse_eids, unique_nids, \
            #             unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
            #             Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst = self.sample_our(g, nids, times)
            #         else:
            #             assert self.num_layers == 2
            #             _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
            #             unique_eids, inverse_eids, unique_nids, \
            #             unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
            #             Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst, \
            #             b2_inv_idx, b2_dstnodes, b2_dsttimes, b2_dstindex, b2_srcnodes, b2_eids, b2_ets, b2_unique_time_delta, b2_inverse_time_delta, \
            #             b2_unique_eids, b2_inverse_eids, b2_unique_nids, \
            #             b2_unique_dstnodes, b2_inverse_dstnodes, b2_unique_srcnodes, b2_inverse_srcnodes, \
            #             b2_Q_node_idx, b2_reindex, b2_reduce_idx, b2_num_src, b2_num_dst = self.sample_our_layer2(g, nids, times)
                    
            #         if len(self._mailboxUpd_samples) > batch._b_id + 1:
            #             mailbox_uniq, mailbox_nbrs, mailbox_ets, mailbox_eid = self._mailboxUpd_samples[batch._b_id + 1]

            #         # loading 
            #         with torch.cuda.StreamContext(torch.cuda.Stream()):
            #             '''
            #             _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
            #             unique_eids, inverse_eids, unique_nids, \
            #             unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
            #             Q_node_idx, reindex
            #             '''
            #             if self.num_layers == 1:
            #                 self.next_data = _inv_idx.to("cuda"), b_dstnodes.to("cuda"), b_dsttimes.to("cuda"), b_dstindex.to("cuda"), b_srcnodes.to("cuda"), b_eids.to("cuda"), b_ets.to("cuda"), \
            #                                 unique_time_delta.to("cuda"), inverse_time_delta.to("cuda"), \
            #                                 unique_eids.to("cuda"), inverse_eids.to("cuda"), unique_nids.to("cuda"), \
            #                                 unique_dstnodes.to("cuda"), inverse_dstnodes.to("cuda"), unique_srcnodes.to("cuda"), inverse_srcnodes.to("cuda"), \
            #                                 Q_node_idx.to("cuda"), reindex.to("cuda"), reduce_idx.to("cuda"), b_num_src, b_num_dst
            #             else:
            #                 assert self.num_layers == 2
            #                 self.next_data = _inv_idx.to("cuda"), b_dstnodes.to("cuda"), b_dsttimes.to("cuda"), b_dstindex.to("cuda"), b_srcnodes.to("cuda"), b_eids.to("cuda"), b_ets.to("cuda"), \
            #                                 unique_time_delta.to("cuda"), inverse_time_delta.to("cuda"), \
            #                                 unique_eids.to("cuda"), inverse_eids.to("cuda"), unique_nids.to("cuda"), \
            #                                 unique_dstnodes.to("cuda"), inverse_dstnodes.to("cuda"), unique_srcnodes.to("cuda"), inverse_srcnodes.to("cuda"), \
            #                                 Q_node_idx.to("cuda"), reindex.to("cuda"), reduce_idx.to("cuda"), b_num_src, b_num_dst, \
            #                                 b2_inv_idx.to("cuda"), b2_dstnodes.to("cuda"), b2_dsttimes.to("cuda"), b2_dstindex.to("cuda"), b2_srcnodes.to("cuda"), b2_eids.to("cuda"), b2_ets.to("cuda"), \
            #                                 b2_unique_time_delta.to("cuda"), b2_inverse_time_delta.to("cuda"), \
            #                                 b2_unique_eids.to("cuda"), b2_inverse_eids.to("cuda"), b2_unique_nids.to("cuda"), \
            #                                 b2_unique_dstnodes.to("cuda"), b2_inverse_dstnodes.to("cuda"), b2_unique_srcnodes.to("cuda"), b2_inverse_srcnodes.to("cuda"), \
            #                                 b2_Q_node_idx.to("cuda"), b2_reindex.to("cuda"), b2_reduce_idx.to("cuda"), b2_num_src, b2_num_dst
            #             self.next_mailboxUpd = mailbox_uniq.to("cuda"), mailbox_nbrs.to("cuda"), mailbox_ets.to("cuda"), mailbox_eid.to("cuda")
            #             TEST_BLKM_preload_sampling_event = torch.cuda.Event()
            #             TEST_BLKM_preload_sampling_event.record()
            #         TEST_BLKM_preload_sampling_event.synchronize()
            #     # 同步curr batch
            #     if self.sampling_thread is not None:
            #         self.sampling_thread.join()
            #         # print(f"sampling thread join")
            #         assert self.next_data is not None
            #         assert self.curr_data == None
            #         self.curr_data = self.next_data
            #         self.curr_mailboxUpd = self.next_mailboxUpd
            #         self.next_data = None # 指示当前batch 可以开始下一轮预采样了
            #     # 总是采样next batch
            #     with nvtx.annotate("thread sampling nxt", color="purple"):
            #         # print(f"    here thread sampling nxt")
            #         assert self.curr_data is not None
            #         assert self.next_data == None
            #         if batch._nxt_idx is not None: # 如果还有next batch
            #             # print(f"sampling thread start")
            #             self.sampling_thread = threading.Thread(target=sampling, args=(self, batch))
            #             self.sampling_thread.start()

            ### accurate kernel performance
            with nvtx.annotate("thread sampling", color="purple"):
                # sampling
                nids = g._edges[batch._beg_idx:batch._end_idx].T.reshape(-1)
                nids = np.concatenate([nids, batch._neg_nodes]).astype(np.int32) # include neg nodes
                times = np.tile(g._times[batch._beg_idx:batch._end_idx], 3).astype(np.float32)
                if self.num_layers == 1:
                    _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
                    unique_eids, inverse_eids, unique_nids, \
                    unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
                    Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst = self.sample_our(g, nids, times)
                else:
                    assert self.num_layers == 2
                    _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
                    unique_eids, inverse_eids, unique_nids, \
                    unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
                    Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst, \
                    b2_inv_idx, b2_dstnodes, b2_dsttimes, b2_dstindex, b2_srcnodes, b2_eids, b2_ets, b2_unique_time_delta, b2_inverse_time_delta, \
                    b2_unique_eids, b2_inverse_eids, b2_unique_nids, \
                    b2_unique_dstnodes, b2_inverse_dstnodes, b2_unique_srcnodes, b2_inverse_srcnodes, \
                    b2_Q_node_idx, b2_reindex, b2_reduce_idx, b2_num_src, b2_num_dst = self.sample_our_layer2(g, nids, times)
                
                if len(self._mailboxUpd_samples) > batch._b_id:
                    # import pdb;pdb.set_trace()
                    mailbox_uniq, mailbox_nbrs, mailbox_ets, mailbox_eid = self._mailboxUpd_samples[batch._b_id]

                # loading 
                with torch.cuda.StreamContext(torch.cuda.Stream()):
                    '''
                    _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, unique_time_delta, inverse_time_delta, \
                    unique_eids, inverse_eids, unique_nids, \
                    unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
                    Q_node_idx, reindex
                    '''
                    if self.num_layers == 1:
                        self.curr_data = _inv_idx.to("cuda"), b_dstnodes.to("cuda"), b_dsttimes.to("cuda"), b_dstindex.to("cuda"), b_srcnodes.to("cuda"), b_eids.to("cuda"), b_ets.to("cuda"), \
                                        unique_time_delta.to("cuda"), inverse_time_delta.to("cuda"), \
                                        unique_eids.to("cuda"), inverse_eids.to("cuda"), unique_nids.to("cuda"), \
                                        unique_dstnodes.to("cuda"), inverse_dstnodes.to("cuda"), unique_srcnodes.to("cuda"), inverse_srcnodes.to("cuda"), \
                                        Q_node_idx.to("cuda"), reindex.to("cuda"), reduce_idx.to("cuda"), b_num_src, b_num_dst
                    else:
                        assert self.num_layers == 2
                        self.curr_data = _inv_idx.to("cuda"), b_dstnodes.to("cuda"), b_dsttimes.to("cuda"), b_dstindex.to("cuda"), b_srcnodes.to("cuda"), b_eids.to("cuda"), b_ets.to("cuda"), \
                                        unique_time_delta.to("cuda"), inverse_time_delta.to("cuda"), \
                                        unique_eids.to("cuda"), inverse_eids.to("cuda"), unique_nids.to("cuda"), \
                                        unique_dstnodes.to("cuda"), inverse_dstnodes.to("cuda"), unique_srcnodes.to("cuda"), inverse_srcnodes.to("cuda"), \
                                        Q_node_idx.to("cuda"), reindex.to("cuda"), reduce_idx.to("cuda"), b_num_src, b_num_dst, \
                                        b2_inv_idx.to("cuda"), b2_dstnodes.to("cuda"), b2_dsttimes.to("cuda"), b2_dstindex.to("cuda"), b2_srcnodes.to("cuda"), b2_eids.to("cuda"), b2_ets.to("cuda"), \
                                        b2_unique_time_delta.to("cuda"), b2_inverse_time_delta.to("cuda"), \
                                        b2_unique_eids.to("cuda"), b2_inverse_eids.to("cuda"), b2_unique_nids.to("cuda"), \
                                        b2_unique_dstnodes.to("cuda"), b2_inverse_dstnodes.to("cuda"), b2_unique_srcnodes.to("cuda"), b2_inverse_srcnodes.to("cuda"), \
                                        b2_Q_node_idx.to("cuda"), b2_reindex.to("cuda"), b2_reduce_idx.to("cuda"), b2_num_src, b2_num_dst
                    self.curr_mailboxUpd = mailbox_uniq.to("cuda"), mailbox_nbrs.to("cuda"), mailbox_ets.to("cuda"), mailbox_eid.to("cuda")
                    TEST_BLKM_preload_sampling_event = torch.cuda.Event()
                    TEST_BLKM_preload_sampling_event.record()
                TEST_BLKM_preload_sampling_event.synchronize()
            # using sampling data
            if self.num_layers == 1:
                _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, \
                unique_time_delta, inverse_time_delta, \
                unique_eids, inverse_eids, unique_nids, \
                unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
                Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst = self.curr_data
            else:
                _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, \
                unique_time_delta, inverse_time_delta, \
                unique_eids, inverse_eids, unique_nids, \
                unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, \
                Q_node_idx, reindex, reduce_idx, b_num_src, b_num_dst, \
                b2_inv_idx, b2_dstnodes, b2_dsttimes, b2_dstindex, b2_srcnodes, b2_eids, b2_ets, \
                b2_unique_time_delta, b2_inverse_time_delta, \
                b2_unique_eids, b2_inverse_eids, b2_unique_nids, \
                b2_unique_dstnodes, b2_inverse_dstnodes, b2_unique_srcnodes, b2_inverse_srcnodes, \
                b2_Q_node_idx, b2_reindex, b2_reduce_idx, b2_num_src, b2_num_dst = self.curr_data
            mailbox_uniq, mailbox_nbrs, mailbox_ets, mailbox_eid = self.curr_mailboxUpd
            self.curr_data = None # curr_data被使用后即赋值为None

            # updating and aggregating
            if self.num_layers == 1:
                with nvtx.annotate("update mem", color="purple"):
                    with nvtx.annotate("cal mail_delta", color="red"): # TODO
                        with nvtx.annotate("get_mailbox_time", color="blue"): # TODO
                            unique_mail_ts = self.ctx.manager_mem_mail.get_mailbox_time(unique_nids)
                        with nvtx.annotate("get_mem_time", color="blue"): # TODO
                            delta = unique_mail_ts - self.ctx.manager_mem_mail.get_mem_time(unique_nids)
                        with nvtx.annotate("precomputed_times", color="blue"): # TODO
                            mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to("cuda"))

                    with nvtx.annotate("concat mail", color="blue"): # TODO
                        mail = torch.cat([self.ctx.manager_mem_mail.get_mailbox_data(unique_nids), mail_delta], dim=1)
                    with nvtx.annotate("concat mem", color="blue"): # TODO
                        mem = self.ctx.manager_mem_mail.get_mem_data(unique_nids)
                    
                    with nvtx.annotate("nn.grucell", color="blue"): # TODO
                        mem = self.mem_cell(mail, mem)
                    with nvtx.annotate("update_mem_batch", color="blue"): # TODO
                        with torch.no_grad():
                            self.ctx.manager_mem_mail.update_mem_batch(unique_nids, mem, unique_mail_ts, mailbox_uniq, mailbox_nbrs)
                
                # with nvtx.annotate("update mem", color="purple"):
                #     unique_mail_ts = g.mailbox.time[unique_nids] # on cpu
                #     delta = unique_mail_ts - g.mem.time[unique_nids]
                #     mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to("cuda"))
                # mail = torch.cat([g.mailbox.mail[unique_nids], mail_delta], dim=1)
                # mem = g.mem.data[unique_nids]
                # with nvtx.annotate("nn.grucell", color="blue"): # TODO
                #     mem = self.mem_cell(mail, mem)
                # # 写回也可以pipeline TODO 不一定有必要，可以和后面的计算掩盖
                # g.mem.update(unique_nids, mem, unique_mail_ts) # 目前为写回CPU

                # no scatter    
                # 替换tail.nfeat() 
                with nvtx.annotate("get_data_batch nfeat efeat", color="purple"):
                    with nvtx.annotate("nfeat get_data_batch", color="purple"):
                        tail_nfeat = self.ctx.manager_nfeat.get_data_batch(unique_nids)
                    with nvtx.annotate("nfeat nfeat_map", color="purple"):
                        nfeat = tail_nfeat if self.nfeat_map is None else self.nfeat_map(tail_nfeat)
                    # 替换tail.efeat()
                    with nvtx.annotate("efeat get_data_batch", color="purple"):
                        tail_efeat = self.ctx.manager_efeat.get_data_batch(unique_eids)
                # using attn0, b_*
                with nvtx.annotate("op.aggregate", color="green"):
                    # def forward(self, blk: TBlock, Q_node_idx, nodeData_dst, node_dst_inverse, nodeData_src, node_src_inverse, efeat_unique, efeat_inverse, unique_time_delta, time_inverse) -> Tensor:
                    output = self.attn0(b_num_src, b_num_dst, reduce_idx, reindex, Q_node_idx, nfeat, mem, unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, tail_efeat, inverse_eids, unique_time_delta, inverse_time_delta)
                embeds = output[_inv_idx]
            else:
                assert self.num_layers == 2
                with nvtx.annotate("update mem", color="purple"):
                    with nvtx.annotate("cal mail_delta", color="red"): # TODO
                        unique_mail_ts = g.mailbox.time[b2_unique_nids] # on cpu
                        delta = unique_mail_ts - g.mem.time[b2_unique_nids]
                        mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to("cuda"))
                    mail = torch.cat([g.mailbox.mail[b2_unique_nids], mail_delta], dim=1)
                    mem = g.mem.data[b2_unique_nids]
                    mem = self.mem_cell(mail, mem)
                    # 写回也可以pipeline TODO 不一定有必要，可以和后面的计算掩盖
                    g.mem.update(b2_unique_nids, mem, unique_mail_ts) # 目前为写回CPU

                # 替换tail.nfeat() 
                tail_nfeat = self.ctx.manager_nfeat.get_data_batch(b2_unique_nids)
                nfeat = tail_nfeat if self.nfeat_map is None else self.nfeat_map(tail_nfeat)
                # 替换tail.efeat()
                tail_efeat = self.ctx.manager_efeat.get_data_batch(b2_unique_eids)
                tail_efeat_attn1 = self.ctx.manager_efeat.get_data_batch(unique_eids)
                # using attn0, b2_*
                # import pdb;pdb.set_trace()
                output = self.attn0(b2_num_src, b2_num_dst, b2_reduce_idx, b2_reindex, b2_Q_node_idx, nfeat, mem, b2_unique_dstnodes, b2_inverse_dstnodes, b2_unique_srcnodes, b2_inverse_srcnodes, tail_efeat, b2_inverse_eids, b2_unique_time_delta, b2_inverse_time_delta)
                # output = output[b2_inv_idx]
                # using attn1, b_*
                output = self.attn1(b_num_src, b_num_dst, reduce_idx, reindex, Q_node_idx, output, b2_inv_idx, unique_dstnodes, inverse_dstnodes, unique_srcnodes, inverse_srcnodes, tail_efeat_attn1, inverse_eids, unique_time_delta, inverse_time_delta)
                embeds = output[_inv_idx]
            
            # print(f"embeds {embeds.shape}")
            # compute scores
            with nvtx.annotate("compute score", color="purple"):
                src, dst, neg = batch.split_data(embeds) # 这里还可以去冗余
                scores = self.edge_predictor(src, dst)
                if neg is not None:
                    scores = (scores, self.edge_predictor(src, neg))
            del embeds
            del src
            del dst
            del neg
            with nvtx.annotate("save raw msgs", color="purple"):
                ### 原版消融
                # self.save_raw_msgs_TEST_BLKM(batch)
                # sdev = batch.g.storage_device()
                # mem = batch.g.mem.data

                # with nvtx.annotate("save raw msgs-block_adj", color="red"):
                #     # blk = batch.block_adj(self.ctx) # return TBlock(ctx, 0, dstnodes, ets, dstindex, srcnodes, eids, ets)
                #     nids = g._edges[batch._beg_idx:batch._end_idx]
                #     dstnodes = nids.T.reshape(-1).astype(np.int32)

                #     nids = g._edges[batch._beg_idx:batch._end_idx]
                #     nids = np.flip(nids, axis=1)
                #     srcnodes = nids.T.reshape(-1).astype(np.int32)

                #     eid = np.tile(np.arange(batch._beg_idx, batch._end_idx, dtype=np.int32), 2)

                #     times = g._times[batch._beg_idx:batch._end_idx]
                #     ets = np.tile(times, 2).astype(np.float32)
                    
                # with nvtx.annotate("save raw msgs-op.coalesce", color="red"):
                #     # blk = tg.op.coalesce(blk, by='latest')
                #     assert len(dstnodes) == len(srcnodes)
                #     uniq_nodes, uniq_idx = np.unique(dstnodes, return_index=True)
                #     idx = _c.find_latest_uniq(uniq_nodes, dstnodes, ets)
                #     srcnodes = srcnodes[idx]
                #     eid = eid[idx]
                #     ets = ets[idx]
                #     dstnodes = uniq_nodes
                #     # dsttimes = dsttimes[uniq_idx]

                # with nvtx.annotate("save raw msgs-uniq nbrs", color="red"):
                #     # 写回数据量实际很小 μs级别
                #     # import pdb;pdb.set_trace()
                #     uniq = torch.from_numpy(dstnodes).long().to(sdev)
                #     nbrs = torch.from_numpy(srcnodes).long().to(sdev)
                    
                #     efeat = self.ctx.manager_efeat.get_data_batch(eid)
                #     mail = torch.cat([mem[uniq], mem[nbrs], efeat], dim=1)
                #     mail_ts = torch.from_numpy(ets).to(sdev)
                # with nvtx.annotate("save raw msgs-store mail_ts", color="red"):
                #     batch.g.mailbox.store(uniq, mail, mail_ts, uniq, nbrs, batch._b_id)
                
                ### 二改消融
                # mem = batch.g.mem.data
                # efeat = self.ctx.manager_efeat.get_data_batch(mailbox_eid)
                # with nvtx.annotate("save raw msgs-torch.cat", color="red"):
                #     mail = torch.cat([mem[mailbox_uniq], mem[mailbox_nbrs], efeat], dim=1)
                # with nvtx.annotate("save raw msgs-store", color="red"):
                #     batch.g.mailbox.store(mailbox_uniq.long(), mail, mailbox_ets, mailbox_uniq.long(), mailbox_nbrs.long(), batch._b_id)
                
                with torch.no_grad():
                    self.ctx.manager_mem_mail.update_mailbox(mailbox_uniq, mailbox_nbrs, mailbox_eid, mailbox_ets)
            return scores


    def _load_new_perfCeilBase(self, batch):
        # print(f"_load_new_perfCeilBase batch {batch._b_id}")
        if self.ctx.next_blk_tail is None:
            head = batch.block(self.ctx)

            for i in range(self.num_layers):
                tail = head if i == 0 \
                    else tail.next_block(include_dst=True, use_dst_times=False)
                tail = tg.op.dedup(tail) if self.dedup else tail
                with nvtx.annotate("sample", color="purple"):
                    tail = self.sampler.sample(tail) # 先去重再采样
                        
            # load data / feats
            with nvtx.annotate("preload data/feat", color="purple"):
                tg.op.preload(head, use_pin=True)

            self.ctx.next_blk_tail = tail
            self.ctx.next_blk_head = head
            self.ctx.next_mem = tail.mem_data()

    def update_memory_newBase(self, blk: tg.TBlock, batch:tg.TBatch, mem) -> Tensor:
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

        # mem = blk.mem_data() # on cpu?
        time_start_1 = tt.start()
        with nvtx.annotate("update mem-mem_cell", color="purple"):
            mem = self.mem_cell(mail, mem)
        tt.t_mem_update_gru_cell += tt.elapsed(time_start_1)
        time_start_2 = tt.start()
        blk.g.mem.update(nodes, mem, mail_ts)
        tt.t_mem_update_after += tt.elapsed(time_start_2)
        return mem

    def forward_origin_newBase(self, batch: tg.TBatch) -> Tensor:
        with nvtx.annotate("forward", color="purple"):
            # print(f"forward batch {batch._b_id} - len(batch) {len(batch)}")
            # setup message passing
            
            tail = self.ctx.curr_blk_tail
            head = self.ctx.curr_blk_head
            mem = self.ctx.curr_mem

            if tail.num_dst() > 0:
                # t_start = tt.start()
                with nvtx.annotate("update mem", color="purple"):
                    mem = self.update_memory_newBase(tail, batch, mem)
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
                # print(f"batch {batch._b_id} embeds {embeds.shape}")
                src, dst, neg = batch.split_data(embeds)
                scores = self.edge_predictor(src, dst)
                if neg is not None:
                    scores = (scores, self.edge_predictor(src, neg))
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
                # print(f"batch {batch._b_id}")
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
            # 由于new blk 我肯定load了很多没用的东西 # 纯offline
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

            # # ! mailbox uniq, nbrs, ets 统计
            # if self.is_train:
            #     add_mailbox_upd_batch((uniq.int(), nbrs.int(), mail_ts, torch.tensor(blk.eid).int()))
            # else:
            #     add_mailbox_upd_batch_eval((uniq.int(), nbrs.int(), mail_ts, torch.tensor(blk.eid).int()))

            batch.g.mailbox.store(uniq, mail, mail_ts)
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
