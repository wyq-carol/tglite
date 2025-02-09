import torch
import tglite as tg

from torch import nn, Tensor
from tglite.nn import TemporalAttnLayer
from tglite._stats import tt

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..')) 
import support
from tglite.gpu_mem_track import *
from tglite.mymodule import *
import time
import nvtx
import tglite.config


class TGN(nn.Module):
    def __init__(self, ctx: tg.TContext,
                 dim_node: int, dim_edge: int, dim_time: int, dim_embed: int,
                 sampler: tg.TSampler, num_layers=2, num_heads=2, dropout=0.1,
                 dedup: bool = True):
        super().__init__()
        self.ctx = ctx
        self.dim_edge = dim_edge
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

    def forward(self, batch: tg.TBatch) -> Tensor:
        if tglite.config.ON_HETER:
            return self.forward0(batch)
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
            for i in range(self.num_layers): # 两层GNN有额外的问题，先处理1层
                tail = head if i == 0 \
                    else tail.next_block(include_dst=True, use_dst_times=False)
                tail = tg.op.dedup(tail) if self.dedup else tail
                with nvtx.annotate("sample", color="purple"):
                    tail = self.sampler.sample(tail)
            
            # load data / feats
            with nvtx.annotate("preload data/feat", color="purple"):
                tg.op.preload(head, use_pin=True)

            if tail.num_dst() > 0:
                with nvtx.annotate("update mem", color="purple"):
                    mem = self.update_memory0(tail)
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
                self.save_raw_msgs(batch)
            return scores
    
    def update_memory0(self, blk: tg.TBlock) -> Tensor:
        # 1. 对update_memory0进行去重
        cdev = blk.g.compute_device()
        nodes = blk.allnodes()

        mail_ts = blk.g.mailbox.time[nodes] # on cpu? no new segments

        delta = mail_ts - blk.g.mem.time[nodes]
        delta = delta.squeeze().to(cdev)
        mail = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta) # on cpu
        mail = torch.cat([blk.mail(), mail], dim=1) # no new segments 不等同于no request

        mem = blk.mem_data() # on cpu?

        nodes = blk.allnodes()
        unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)
        unique_mail_ts = blk.g.mailbox.time[unique_nodes]
        delta = unique_mail_ts - blk.g.mem.time[unique_nodes]
        mail = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta) # on cpu
        # mail = torch.cat([_g_mail[unique_nodes], mail], dim=1)
        
        # if mail.

        mem = self.mem_cell(mail, mem)
        blk.g.mem.update(nodes, mem, mail_ts)
        return mem

    def forward_origin(self, batch: tg.TBatch) -> Tensor:
        with nvtx.annotate("forward", color="purple"):
            # setup message passing
            with nvtx.annotate("dedup and cache", color="purple"):

                head = batch.block(self.ctx)

                for i in range(self.num_layers):
                    tail = head if i == 0 \
                        else tail.next_block(include_dst=True, use_dst_times=False)
                    tail = tg.op.dedup(tail) if self.dedup else tail
                    with nvtx.annotate("sample", color="purple"):
                        tail = self.sampler.sample(tail)

            # load data / feats
            with nvtx.annotate("preload data/feat", color="purple"):

                tg.op.preload(head, use_pin=True)
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            if tail.num_dst() > 0:
                # t_start = tt.start()
                with nvtx.annotate("update mem", color="purple"):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    mem = self.update_memory(tail)
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

    def update_memory(self, blk: tg.TBlock) -> Tensor:
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
