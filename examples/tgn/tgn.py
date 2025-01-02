import torch
import tglite as tg

from torch import nn, Tensor
from tglite.nn import TemporalAttnLayer
from tglite._stats import tt

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..')) 
import support
import time
import nvtx


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
        self.mem_cell = nn.GRUCell(2 * dim_embed + dim_edge + dim_time, dim_embed)
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
            if tail.num_dst() > 0:
                # t_start = tt.start()
                # import pdb;pdb.set_trace()
                with nvtx.annotate("update mem", color="purple"):
                    mem = self.update_memory(tail)
                # tt.t_update_memory += tt.elapsed(t_start)
                nfeat = tail.nfeat() if self.nfeat_map is None else self.nfeat_map(tail.nfeat())
                tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
                tail.srcdata['h'] = nfeat[tail.num_dst():] + mem[tail.num_dst():]
                # tt.t_mem_update += tt.elapsed(t_start)
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

            # memory messages
            t_start = tt.start()
            with nvtx.annotate("save raw msgs", color="purple"):
                self.save_raw_msgs(batch)
            tt.t_post_update += tt.elapsed(t_start)

            return scores

    def update_memory(self, blk: tg.TBlock) -> Tensor:
        cdev = blk.g.compute_device()
        # import pdb;pdb.set_trace()
        nodes = blk.allnodes()

        # index, reverse = torch.unique(nodes, return_inverse=True)
        # mail_ts = blk.g.mailbox.time[index][reverse]
        time_start_0 = tt.start()
        mail_ts = blk.g.mailbox.time[nodes]
        # index, reverse = torch.unique(nodes, return_inverse=True)
        # mail_ts = blk.g.mailbox.time[index][reverse]
        tt.tt_mail_ts_load += tt.elapsed(time_start_0)

        delta = mail_ts - blk.g.mem.time[nodes]
        delta = delta.squeeze().to(cdev)
        with nvtx.annotate("update mem-precompute_times", color="purple"):
            mail = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta)
        mail = torch.cat([blk.mail(), mail], dim=1)

        mem = blk.mem_data()
        time_start_1 = tt.start()
        with nvtx.annotate("update mem-mem_cell", color="purple"):
            mem = self.mem_cell(mail, mem)
        tt.t_mem_update_gru_cell += tt.elapsed(time_start_1)
        time_start_2 = tt.start()
        blk.g.mem.update(nodes, mem, mail_ts)
        tt.t_mem_update_after += tt.elapsed(time_start_2)
        return mem

    def save_raw_msgs(self, batch: tg.TBatch):
        sdev = batch.g.storage_device()
        mem = batch.g.mem.data

        with nvtx.annotate("save raw msgs-block_adj", color="red"):
            blk = batch.block_adj(self.ctx)
        with nvtx.annotate("save raw msgs-op.coalesce", color="red"):
            blk = tg.op.coalesce(blk, by='latest')

        with nvtx.annotate("save raw msgs-uniq nbrs", color="red"):
            uniq = torch.from_numpy(blk.dstnodes).long().to(sdev)
            nbrs = torch.from_numpy(blk.srcnodes).long().to(sdev)
            if self.dim_edge > 0:
                eids = torch.from_numpy(blk.eid).long().to(sdev)
                mail = torch.cat([mem[uniq], mem[nbrs], batch.g.efeat[eids]], dim=1)
            else:
                mail = torch.cat([mem[uniq], mem[nbrs]], dim=1)
            mail_ts = torch.from_numpy(blk.ets).to(sdev)
        with nvtx.annotate("save raw msgs-store mail_ts", color="red"):
            batch.g.mailbox.store(uniq, mail, mail_ts)