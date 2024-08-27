import tglite as tg

from torch import nn, Tensor
from tglite.nn import TemporalAttnLayer

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..')) 
import support

from tglite._stats import tt


class TGAT(nn.Module):
    def __init__(self, ctx: tg.TContext,
                 dim_node: int, dim_edge: int, dim_time: int, dim_embed: int,
                 sampler: tg.TSampler, num_layers=2, num_heads=2, dropout=0.1,
                 dedup: bool = True):
        super().__init__()
        self.ctx = ctx
        self.num_layers = num_layers
        self.attn = nn.ModuleList([
            TemporalAttnLayer(ctx,
                num_heads=num_heads,
                dim_node=dim_node if i == 0 else dim_embed,
                dim_edge=dim_edge,
                dim_time=dim_time,
                dim_out=dim_embed,
                dropout=dropout)
            for i in range(num_layers)])
        self.sampler = sampler
        self.edge_predictor = support.EdgePredictor(dim=dim_embed)
        self.dedup = dedup

    def forward(self, batch: tg.TBatch) -> Tensor:
        # import pdb; pdb.set_trace()
        t_block = tt.start()
        # wyq- start head
        head = batch.block(self.ctx)
        tt.t_block = tt.elapsed(t_block)
        for i in range(self.num_layers):
            # wyq- 遍历多个tail
            t_next = tt.start()
            # wyq- start tail
            # 如果i=0, tail=head; 
            # 否则   , tail=tail.next_block, 自动设置 (include_dst=True)
            tail = head if i == 0 \
                else tail.next_block(include_dst=True)
            tt.t_next.append(tt.elapsed(t_next))
            t_dedup = tt.start()
            # 更新tail- dedup
            tail = tg.op.dedup(tail) if self.dedup else tail
            t_end_dedup = tt.elapsed(t_dedup)
            tt.t_dedup.append(t_end_dedup)
            tt.t_dedup_total += t_end_dedup
            t_cache = tt.start()
            # 更新tail- cache; for 采样？
            tail = tg.op.cache(self.ctx, tail.layer, tail)
            tt.t_cache.append(tt.elapsed(t_cache))
            # after sample is over
            tail = self.sampler.sample(tail)
            print(f"tail over")
            print(f"tail _layer={tail._layer}")
            print(f"tail dstnodes.size={tail._dstnodes.size}, times.size={tail._dsttimes.size}")
            if tail._srcnodes is not None:
                print(f"tail srcnodes.size={tail._srcnodes.size}")
            if tail.eid is not None:
                print(f"tail eid.size={tail.eid.size}, ets.size={tail.ets.size}")
            print()

        t_preload = tt.start()
        tg.op.preload(head, use_pin=True)
        tt.t_preload = tt.elapsed(t_preload)
        if tail.num_dst() > 0:
            tail.dstdata['h'] = tail.dstfeat()
            tail.srcdata['h'] = tail.srcfeat()
        t_aggregate = tt.start()
        # import pdb; pdb.set_trace()
        embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
        time = tt.elapsed(t_aggregate)
        tt.t_aggregate = time
        tt.t_aggregate_total += time
        # import pdb; pdb.set_trace()
        del head
        del tail

        tt.print_model()
        tt.reset_model()
        src, dst, neg = batch.split_data(embeds)
        scores = self.edge_predictor(src, dst)
        if batch.neg_nodes is not None:
            scores = (scores, self.edge_predictor(src, neg))

        return scores