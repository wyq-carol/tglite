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
        self.dim_embed = dim_embed
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
        # wyq add record uniq-nbrs
        self.dstnodes2latestNbrs = torch.zeros([tglite.config.num_nodes, 2], dtype=torch.long) # 可以用int # TODO **去冗余重建_mailbox**

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
            for i in range(self.num_layers): # TODO 两层GNN有额外的问题，先处理1层
                tail = head if i == 0 \
                    else tail.next_block(include_dst=True, use_dst_times=False)
                tail = tg.op.dedup(tail) if self.dedup else tail
                with nvtx.annotate("sample", color="purple"):
                    tail = self.sampler.sample(tail)
            # TODO 这里可以暂时整体offload -> 未来加入采样动态性的考虑
            
            # load data / feats
            with nvtx.annotate("preload data/feat", color="purple"):
                tg.op.preload(head, use_pin=True)
                
                # TODO pipeline

            if tail.num_dst() > 0:
                with nvtx.annotate("update mem", color="purple"):
                    ## pipeline_memUpd comm(mem, mailbox) comp(update mem)
                    mem = self.update_memory0_pipeline_memUpd(tail, batch)
                    ## no_redundant_memUpd
                    # mem = self.update_memory0_no_redundant_memUpd(tail, batch) # [TODO @ZZZ]
                    ## unique_nodes_memUpd
                    # mem = self.update_memory0(tail, batch)
                    ## no pipeline
                    # mem = self.update_memory(tail, batch)
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

    # pipeline_memUpd comm(mem, mailbox) comp(update mem)
    def update_memory0_pipeline_memUpd(self, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # TODO 写得没什么问题，但是通信太长计算太短，pipeline 不起来
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
        stage0_index = math.floor(len(unique_nodes)/3)
        stage_indices = [(0, stage0_index), (stage0_index, math.floor(len(unique_nodes)*2/3)), (math.floor(len(unique_nodes)*2/3), len(unique_nodes))] 
        # import pdb;pdb.set_trace()
        # 0阶段通信
        with nvtx.annotate(f"memUpdStage_comm_{0}", color="green"):
            unique_nodes_slice = unique_nodes[0:stage0_index]
            # all on cpu
            cur_mail = torch.cat([blk._g.mailbox.mail[unique_nodes_slice].to(cdev), mail_delta[0:stage0_index]], dim=1)
            cur_mem = blk._g.mem._data[unique_nodes_slice].to(cdev)
        # for 循环
        comp_stream_memUpd = torch.cuda.Stream()
        comm_stream_memUpd = torch.cuda.Stream()
        nxt_mail = None
        nxt_mem = None
        mems = []
        for i, _ in enumerate(stage_indices):
            with nvtx.annotate(f"Stage_{i}", color="blue"):
                # nxt阶段通信
                if i < len(stage_indices) - 1:
                    with torch.cuda.stream(comm_stream_memUpd), nvtx.annotate(f"memUpdStage_comm_{i+1}", color="green"):
                        nxt_start_idx, nxt_end_idx = stage_indices[i + 1]
                        nxt_unique_nodes_slice = unique_nodes[nxt_start_idx:nxt_end_idx]
                        # all on cpu
                        print(f"blk._g.mailbox.mail.is_pinned() {blk._g.mailbox.mail.is_pinned()}")
                        nxt_mail = torch.cat([blk._g.mailbox.mail[nxt_unique_nodes_slice].to(cdev, non_blocking=True), mail_delta[nxt_start_idx:nxt_end_idx]], dim=1)
                        print(f"blk._g.mem._data.is_pinned() {blk._g.mem._data.is_pinned()}")
                        nxt_mem = blk._g.mem._data[nxt_unique_nodes_slice].to(cdev, non_blocking=True)
                # cur阶段计算
                with torch.cuda.stream(comp_stream_memUpd), nvtx.annotate(f"memUpdStage_comp_{i}", color="red"):
                    mem = self.mem_cell(cur_mail, cur_mem)
                    ans = torch.matmul(cur_mail, cur_mail[0:472])
                    # print(ans) TODO
                mems.append(mem)
                torch.cuda.synchronize()
                # 在nxt阶段使用通信好的张量
                if nxt_mail is not None:
                    cur_mail = nxt_mail
                    cur_mem = nxt_mem
                    nxt_mail = None
                    nxt_mem = None
        torch.cuda.current_stream().wait_stream(comp_stream_memUpd)
        mem = torch.cat(mems)

        # assert torch.allclose(mem_standard, mem) # accuracy test pass! \O/

        # 写回也可以pipeline TODO
        blk.g.mem.update(unique_nodes, mem, unique_mail_ts) # 目前为写回CPU

        return mem[inverse_indices]
                
    # no_redundant_memUpd
    def update_memory0_no_redundant_memUpd(self, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
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
    def update_memory0(self, blk: tg.TBlock, batch:tg.TBatch) -> Tensor:
        # 1. [DONE]对update_memory0进行去重
        # 2. [TODO]将下个batch使用的特征放在GPU上，其余放在CPU上
        # 3. [TODO]将热节点放在GPU上，其余放在CPU上
        cdev = blk.g.compute_device()
        nodes = blk.allnodes()
        unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)
        unique_mail_ts = blk.g.mailbox.time[unique_nodes]
        delta = unique_mail_ts - blk.g.mem.time[unique_nodes]
        mail_delta = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta.squeeze().to(cdev)) # on GPU
        mem = blk._g.mem._data[unique_nodes].to(cdev)
        mail = torch.cat([blk._g.mailbox.mail[unique_nodes].to(cdev), mail_delta], dim=1)


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
                    if tglite.config.ON_STATISTIC:
                        # 1. 统计热节点
                        node_centric_skew(tail._dstnodes, tail._srcnodes)
                        # 2. 将采样完整offload 出去 WYQ_TODO: 保存一整个sample文件
                        add_samples((tail._dstnodes, tail._dstindex, tail._srcnodes, tail._eid, tail._ets))

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
