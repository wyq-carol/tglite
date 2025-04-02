import torch
import numpy as np
import torch_scatter
from torch import Tensor
from typing import Any, Callable, List, Union

from . import _c
from ._core import TError
from ._block import TBlock
from ._context import TContext
from ._stats import tt
import nvtx
from tglite.gpu_mem_track import *
from tglite._utils import INFO_LOG


# def find_last_message(uniq_nodes: np.ndarray, sorted_edges: np.ndarray):
#     msg_order, msg_index = _c.find_last_message(uniq_nodes, sorted_edges)
#     msg_order = msg_order.reshape(-1, 2)
#     return msg_order, msg_index


def edge_view(blk: TBlock, data: Tensor) -> Tensor:
    '''
    Reindex the data of edges based on the target node indices in the TBlock.
    
    :param blk:
    :param data:
    '''
    # blk._check_has_nbrs()
    # assert data.shape[0] == blk._dstdata.dim()
    if blk._g.storage_device() != torch.device("cpu"):
        # print(f"torch.save {blk._g_dstindex.shape}")
        # torch.save(blk._g_dstindex, "blk._g_dstindex.pt")
        return data[blk._g_dstindex]

    idx = torch.from_numpy(blk._dstindex)
    idx = idx.to(device=data.device, dtype=torch.long)
    return data[idx]


def edge_softmax(blk: TBlock, data: Tensor) -> Tensor:
    '''
    Computes segmented softmax on given data using edge information from the block.
    
    :param blk:
    :param data:
    '''
    # blk._check_has_nbrs()
    # size = blk._edata.dim()
    size = blk.num_src()
    # assert blk._edata.dim() == blk.num_src()
    if blk._g.storage_device() != torch.device("cpu"):
        reindex = torch.unique(blk._g_dstindex, return_inverse=True)[1]
        # print(f"edge_softmax reindex {reindex}")
        return torch_scatter.scatter_softmax(data, reindex, dim=0, dim_size=size)

    # 在这等我！
    reindex = torch.from_numpy(blk._dstindex)
    reindex = reindex.to(device=data.device, dtype=torch.long)
    reindex = torch.unique(reindex, return_inverse=True)[1]
    # reindex = reindex.to(device=data.device, dtype=torch.long)
    return torch_scatter.scatter_softmax(data, reindex, dim=0, dim_size=size)


def edge_reduce(blk: TBlock, data: Tensor, op='sum') -> Tensor:
    '''
    Computes segmented reduction (e.g. sum or mean) on given data 
    using edge information from the block.
    
    :param blk:
    :param data:
    '''
    # blk._check_has_nbrs()
    assert op in ['sum', 'mean'], "currently only supports sum or mean"
    # size = blk._dstdata.dim()
    size = blk.num_dst()
    # assert blk._dstdata.dim() == blk.num_dst()
    if blk._g.storage_device() != torch.device("cpu"):
        return torch_scatter.segment_coo(data, blk._g_dstindex, dim_size=size, reduce=op)

    scatter_idx = torch.from_numpy(blk._dstindex)
    scatter_idx = scatter_idx.to(device=data.device, dtype=torch.long)
    return torch_scatter.segment_coo(data, scatter_idx, dim_size=size, reduce=op)


def src_scatter(blk: TBlock, data: Tensor, op='sum') -> Tensor:
    '''
    Aggregates the features of the source node indices in TBlock, 
    using the specified aggregation operation ('sum' or 'mean').
    
    :param blk:
    :param data:
    :param op:
    '''
    blk._check_has_nbrs()
    assert op in ['sum', 'mean'], "currently only supports sum or mean"
    assert data.shape[0] == len(blk._srcnodes)
    uniq_nids, idx = blk.uniq_src()
    idx = idx.to(data.device)
    return torch_scatter.scatter(data, idx, dim=0, dim_size=len(uniq_nids), reduce=op)


def coalesce(blk: TBlock, by='latest') -> TBlock:
    '''
    Segmented operation to reduce source nodes for each destination node 
    by a certain property, such as latest timestamp.
    
    :param blk:
    :param by: method to sample source nodes
    '''
    assert by == 'latest', "currently only supports latest"
    assert blk.has_nbrs() and len(blk.dstnodes) == len(blk.srcnodes)
    uniq_nodes, uniq_idx = np.unique(blk.dstnodes, return_index=True)
    idx = _c.find_latest_uniq(uniq_nodes, blk.dstnodes, blk.ets)
    src = blk.srcnodes[idx]
    eid = blk.eid[idx]
    ets = blk.ets[idx]
    blk._replace_dst(uniq_nodes, blk.dsttimes[uniq_idx])
    blk.set_nbrs(np.arange(len(src)), src, eid, ets)
    return blk


def preload0_noMailMem(blk: TBlock, use_pin=True):
    '''
    Prefetch data (e.g. features, memory, mails) needed by the TBlock 
    and its subsequent blocks for computations.

    :param blk:
    :param use_pin: whether to pin memory
    '''
    curr = blk
    while curr.next is not None:
        curr = curr.next
    while curr is not None:
        if curr.num_dst() > 0:
            if curr.has_nbrs():
                if curr.next is None:
                    with nvtx.annotate("preload nfeat", color="red"):
                        curr._load_nfeat(use_pin=use_pin)
                with nvtx.annotate("preload efeat", color="red"):
                    curr._load_efeat(use_pin=use_pin)
        curr = curr.prev

def preload0_uniqLoadFeat_noMailMem(blk: TBlock, use_pin=True):
    '''
    Prefetch data (e.g. features, memory, mails) needed by the TBlock 
    and its subsequent blocks for computations.

    :param blk:
    :param use_pin: whether to pin memory
    '''
    curr = blk
    while curr.next is not None:
        curr = curr.next
    while curr is not None:
        if curr.num_dst() > 0:
            if curr.has_nbrs():
                if curr.next is None:
                    with nvtx.annotate("preload nfeat", color="red"):
                        curr._load_nfeat0_uniqLoadFeat(use_pin=use_pin)
                with nvtx.annotate("preload efeat", color="red"):
                    curr._load_efeat0_uniqLoadFeat(use_pin=use_pin)
        curr = curr.prev

def preload0_uniqLoadFeat(blk: TBlock, use_pin=True):
    '''
    Prefetch data (e.g. features, memory, mails) needed by the TBlock 
    and its subsequent blocks for computations.

    :param blk:
    :param use_pin: whether to pin memory
    '''
    curr = blk
    while curr.next is not None:
        curr = curr.next
    while curr is not None:
        if curr.num_dst() > 0:
            if curr.next is None:
                with nvtx.annotate("preload mail", color="red"):
                    curr._load_mail(use_pin=use_pin)
                with nvtx.annotate("preload mem_data", color="red"):
                    curr._load_mem_data(use_pin=use_pin)
            if curr.has_nbrs():
                if curr.next is None:
                    with nvtx.annotate("preload nfeat", color="red"):
                        curr._load_nfeat0_uniqLoadFeat(use_pin=use_pin)
                with nvtx.annotate("preload efeat", color="red"):
                    curr._load_efeat0_uniqLoadFeat(use_pin=use_pin)
        curr = curr.prev

def preload(blk: TBlock, use_pin=True):
    '''
    Prefetch data (e.g. features, memory, mails) needed by the TBlock 
    and its subsequent blocks for computations.

    :param blk:
    :param use_pin: whether to pin memory
    '''
    curr = blk
    while curr.next is not None:
        curr = curr.next
    while curr is not None:
        if curr.num_dst() > 0:
            if curr.next is None:
                with nvtx.annotate("preload mail", color="red"):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    curr._load_mail(use_pin=use_pin)
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                with nvtx.annotate("preload mem_data", color="red"):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    curr._load_mem_data(use_pin=use_pin)
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            if curr.has_nbrs():
                if curr.next is None:
                    with nvtx.annotate("preload nfeat", color="red"):
                        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                        curr._load_nfeat(use_pin=use_pin)
                        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                with nvtx.annotate("preload efeat", color="red"):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    curr._load_efeat(use_pin=use_pin)
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        curr = curr.prev


def aggregate(blk: TBlock, fn_or_list: Union[Callable, List[Callable]], key: str = None) -> Any:
    '''
    Performs pull-style multi-hop aggregation from the tail block 
    back towards the given block by applying function to each block, 
    using the key to pass along results.
    
    :param blk:
    :param fn_or_list:
    :param key:
    '''
    with nvtx.annotate("op.aggregate", color="green"):
        while blk.next is not None:
            blk = blk.next
        output = None
        while blk is not None:
            if blk.num_dst() == 0:
                with nvtx.annotate("op.aggregate run_hooks", color="green"):
                    output = blk.run_hooks(output)
            elif isinstance(fn_or_list, List):
                with nvtx.annotate("op.aggregate apply", color="green"):
                    output = blk.apply(fn_or_list[blk.layer])
            else:
                with nvtx.annotate("op.aggregate apply2", color="green"):
                    output = blk.apply(fn_or_list)
            t_start = tt.start()
            if blk.prev is not None and output is not None and key:
                if blk._include_prev_dst:
                    num_dst = blk.prev.num_dst()
                    with nvtx.annotate("aggregate blk.prev.dstdata/srcdata", color="red"):
                        blk.prev.dstdata[key] = output[:num_dst]
                        blk.prev.srcdata[key] = output[num_dst:]
                else:
                    with nvtx.annotate("op.aggregate blk.prev.dstdata", color="green"):
                        blk.prev.srcdata[key] = output
            with nvtx.annotate("op.aggregate clear_data", color="red"):
                blk.clear_data()
            with nvtx.annotate("op.aggregate clear_hooks", color="red"):
                blk.clear_hooks()
            tt.t_prep_input += tt.elapsed(t_start)
            blk = blk.prev
        return output


def propagate(blk: TBlock, fn_or_list: Union[Callable, List[Callable]]) -> Any:
    '''
    Performs push-style multi-hop propagation from the given block 
    to the tail block by applying function to each block.
    
    :param blk:
    :param fn_or_list:
    '''
    output = None
    while blk is not None:
        if blk.num_dst() == 0:
            output = blk.run_hooks(output)
        elif isinstance(fn_or_list, List):
            output = blk.apply(fn_or_list[blk.layer])
        else:
            output = blk.apply(fn_or_list)
        blk.clear_data()
        blk.clear_hooks()
        blk = blk.next
    return output


def dedup(blk: TBlock) -> TBlock:
    '''
    Applies the deduplication optimization to the TBlock 
    by rewriting the destination nodes.

    :param blk:
    '''
    if blk.num_dst() == 0:
        return blk
    nodes = blk._dstnodes
    times = blk._dsttimes
    has_dups, nodes, times, inv_idx = _c.dedup_targets(nodes, times)
    if has_dups:
        blk._replace_dst(nodes, times)
        # print("register-hook _DedupInvertHook")
        with nvtx.annotate("register-hook _DedupInvertHook", color="red"):
            blk.register_hook(_DedupInvertHook(inv_idx))
    return blk

def dedup_statistic(blk: TBlock) -> TBlock:
    '''
    Applies the deduplication optimization to the TBlock 
    by rewriting the destination nodes.

    :param blk:
    '''
    if blk.num_dst() == 0:
        return blk
    nodes = blk._dstnodes
    times = blk._dsttimes
    has_dups, nodes, times, inv_idx = _c.dedup_targets(nodes, times)
    if has_dups:
        blk._replace_dst(nodes, times)
        # print("register-hook _DedupInvertHook")
        with nvtx.annotate("register-hook _DedupInvertHook", color="red"):
            blk.register_hook(_DedupInvertHook(inv_idx))
    return blk, inv_idx

def dedup1_offlineSample(blk: TBlock, inv_idx) -> TBlock:
    '''
    Applies the deduplication optimization to the TBlock 
    by rewriting the destination nodes.

    :param blk:
    '''
    if blk.num_dst() == 0:
        return blk
    # print("register-hook _DedupInvertHook")
    with nvtx.annotate("register-hook _DedupInvertHook", color="red"):
        blk.register_hook(_DedupInvertHook(inv_idx))
    return blk


class _DedupInvertHook(object):
    def __init__(self, inv_idx: np.ndarray):
        self.inv_idx = inv_idx

    def __call__(self, _blk: TBlock, output: Tensor) -> Tensor:
        return output[self.inv_idx]


def cache(ctx: TContext, id: int, blk: TBlock, include_first=False):
    '''
    Applies the caching optimization to the TBlock by rewriting 
    the destination nodes and using ctx as scratch space.
    
    :param ctx:
    :param id:
    :param blk:
    :param include_first:
    '''
    # if ctx.graph.storage_device() != torch.device("cpu"):
    #     return blk
    # TGN 只有dedup 没有cache！
    if ctx._training or not ctx._cache_enabled:
        return blk
    if blk.prev is None and not include_first:
        # usually not worth it for the head block
        return blk
    if blk.num_dst() == 0:
        return blk

    cache_table = ctx._cache_tables.get(id)
    if cache_table is None:
        cache_table = _c.EmbedTable(ctx._cache_dim_emb, ctx._cache_limit)
        ctx._cache_tables[id] = cache_table

    nodes = blk._dstnodes
    times = blk._dsttimes

    keys = _c.compute_cache_keys(nodes, times)
    hit_idx, embeds = cache_table.lookup(keys, blk.g.compute_device())
    hit_count = torch.sum(hit_idx).item()

    if hit_count == len(nodes):
        print("register-hook _CacheAllHitsHook")
        with nvtx.annotate("register-hook _CacheAllHitsHook", color="red"):
            blk._replace_dst_empty()
            blk.register_hook(_CacheAllHitsHook(embeds))
    elif hit_count == 0:
        # no need to adjust dst nodes
        print("register-hook _CacheAllMissHook")
        with nvtx.annotate("register-hook _CacheAllMissHook", color="red"):
            blk.register_hook(_CacheAllMissHook(cache_table, keys))
    else:
        print("register-hook _CachePartialHitsHook")
        with nvtx.annotate("register-hook _CachePartialHitsHook", color="red"):
            miss_idx = (~ hit_idx)
            miss_idx_np = miss_idx.cpu().numpy()
            nodes = nodes[miss_idx_np]
            times = times[miss_idx_np]
            keys = keys[miss_idx_np]
            blk._replace_dst(nodes, times)
            blk.register_hook(_CachePartialHitsHook(cache_table, embeds, miss_idx, keys))

    return blk


class _CacheAllHitsHook(object):
    def __init__(self, embeds: Tensor):
        self.embeds = embeds

    def __call__(self, _blk: TBlock, output: Tensor) -> Tensor:
        '''
        Returns the cached embeddings for all destination nodes.
        
        :param blk:
        :param output:

        '''
        return self.embeds


class _CacheAllMissHook(object):
    def __init__(self, table, miss_keys: np.ndarray):
        self.table = table
        self.miss_keys = miss_keys

    def __call__(self, _blk: TBlock, output: Tensor) -> Tensor:
        '''
        Stores embeddings for all destination nodes.
        
        :param blk:
        :param output:
        '''
        if output.shape[0] != len(self.miss_keys):
            raise TError('dimension mismatch')
        self.table.store(self.miss_keys, output)
        return output


class _CachePartialHitsHook(object):
    def __init__(self, table, embeds: Tensor, miss_idx: Tensor, miss_keys: np.ndarray):
        self.table = table
        self.embeds = embeds
        self.miss_idx = miss_idx
        self.miss_keys = miss_keys

    def __call__(self, _blk: TBlock, output: Tensor) -> Tensor:
        '''
        Returns the cached embeddings for destination nodes with hits,
        and stores the embeddings for destination nodes with misses.
        
        :param blk:
        :param output:
        '''
        if output.shape[0] != len(self.miss_keys):
            raise TError('dimension mismatch')
        self.table.store(self.miss_keys, output)
        self.embeds[self.miss_idx] = output
        return self.embeds


def precomputed_zeros(ctx: TContext, id: int, encoder: Callable, num: int) -> Tensor:

    '''
    Generates a tensor of precomputed zero values encoded by the specified encoder, 
    used for creating a batch of zero time encodings.

    :param ctx:
    :param id:
    :param encoder:
    :param num:
    :return: precomputed zero-initialized tensor of the given size
    '''
    with nvtx.annotate("precompute_zeros", color="red"):
        cdev = ctx._g.compute_device()
        if ctx._training or not ctx._time_enabled:
            return encoder(True, num)
            
            if ctx._z is not None:
                return encoder.preload_zeros(ctx._z.expand(num))
            else: 
                # 写kernel 不要真的生成torch.zeros
                if getattr(encoder, '__tg_builtin_encoder__', False):
                    return encoder.zeros(num, cdev)
                else:
                    return encoder(False, torch.zeros(num, dtype=torch.float, device=cdev))

        time_table = ctx._time_tables.get(id) # id 为layer id
        if time_table is None:
            print(f"encoder: {encoder}")
            print(torch.arange(
                ctx._time_window + 1, dtype=torch.float, device=cdev).shape)
            time_table = encoder(torch.arange(
                ctx._time_window + 1, dtype=torch.float, device=cdev))
            ctx._time_tables[id] = time_table

        output = time_table[0].repeat(num, 1)
        output = output.view(num, -1)
        return output


def precomputed_times(ctx: TContext, id: int, encoder: Callable, times: Tensor) -> Tensor:
    '''
    Encodes a tensor of time values using the provided encoder, leverage precomputed values 
    if available in the context's time table. 
    
    :param ctx:
    :param id:
    :param encoder:
    :param times:
    :return: a precomputed tensor of the given times
    '''
    with nvtx.annotate("precompute_times", color="red"):
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        if ctx._training or not ctx._time_enabled:
            return encoder(False, times)
    
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        time_table = ctx._time_tables.get(id)
        if time_table is None:
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            time_table = encoder(torch.arange(
                ctx._time_window + 1, dtype=torch.float, device=ctx._g.compute_device()))
            ctx._time_tables[id] = time_table
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        size = times.shape[0]
        hit_count, hit_idx, output, times, inv_idx = \
            _c.find_dedup_time_hits(times, time_table, ctx._time_window)
        uniq_size = times.shape[0]
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        if hit_count != uniq_size:
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            miss_idx = (~ hit_idx)
            times = times[miss_idx]
            output[miss_idx] = encoder(times.squeeze())
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

        output = output[inv_idx]
        output = output.view(size, -1)
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        return output
