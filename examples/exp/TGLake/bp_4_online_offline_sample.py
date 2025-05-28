
from tglite._sampler import TSampler
from tglite._utils import create_tcsr, check_num_nodes
import numpy as np
import nvtx
import torch
import pandas as pd
import os
import time
import tglite._c as _c

def dedup(blk):
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
    return blk, inv_idx


class MockGraph:
    def __init__(self, edges, times):
        self._tcsr = None
        self._edges = edges
        self._times = times
        self._num_nodes = check_num_nodes(edges)
        self._tcsr = None

    def num_nodes(self):
        return self._num_nodes

    def num_edges(self):
        return self._edges.shape[0]

    def _init_tcsr(self):
        """Creates tcsr of the graph if it doesn't exist"""
        if self._tcsr is None:
            self._tcsr = create_tcsr(self._edges, self._times, num_nodes=self._num_nodes)

    def _get_tcsr(self):
        """Returns the tcsr of the graph"""
        self._init_tcsr()
        return self._tcsr

class MockBlock:
    """模拟TBlock的数据结构"""
    def __init__(self, g, dstnodes, dsttimes):
        self._g = g
        self._dstnodes = dstnodes
        self._dsttimes = dsttimes
        self._has_nbrs = None
        self._dstindex = None
        self._srcnodes = None
        self._eid = None
        self._ets = None
    
    def set_nbrs(self, dstindex, srcnodes,
                      eid, ets):
        """Sets the neighbor attributes for the block."""
        self.clear_nbrs()
        self._has_nbrs = True
        self._dstindex = dstindex
        self._srcnodes = srcnodes
        self._eid = eid
        self._ets = ets

    def num_dst(self) -> int:
        return len(self._dstnodes)
    
    def clear_nbrs(self):
        """Clears the neighbor attributes and related cache."""
        self._has_nbrs = False
        self._dstindex = None
        self._srcnodes = None
        self._eid = None
        self._ets = None

    def _replace_dst(self, dstnodes, dsttimes):
        """Replaces destination nodes and timestamps with given arrays."""
        self.clear_nbrs()
        self._dstnodes = dstnodes
        self._dsttimes = dsttimes
    
class EdgesIter(object):
    """ An edge iterator of a TGraph."""
    def __init__(self, g: MockGraph, size=1, start=None, end=None):
        self._g = g
        self._size = size
        self._curr = 0 if start is None else start
        self._last = g.num_edges() if end is None else end
        self._b_id = -1

    def __iter__(self):
        return self

    def __next__(self):
        self._b_id += 1
        if self._curr < self._last:
            idx = self._curr
            self._curr += self._size
            end = min(self._curr, self._last)
            return self._b_id, idx, end
        raise StopIteration

def iter_edges(g: MockGraph, size=1, start=None, end=None) -> EdgesIter:
    return EdgesIter(g, size=size, start=start, end=end)

# TODO Sampler
# _inv_idx, 去重<dstnode, dsttime> 后的图
# blk.num_src, blk.num_dst
# b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta,
# unique_eids, _reverse_eids, _unique_nids,
# _unique_dst_nodes, _reverse_dst_nodes, _unique_src_nodes, _reverse_src_nodes: 对node_inversed_indices 进行torch.unique
# Q_node_idx: reverse_dst_nodes[b_dstindex[x]] x.arange(len(b_srcnodes))
# reindex: 对b_dstindex 取torch.unique
## on CPU or on GPU? / 三类去重如何一次完成? / 如何结合offline 结果处理online 的负采样?

if __name__ == "__main__":
    time_counts = []
    time_counts_our = []
    
    ### load graph
    df = pd.read_csv(str(os.path.join('', f'/home/volume/wiki-talk/edges.csv')))
    src = df['src'].to_numpy().astype(np.int32).reshape(-1, 1)
    dst = df['dst'].to_numpy().astype(np.int32).reshape(-1, 1)
    etime = df['time'].to_numpy().astype(np.float32)
    edges = np.concatenate([src, dst], axis=1)
    g = MockGraph(edges, etime)

    ### sampling config
    # train_end, val_end = support.data_split(g.num_edges(), 0.7, 0.15)
    num_samples = g.num_edges()
    train_percent = 0.7
    val_percent = 0.15
    train_end = int(np.ceil(num_samples * train_percent))
    val_end = int(np.ceil(num_samples * (train_percent + val_percent)))
    neg_sampler = lambda size: np.random.randint(0, g.num_nodes(), size)
    N_NBRS=10
    SAMPLING='recent'
    N_THREADS=8
    LAYER=2
    TEST_B_ID=5
    bs = 6000 # 2000

    ### sampling and training
    edge_iter = iter_edges(g, size=bs, end=train_end)
    try:
        b_id, idx, end = next(edge_iter)
        neg_nodes = neg_sampler(bs)
        while True:
            try: # 这种写法是为了提前拿到下一个batch的采样结果
                next_b_id, next_idx, next_end = next(edge_iter)
                next_neg_nodes = neg_sampler(bs)
            except StopIteration:
                print("epoch end")
                break
            ### DO STH.
            if b_id > TEST_B_ID: # for short test
                break
            ### build blk
            # nodes()
            nids = g._edges[idx:end]
            nids = nids.T.reshape(-1)
            nids = np.concatenate([nids, neg_nodes]).astype(np.int32)
            # times()
            n_repeats = 3
            times = g._times[idx:end]
            times = np.tile(times, n_repeats).astype(np.float32)
            # blk
            blk = MockBlock(g, nids, times)
            sampler = TSampler(N_NBRS, strategy=SAMPLING, num_threads=N_THREADS)
            # Sample baseline
            start_time = time.time()
            with nvtx.annotate("sample baseline", color="red"):
                blk, _ = dedup(blk) # dedup and cache
                blk = sampler.sample(blk)
                time_delta = blk._dsttimes[blk._dstindex] - blk._ets
                unique_time_delta, inverse_time_delta = np.unique(time_delta, return_inverse=True)
                all_nids = np.concatenate([blk._dstnodes, blk._srcnodes])
                unique_nids, inverse_nids = np.unique(all_nids, return_inverse=True)
                unique_eids, inverse_eids = np.unique(blk._eid, return_inverse=True)
                unique_dstnodes, inverse_dstnodes = np.unique(inverse_nids[:blk._dstnodes.shape[0]], return_inverse=True)
                unique_srcnodes, inverse_srcnodes = np.unique(inverse_nids[blk._dstnodes.shape[0]:], return_inverse=True)
                x = np.arange(blk._srcnodes.shape[0])
                Q_node_idx = inverse_dstnodes[blk._dstindex[x]]
                reindex = np.unique(blk._dstindex, return_inverse=True)[1]
            time_count = time.time() - start_time
            if b_id > 1:
                time_counts.append(time_count)
            # Our sample
            blk = MockBlock(g, nids, times)
            start_time = time.time()
            with nvtx.annotate("sample our", color="red"):
                blk = sampler.sample(blk) # TODO 
            time_count = time.time() - start_time
            if b_id > 1:
                time_counts_our.append(time_count)
            if LAYER > 1:
                # 2 layer
                next_dstnodes = np.concatenate([blk._dstnodes, blk._srcnodes])
                next_dsttimes = np.concatenate([blk._dsttimes, blk._ets])
                blk = MockBlock(g, next_dstnodes, next_dsttimes)
                # Sample baseline
                start_time = time.time()
                with nvtx.annotate("sample baseline", color="red"):
                    blk, _ = dedup(blk) # dedup and cache
                    blk = sampler.sample(blk)
                    time_delta = blk._dsttimes[blk._dstindex] - blk._ets
                    unique_time_delta, inverse_time_delta = np.unique(time_delta, return_inverse=True)
                    all_nids = np.concatenate([blk._dstnodes, blk._srcnodes])
                    unique_nids, inverse_nids = np.unique(all_nids, return_inverse=True)
                    unique_eids, inverse_eids = np.unique(blk._eid, return_inverse=True)
                    unique_dstnodes, inverse_dstnodes = np.unique(inverse_nids[:blk._dstnodes.shape[0]], return_inverse=True)
                    unique_srcnodes, inverse_srcnodes = np.unique(inverse_nids[blk._dstnodes.shape[0]:], return_inverse=True)
                    x = np.arange(blk._srcnodes.shape[0])
                    Q_node_idx = inverse_dstnodes[blk._dstindex[x]]
                    reindex = np.unique(blk._dstindex, return_inverse=True)[1]
                time_count = time.time() - start_time
                if b_id > 1:
                    time_counts.append(time_count)
                # Our sample
                blk = MockBlock(g, next_dstnodes, next_dsttimes)
                start_time = time.time()
                with nvtx.annotate("sample our", color="red"):
                    blk = sampler.sample(blk) # TODO
                time_count = time.time() - start_time
                if b_id > 1:
                    time_counts_our.append(time_count)
            ## INIT NXT_Batch
            b_id = next_b_id
            idx = next_idx
            end = next_end
            neg_nodes = next_neg_nodes
    except StopIteration:
        pass

    # dedup and cache + sample + get our format
    print(f"baseline {1000*sum(time_counts)/len(time_counts):>10.4f}ms")
    print(f"Our      {1000*sum(time_counts_our)/len(time_counts_our):>10.4f}ms")
    