import numpy as np
import pandas as pd
import os
import time
import torch
import nvtx
from tglite._sampler import TSampler
from tglite._utils import create_tcsr, check_num_nodes
import tglite._c as _c
import custom_unique
# TODO offline+online?

def dedup(blk):
    """
    Applies the deduplication optimization to the TBlock by rewriting the destination nodes.

    :param blk: MockBlock instance
    :return: Deduplicated block and inverse index
    """
    if blk.num_dst() == 0:
        return blk, None
    nodes = blk._dstnodes
    times = blk._dsttimes
    has_dups, nodes, times, inv_idx = _c.dedup_targets(nodes, times)
    if has_dups:
        blk._replace_dst(nodes, times)
    return blk, inv_idx

class MockGraph:
    """
    A mock graph class for testing purposes.
    """
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
    """
    A mock block class for testing purposes.
    """
    def __init__(self, g, dstnodes, dsttimes):
        self._g = g
        self._dstnodes = dstnodes
        self._dsttimes = dsttimes
        self._has_nbrs = None
        self._dstindex = None
        self._srcnodes = None
        self._eid = None
        self._ets = None

    def set_nbrs(self, dstindex, srcnodes, eid, ets):
        """Sets the neighbor attributes for the block."""
        self.clear_nbrs()
        self._has_nbrs = True
        self._dstindex = dstindex
        self._srcnodes = srcnodes
        self._eid = eid
        self._ets = ets

    def num_dst(self):
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


class EdgesIter:
    """
    An edge iterator of a TGraph.
    """
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


def iter_edges(g: MockGraph, size=1, start=None, end=None):
    """
    Creates an edge iterator for the given graph.

    :param g: MockGraph instance
    :param size: Batch size
    :param start: Start index
    :param end: End index
    :return: EdgesIter instance
    """
    return EdgesIter(g, size=size, start=start, end=end)


def sample_baseline(blk, sampler):
    """
    Performs the baseline sampling method.

    :param blk: MockBlock instance
    :param sampler: TSampler instance
    :return: Time taken for sampling
    """
    start_time = time.time()
    with nvtx.annotate("sample baseline", color="red"):
        blk, _ = dedup(blk)  # dedup and cache
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
    return time.time() - start_time

# TODO
def sample_our(blk, sampler):
    """
    Performs the custom sampling method.

    :param blk: MockBlock instance
    :param sampler: TSampler instance
    :return: Time taken for sampling
    """
    with nvtx.annotate("sample our", color="red"):
        # 先全转为tensor进行测试
        start_time = time.time() # 重写数据结构，全部使用Tensor # 如果掩盖不住就没法prefetch，放上去看一下吧 不然没必要拆分offline和online
        blk, _ = dedup(blk)  # dedup and cache
        blk = sampler.sample(blk)
        b_dstindex = torch.tensor(blk._dstindex)
        b_dsttimes = torch.tensor(blk._dsttimes)
        b_ets = torch.tensor(blk._ets)
        b_dstnodes = torch.tensor(blk._dstnodes)
        b_srcnodes = torch.tensor(blk._srcnodes)
        b_eid = torch.tensor(blk._eid)
        time_delta = b_dsttimes[b_dstindex] - b_ets
        all_nids = torch.cat([b_dstnodes, b_srcnodes])
        unique_time_delta, inverse_time_delta = torch.unique_consecutive(time_delta, return_inverse=True)
        unique_nids, inverse_nids = torch.unique_consecutive(all_nids, return_inverse=True)
        unique_eids, inverse_eids = torch.unique_consecutive(b_eid, return_inverse=True)
        unique_dstnodes, inverse_dstnodes = torch.unique_consecutive(inverse_nids[:b_dstnodes.shape[0]], return_inverse=True)
        unique_srcnodes, inverse_srcnodes = torch.unique_consecutive(inverse_nids[b_dstnodes.shape[0]:], return_inverse=True)
        x = torch.arange(b_srcnodes.shape[0])
        Q_node_idx = inverse_dstnodes[b_dstindex[x]]
        reindex = torch.unique(b_dstindex, return_inverse=True)[1]
    return time.time() - start_time


def run_sampling_test(g, sampler, sampling_config):
    """
    Runs the sampling test for the given graph and sampler.

    :param g: MockGraph instance
    :param sampler: TSampler instance
    :param sampling_config: Dictionary containing sampling configuration
    """
    time_counts_baseline = []
    time_counts_our = []

    edge_iter = iter_edges(g, size=sampling_config['batch_size'], end=sampling_config['train_end'])

    for b_id, idx, end in edge_iter:
        if b_id > sampling_config['test_batch_id']:
            break

        neg_nodes = sampling_config['neg_sampler'](sampling_config['batch_size'])

        # Build block
        nids = g._edges[idx:end].T.reshape(-1)
        nids = np.concatenate([nids, neg_nodes]).astype(np.int32)
        times = np.tile(g._times[idx:end], 3).astype(np.float32)
        blk = MockBlock(g, nids, times)

        # Sample baseline
        time_count = sample_baseline(blk, sampler)
        if b_id > 1:
            time_counts_baseline.append(time_count)

        # Sample our
        blk = MockBlock(g, nids, times)
        time_count = sample_our(blk, sampler)
        if b_id > 1:
            time_counts_our.append(time_count)

        if sampling_config['LAYER'] > 1:
            # 2 layer
            next_dstnodes = np.concatenate([blk._dstnodes, blk._srcnodes])
            next_dsttimes = np.concatenate([blk._dsttimes, blk._ets])
            blk = MockBlock(g, next_dstnodes, next_dsttimes)

            # Sample baseline
            time_count = sample_baseline(blk, sampler)
            if b_id > 1:
                time_counts_baseline.append(time_count)

            # Sample our
            blk = MockBlock(g, next_dstnodes, next_dsttimes)
            time_count = sample_our(blk, sampler)
            if b_id > 1:
                time_counts_our.append(time_count)

    print(f"Baseline: {1000 * sum(time_counts_baseline) / len(time_counts_baseline):>10.4f}ms")
    print(f"Our     : {1000 * sum(time_counts_our) / len(time_counts_our):>10.4f}ms")


if __name__ == "__main__":
    # Load graph
    df = pd.read_csv(os.path.join('', '/home/volume/wiki-talk/edges.csv'))
    src = df['src'].to_numpy().astype(np.int32).reshape(-1, 1)
    dst = df['dst'].to_numpy().astype(np.int32).reshape(-1, 1)
    etime = df['time'].to_numpy().astype(np.float32)
    edges = np.concatenate([src, dst], axis=1)
    g = MockGraph(edges, etime)

    # Sampling configuration
    sampling_config = {
        'num_samples': g.num_edges(),
        'train_percent': 0.7,
        'val_percent': 0.15,
        'neg_sampler': lambda size: np.random.randint(0, g.num_nodes(), size),
        'N_NBRS': 10,
        'SAMPLING': 'recent',
        'N_THREADS': 8,
        'LAYER': 2,
        'TEST_B_ID': 5,
        'batch_size': 6000,
        'train_end': int(np.ceil(g.num_edges() * 0.7)),
        'val_end': int(np.ceil(g.num_edges() * (0.7 + 0.15))),
        'test_batch_id': 5
    }
    print(f"BS {sampling_config['batch_size']} LAYER {sampling_config['LAYER']} DATA {'wiki-talk'}")

    # Initialize sampler
    sampler = TSampler(sampling_config['N_NBRS'], strategy=sampling_config['SAMPLING'], num_threads=sampling_config['N_THREADS'])

    # Run sampling test
    run_sampling_test(g, sampler, sampling_config)