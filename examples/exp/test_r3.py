import numpy as np
import pandas as pd
import os
import time
import torch
import nvtx
from tglite._sampler import TSampler
from tglite._utils import create_tcsr, check_num_nodes
import tglite._c as _c
import torch

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

# TODO
def sample_our(blk, sampler):
    """
    Performs the custom sampling method.

    :param blk: MockBlock instance
    :param sampler: TSampler instance
    :return: Time taken for sampling
    """
    with nvtx.annotate("sample our", color="red"):
        blk, b_inv_idx = dedup(blk)  # dedup and cache
        blk = sampler.sample(blk)

def sample_our_21(blk, sampler):
    """
    Performs the custom sampling method.

    :param blk: MockBlock instance
    :param sampler: TSampler instance
    :return: Time taken for sampling
    """
    with nvtx.annotate("sample our", color="red"):
        # 先全转为tensor进行测试
        blk, b_inv_idx = dedup(blk)  # dedup and cache
        blk = sampler.sample(blk)

_mailboxUpd_samples = None

class MemSimulator:
    def __init__(self, max_nids):
        self.max_nids = max_nids
        self.mem = torch.zeros((max_nids), dtype=torch.int32, device='cuda')
        self.mailbox = torch.zeros((max_nids, 2), dtype=torch.int32, device='cuda') - 1

    def simulate_mangaer(self, idx, uniq, nbr, bid):
        unique, counts = torch.unique(self.mailbox, return_counts=True)
        cache_mask = unique >= self.max_nids
        cache_idx = unique[cache_mask] - self.max_nids
        counts = counts[cache_mask].to(torch.int32)
        cache = torch.zeros((2 * self.max_nids), dtype=torch.int32, device='cuda')
        if cache_idx.shape[0] > 0:
            cache[cache_idx] = counts
        
        all_indices = torch.arange(self.max_nids, device='cuda')
        keep_indices = all_indices[~torch.isin(all_indices, uniq)]
        un_updated = self.mailbox[keep_indices]
        unique, counts = torch.unique(un_updated, return_counts=True)
        mask = torch.isin(unique, idx)
        cached = unique[mask]
        counts = counts[mask]
        
        free_cache = torch.nonzero(cache == 0, as_tuple=True)[0].to(torch.int32) #id
        cid = free_cache[: cached.shape[0]] + self.max_nids
        
        cached = cached.sort().values
        if cached.shape[0] > 0:
            flat = self.mailbox.flatten()
            idx = torch.bucketize(flat, cached, right=False)
            valid_idx = idx < cached.size(0)
            matched = torch.zeros_like(flat, dtype=torch.bool)
            matched[valid_idx] = flat[valid_idx] == cached[idx[valid_idx]]
            flat[matched] = cid[idx[matched]]
            table_replaced = flat.view_as(self.mailbox)
            self.mailbox = table_replaced

        self.mailbox[uniq] = torch.stack((uniq, nbr), dim = 1).to(torch.int32)
        self.mem[idx] += 1
        
        # print(f"{}")
        with open("stack_l2_b2_output.txt", "a") as f:
            print(f"mem_mailbox usage in [round {bid}] : [{(cache > 0).sum().item() + (self.mem > 0).sum().item()}]", file=f, flush=True)
    
    
    
    
    
    

def run_sampling_test(g, sampler, sampling_config):
    """
    Runs the sampling test for the given graph and sampler.

    :param g: MockGraph instance
    :param sampler: TSampler instance
    :param sampling_config: Dictionary containing sampling configuration
    """
    edge_iter = iter_edges(g, size=sampling_config['batch_size'], end=sampling_config['train_end'])

    mem_sim = MemSimulator(2601977)
    for b_id, idx, end in edge_iter:
        print(f"b_id {b_id}")
        if b_id > 500:
            break

        neg_nodes = sampling_config['neg_sampler'](sampling_config['batch_size'])

        # Build block
        nids = g._edges[idx:end].T.reshape(-1)
        nids = np.concatenate([nids, neg_nodes]).astype(np.int32)
        times = np.tile(g._times[idx:end], 3).astype(np.float32)
        blk = MockBlock(g, nids, times)

        # Sample our
        if sampling_config['LAYER'] == 1:
            blk = MockBlock(g, nids, times)
            time_count = sample_our(blk, sampler)
            # memory 1. nids torch.cat([blk._dstnodes, blk._srcnodes]).unique
            # mailbox 
            

        if sampling_config['LAYER'] == 2:
            blk = MockBlock(g, nids, times)
            time_count = sample_our_21(blk, sampler)
            next_dstnodes = np.concatenate([blk._dstnodes, blk._srcnodes])
            next_dsttimes = np.concatenate([blk._dsttimes, blk._ets])
            blk = MockBlock(g, next_dstnodes, next_dsttimes)

            # Sample our
            blk = MockBlock(g, next_dstnodes, next_dsttimes)
            time_count = sample_our(blk, sampler)
            
        

        # here
        # memory data indices torch.unique([np.concatenate([blk._dstnodes, blk._srcnodes])])
        # mailbox_ref_uniq, mailbox_ref_counts, mailbox_uniq, mailbox_nbrs, mailbox_ets, mailbox_eid = _mailboxUpd_samples[b_id]
        idx = torch.unique(torch.from_numpy(np.concatenate([blk._dstnodes, blk._srcnodes])).to('cuda'))
        # print(blk._dstnodes, blk._srcnodes)
        # mem_sim.simulate_mangaer(idx, mailbox_uniq.to('cuda'), mailbox_nbrs.to('cuda'), b_id)
        uniq = torch.randperm(2601977, dtype=torch.int32)[:5000].to('cuda')
        nbr = torch.randperm(2601977, dtype=torch.int32)[:5000].to('cuda')


        mem_sim.simulate_mangaer(idx, uniq,nbr, b_id)


if __name__ == "__main__":
    
    # simulate_mangaer(None, None, None, torch.tensor([0, 1, 2], device='cuda'), torch.tensor([2, 1], device='cuda'), torch.tensor([0, 2], device='cuda'), 3)
    
    print("here")
    # ! Load graph
    # df = pd.read_csv(os.path.join('', '/home/volume/wiki-talk/edges.csv'))
    # df = pd.read_csv(os.path.join('', '/home/volume/lastfm/edges.csv'))
    df = pd.read_csv(os.path.join('', '/home/volume/stackoverflow/edges.csv'))
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
        # ! layer
        'LAYER': 2,
        'TEST_B_ID': 5,
        # ! batchsize
        'batch_size': 2000,
        'train_end': int(np.ceil(g.num_edges() * 0.7)),
        'val_end': int(np.ceil(g.num_edges() * (0.7 + 0.15))),
        'test_batch_id': 5
    }
    print(f"BS {sampling_config['batch_size']} LAYER {sampling_config['LAYER']} DATA {''}")

    # Initialize sampler
    sampler = TSampler(sampling_config['N_NBRS'], strategy=sampling_config['SAMPLING'], num_threads=sampling_config['N_THREADS'])

    # ! add pt
    # _mailboxUpd_samples = torch.load('/home/volume/tglake_res/tglake_res_mailboxUpd/TRAIN_mailboxUpdBatchs_DATA_lastfm_BS_2000_NLAYER_1_NBR_10_NHEAD_2.pt')
    # Run sampling test
    run_sampling_test(g, sampler, sampling_config)
