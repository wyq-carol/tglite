from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._block import TBlock
    from ._context import TContext

import torch
import numpy as np
from torch import Tensor

from ._stats import tt
from .op import precomputed_zeros, precomputed_times, edge_reduce, edge_view, edge_softmax
import nvtx
from .mymodule import LinearHandleZeroInput
from tglite.gpu_mem_track import *

def is_tensor_all_zeros(tensor):
    """
    判断一个Tensor是否全为0。

    参数:
    tensor (torch.Tensor): 需要判断的Tensor。

    返回:
    bool: 如果Tensor全为0，返回True；否则返回False。
    """
    return torch.all(tensor == 0).item()

class TimeEncode(torch.nn.Module):

    # A hidden tag used to detect if we have an instance of this builtin encoder
    # so that we can call the more optimized method of generating zeros, without
    # running into situation with circular dependencies.
    __tg_builtin_encoder__ = True

    def __init__(self, dim_time: int):
        '''
        Initializes the TimeEncode module, which encodes time information into a higher-dimensional space.
        
        :param dim_time: dimensionality of the encoded time
        '''
        super().__init__()
        # self.w = torch.nn.Linear(1, dim_time)
        self.w = LinearHandleZeroInput(1, dim_time)
        self.w.weight = torch.nn.Parameter(torch
            .from_numpy(1 / 10 ** np.linspace(0, 9, dim_time))
            .float().reshape(dim_time, 1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim_time).float())
        self._z = torch.zeros(1).float()

    def preload_zeros(self, view):
        return self(view)

    def zeros(self, size: int, device):
        '''
        Generates a tensor of zeros with the encoded time dimensionality.
        
        :param size:
        :param device:
        '''
        # 在这等着我呢！
        if self._z.device != torch.device(device):
            self._z = self._z.to(device)
        # expand does not allocate memory
        view = self._z.expand(size)
        return self(False, view)

    def forward(self, is_zero_tensor, ts: Tensor) -> Tensor:
        '''
        Forward pass of the TimeEncode module. Encodes the input time stamps into a high-dimensional space.
        
        :param ts: input time stamps
        '''
        # here
        with nvtx.annotate("time-encode", color="red"):
            # wyq_refine
            # ans = torch.cos(self.w(is_zero_tensor, ts.unsqueeze(-1)))
            # WYQ_TODO fusion
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            tmp = self.w(is_zero_tensor, ts)
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            ans = torch.cos(tmp)
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            return ans


class TemporalAttnLayer(torch.nn.Module):
    def __init__(self, ctx: TContext, num_heads: int,
                 dim_node: int, dim_edge: int, dim_time: int, dim_out: int,
                 dropout=0.1):
        """
        Initializes the Temporal Attention Layer for processing dynamic graphs with temporal features.
        This layer uses multi-head attention mechanism to incorporate node, edge, and time features.

        :param ctx: context object
        :param num_heads: number of heads
        :param dim_node: dimension of node features
        :param dim_edge: dimension of edge features
        :param dim_time: dimension of time features
        :param dim_out: dimension of output features
        :param dropout: dropout rate
        """
        super().__init__()
        assert (dim_out % num_heads == 0)
        self.ctx = ctx
        self.num_heads = num_heads
        self.dim_edge = dim_edge
        self.dim_out = dim_out
        self.time_encode = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_node + dim_time, dim_out)
        self.w_kv = torch.nn.Linear(dim_node + dim_edge + dim_time, dim_out * 2)
        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)
        self.attn_act = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, blk: TBlock) -> Tensor:
        '''
        Forward pass of the Temporal Attention Layer. Applies a time-sensitive attention mechanism over 
        the input graph block (blk) to produce node embeddings. The method handles both cases of graph blocks 
        with and without edges.

        If the block has no edges, a zero-initialized tensor is concatenated with the destination node features.
        For blocks with edges, the method computes attention scores and aggregates neighbor features using 
        the computed attention. 

        :param blk: input graph block
        '''
        if blk.num_edges() == 0:
            dev = blk.dstdata['h'].device
            out = torch.zeros(blk.num_dst(), self.dim_out, dtype=torch.float32, device=dev)
            out = torch.cat([out, blk.dstdata['h']], dim=1)
        else:
            with nvtx.annotate("precompute", color="blue"):
                t_start = tt.start()
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                zero_time_feat = precomputed_zeros(self.ctx, blk.layer, self.time_encode, blk.num_dst())
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                tt.t_time_zero += tt.elapsed(t_start)
                t_start = tt.start()
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                nbrs_time_feat = precomputed_times(self.ctx, blk.layer, self.time_encode, blk.time_deltas())
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                tt.t_time_nbrs += tt.elapsed(t_start)
                t_start = tt.start()
            
            with nvtx.annotate("redundancy-mul", color="red"): # TODO 这段的执行时间确实长
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                Q = torch.cat([blk.dstdata['h'], zero_time_feat], dim=1)
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                if self.dim_edge > 0:
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    Z = torch.cat([blk.srcdata['h'], blk.efeat(), nbrs_time_feat], dim=1)
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                else:
                    Z = torch.cat([blk.srcdata['h'], nbrs_time_feat], dim=1)
                del zero_time_feat
                del nbrs_time_feat
                # torch.cuda.empty_cache()
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                Q = self.w_q(Q)
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                Z = self.w_kv(Z)
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            
            with nvtx.annotate("else", color="red"): # else这段没有显存变化 torch的机制？
                with nvtx.annotate("else-K", color="red"):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    K = Z[:, :self.dim_out]
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                with nvtx.annotate("else-V", color="red"):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    V = Z[:, self.dim_out:]
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    del Z
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    tt.t_sum += tt.elapsed(t_start)
                    
                    t_start = tt.start()
                with nvtx.annotate("else-Q", color="red"):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    Q = edge_view(blk, Q) # 对Q进行scatter
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                with nvtx.annotate("else-reshape", color="red"):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    Q = torch.reshape(Q, (Q.shape[0], self.num_heads, -1))
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    # print(Q.shape)
                    K = torch.reshape(K, (K.shape[0], self.num_heads, -1))
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    V = torch.reshape(V, (V.shape[0], self.num_heads, -1))
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

                with nvtx.annotate("else-attn", color="red"):
                    attn = torch.sum(Q * K, dim=2)
                    # print(attn.shape)
                    del Q
                    del K
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

                with nvtx.annotate("else-leakyRelu", color="red"):
                    attn = self.attn_act(attn)
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            with nvtx.annotate("edge-softmax", color="blue"):
                with nvtx.annotate("edge-softmax", color="blue"):
                    attn = edge_softmax(blk, attn)  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                with nvtx.annotate("dropout", color="blue"):
                    attn = self.dropout(attn)
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                with nvtx.annotate("reshape", color="blue"):
                    out = torch.reshape(V * attn[:, :, None], (V.shape[0], -1))  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                    del attn
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)

                with nvtx.annotate("edge-reduce", color="blue"):
                    out = edge_reduce(blk, out, op='sum') # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                out = torch.cat([out, blk.dstdata['h']], dim=1)
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            tt.t_self_attn += tt.elapsed(t_start)

            with nvtx.annotate("output", color="blue"):
                out = self.w_out(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                # torch.cuda.empty_cache()
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                out = torch.nn.functional.relu(self.dropout(out))
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                out = self.layer_norm(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        return out
    
