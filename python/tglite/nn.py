from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._block import TBlock
    from ._context import TContext

import torch
import numpy as np
from ._utils import INFO_LOG
from torch import Tensor

from ._stats import tt
from .op import precomputed_zeros, precomputed_times, edge_reduce, edge_view, edge_softmax
import nvtx
from .mymodule import LinearHandleZeroInput
from tglite.gpu_mem_track import *
from dgNN.src.tgn_kernel_fuse.test import FusedTGNFunction
import torch_scatter

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
            tmp = self.w(is_zero_tensor, ts) # TODO ts
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            ans = torch.cos(tmp)
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            return ans


class TemporalAttnLayer_precompute(torch.nn.Module):
    # 统计一系列输入输出的shape
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
        
        INFO_LOG("Layer init")
        INFO_LOG(f"num_heads: {num_heads}")
        INFO_LOG(f"dim_node: {dim_node}")
        INFO_LOG(f"dim_edge: {dim_edge}")
        INFO_LOG(f"dim_time: {dim_time}")
        INFO_LOG(f"dim_out: {dim_out}")
        INFO_LOG(f"dropout: {dropout}")
        
        
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
        INFO_LOG("FORWARD BLK")
        INFO_LOG(f"blk.num_dst(): {blk.num_dst()}")
        INFO_LOG(f"blk.num_src(): {blk.num_src()}")
        INFO_LOG(f"blk.num_edges(): {blk.num_edges()}")
        INFO_LOG(f"blk.layer: {blk.layer}")
        INFO_LOG(f"blk.time_deltas().shape: {blk.time_deltas().shape}")
        INFO_LOG(f"blk.dstdata['h'].shape: {blk.dstdata['h'].shape}")
        INFO_LOG(f"blk.srcdata['h'].shape: {blk.srcdata['h'].shape}")
        INFO_LOG(f"blk.efeat().shape: {blk.efeat().shape}")
        INFO_LOG("FORWARD CTX")
        INFO_LOG(f"ctx._training: {self.ctx._training}")
        INFO_LOG(f"ctx._time_enabled: {self.ctx._time_enabled}")
        INFO_LOG(f"ctx._time_window: {self.ctx._time_window}")
        INFO_LOG(f"ctx._g: {self.ctx._g}")
        INFO_LOG(f"ctx._time_tables: {self.ctx._time_tables}")
        INFO_LOG(f"ctx._z: {self.ctx._z}")

        # print(f"blk._g_dstindex {blk._g_dstindex.shape}")
        # torch.save(blk._g_dstindex, "blk._g_dstindex.pt")
        # print(f"blk.dstdata['h'] {blk.dstdata['h'].shape}")
        # torch.save(blk.dstdata['h'], "blk.dstdata_h.pt")
        # print(f"blk.srcdata['h'] {blk.srcdata['h'].shape}")
        # torch.save(blk.srcdata['h'], "blk.srcdata_h.pt")
        # torch.save(blk.efeat(), "blk.efeat.pt")
        # torch.save(blk.time_deltas(), "blk.time_deltas.pt")

        with nvtx.annotate("precompute", color="blue"):
            t_start = tt.start()
            INFO_LOG("FOWARD PRECOMPUTE ZERO")
            zero_time_feat = precomputed_zeros(self.ctx, blk.layer, self.time_encode, blk.num_dst())
            INFO_LOG(f"zero_time_feat.shape {zero_time_feat.shape}")
            tt.t_time_zero += tt.elapsed(t_start)
            t_start = tt.start()
            INFO_LOG("FOWARD PRECOMPUTE NBR")
            nbrs_time_feat = precomputed_times(self.ctx, blk.layer, self.time_encode, blk.time_deltas())
            INFO_LOG(f"nbrs_time_feat.shape {nbrs_time_feat.shape}")
            tt.t_time_nbrs += tt.elapsed(t_start)
            t_start = tt.start()
        
        with nvtx.annotate("redundancy-mul", color="red"): # TODO 这段的执行时间确实长
            Q = torch.cat([blk.dstdata['h'], zero_time_feat], dim=1)
            INFO_LOG(f"Q {Q.dtype}")
            if self.dim_edge > 0:
                Z = torch.cat([blk.srcdata['h'], blk.efeat(), nbrs_time_feat], dim=1)
            else:
                Z = torch.cat([blk.srcdata['h'], nbrs_time_feat], dim=1)
            del zero_time_feat
            del nbrs_time_feat
            # torch.cuda.empty_cache()

            Q = self.w_q(Q)
            Z = self.w_kv(Z)
        
        with nvtx.annotate("else", color="red"): # else这段没有显存变化 torch的机制？
            with nvtx.annotate("else-K", color="red"):
                K = Z[:, :self.dim_out]
            with nvtx.annotate("else-V", color="red"):
                V = Z[:, self.dim_out:]
                del Z
                tt.t_sum += tt.elapsed(t_start)
                
                t_start = tt.start()
            with nvtx.annotate("else-Q", color="red"):
                Q = edge_view(blk, Q) # 对Q进行scatter
            with nvtx.annotate("else-reshape", color="red"):
                Q = torch.reshape(Q, (Q.shape[0], self.num_heads, -1))
                K = torch.reshape(K, (K.shape[0], self.num_heads, -1))
                V = torch.reshape(V, (V.shape[0], self.num_heads, -1))

            with nvtx.annotate("else-attn", color="red"):
                attn = torch.sum(Q * K, dim=2)
                del Q
                del K

            with nvtx.annotate("else-leakyRelu", color="red"):
                attn = self.attn_act(attn)
        with nvtx.annotate("edge-softmax", color="blue"):
            with nvtx.annotate("edge-softmax", color="blue"):
                attn = edge_softmax(blk, attn)  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            with nvtx.annotate("dropout", color="blue"):
                attn = self.dropout(attn)
            with nvtx.annotate("reshape", color="blue"):
                out = torch.reshape(V * attn[:, :, None], (V.shape[0], -1))  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                del attn

            with nvtx.annotate("edge-reduce", color="blue"):
                out = edge_reduce(blk, out, op='sum') # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out = torch.cat([out, blk.dstdata['h']], dim=1)
        tt.t_self_attn += tt.elapsed(t_start)

        with nvtx.annotate("output", color="blue"):
            out = self.w_out(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            # torch.cuda.empty_cache()
            out = torch.nn.functional.relu(self.dropout(out))
            out = self.layer_norm(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        return out


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
        # if blk.num_edges() == 0:
        #     dev = blk.dstdata['h'].device
        #     out = torch.zeros(blk.num_dst(), self.dim_out, dtype=torch.float32, device=dev)
        #     out = torch.cat([out, blk.dstdata['h']], dim=1)
        # else:
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
    

class TemporalAttnLayer_2_perfCeil(torch.nn.Module):
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
                zero_time_feat = precomputed_zeros(self.ctx, blk.layer, self.time_encode, blk.num_dst())
                nbrs_time_feat = precomputed_times(self.ctx, blk.layer, self.time_encode, blk.time_deltas())
            
            with nvtx.annotate("redundancy-mul", color="red"): # TODO 这段的执行时间确实长
                Q = torch.cat([blk.dstdata['h'], zero_time_feat], dim=1)
                if self.dim_edge > 0:
                    Z = torch.cat([blk.srcdata['h'], blk.efeat(), nbrs_time_feat], dim=1)
                else:
                    Z = torch.cat([blk.srcdata['h'], nbrs_time_feat], dim=1)
                del zero_time_feat
                del nbrs_time_feat

                Q = self.w_q(Q)
                Z = self.w_kv(Z)
            
            with nvtx.annotate("else", color="red"): # else这段没有显存变化 torch的机制？
                with nvtx.annotate("else-K", color="red"):
                    K = Z[:, :self.dim_out]
                with nvtx.annotate("else-V", color="red"):
                    V = Z[:, self.dim_out:]
                    del Z
                    
                with nvtx.annotate("else-Q", color="red"):
                    Q = edge_view(blk, Q) # 对Q进行scatter
                with nvtx.annotate("else-reshape", color="red"):
                    Q = torch.reshape(Q, (Q.shape[0], self.num_heads, -1))
                    K = torch.reshape(K, (K.shape[0], self.num_heads, -1))
                    V = torch.reshape(V, (V.shape[0], self.num_heads, -1))

                with nvtx.annotate("else-attn", color="red"):
                    attn = torch.sum(Q * K, dim=2)
                    del Q
                    del K

                with nvtx.annotate("else-leakyRelu", color="red"):
                    attn = self.attn_act(attn)

            with nvtx.annotate("edge-softmax", color="blue"):
                with nvtx.annotate("edge-softmax", color="blue"):
                    attn = edge_softmax(blk, attn)  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                with nvtx.annotate("dropout", color="blue"):
                    attn = self.dropout(attn)
                with nvtx.annotate("reshape", color="blue"):
                    out = torch.reshape(V * attn[:, :, None], (V.shape[0], -1))  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                    del attn

                with nvtx.annotate("edge-reduce", color="blue"):
                    out = edge_reduce(blk, out, op='sum') # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                out = torch.cat([out, blk.dstdata['h']], dim=1)

            with nvtx.annotate("output", color="blue"):
                out = self.w_out(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                out = torch.nn.functional.relu(self.dropout(out))
                out = self.layer_norm(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        return out


class TemporalAttnLayer_0_fusion1_testblkm(torch.nn.Module): # our tmpAttnLayer
    def __init__(self, ctx: TContext, num_heads: int,
                 dim_node: int, dim_edge: int, dim_time: int, dim_out: int, layer: int,
                 dropout=0.1):
        super().__init__()
        assert (dim_out % num_heads == 0)
        self.ctx = ctx
        self.num_heads = num_heads
        self.dim_edge = dim_edge
        self.dim_out = dim_out
        self.dim_time = dim_time
        self.time_encode = TimeEncode(dim_time)
        # 拆分后前反向的性能开销 TODO
        self.w_q_node = torch.nn.Linear(dim_node, dim_out)
        self.w_q_time = torch.nn.Linear(dim_time, dim_out)
        self.w_kv_node = torch.nn.Linear(dim_node, dim_out * 2)
        self.w_kv_edge = torch.nn.Linear(dim_edge, dim_out * 2)
        self.w_kv_time = torch.nn.Linear(dim_time, dim_out * 2)
        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)
        self.w_out_dstData = torch.nn.Linear(dim_node, dim_out)
        self.w_out_edgerdc = torch.nn.Linear(dim_out, dim_out)
        self.attn_act = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.layer = layer

    @torch.compile
    def fusion_precompute(self, layer, _unique_time_delta):
        return precomputed_times(self.ctx, layer, self.time_encode, _unique_time_delta) # precompute是elementwise 理论上讲行不变

    @torch.compile
    def fusion_1(self, Q_node_idx, Q_node, Q_time, K_node, K_edge, K_time, Z_node_inverse, Z_edge_inverse, Z_time_inverse):
        Q = torch.index_select(Q_node, 0, Q_node_idx) + Q_time.expand(len(Q_node_idx), -1)
        K = torch.index_select(K_node, 0, Z_node_inverse) + torch.index_select(K_edge, 0, Z_edge_inverse) + torch.index_select(K_time, 0, Z_time_inverse)
        Q = Q.reshape(Q.shape[0], self.num_heads, -1)
        K = K.reshape(K.shape[0], self.num_heads, -1)
        attn = torch.sum(Q * K, dim=2)
        attn = self.attn_act(attn)
        return attn
    
    def fusion_2(self, num_dst, attn, V_node, Z_node_inverse, V_edge, Z_edge_inverse, V_time, Z_time_inverse, reduce_idx):
        out = FusedTGNFunction.apply(num_dst, attn, V_node, Z_node_inverse, V_edge, Z_edge_inverse, V_time, Z_time_inverse, reduce_idx)
        return out
    
    @torch.compile
    def fusion_3(self, out, nodeData_dst, node_dst_inverse):
        out_dstdata = self.w_out_dstData(nodeData_dst)
        out_edgerdc = self.w_out_edgerdc(out)
        out = torch.index_select(out_dstdata, 0, node_dst_inverse) + out_edgerdc
        out = torch.nn.functional.relu(self.dropout(out))
        out = self.layer_norm(out)
        return out

    @torch.compile
    def fusion_0(self, nfeat, mem, unique_dstnodes, unique_srcnodes):
        nodeData = nfeat + mem
        nodeData_dst = nodeData[unique_dstnodes]
        nodeData_src = nodeData[unique_srcnodes]
        return nodeData_dst, nodeData_src
    
    @torch.compile
    def fusion_0_1(self, nfeat, unique_dstnodes, unique_srcnodes):
        nodeData_dst = nfeat[unique_dstnodes]
        nodeData_src = nfeat[unique_srcnodes]
        return nodeData_dst, nodeData_src

    # ! 先以压缩为核心写kernel，即相同的只存储一次
    def forward(self, num_src, num_dst, reduce_idx, reindex, Q_node_idx, nfeat, mem, unique_dstnodes, node_dst_inverse, unique_srcnodes, node_src_inverse, efeat_unique, efeat_inverse, _unique_time_delta, time_inverse) -> Tensor:
        # TODO _g_dstindex 可以进一步预处理
        ## fusion0
        with nvtx.annotate("fusion_0", color="blue"):
            if self.layer == 0:
                nodeData_dst, nodeData_src = self.fusion_0(nfeat, mem, unique_dstnodes, unique_srcnodes)
                ## fusion0 消融
                # nodeData = nfeat + mem
                # nodeData_dst = nodeData[unique_dstnodes]
                # nodeData_src = nodeData[unique_srcnodes]
            else:
                assert self.layer == 1
                nodeData_dst, nodeData_src = self.fusion_0_1(nfeat, unique_dstnodes, unique_srcnodes)
        
        time_unique = self.fusion_precompute(self.layer, _unique_time_delta)

        # nodeData_src = nodeData_src.contiguous()
        # nodeData_dst = nodeData_dst.contiguous()
        # efeat_unique = efeat_unique.contiguous()
        # time_unique = time_unique.contiguous()
        # print(f"nodeData_dst {nodeData_dst.shape} {nodeData_dst.is_contiguous()}")
        # print(f"nodeData_src {nodeData_src.shape} {nodeData_src.is_contiguous()}")
        # print(f"efeat_unique {efeat_unique.shape} {efeat_unique.is_contiguous()}")
        # print(f"time_unique  {time_unique.shape} {time_unique.is_contiguous()}")
        
        with nvtx.annotate("Q_node", color="blue"):
            Q_node = self.w_q_node(nodeData_dst) # TODO 但是现在srcnode算的变多了，其实应该分开去重 -> 但矩阵乘的时间可以被pipeline掩盖/过于短的kernel对GPU而言也不友好，所以无所谓 -> 同一份nodeData 把两种W(三组W)拼在一起
        # print(f"Q_node {Q_node.shape}") # Q_node torch.Size([7018, 100])
        # print(f"Q_time {Q_time.shape}") # Q_time torch.Size([1, 100])

        with nvtx.annotate("Q_time", color="blue"):
            time_dst_unique = torch.ones([1, self.dim_time], dtype=torch.float, device="cuda") # 放弃更新bias
            Q_time = self.w_q_time(time_dst_unique)

        with nvtx.annotate("KV_node/edge/time", color="blue"):
            Z_node = self.w_kv_node(nodeData_src) # TODO
            Z_node_inverse = node_src_inverse
            K_node = Z_node[:, :self.dim_out]
            V_node = Z_node[:, self.dim_out:] # TODO 不一定要在这里把V算出来 显存换时间

            Z_edge = self.w_kv_edge(efeat_unique)
            Z_edge_inverse = efeat_inverse
            K_edge = Z_edge[:, :self.dim_out]
            V_edge = Z_edge[:, self.dim_out:]

            Z_time = self.w_kv_time(time_unique)
            Z_time_inverse = time_inverse
            K_time = Z_time[:, :self.dim_out]
            V_time = Z_time[:, self.dim_out:]
        # print(f"Z_node {Z_node.shape}") # Z_node torch.Size([1175, 200])
        # print(f"Z_node_inverse {Z_node_inverse.shape} {max(Z_node_inverse)}") # torch.Size([90180]) 1174
        # print(f"Z_edge {Z_edge.shape}") # Z_edge torch.Size([8168, 200])
        # print(f"Z_time {Z_time.shape}") # Z_time torch.Size([62723, 200])


        ### fusion1
        # A,B,C,D,E
        # Q_node, Q_time, K_node, K_edge, K_time
        # attn(x, y) 
        # K_node[Z_node_inverse[x], y*50:(y+1)*50] + K_edge[Z_edge_inverse[x], y*50:(y+1)*50] + K_time[Z_time_inverse[x], y*50:(y+1)*50] shape(50, 1)
        # Q_node[node_dst_inverse[x], y*50:(y+1)*50] + Q_time[0, y*50:(y+1)*50] shape(50, 1)
        # 一个thread_block shared_mem 164KB 并行度 41000*float 1*SM(持有的资源, sharedmem和thread)
        with nvtx.annotate("fusion_1", color="blue"):
            attn = self.fusion_1(Q_node_idx, Q_node, Q_time, K_node, K_edge, K_time, Z_node_inverse, Z_edge_inverse, Z_time_inverse)

        ### fusion2
        # forward(self, num_src, reindex, m, attn, unique_node, unique_node_idx, unique_edge, unique_edge_idx, unique_time, unique_time_idx, reduce_idx) -> Tensor:
        with nvtx.annotate("fusion_2", color="blue"):
            # print(f"attn {attn.shape} {attn.is_contiguous()}")
            # print(f"num_src {num_src}")
            # print(f"reindex {reindex.shape} {reindex.is_contiguous()}")
            # print(f"Z_node_inverse {Z_node_inverse.shape} {Z_node_inverse.is_contiguous()}")
            # print(f"Z_edge_inverse {Z_edge_inverse.shape} {Z_edge_inverse.is_contiguous()}")
            # print(f"Z_time_inverse {Z_time_inverse.shape} {Z_time_inverse.is_contiguous()}")
            # torch.cuda.synchronize()
            attn = torch_scatter.scatter_softmax(attn, reindex, dim=0, dim_size=num_src)
            Z_node_inverse = Z_node_inverse.int()
            Z_edge_inverse = Z_edge_inverse.int()
            Z_time_inverse = Z_time_inverse.int()
            # torch.cuda.synchronize()
            out = FusedTGNFunction.apply(num_dst, attn, V_node, Z_node_inverse, V_edge, Z_edge_inverse, V_time, Z_time_inverse, reduce_idx)
            # torch.cuda.synchronize()
        
        ## fusion2 消融
        # with nvtx.annotate("edge_softmax", color="blue"):
        #     attn = edge_softmax(blk, attn)  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        #     attn = self.dropout(attn)
        # with nvtx.annotate("out and edge_reduce", color="blue"):
        #     V = torch.index_select(V_node, 0, Z_node_inverse) + torch.index_select(V_edge, 0, Z_edge_inverse) + torch.index_select(V_time, 0, Z_time_inverse)
        #     V = torch.reshape(V, (V.shape[0], self.num_heads, -1))
        #     out = torch.reshape(V * attn[:, :, None], (V.shape[0], -1))  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        #     out = edge_reduce(blk, out, op='sum') # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):]
        
            
        ### fusion3 w_out 我还能再融一次
        with nvtx.annotate("fusion_3", color="blue"):
            out = self.fusion_3(out, nodeData_dst, node_dst_inverse)
        ## fusion3 消融
        # with nvtx.annotate("else", color="blue"):
        #     # blk.dstdata['h'] = torch.index_select(nodeData_dst, 0, node_dst_inverse)
        #     dstdata_h = nodeData_dst[node_dst_inverse]
        #     out = torch.cat([out, dstdata_h], dim=1)
        #     out = self.w_out(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        #     out = torch.nn.functional.relu(self.dropout(out))
        #     out = self.layer_norm(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        return out

class TemporalAttnLayer_1_fusion1_testblkm(torch.nn.Module): # our tmpAttnLayer
    def __init__(self, ctx: TContext, num_heads: int,
                 dim_node: int, dim_edge: int, dim_time: int, dim_out: int, layer: int,
                 dropout=0.1):
        super().__init__()
        assert (dim_out % num_heads == 0)
        self.ctx = ctx
        self.num_heads = num_heads
        self.dim_edge = dim_edge
        self.dim_out = dim_out
        self.dim_time = dim_time
        self.time_encode = TimeEncode(dim_time)
        # 拆分后前反向的性能开销 TODO
        self.w_q_node = torch.nn.Linear(dim_node, dim_out)
        self.w_q_time = torch.nn.Linear(dim_time, dim_out)
        self.w_kv_node = torch.nn.Linear(dim_node, dim_out * 2)
        self.w_kv_edge = torch.nn.Linear(dim_edge, dim_out * 2)
        self.w_kv_time = torch.nn.Linear(dim_time, dim_out * 2)
        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)
        self.w_out_dstData = torch.nn.Linear(dim_node, dim_out)
        self.w_out_edgerdc = torch.nn.Linear(dim_out, dim_out)
        self.attn_act = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.layer = layer

    @torch.compile()
    def fusion_precompute(self, layer, _unique_time_delta):
        return precomputed_times(self.ctx, layer, self.time_encode, _unique_time_delta) # precompute是elementwise 理论上讲行不变

    @torch.compile()
    def fusion_1(self, Q_node_idx, Q_node, Q_time, K_node, K_edge, K_time, Z_node_inverse, Z_edge_inverse, Z_time_inverse):
        # print(f"Q_node_idx {Q_node_idx.shape}")
        # print(f"Q_node {Q_node.shape}")
        # print(f"Q_time {Q_time.shape}")
        # print(f"K_node {K_node.shape}")
        # print(f"K_edge {K_edge.shape}")
        # print(f"K_time {K_time.shape}")
        # print(f"Z_node_inverse {Z_node_inverse.shape}")
        # print(f"Z_edge_inverse {Z_edge_inverse.shape}")
        # print(f"Z_time_inverse {Z_time_inverse.shape}")
        
        Q = torch.index_select(Q_node, 0, Q_node_idx) + Q_time.expand(len(Q_node_idx), -1)
        K = torch.index_select(K_node, 0, Z_node_inverse) + torch.index_select(K_edge, 0, Z_edge_inverse) + torch.index_select(K_time, 0, Z_time_inverse)
        Q = Q.reshape(Q.shape[0], self.num_heads, -1)
        K = K.reshape(K.shape[0], self.num_heads, -1)
        attn = torch.sum(Q * K, dim=2)
        attn = self.attn_act(attn)
        return attn
    
    def fusion_2(self, num_dst, attn, V_node, Z_node_inverse, V_edge, Z_edge_inverse, V_time, Z_time_inverse, reduce_idx):
        out = FusedTGNFunction.apply(num_dst, attn, V_node, Z_node_inverse, V_edge, Z_edge_inverse, V_time, Z_time_inverse, reduce_idx)
        return out
    
    @torch.compile()
    def fusion_3(self, out, nodeData_dst, node_dst_inverse):
        out_dstdata = self.w_out_dstData(nodeData_dst)
        out_edgerdc = self.w_out_edgerdc(out)
        out = torch.index_select(out_dstdata, 0, node_dst_inverse) + out_edgerdc
        out = torch.nn.functional.relu(self.dropout(out))
        out = self.layer_norm(out)
        return out

    @torch.compile()
    def fusion_0(self, nfeat, mem, unique_dstnodes, unique_srcnodes):
        nodeData = nfeat + mem
        nodeData_dst = nodeData[unique_dstnodes]
        nodeData_src = nodeData[unique_srcnodes]
        return nodeData_dst, nodeData_src
    
    @torch.compile()
    def fusion_0_1(self, output, b2_inv_idx, unique_dstnodes, unique_srcnodes):
        nodeData_dst = output[b2_inv_idx][unique_dstnodes]
        nodeData_src = output[b2_inv_idx][unique_srcnodes]
        return nodeData_dst, nodeData_src

    # ! 先以压缩为核心写kernel，即相同的只存储一次
    def forward(self, num_src, num_dst, reduce_idx, reindex, Q_node_idx, output, b2_inv_idx, unique_dstnodes, node_dst_inverse, unique_srcnodes, node_src_inverse, efeat_unique, efeat_inverse, _unique_time_delta, time_inverse) -> Tensor:
        # TODO _g_dstindex 可以进一步预处理
        ## fusion0
        with nvtx.annotate("fusion_0", color="blue"):
            nodeData_dst, nodeData_src = self.fusion_0_1(output, b2_inv_idx, unique_dstnodes, unique_srcnodes)
            # ## fusion0 消融
            # nodeData_dst = output[b2_inv_idx][unique_dstnodes]
            # nodeData_src = output[b2_inv_idx][unique_srcnodes]

        with nvtx.annotate("Q_node", color="blue"):
            Q_node = self.w_q_node(nodeData_dst) # TODO 但是现在srcnode算的变多了，其实应该分开去重 -> 但矩阵乘的时间可以被pipeline掩盖/过于短的kernel对GPU而言也不友好，所以无所谓 -> 同一份nodeData 把两种W(三组W)拼在一起
        # print(f"Q_node {Q_node.shape}") # Q_node torch.Size([7018, 100])
        # print(f"Q_time {Q_time.shape}") # Q_time torch.Size([1, 100])

        with nvtx.annotate("Q_time", color="blue"):
            time_dst_unique = torch.ones([1, self.dim_time], dtype=torch.float, device="cuda") # 放弃更新bias
            Q_time = self.w_q_time(time_dst_unique)


        with nvtx.annotate("KV_node/edge/time", color="blue"):
            Z_node = self.w_kv_node(nodeData_src)
            Z_node_inverse = node_src_inverse
            K_node = Z_node[:, :self.dim_out]
            V_node = Z_node[:, self.dim_out:] # TODO 不一定要在这里把V算出来 显存换时间

            Z_edge = self.w_kv_edge(efeat_unique)
            Z_edge_inverse = efeat_inverse
            K_edge = Z_edge[:, :self.dim_out]
            V_edge = Z_edge[:, self.dim_out:]

            time_unique = self.fusion_precompute(self.layer, _unique_time_delta)
            Z_time = self.w_kv_time(time_unique)
            Z_time_inverse = time_inverse
            K_time = Z_time[:, :self.dim_out]
            V_time = Z_time[:, self.dim_out:]
        # print(f"Z_node {Z_node.shape}") # Z_node torch.Size([1175, 200])
        # print(f"Z_node_inverse {Z_node_inverse.shape} {max(Z_node_inverse)}") # torch.Size([90180]) 1174
        # print(f"Z_edge {Z_edge.shape}") # Z_edge torch.Size([8168, 200])
        # print(f"Z_time {Z_time.shape}") # Z_time torch.Size([62723, 200])


        ### fusion1
        # A,B,C,D,E
        # Q_node, Q_time, K_node, K_edge, K_time
        # attn(x, y) 
        # K_node[Z_node_inverse[x], y*50:(y+1)*50] + K_edge[Z_edge_inverse[x], y*50:(y+1)*50] + K_time[Z_time_inverse[x], y*50:(y+1)*50] shape(50, 1)
        # Q_node[node_dst_inverse[x], y*50:(y+1)*50] + Q_time[0, y*50:(y+1)*50] shape(50, 1)
        # 一个thread_block shared_mem 164KB 并行度 41000*float 1*SM(持有的资源, sharedmem和thread)
        with nvtx.annotate("fusion_1", color="blue"):
            attn = self.fusion_1(Q_node_idx, Q_node, Q_time, K_node, K_edge, K_time, Z_node_inverse, Z_edge_inverse, Z_time_inverse)


        ### fusion2
        # forward(self, num_src, reindex, m, attn, unique_node, unique_node_idx, unique_edge, unique_edge_idx, unique_time, unique_time_idx, reduce_idx) -> Tensor:
        with nvtx.annotate("fusion_2", color="blue"):
            attn = torch_scatter.scatter_softmax(attn, reindex, dim=0, dim_size=num_src)
            Z_node_inverse = Z_node_inverse.int()
            Z_edge_inverse = Z_edge_inverse.int()
            Z_time_inverse = Z_time_inverse.int()
            out = FusedTGNFunction.apply(num_dst, attn, V_node, Z_node_inverse, V_edge, Z_edge_inverse, V_time, Z_time_inverse, reduce_idx)
        ## fusion2 消融
        # with nvtx.annotate("edge_softmax", color="blue"):
        #     attn = edge_softmax(blk, attn)  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        #     attn = self.dropout(attn)
        # with nvtx.annotate("out and edge_reduce", color="blue"):
        #     V = torch.index_select(V_node, 0, Z_node_inverse) + torch.index_select(V_edge, 0, Z_edge_inverse) + torch.index_select(V_time, 0, Z_time_inverse)
        #     V = torch.reshape(V, (V.shape[0], self.num_heads, -1))
        #     out = torch.reshape(V * attn[:, :, None], (V.shape[0], -1))  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        #     out = edge_reduce(blk, out, op='sum') # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):]
        
            
        ### fusion3 w_out 我还能再融一次
        with nvtx.annotate("fusion_3", color="blue"):
            out = self.fusion_3(out, nodeData_dst, node_dst_inverse)
        ## fusion3 消融
        # with nvtx.annotate("else", color="blue"):
        #     # blk.dstdata['h'] = torch.index_select(nodeData_dst, 0, node_dst_inverse)
        #     dstdata_h = nodeData_dst[node_dst_inverse]
        #     out = torch.cat([out, dstdata_h], dim=1)
        #     out = self.w_out(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        #     out = torch.nn.functional.relu(self.dropout(out))
        #     out = self.layer_norm(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        return out

class TemporalAttnLayer0_2_perfCeil(torch.nn.Module):
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
        # 拆分后前反向的性能开销 TODO
        self.w_q = torch.nn.Linear(dim_node + dim_time, dim_out)
        self.w_kv = torch.nn.Linear(dim_node + dim_edge + dim_time, dim_out * 2)
        '''
        self.w_kv_node = self.w_kv[dim_node, :] # TypeError: 'Linear' object is not subscriptable
        self.w_kv_edge = self.w_kv[dim_node : dim_edge, :]
        self.w_kv_time = self.w_kv[dim_node + dim_edge :, :]
        print(f"self.w_kv_node {self.w_kv_node}")
        print(f"self.w_kv_edge {self.w_kv_edge}")
        print(f"self.w_kv_time {self.w_kv_time}")'''
        self.w_q_node = torch.nn.Linear(dim_node, dim_out)
        self.w_q_time = torch.nn.Linear(dim_time, dim_out)
        self.w_kv_node = torch.nn.Linear(dim_node, dim_out * 2)
        self.w_qkv_node = torch.nn.Linear(dim_node, dim_out * 3)
        self.w_kv_edge = torch.nn.Linear(dim_edge, dim_out * 2)
        self.w_kv_time = torch.nn.Linear(dim_time, dim_out * 2)
        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)
        self.attn_act = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
    
 
    # @torch.compile(dynamic=True)
    def compute_z_ours(self, idx, nodeData, node_inverse, node_dst_inverse, efeat_unique, efeat_inverse, time_unique, time_inverse, time_dst_unique, time_dst_inverse):
        # print(f"Q= {Q.shape}")
        # print(f"idx= {idx.shape}")
        # print(f"node_unique= {node_unique.shape}")
        # print(f"node_inverse= {node_inverse.shape}")
        # print(f"efeat_unique= {efeat_unique.shape}")
        # print(f"efeat_inverse= {efeat_inverse.shape}")
        # print(f"time_unique= {time_unique.shape}")
        # print(f"time_inverse= {time_inverse.shape}")
        # QKV_node = self.w_qkv_node(nodeData)

        Q_node = self.w_q_node(nodeData)
        # Q_node = QKV_node[:, :self.dim_out]
        Q_time = self.w_q_time(time_dst_unique)

        Q_node = Q_node[node_dst_inverse]
        Q_time = Q_time[time_dst_inverse]
        Q_our = Q_node + Q_time
        
        Q = Q_our[idx]


        Z_node = self.w_kv_node(nodeData)
        # Z_node = QKV_node[:, self.dim_out:]
        Z_edge = self.w_kv_edge(efeat_unique)
        Z_time = self.w_kv_time(time_unique)
        
        Z_node = Z_node[node_inverse]
        Z_edge = Z_edge[efeat_inverse]
        Z_time = Z_time[time_inverse]
        Z_our = Z_node + Z_edge + Z_time

        K = Z_our[:, :self.dim_out]
        V = Z_our[:, self.dim_out:]
        

        assert Q.shape[0] == K.shape[0]

        Q = torch.reshape(Q, (Q.shape[0], self.num_heads, -1))
        K = torch.reshape(K, (K.shape[0], self.num_heads, -1))
        V = torch.reshape(V, (V.shape[0], self.num_heads, -1))

        attn = torch.sum(Q * K, dim=2)
        attn = self.attn_act(attn)
        return attn, V

    def compute_z_ours_nvtx(self, blk: TBlock, zero_time_feat, nbrs_time_feat):
        with nvtx.annotate("redundancy-mul", color="red"): # TODO 这段的执行时间确实长
            # concat < mul
            with nvtx.annotate("redundancy-concat: Q", color="red"):
                Q = torch.cat([blk.dstdata['h'], zero_time_feat], dim=1)
            with nvtx.annotate("redundancy-concat: KV", color="red"): # TODO 这段的执行时间很意外 torch.cat 原理
                Z = torch.cat([blk.srcdata['h'], blk.efeat(), nbrs_time_feat], dim=1)

            del zero_time_feat
            # del nbrs_time_feat

            with nvtx.annotate("redundancy-mul: Q", color="red"):
                Q = self.w_q(Q)
            with nvtx.annotate("redundancy-mul: KV", color="red"): # TODO 这段的执行时间确实长
                Z = self.w_kv(Z) # 消融一下
            

            with torch.no_grad():
                node_unique, node_inverse = torch.unique(blk.srcdata['h'], dim=0, return_inverse=True)
                efeat_unique, efeat_inverse = torch.unique(blk.efeat(), dim=0, return_inverse=True) # 172的长度可能乘起来不够高效
                time_unique, time_inverse = torch.unique(nbrs_time_feat, dim=0, return_inverse=True)
            # print(f"node_unique {node_unique.shape} efeat_unique {efeat_unique.shape} time_unique {time_unique.shape}")
            
            with nvtx.annotate("redundancy-our: KV cal", color="red"): # TODO 这段的执行时间很意外 torch.cat 原理
                # concat > mul

                # Z_node = self.w_kv_node(blk.srcdata['h'])
                # Z_edge = self.w_kv_edge(blk.efeat())
                # Z_time = self.w_kv_time(nbrs_time_feat)
                # [DOING] 怎么把unique提前拿到
                with nvtx.annotate("redundancy-our: KV cal node", color="red"): # TODO 减少计算真的能减少执行时间
                    Z_node = self.w_kv_node(node_unique)
                with nvtx.annotate("redundancy-our: KV cal edge", color="red"):
                    Z_edge = self.w_kv_edge(efeat_unique)
                with nvtx.annotate("redundancy-our: KV cal time", color="red"):
                    Z_time = self.w_kv_time(time_unique)
            with nvtx.annotate("redundancy-our: KV reverse", color="red"): 
                with nvtx.annotate("redundancy-our: KV reverse node", color="red"):
                    Z_node = Z_node[node_inverse]
                with nvtx.annotate("redundancy-our: KV reverse edge", color="red"):
                    Z_edge = Z_edge[efeat_inverse]
                with nvtx.annotate("redundancy-our: KV reverse time", color="red"):
                    Z_time = Z_time[time_inverse]
            with nvtx.annotate("redundancy-our: KV add", color="red"): # TODO 这段的执行时间很意外 torch.cat 原理(index_elementwise + catArrayBatchedCopy)
                # TODO 怎么把unique提前拿到; element-wise
                Z_our = Z_node + Z_edge + Z_time

            # print(f"Z {Z.shape}")
            # print(f"Z {Z.shape} Z_our {Z_our.shape}")
            # print(f"Z = {Z}")
            # print(f"Z_our = {Z_our}")
            
            # assert Z.shape == Z_our.shape
                
            del nbrs_time_feat
        

        with nvtx.annotate("else", color="red"): # else这段没有显存变化 torch的机制？
            with nvtx.annotate("else-K", color="red"):
                K = Z_our[:, :self.dim_out]
            with nvtx.annotate("else-V", color="red"):
                V = Z_our[:, self.dim_out:]
                del Z
                
            with nvtx.annotate("else-Q", color="red"):
                # Q = edge_view(blk, Q) # 对Q进行scatter
                idx = torch.from_numpy(blk._dstindex)
                idx = idx.to(device=Q.device, dtype=torch.long)
                Q = Q[idx]
            with nvtx.annotate("else-reshape", color="red"):
                Q = torch.reshape(Q, (Q.shape[0], self.num_heads, -1))
                K = torch.reshape(K, (K.shape[0], self.num_heads, -1))
                V = torch.reshape(V, (V.shape[0], self.num_heads, -1))

            with nvtx.annotate("else-attn", color="red"):
                attn = torch.sum(Q * K, dim=2)
                del Q
                del K
        return attn, V
    
    # @torch.compile TODO
    # input: tail, nodeData, _reverse_nids
    def forward_redundancy_mul(self, blk: TBlock, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta): # TODO build pipeline
        # [DOING] 先拆分计算；纵切只要传入对应的indices范围即可
        # 处理bottleneck 的KV 部分
        with nvtx.annotate("forward_redundancy_mul", color="blue"):
            with nvtx.annotate("precompute", color="blue"):
                # no scatter
                # zero_time_feat = precomputed_zeros(self.ctx, blk.layer, self.time_encode, blk.num_dst()) # TODO 行全0 优化
                time_dst_unique = precomputed_zeros(self.ctx, blk.layer, self.time_encode, 1) 
                time_dst_inverse = torch.zeros(blk.num_dst(), dtype=torch.int64, device="cuda")
                # nbrs_time_feat_st = precomputed_times(self.ctx, blk.layer, self.time_encode, blk.time_deltas()) # TODO
                nbrs_time_feat = precomputed_times(self.ctx, blk.layer, self.time_encode, _unique_time_delta.to("cuda")) # precompute是elementwise 理论上讲行不变

            # Q = edge_view(blk, Q) # 对Q进行scatter
            idx = torch.from_numpy(blk._dstindex)
            idx = idx.to(device="cuda", dtype=torch.long)

            '''
            # 给具体可能的值 TODO
            # with torch.no_grad():
                # node_unique, node_inverse = torch.unique(blk.srcdata['h'], dim=0, return_inverse=True)
                # efeat_unique, efeat_inverse = torch.unique(blk.efeat(), dim=0, return_inverse=True) # 172的长度可能乘起来不够高效
                # time_unique, time_inverse = torch.unique(nbrs_time_feat_st, dim=0, return_inverse=True)
                # print(f"nbrs_time_feat_st.shape {nbrs_time_feat_st.shape}")
                # print(f"nbrs_time_feat_st = {nbrs_time_feat_st}")
                # print(f"nbrs_time_feat[_reverse_time_delta].shape {nbrs_time_feat[_reverse_time_delta].shape}")
                # print(f"nbrs_time_feat[_reverse_time_delta] = {nbrs_time_feat[_reverse_time_delta]}")
                # time_d_unique, time_d_inverse = torch.unique(blk.time_deltas(), dim=0, return_inverse=True)
                # print(f"_unique_time_delta.shape {_unique_time_delta[_reverse_time_delta].shape}")
                # print(f"_unique_time_delta[_reverse_time_delta] = {_unique_time_delta[_reverse_time_delta]}")
                # print(f"blk.time_deltas().shape {blk.time_deltas().shape}")
                # print(f"blk.time_deltas() = {blk.time_deltas()}")
                # print(f"_reverse_time_delta.shape {_reverse_time_delta.shape}")
                # print(f"_reverse_time_delta = {_reverse_time_delta}")
                # print(f"time_inverse.shape {time_inverse.shape}")
                # print(f"time_inverse = {time_inverse}")
                # print(f"nbrs_time_feat.shape {nbrs_time_feat.shape}")
                # print(f"nbrs_time_feat = {nbrs_time_feat}")
                # print(f"time_unique.shape {time_unique.shape}")
                # print(f"time_unique = {time_unique}")
                # assert torch.allclose(_unique_time_delta[_reverse_time_delta].to("cuda"), blk.time_deltas())
                # assert torch.allclose(nbrs_time_feat[_reverse_time_delta], nbrs_time_feat_st) # \O/ acc test passed!
                # print(f"time_unique {time_unique.shape}")
                # print(f"time_inverse = {time_inverse}")
                # print(f"time_delta_unique {time_d_unique.shape}")
                # print(f"time_delta_inverse = {time_d_inverse}")

                # node_dst_unique, node_dst_inverse = torch.unique(blk.dstdata['h'], dim=0, return_inverse=True)
                ## time_dst_unique_st, time_dst_inverse_st = torch.unique(zero_time_feat, dim=0, return_inverse=True) # 优先处理zero time feat
                ## assert torch.allclose(time_dst_unique, time_dst_unique_st)
                ## assert torch.allclose(time_dst_inverse, time_dst_inverse_st) # \O/ acc test passed!

                ## print("===") # TODO dst的冗余也有很多
                ## print(f"blk.dstdata['h'] {blk.dstdata['h'].shape}")
                ## print(f"node_dst_unique {node_dst_unique.shape}")
                ## print()
            '''

        # TODO 但是现在srcnode算的变多了，其实应该分开去重 -> 但矩阵乘的时间可以被pipeline掩盖/过于短的kernel对GPU而言也不友好，所以无所谓 -> 同一份nodeData 把两种W(三组W)拼在一起
        return self.ctx.compiled_forward_redundancy_mul(idx, nodeData, _reverse_nids[blk.num_dst():], _reverse_nids[:blk.num_dst()], efeat_unique, _reverse_eids, nbrs_time_feat, _reverse_time_delta, time_dst_unique, time_dst_inverse)
        ## no scatter
        # return self.ctx.compiled_forward_redundancy_mul(idx, node_unique, node_inverse, efeat_unique, efeat_inverse, time_unique, time_inverse, node_dst_unique, node_dst_inverse, time_dst_unique, time_dst_inverse)
        ## no-bug @torch.compile
        # return self.compute_z_ours(blk, zero_time_feat, nbrs_time_feat, idx) # TODO
        ## with nvtx
        # return self.compute_z_ours_nvtx(blk, zero_time_feat, nbrs_time_feat)
        
    # input: nodeData, _reverse_nids
    def forward(self, blk: TBlock, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta) -> Tensor:
        attn, V = self.forward_redundancy_mul(blk, nodeData, _reverse_nids, efeat_unique, _reverse_eids, _unique_time_delta, _reverse_time_delta)

        with nvtx.annotate("edge-softmax", color="blue"):
            with nvtx.annotate("edge-softmax", color="blue"):
                attn = edge_softmax(blk, attn)  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            with nvtx.annotate("dropout", color="blue"):
                attn = self.dropout(attn)
            with nvtx.annotate("reshape", color="blue"):
                out = torch.reshape(V * attn[:, :, None], (V.shape[0], -1))  # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                del attn

            with nvtx.annotate("edge-reduce", color="blue"):
                out = edge_reduce(blk, out, op='sum') # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):]
            # tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
            blk.dstdata['h'] = nodeData[_reverse_nids[:blk.num_dst()]]
            out = torch.cat([out, blk.dstdata['h']], dim=1)

        with nvtx.annotate("output", color="blue"):
            out = self.w_out(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out = torch.nn.functional.relu(self.dropout(out))
            out = self.layer_norm(out) # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        return out