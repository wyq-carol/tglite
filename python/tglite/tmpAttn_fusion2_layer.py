from dgNN.src.tgn_kernel_fuse.test import FusedTGNFunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.linear import Identity
from tglite._block import TBlock
from tglite._context import TContext
from tglite.nn import TimeEncode
from tglite.op import edge_softmax, edge_reduce
import nvtx
import torch_scatter

class TemporalAttnLayer_FUSION2Opt(torch.nn.Module): # our tmpAttnLayer
    def __init__(self, ctx: TContext, num_heads: int,
                 dim_node: int, dim_edge: int, dim_time: int, dim_out: int,
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
        self.w_qkv_node = torch.nn.Linear(dim_node, dim_out * 3)
        self.w_kv_edge = torch.nn.Linear(dim_edge, dim_out * 2)
        self.w_kv_time = torch.nn.Linear(dim_time, dim_out * 2)
        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)
        self.attn_act = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, num_src, reindex, m, attn, unique_node, unique_node_idx, unique_edge, unique_edge_idx, unique_time, unique_time_idx, reduce_idx) -> Tensor:
        with nvtx.annotate("scatter_softmax", color="red"):
            attn = torch_scatter.scatter_softmax(attn, reindex, dim=0, dim_size=num_src)
        print(f"m {m}")
        print(f"attn {attn.shape}")
        print(f"unique_node {unique_node.shape}")
        print(f"unique_node_idx {unique_node_idx.shape}")
        print(f"unique_edge {unique_edge.shape}")
        print(f"unique_edge_idx {unique_edge_idx.shape}")
        print(f"unique_time {unique_time.shape}")
        print(f"unique_time_idx {unique_time_idx.shape}")
        print(f"reduce_idx {reduce_idx.shape}")
        output = FusedTGNFunction.apply(m, attn, unique_node, unique_node_idx, unique_edge, unique_edge_idx, unique_time, unique_time_idx, reduce_idx)
        # print(f"FUSION2Opt-out {output}")
        return output