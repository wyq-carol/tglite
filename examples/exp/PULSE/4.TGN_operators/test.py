
import argparse
import time
import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
import GPUtil
import scipy.sparse as sp
import tglite as tg
from tglite.mymodule import *
import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Callable
import sys
import os
from tglite.nn import TemporalAttnLayer as OriginalLayer
from tglite._context import TContext
from tglite._sampler import TSampler
from tglite.op import precomputed_zeros, precomputed_times
from tglite import _c
# from tglite.nn import TemporalAttnLayer, TemporalAttnLayer0_2_perfCeil, TemporalAttnLayer_2_perfCeil, TemporalAttnLayer_precompute, TemporalAttnLayer_0_fusion1_testblkm, TemporalAttnLayer_1_fusion1_testblkm
from tglite.nn import TemporalAttnLayer_0_fusion1_testblkm as OptimizedLayer
from torch import Tensor
from tglite._block import TBlock

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# from examples.compute_z_ours.python.nn_optimized import TemporalAttnLayer as OptimizedLayer
# from examples.compute_z_ours.python.nn_ablation1 import TemporalAttnLayer as Ablation1Layer
# from examples.compute_z_ours.python.nn_ablation2 import TemporalAttnLayer as Ablation2Layer

class MockGraph:
    def __init__(self, device):
        self._device = device
        
    def compute_device(self):
        return self._device
    def device(self):
        return self._device
    def storage_device(self):
        return self._device

class MockTContext:
    def __init__(self, training=True, time_enabled=False, time_window=1000, device='cuda'):
        self._training = training
        self._time_enabled = time_enabled
        self._time_window = time_window
        self._g = MockGraph(device)
        self._time_tables = {}
        self._z = None

class MockBlock:
    """模拟TBlock的数据结构"""
    def __init__(self, index, num_src: int, num_dst: int, num_edges: int, device: torch.device):
        self._g = MockGraph(device)
        self._g_dstindex = index
        self._num_src = num_src
        self._num_dst = num_dst
        self._num_edges = num_edges
        self._device = device
        self._layer = 0
        
        # 初始化数据
        # 根据tmp.log数据，srcdata和dstdata的h特征维度为100
        self._srcdata = {'h': torch.randn(num_src, 100, device=device)}
        self._dstdata = {'h': torch.randn(num_dst, 100, device=device)}
        
        # 根据tmp.log数据，efeat维度为172
        self._efeat = torch.randn(num_edges, 172, device=device) if num_edges > 0 else None
        
        # 根据tmp.log数据，time_deltas是一维的
        self._time_deltas = torch.randn(num_edges, device=device) if num_edges > 0 else None
    def num_src(self) -> int:
        return self._num_src
    def num_dst(self) -> int:
        return self._num_dst
    def num_edges(self) -> int:
        return self._num_edges
    @property
    def srcdata(self):
        return self._srcdata
    @property
    def dstdata(self):
        return self._dstdata
    def efeat(self) -> torch.Tensor:
        return self._efeat
    def time_deltas(self) -> torch.Tensor:
        return self._time_deltas
        
    @property
    def layer(self) -> int:
        return self._layer

def test_precomputed_zeros():
    # 测试场景1: 训练模式
    ctx = MockTContext(training=True)
    id = 0
    num = 5
    def encoder(is_zero: bool, ts: torch.Tensor) -> torch.Tensor:
        if is_zero:
            return torch.zeros(num, 10, device=ctx._g.compute_device())
        else:
            return torch.randn(num, 10, device=ctx._g.compute_device())
            
    # result1 = precomputed_zeros(ctx, 0, encoder, 5)
    # def precomputed_zeros(ctx: TContext, id: int, encoder: Callable, num: int) -> Tensor:
    cdev = ctx._g.compute_device()
    if ctx._training or not ctx._time_enabled:
        # return encoder(True, None)
        return encoder(True, torch.zeros(num, dtype=torch.float, device=cdev))
        
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
        time_table = encoder(False, torch.arange(
            ctx._time_window + 1, dtype=torch.float, device=cdev))
        ctx._time_tables[id] = time_table

    output = time_table[0].repeat(num, 1)
    result1 = output.view(num, -1)
    print("训练模式结果形状:", result1.shape)
    
    # 测试场景2: 推理模式
    ctx = MockTContext(training=False)
    # result2 = precomputed_zeros(ctx2, 0, encoder, 5)
    # def precomputed_zeros(ctx: TContext, id: int, encoder: Callable, num: int) -> Tensor:
    cdev = ctx._g.compute_device()
    if ctx._training or not ctx._time_enabled:
        # return encoder(True, None)
        return encoder(True, torch.zeros(num, dtype=torch.float, device=cdev))
        
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
        time_table = encoder(False, torch.arange(
            ctx._time_window + 1, dtype=torch.float, device=cdev))
        ctx._time_tables[id] = time_table

    output = time_table[0].repeat(num, 1)
    result2 = output.view(num, -1)
    print("推理模式结果形状:", result2.shape)

def test_precomputed_times():
    # 测试场景1: 训练模式
    ctx = MockTContext(training=True)
    def encoder(is_zero, times):
        if is_zero:
            return torch.zeros(len(times), 10, device=ctx._g.compute_device())
        else:
            return torch.randn(len(times), 10, device=ctx._g.compute_device())
    times = torch.tensor([1.0, 2.0, 3.0], device=ctx._g.compute_device())
    id = 0
    # result1 = precomputed_times(ctx1, 0, encoder, times)
    # def precomputed_times(ctx: TContext, id: int, encoder: Callable, times: Tensor) -> Tensor:
    with nvtx.annotate("precompute_times", color="red"):
        if ctx._training or not ctx._time_enabled:
            return encoder(False, times)
    
        time_table = ctx._time_tables.get(id)
        if time_table is None:
            time_table = encoder(torch.arange(
                ctx._time_window + 1, dtype=torch.float, device=ctx._g.compute_device()))
            ctx._time_tables[id] = time_table

        size = times.shape[0]
        hit_count, hit_idx, output, times, inv_idx = \
            _c.find_dedup_time_hits(times, time_table, ctx._time_window)
        uniq_size = times.shape[0]

        if hit_count != uniq_size:
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            miss_idx = (~ hit_idx)
            times = times[miss_idx]
            output[miss_idx] = encoder(times.squeeze())

        output = output[inv_idx]
        result1 = output.view(size, -1)
    print("训练模式结果形状:", result1.shape)
    
    # 测试场景2: 推理模式
    ctx = MockTContext(training=False)
    # result2 = precomputed_times(ctx2, 0, encoder, times)
    # def precomputed_times(ctx: TContext, id: int, encoder: Callable, times: Tensor) -> Tensor:
    with nvtx.annotate("precompute_times", color="red"):
        if ctx._training or not ctx._time_enabled:
            return encoder(False, times)
    
        time_table = ctx._time_tables.get(id)
        if time_table is None:
            time_table = encoder(torch.arange(
                ctx._time_window + 1, dtype=torch.float, device=ctx._g.compute_device()))
            ctx._time_tables[id] = time_table

        size = times.shape[0]
        hit_count, hit_idx, output, times, inv_idx = \
            _c.find_dedup_time_hits(times, time_table, ctx._time_window)
        uniq_size = times.shape[0]

        if hit_count != uniq_size:
            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
            miss_idx = (~ hit_idx)
            times = times[miss_idx]
            output[miss_idx] = encoder(times.squeeze())

        output = output[inv_idx]
        result2 = output.view(size, -1)
    print("推理模式结果形状:", result2.shape)

class PerformanceMetrics:
    def __init__(self):
        self.train_times: List[float] = []
        self.inference_times: List[float] = []
        self.memory_usage: List[float] = []
        
    def add_metrics(self, train_time: float, inference_time: float, memory_usage: float):
        self.train_times.append(train_time)
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_usage)
        
    def get_average_metrics(self) -> Tuple[float, float, float]:
        return (
            np.mean(self.train_times),
            np.mean(self.inference_times),
            np.mean(self.memory_usage)
        )

def generate_test_data(
    _g_dstindex: torch.Tensor,
    dstdata_h: torch.Tensor,
    srcdata_h: torch.Tensor,
    efeat: torch.Tensor,
    time_deltas: torch.Tensor,
    num_nodes: int,
    num_dsts: int,
    num_edges: int, # num_srcs
    dim_node: int,
    dim_edge: int,
    dim_time: int,
    device: torch.device,
    inputs
) -> Dict[str, torch.Tensor]:
    """生成测试数据"""
    # 创建测试数据
    blk = MockBlock(
        index=_g_dstindex,
        num_src=num_edges,  # 源节点数
        num_dst=num_dsts,  # 目标节点数
        num_edges=num_edges,  # 边数
        device=device
    )
    
    # 创建Context
    ctx = MockTContext(device=device)

    with torch.no_grad():
        print("here")
        nodeData, _reverse_nids = torch.unique(torch.cat((srcdata_h, dstdata_h), dim=0), dim=0, return_inverse=True)
        nodeData_src, _reverse_src_nids = torch.unique(srcdata_h, dim=0, return_inverse=True)
        nodeData_dst, _reverse_dst_nids = torch.unique(dstdata_h, dim=0, return_inverse=True)
        # print(f"nodeData_src {nodeData_src.shape}") # nodeData_src torch.Size([1175, 100])
        # print(f"nodeData_dst {nodeData_dst.shape}") nodeData_dst torch.Size([7018, 100])
        efeat_unique, _reverse_eids = torch.unique(efeat, dim=0, return_inverse=True)
        _unique_time_delta, _reverse_time_delta = torch.unique(time_deltas, dim=0, return_inverse=True)

        x = torch.arange(srcdata_h.shape[0])
        Q_node_idx = _reverse_dst_nids[blk._g_dstindex[x]]
        # print(f"Q_node_idx {Q_node_idx.shape} {Q_node_idx}") # TODO 可以用CSR那种方式存储?

    return {
        'blk': blk,
        'ctx': ctx,
        'nodeData': nodeData,
        'reverse_nids': _reverse_nids,
        'nodeData_src': nodeData_src, 
        'reverse_src_nids': _reverse_src_nids,
        'nodeData_dst': nodeData_dst, 
        'reverse_dst_nids': _reverse_dst_nids,
        'efeat_unique': efeat_unique,
        'reverse_eids': _reverse_eids,
        'unique_time_delta': _unique_time_delta,
        'reverse_time_delta': _reverse_time_delta,
        'Q_node_idx': Q_node_idx,
        'inputs': inputs
    }

def measure_memory_usage() -> float:
    """测量当前GPU内存使用情况"""
    if torch.cuda.is_available():
        # 同步所有CUDA流
        torch.cuda.synchronize()
        # 获取当前内存使用
        current_memory = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
        # 获取峰值内存使用
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # 转换为MB
        return max(current_memory, peak_memory)
    return 0.0

def get_layer_input(layer, test_data):
    """根据层类型返回对应的输入参数"""
    if isinstance(layer, (OriginalLayer, )):
        return (test_data['blk'],)  # 返回一个单元素元组
    elif isinstance(layer, (OptimizedLayer, )):
        # return (test_data['blk'], *test_data['inputs'])
        return test_data['inputs']
    # def forward(self, num_src, num_dst, reduce_idx, reindex, Q_node_idx, nfeat, mem, unique_dstnodes, node_dst_inverse, unique_srcnodes, Z_node_inverse, efeat_unique, Z_edge_inverse, _unique_time_delta, Z_time_inverse) -> Tensor:
    else:  # GATConv
        return (test_data['row_ptr'], test_data['col_idx'], test_data['col_ptr'], 
                test_data['row_idx'], test_data['permute'], test_data['features'])

def test_layer_performance(layer, test_data, num_epochs=30, device='cuda'):
    """测试单个层的性能"""
    # 重置GPU内存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 将层移动到指定设备
    layer = layer.to(device)
    
    # 准备优化器和损失函数
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    # 获取输入参数
    input_args = get_layer_input(layer, test_data)

    # 预热
    layer.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        output = layer(*input_args)
        loss = criterion(output, torch.randn_like(output))
        loss.backward()
        optimizer.step()

    import threading
    import time
    import matplotlib.pyplot as plt

    # 显存采样配置
    sampled_alloc = []
    sampled_reserved = []
    sampled_phase = []
    sampled_time = []
    sampling = True
    interval = 0.00001  # 0.01ms 采样间隔

    def memory_sampler():
        while sampling:
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            sampled_alloc.append(alloc)
            sampled_reserved.append(reserved)
            sampled_phase.append(current_phase)
            sampled_time.append(time.time() - mask_start_time)
            time.sleep(interval)

    # 执行一次训练迭代并采样
    layer.train()
    sampling = True
    sampled_alloc.clear()
    sampled_reserved.clear()
    sampled_phase.clear()
    sampled_time.clear()
    current_phase = 'idle'

    mask_start_time = time.time()  # 放到 start 前面！
    sampler_thread = threading.Thread(target=memory_sampler)
    sampler_thread.start()

    optimizer.zero_grad()

    # 1. forward
    current_phase = 'forward'
    torch.cuda.synchronize()
    output = layer(*input_args)
    torch.cuda.synchronize()

    # 2. backward
    current_phase = 'backward'
    torch.cuda.synchronize()
    loss = criterion(output, torch.randn_like(output))
    loss.backward()
    torch.cuda.synchronize()

    # 3. optimizer step
    current_phase = 'step'
    torch.cuda.synchronize()
    optimizer.step()
    torch.cuda.synchronize()

    # 停止采样
    # current_phase = 'idle'
    sampling = False
    sampler_thread.join()

    # 绘图
    # from matplotlib.font_manager import _rebuild
    # _rebuild()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20  # 设置全局字体大小为14，你可以根据需要调整这个值

    colors = {
        # 'idle': 'gray',
        'forward': 'blue',
        'backward': 'red',
        'step': 'green'
    }

    # print(f"sampled_time {sampled_time}")
    x = np.array(sampled_time)
    alloc = np.array(sampled_alloc)
    reserved = np.array(sampled_reserved)

    plt.figure(figsize=(12, 6))
    for phase in ['forward', 'backward', 'step']:
        mask = np.array([p == phase for p in sampled_phase])
        plt.plot(x[mask], alloc[mask], label=f'{phase} (alloc)', color=colors[phase], linestyle='-')
        plt.plot(x[mask], reserved[mask], label=f'{phase} (reserved)', color=colors[phase], linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('GPU Memory Usage (MB)')
    plt.title('GPU Memory Usage During One TGN Layer')
    plt.legend(loc='center right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tmp.png")
    plt.show()
    # # 显存采样配置
    # mask_start_time = 0
    # sampled_alloc = []
    # sampled_reserved = []
    # sampled_phase = []
    # sampled_time = []
    # sampling = True
    # interval = 0.001  # 1ms 采样间隔
    # def memory_sampler():
    #     while sampling:
    #         torch.cuda.synchronize()
    #         alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    #         reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    #         sampled_alloc.append(alloc)
    #         sampled_reserved.append(reserved)
    #         sampled_phase.append(current_phase)
    #         sampled_time.append(time.time() - mask_start_time)
    #         time.sleep(interval)

    # # 执行一次训练迭代并采样
    # layer.train()
    # sampling = True
    # sampled_alloc.clear()
    # sampled_reserved.clear()
    # sampled_phase.clear()
    # current_phase = 'idle'
    # sampler_thread = threading.Thread(target=memory_sampler)
    # sampler_thread.start()

    # # Start training iteration
    # optimizer.zero_grad()
    # mask_start_time = time.time()
    # # 1. forward
    # current_phase = 'forward'
    # torch.cuda.synchronize()
    # output = layer(*input_args)
    # torch.cuda.synchronize()

    # # 2. backward
    # current_phase = 'backward'
    # torch.cuda.synchronize()
    # loss = criterion(output, torch.randn_like(output))
    # loss.backward()
    # torch.cuda.synchronize()

    # # 3. optimizer step
    # current_phase = 'step'
    # torch.cuda.synchronize()
    # optimizer.step()
    # torch.cuda.synchronize()

    # # Stop sampling
    # current_phase = 'idle'
    # sampling = False
    # sampler_thread.join()

    # import numpy as np

    # colors = {
    #     'idle': 'gray',
    #     'forward': 'blue',
    #     'backward': 'red',
    #     'step': 'green'
    # }
    # x = np.arange(len(sampled_alloc))
    # alloc = np.array(sampled_alloc)
    # reserved = np.array(sampled_reserved)

    # plt.figure(figsize=(12, 6))
    # for phase in ['forward', 'backward', 'step', 'idle']:
    #     count = sampled_phase.count(phase)
    #     print(f"{phase}: {count} samples")

    # for phase in ['forward', 'backward', 'step', 'idle']:
    #     mask = np.array([p == phase for p in sampled_phase])
    #     x = np.array(sampled_time)
    #     plt.plot(x[mask], alloc[mask], label=f'{phase} (alloc)', color=colors[phase], linestyle='-')
    #     plt.plot(x[mask], reserved[mask], label=f'{phase} (reserved)', color=colors[phase], linestyle='--')

    # plt.xlabel('Sample Index')
    # plt.ylabel('GPU Memory Usage (MB)')
    # plt.title('GPU Memory Usage During One Training Iteration')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("tmp.png")

    
    

    # 测试训练时间
    torch.cuda.synchronize()
    start_time = time.time()
    
    layer.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        with nvtx.annotate("forward", color="red"):
            output = layer(*input_args)
        with nvtx.annotate("backward", color="red"):
            loss = criterion(output, torch.randn_like(output))
            loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    train_time = (time.time() - start_time) / num_epochs
    print(f"train_time {train_time}")
    
    # # 预热
    # layer.eval()
    # with torch.no_grad():
    #     for _ in range(num_epochs):
    #         output = layer(*input_args)

    # # 测试推理时间
    # layer.eval()
    # torch.cuda.synchronize()
    # start_time = time.time()
    
    # with torch.no_grad():
    #     for _ in range(num_epochs):
    #         output = layer(*input_args)
    
    # torch.cuda.synchronize()
    # inference_time = (time.time() - start_time) / num_epochs
    
    # 测量内存使用
    memory_usage = measure_memory_usage()
    
    return train_time, 1, memory_usage

def run_ablation_study(test_data, num_epochs=10, device='cuda'):
    """运行消融实验"""
    results = {}
    
    # 创建所有层
    # ! HERE
    layers = {
        # 'optimized': OptimizedLayer(
        #     ctx=test_data['ctx'],
        #     dim_node=100,
        #     dim_edge=172,
        #     dim_time=100,
        #     dim_out=100,
        #     layer=0,
        #     num_heads=2,
        #     dropout=0.1
        # ),
        'original': OriginalLayer(
            ctx=test_data['ctx'],
            dim_node=100,
            dim_edge=172,
            dim_time=100,
            dim_out=100,
            num_heads=2,
            dropout=0.1
        ),
    }
    
    # 测试每个层
    for name, layer in layers.items():
        print(f"\n测试 {name} 层...")
        with nvtx.annotate(f"test_layer_{name}", color="red"):
            train_time, inference_time, memory_usage = test_layer_performance(
                layer, test_data, num_epochs, device
            )
        results[name] = {
            'train_time': train_time,
            'inference_time': inference_time,
            'memory_usage': memory_usage
        }
        
        # 清理GPU内存
        del layer
        torch.cuda.empty_cache()
    
    return results

def print_results(results: Dict[str, Dict[str, float]]):
    """打印实验结果"""
    print("\n=== 性能对比结果 ===")
    print("版本\t\t训练时间(s)\t推理时间(s)\t内存使用(MB)\t训练加速比\t推理加速比")
    print("-" * 70)
    
    # 使用原始版本作为基准
    base_metrics = results['original']
    
    for name, metrics in results.items():
        if name != 'warmup':
            train_time = metrics['train_time']
            inference_time = metrics['inference_time']
            memory_usage = metrics['memory_usage']
            train_speedup = base_metrics['train_time'] / train_time
            inference_speedup = base_metrics['inference_time'] / inference_time
            
            print(f"{name:<12} {train_time:>10.4f} {inference_time:>10.4f} "
                f"{memory_usage:>10.2f} {train_speedup:>10.2f}x {inference_speedup:>10.2f}x")

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 生成测试数据
    print("\n生成测试数据...")

    # _g_dstindex=torch.load("blk._g_dstindex.pt")
    # dstdata_h=torch.load("blk.dstdata_h.pt")
    # srcdata_h=torch.load("blk.srcdata_h.pt")
    # efeat=torch.load("blk.efeat.pt")
    # time_deltas=torch.load("blk.time_deltas.pt")
    # print(f"_g_dstindex {_g_dstindex.shape}")
    # print(f"dstdata_h {dstdata_h.shape}")
    # print(f"srcdata_h {srcdata_h.shape}")
    # print(f"efeat {efeat.shape}")
    # print(f"time_deltas {time_deltas.shape}")

    # ! HERE
    inputs = torch.load("wiki-talk-bs6000-layer1/input.pt")
    num_src, num_dst, reduce_idx, reindex, Q_node_idx, nfeat, mem, unique_dstnodes, node_dst_inverse, unique_srcnodes, Z_node_inverse, efeat_unique, Z_edge_inverse, _unique_time_delta, Z_time_inverse = inputs
    # 假设 Q_node_idx 和 inverse_dstnodes 已经定义
    # 初始化 b_dstindex
    b_dstindex = torch.zeros_like(Q_node_idx)

    # 遍历 Q_node_idx，找到对应的 b_dstindex
    for i, q_idx in enumerate(Q_node_idx):
        # 找到 inverse_dstnodes 中值为 q_idx 的索引
        b_dstindex[i] = torch.where(node_dst_inverse == q_idx)[0][0]
    assert torch.all(node_dst_inverse[b_dstindex] == Q_node_idx)
    # print(f"====")
    # print(f"_g_dstindex {b_dstindex.shape}")
    # print(f"dstdata_h {torch.rand(num_dst, 100, dtype=torch.float32).shape}")
    # print(f"srcdata_h {torch.rand(num_src, 100, dtype=torch.float32).shape}")
    # print(f"efeat {efeat_unique[Z_edge_inverse].shape}")
    # print(f"time_deltas {_unique_time_delta[Z_time_inverse].shape}")

    # test_data = generate_test_data(
    #     _g_dstindex=torch.load("blk._g_dstindex.pt"),
    #     dstdata_h=torch.load("blk.dstdata_h.pt"),
    #     srcdata_h=torch.load("blk.srcdata_h.pt"),
    #     efeat=torch.load("blk.efeat.pt"),
    #     time_deltas=torch.load("blk.time_deltas.pt"),
    #     num_nodes=90180,
    #     num_dsts=16350,
    #     num_edges=90180,
    #     dim_node=100,
    #     dim_edge=172,
    #     dim_time=100,
    #     device=device,
    #     inputs=inputs
    # )

    test_data = generate_test_data(
        _g_dstindex=torch.zeros_like(Q_node_idx, device="cuda"),
        dstdata_h=torch.rand(num_dst, 100, dtype=torch.float32, device="cuda"),
        srcdata_h=torch.rand(num_src, 100, dtype=torch.float32, device="cuda"),
        efeat=efeat_unique[Z_edge_inverse],
        time_deltas=_unique_time_delta[Z_time_inverse],
        num_nodes=num_src,
        num_dsts=num_dst,
        num_edges=num_src,
        dim_node=100,
        dim_edge=172,
        dim_time=100,
        device=device,
        inputs=inputs
    )
    
    # 运行消融实验
    print("\n开始性能测试...")
    results = run_ablation_study(test_data, num_epochs=10, device=device)
    
    # 打印结果
    print_results(results)

        
    # # 测试 precomputed_zeros
    # print("\n测试 precomputed_zeros:")
    # test_precomputed_zeros()
    # print("\n测试 precomputed_times:")
    # test_precomputed_times()


if __name__ == "__main__":
    main()

