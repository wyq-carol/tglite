import torch
import os
import tglite.config
import subprocess
import math
import numpy as np
from tqdm import tqdm

def remove_duplicate_eids(dstindex, eids):
    """
    对指向相同 dstindex 的 eid 进行去重，返回需要保留的索引
    :param dstindex: 目标索引列表
    :param eids: 事件 ID 列表
    :return: 需要保留的索引列表
    """
    # 存储需要保留的索引
    keep_indices = []
    # 用于记录每个 dstindex 对应的 eids
    dstindex_eids = {}

    for i in range(len(dstindex)):
        index = dstindex[i]
        eid = eids[i]
        if index not in dstindex_eids:
            # 如果 dstindex 还未记录，则记录该 eid 并保留当前索引
            dstindex_eids[index] = eid
            keep_indices.append(i)
        elif dstindex_eids[index] != eid:
            # 如果 dstindex 已记录且当前 eid 不同，则保留当前索引
            keep_indices.append(i)

    return keep_indices

import torch

def get_common_and_non_common_eids(_unique_eids, prev_eids, device):
    _unique_eids = torch.tensor(_unique_eids, device=device)
    prev_eids = torch.tensor(prev_eids, device=device)

    # 计算公共 eid 及其索引
    mask_common = torch.isin(_unique_eids, prev_eids)
    common_eids_tensor = _unique_eids[mask_common]
    common_eid_indices_tensor = torch.nonzero(mask_common).flatten()

    # 计算非公共 eid 及其索引
    mask_non_common = ~mask_common
    non_common_eids_tensor = _unique_eids[mask_non_common]
    non_common_eid_indices_tensor = torch.nonzero(mask_non_common).flatten()

    # 计算公共 eid 在上一个 batch 中的索引
    common_eid_prev_indices = []
    for eid in common_eids_tensor.cpu().tolist():
        prev_index = (prev_eids == eid).nonzero(as_tuple=True)[0].item()
        common_eid_prev_indices.append(prev_index)
    # common_eid_prev_indices_tensor = torch.tensor(common_eid_prev_indices, device=device)

    return common_eids_tensor, common_eid_indices_tensor, non_common_eids_tensor, non_common_eid_indices_tensor
    # return _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu

import torch

def get_common_and_non_common_eids2(_unique_eids, prev_eids, device):
    # 将输入转换为 PyTorch 张量
    _unique_eids = torch.tensor(_unique_eids, device=device)
    prev_eids = torch.tensor(prev_eids, device=device)

    # 使用 torch.isin 函数找出公共元素的掩码
    common_mask = torch.isin(_unique_eids, prev_eids)

    # 根据掩码提取公共元素
    common_eids_tensor = _unique_eids[common_mask]

    # 找出公共元素在 _unique_eids 中的索引
    common_eid_indices_tensor = torch.nonzero(common_mask, as_tuple=True)[0]

    return common_eids_tensor, common_eid_indices_tensor
    # return _nids_nxt, _idx_nids_nxt



if __name__ == "__main__":
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 解析samples 获得两个mini-batch 之间重叠信息
    # 2. 替换原有采样逻辑，直接变线下这一套(后续对动态计算有想法再说)
    # 3. 根据重叠信息管理数据placement，和数据使用即后续pipeline（根据数据传输时间，计算时间，数据的非欧性 进行pipeline 划分）
    log_dir0 = f"/home/volume/tglake_res"
    log_dir = f"/home/volume/tglake_res/tglake_res_tmp"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for i in range(len(tglite.config.DATA)):
        DATA = tglite.config.DATA[i]
        BATCH_SIZE = tglite.config.BATCH_SIZE[i]
        N_LAYERS = tglite.config.N_LAYERS[i]
        N_NBRS = tglite.config.N_NBRS[i]
        N_HEADS = tglite.config.N_HEADS[i]
        log_name = f"DATA_{DATA}_BS_{BATCH_SIZE}_NLAYER_{N_LAYERS}_NBR_{N_NBRS}_NHEAD_{N_HEADS}"

        file_path = os.path.join(log_dir0, f"samples_{log_name}.pt")
        # # for quick test
        # file_path = os.path.join(log_dir0, f"test_new_samples.pt")

        samples = torch.load(file_path)
        #            ((tail._dstnodes, tail._dstindex, tail._srcnodes, tail._eid, tail._ets))
        # add_samples((inv_idx, tail._dstnodes, tail._dsttimes, tail._dstindex, tail._srcnodes, tail._eid, tail._ets))

        # 每个blk 的sample (TODO 目前只考虑单层)
        # # 去重后的_dstnodes 与_srcnodes 的对应关系不是顺序的
        _inv_idx = [inv_idx for inv_idx, _, _, _, _, _, _ in samples]
        _dstnodes = [_dstnode for _, _dstnode, _, _, _, _, _ in samples]
        # # self._dsttimes 为目标节点采样时间rept*3(include_neg)，去冗余后才执行采样，即为_dstnodes 的绝对位置
        # # 指的是位置(outnodes, sizes()) 所以不一定从0开始，使用时往往采用reindex 用于edge-centric 相关的计算
        # # self._dsttimes         使用直接索引 dts = self._dsttimes[self._dstindex] 对着原有的_dsttimes 索引(表示nid 在dstnodes 中的绝对位置)
        # # edge_softmax / scatter 使用相对位置 reindex = torch.unique(blk._g_dstindex, return_inverse=True)[1]
        _dsttimes = [dsttime for _, _, dsttime, _, _, _, _ in samples]
        _dstindex = [dstindex for _, _, _, dstindex, _, _, _ in samples]
        _srcnodes = [_srcnode for _, _, _, _, _srcnode, _, _ in samples]
        _eids = [_eid for _, _, _, _, _, _eid, _ in samples]
        _ets = [_et for _, _, _, _, _, _, _et in samples]

        new_samples = []
        node_load_percents = []
        dstnodes_load_percents = []
        srcnodes_load_percents = []
        edge_load_percents = []
        time_nbrs_load_percents = []
        # for b_i in range(len(_dstnodes)):
        for b_i in tqdm(range(len(_dstnodes)), desc="任务进度", unit="项"):
            # get batch blk items
            b_inv_idx = torch.tensor(_inv_idx[b_i]).to(device)
            b_dstnodes = torch.tensor(_dstnodes[b_i]).to(device)
            b_dsttimes = torch.tensor(_dsttimes[b_i]).to(device)
            b_dstindex = torch.tensor(_dstindex[b_i]).to(device)
            b_srcnodes = torch.tensor(_srcnodes[b_i]).to(device)
            b_eids = torch.tensor(_eids[b_i]).to(device)
            b_ets = torch.tensor(_ets[b_i]).to(device)

            b_dsttimes_scatter = b_dsttimes[b_dstindex]
            # dts - ets
            time_delta = b_dsttimes_scatter - b_ets
            _unique_time_delta, _reverse_time_delta = torch.unique(time_delta, return_inverse=True)

            # print(f"b_dstnodes {b_dstnodes.size()}") # (dstnodes, dsttimes) 是unique的
            # print(f"b_dsttimes {b_dsttimes.size()}")
            # print(f"b_dstindex {b_dstindex.size()}")
            # print(f"b_srcnodes {b_srcnodes.size()}")
            # print(f"b_eids {b_eids.size()}")
            # print(f"b_ets {b_ets.size()}")

            # 2.2. 为每个mini batch 的数据结构增加 eids 整体去重的_unique_eids, _reverse_eids ;
            ## _unique_eids 2 b_eids(新一轮去重后)
            _unique_eids, _reverse_eids = torch.unique(b_eids, return_inverse=True)

            all_nodes = torch.cat([b_dstnodes, b_srcnodes])
            
            _unique_nids, _reverse_nids = torch.unique(all_nodes, return_inverse=True)
            _unique_dst_nodes, _reverse_dst_nodes = torch.unique(_reverse_nids[:len(b_dstnodes)], return_inverse=True)
            _unique_src_nodes, _reverse_src_nodes = torch.unique(_reverse_nids[len(b_dstnodes):], return_inverse=True)
            x = torch.arange(len(b_srcnodes)).to("cuda")
            Q_node_idx = _reverse_dst_nodes[b_dstindex[x]]
            reindex = torch.unique(b_dstindex, return_inverse=True)[1]

            _unique_ets, _reverse_ets = torch.unique(b_ets, return_inverse=True)

            # 2.3. 为每个mini batch 的数据结构增加 与上一个/下一个batch 重叠的 _eids_pre, _eids_nxt, _nids_pre, _nids_nxt
            if b_i == 0:
                prev_eids = torch.tensor([], device=device)
                prev_nodes = torch.tensor([], device=device)

                _eids_pre = torch.tensor([], device=device)
                _nids_pre = torch.tensor([], device=device)

                _idx_eids_pre = torch.tensor([]).to(device)
                _idx_nids_pre = torch.tensor([]).to(device)

                _eids_cpu = _unique_eids
                _idx_eids_cpu = torch.arange(len(_unique_eids))
                _nids_cpu = _unique_nids
                _idx_nids_cpu = torch.arange(len(_unique_nids))
            else:
                prev_eids = torch.unique(torch.tensor(_eids[b_i - 1]).to(device))
                _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu = get_common_and_non_common_eids(_unique_eids, prev_eids, device)

                prev_nodes = torch.unique(torch.cat([torch.tensor(_dstnodes[b_i - 1]).to(device), torch.tensor(_srcnodes[b_i - 1]).to(device)]))
                # import pdb;pdb.set_trace()
                _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu = get_common_and_non_common_eids(_unique_nids, prev_nodes, device)

            if b_i == len(_dstnodes) - 1:
                _eids_nxt = torch.tensor([], device=device)
                _nids_nxt = torch.tensor([], device=device)

                next_eids = torch.tensor([]).to(device)
                next_nodes = torch.tensor([]).to(device)

                _idx_eids_nxt = torch.tensor([]).to(device)
                _idx_nids_nxt = torch.tensor([]).to(device)

            else:
                next_eids = torch.unique(torch.tensor(_eids[b_i + 1]).to(device))
                _eids_nxt, _idx_eids_nxt = get_common_and_non_common_eids2(_unique_eids, next_eids, device)

                next_nodes = torch.unique(torch.cat([torch.tensor(_dstnodes[b_i + 1]).to(device), torch.tensor(_srcnodes[b_i + 1]).to(device)]))
                _nids_nxt, _idx_nids_nxt = get_common_and_non_common_eids2(_unique_nids, next_nodes, device)

            edge_load_percents.append((_unique_eids.shape[0]) * 100 / _reverse_eids.shape[0])
            node_load_percents.append((_unique_nids.shape[0]) * 100 / _reverse_nids.shape[0])
            dstnodes_load_percents.append((_unique_dst_nodes.shape[0]) * 100 / _reverse_dst_nodes.shape[0])
            srcnodes_load_percents.append((_unique_src_nodes.shape[0]) * 100 / _reverse_src_nodes.shape[0])
            time_nbrs_load_percents.append((_unique_ets.shape[0]) * 100 / _reverse_ets.shape[0])

            '''
            print(f"_unique_eids.shape {_unique_eids.shape[0]}")
            print(f"_reverse_eids.shape {_reverse_eids.shape[0]}")
            print(f"_unique_nids.shape {_unique_nids.shape[0]}")
            print(f"_reverse_nids.shape {_reverse_nids.shape[0]}")
            print(f"_unique_dst_nodes.shape {_unique_dst_nodes.shape[0]}")
            print(f"_reverse_dst_nodes.shape {_reverse_dst_nodes.shape[0]}")
            print(f"_unique_src_nodes.shape {_unique_src_nodes.shape[0]}")
            print(f"_reverse_src_nodes.shape {_reverse_src_nodes.shape[0]}")
            print(f"_unique_ets.shape {_unique_ets.shape[0]}")
            print(f"_reverse_ets.shape {_reverse_ets.shape[0]}")

            print(f"b_inv_idx {len(b_inv_idx)}")
            print(f"b_dstnodes {len(b_dstnodes)}")
            print(f"b_dsttimes {len(b_dsttimes)}")
            print(f"b_dstindex {len(b_dstindex)}")
            print(f"b_srcnodes {len(b_srcnodes)}")
            print(f"b_eids {len(b_eids)}")
            print(f"b_ets {len(b_ets)}")
            print(f"_unique_eids {len(_unique_eids)}")
            print(f"_reverse_eids {len(_reverse_eids)}")
            print(f"_unique_nids {len(_unique_nids)}")
            print(f"_reverse_nids {len(_reverse_nids)}")
            print(f"_unique_ets {len(_unique_ets)}")
            print(f"_reverse_ets {len(_reverse_ets)}")
            print(f"_eids_pre {len(_eids_pre)}")
            print(f"_nids_pre {len(_nids_pre)}")
            print(f"_eids_nxt {len(_eids_nxt)}")
            print(f"_nids_nxt {len(_nids_nxt)}")
            '''
            '''
            print(f"b_i {b_i}")
            print(f"***")
            print(f"prev_eids {len(prev_eids)}")
            print(f"prev_nodes {len(prev_nodes)}")
            print(f"b_dstnodes {len(b_dstnodes)}")
            print(f"b_srcnodes {len(b_srcnodes)}")
            print(f"b_eids {len(b_eids)}")
            print(f"next_eids {len(next_eids)}")
            print(f"next_nodes {len(next_nodes)}")
            print(f"***")
            print(f"_eids_pre {_eids_pre}")
            print(f"_idx_eids_pre {_idx_eids_pre}")
            print(f"_eids_cpu {_eids_cpu}")
            print(f"_idx_eids_cpu {_idx_eids_cpu}")
            print(f"_eids_nxt {_eids_nxt}")
            print(f"_idx_eids_nxt {_idx_eids_nxt}")
            print(f"***")
            print(f"_nids_pre {_nids_pre}")
            print(f"_idx_nids_pre {_idx_nids_pre}")
            print(f"_nids_cpu {_nids_cpu}")
            print(f"_idx_nids_cpu {_idx_nids_cpu}")
            print(f"_nids_nxt {_nids_nxt}")
            print(f"_idx_nids_nxt {_idx_nids_nxt}")
            print()
            assert len(_idx_eids_pre)+len(_idx_eids_cpu) == len(_unique_eids)
            assert len(_idx_nids_pre)+len(_idx_nids_cpu) == len(_unique_nids)
            '''
            new_sample = (
                b_inv_idx,
                b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta,
                prev_eids, next_eids,
                prev_nodes, next_nodes,
                _unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets,
                _unique_dst_nodes, _reverse_dst_nodes, _unique_src_nodes, _reverse_src_nodes,
                Q_node_idx, reindex,
                _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu,
                _eids_nxt, _idx_eids_nxt,
                _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu,
                _nids_nxt, _idx_nids_nxt
            )
            new_samples.append(new_sample)

        # motiv_wiki-talk_bs-6000: 对dstnodes, srcnodes, edges, time_nbrs的冗余度进行统计
        print(f"dstnodes 冗余度 max {max(dstnodes_load_percents):.2f}; min {min(dstnodes_load_percents):.2f}; avg {sum(dstnodes_load_percents) / len(dstnodes_load_percents):.2f}")
        print(f"srcnodes 冗余度 max {max(srcnodes_load_percents):.2f}; min {min(srcnodes_load_percents):.2f}; avg {sum(srcnodes_load_percents) / len(srcnodes_load_percents):.2f}")
        print(f"edges 冗余度 max {max(edge_load_percents):.2f}; min {min(edge_load_percents):.2f}; avg {sum(edge_load_percents) / len(edge_load_percents):.2f}")
        print(f"time_nbrs 冗余度 max {max(time_nbrs_load_percents):.2f}; min {min(time_nbrs_load_percents):.2f}; avg {sum(time_nbrs_load_percents) / len(time_nbrs_load_percents):.2f}")
        print(f"node 冗余度 max {max(node_load_percents):.2f}; min {min(node_load_percents):.2f}; avg {sum(node_load_percents) / len(node_load_percents):.2f}")

        # 将 new_samples 中的张量移回 CPU 再保存
        new_samples_cpu = []
        for sample in new_samples:
            new_sample_cpu = tuple(tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor for tensor in sample)
            new_samples_cpu.append(new_sample_cpu)

        # 你可以在这里对 new_samples 进行进一步处理，比如保存到文件
        torch.save(new_samples_cpu, os.path.join(log_dir, f"new_samples_{log_name}.pt"))