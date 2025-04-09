import random
import time
import os
import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Callable, Optional, Tuple, Union

import tglite as tg
from tglite._stats import tt
from tglite.gpu_mem_track import *
import nvtx
import tglite.config
import threading
import pycuda.driver as cuda
import pycuda.autoinit

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_device(gpu: int) -> torch.device:
    return torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')


def make_model_path(model: str, prefix: str, data: str) -> str:
    """If prefix is not empty, return 'models/{model}/{prefix}-{data}.pt', else return
    'models/{model}/{data}-{time.time()}.pt'."""
    Path(f'models/{model}').mkdir(parents=True, exist_ok=True)
    if prefix:
        return f'models/{model}/{prefix}-{data}.pt'
    else:
        return f'models/{model}/{data}-{time.time()}.pt'


def make_model_mem_path(model: str, prefix: str, data: str) -> str:
    Path(f'models/{model}').mkdir(parents=True, exist_ok=True)
    if prefix:
        return f'models/{model}/{prefix}-{data}-mem.pt'
    else:
        return f'models/{model}/{data}-mem-{time.time()}.pt'


def load_graph(path: Union[str, Path]) -> tg.TGraph:
    """Create a TGraph with edges and timestamps loaded from path. Provided data should include
    'src' 'dst' and 'time' columns."""
    df = pd.read_csv(str(path))

    src = df['src'].to_numpy().astype(np.int32).reshape(-1, 1)
    dst = df['dst'].to_numpy().astype(np.int32).reshape(-1, 1)
    etime = df['time'].to_numpy().astype(np.float32)
    del df

    edges = np.concatenate([src, dst], axis=1)
    del src
    del dst

    g = tg.TGraph(edges, etime)
    print('num edges:', g.num_edges())
    print('num nodes:', g.num_nodes())
    return g

def load_feats(g: tg.TGraph, device, d: str, data_path: str=''):
    """
    Load edge features and node features to g from /home/volume/{d}/edge_features.pt and
    /home/volume/{d}/edge_features.pt. If no file, create random edge and node features for data 'mooc',
    'lastfm' and 'wiki-talk', create random edge features for data 'wiki' and 'reddit', None for
    other data.
    """
    if tglite.config.ON_HETER:
        load_feats0(g, device, d, data_path)
    elif tglite.config.ALL_ON_GPU:
        assert tglite.config.ON_HETER == 0
        load_feats_all_on_gpu(g, device, d, data_path)
    else:
        assert tglite.config.ON_HETER == 0 and tglite.config.ALL_ON_GPU == 0
        load_feats_origin(g, device, d, data_path)

def load_feats0(g: tg.TGraph, device, d: str, data_path: str=''):
    """
    Load edge features and node features to g from /home/volume/{d}/edge_features.pt and
    /home/volume/{d}/edge_features.pt. If no file, create random edge and node features for data 'mooc',
    'lastfm' and 'wiki-talk', create random edge features for data 'wiki' and 'reddit', None for
    other data.
    """
    edge_feats = None
    node_feats = None

    if Path(os.path.join(data_path, f'/home/volume/{d}/edge_features.pt')).exists():
        edge_feats = torch.load(os.path.join(data_path, f'/home/volume/{d}/edge_features.pt'))
        edge_feats = edge_feats.type(torch.float32)
    elif d in ['mooc', 'lastfm']:
        edge_feats = torch.randn(g.num_edges(), 128, dtype=torch.float32)
    elif d in ['wiki-talk', 'stackoverflow']:
        edge_feats = torch.randn(g.num_edges(), 172, dtype=torch.float32)

    if Path(os.path.join(data_path, f'/home/volume/{d}/node_features.pt')).exists():
        node_feats = torch.load(os.path.join(data_path, f'/home/volume/{d}/node_features.pt'))
        node_feats = node_feats.type(torch.float32)
    elif d in ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk', 'stackoverflow']:
        node_feats = torch.randn(g.num_nodes(), edge_feats.shape[1], dtype=torch.float32)

    print('edge feat:', None if edge_feats is None else edge_feats.shape)
    print('node feat:', None if node_feats is None else node_feats.shape)
    # WYQ TODO
    g.efeat = edge_feats.to(device)
    g.nfeat = node_feats.to(device)

def load_feats_all_on_gpu(g: tg.TGraph, device, d: str, data_path: str=''):
    """
    Load edge features and node features to g from /home/volume/{d}/edge_features.pt and
    /home/volume/{d}/edge_features.pt. If no file, create random edge and node features for data 'mooc',
    'lastfm' and 'wiki-talk', create random edge features for data 'wiki' and 'reddit', None for
    other data.
    """
    edge_feats = None
    node_feats = None

    if Path(os.path.join(data_path, f'/home/volume/{d}/edge_features.pt')).exists():
        edge_feats = torch.load(os.path.join(data_path, f'/home/volume/{d}/edge_features.pt'))
        edge_feats = edge_feats.type(torch.float32)
    elif d in ['mooc', 'lastfm']:
        edge_feats = torch.randn(g.num_edges(), 128, dtype=torch.float32)
    elif d in ['wiki-talk', 'stackoverflow']:
        edge_feats = torch.randn(g.num_edges(), 172, dtype=torch.float32)

    if Path(os.path.join(data_path, f'/home/volume/{d}/node_features.pt')).exists():
        node_feats = torch.load(os.path.join(data_path, f'/home/volume/{d}/node_features.pt'))
        node_feats = node_feats.type(torch.float32)
    elif d in ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk', 'stackoverflow']:
        node_feats = torch.randn(g.num_nodes(), edge_feats.shape[1], dtype=torch.float32)

    print('edge feat:', None if edge_feats is None else edge_feats.shape)
    print('node feat:', None if node_feats is None else node_feats.shape)
    g.efeat = edge_feats.to(device)
    g.nfeat = node_feats.to(device)

    if device != torch.device("cpu"):
        g._g_efeat = edge_feats.to(device)
        g._g_nfeat = node_feats.to(device)

def load_feats_origin(g: tg.TGraph, device, d: str, data_path: str=''):
    """
    Load edge features and node features to g from /home/volume/{d}/edge_features.pt and
    /home/volume/{d}/edge_features.pt. If no file, create random edge and node features for data 'mooc',
    'lastfm' and 'wiki-talk', create random edge features for data 'wiki' and 'reddit', None for
    other data.
    """
    edge_feats = None
    node_feats = None

    if Path(os.path.join(data_path, f'/home/volume/{d}/edge_features.pt')).exists():
        edge_feats = torch.load(os.path.join(data_path, f'/home/volume/{d}/edge_features.pt'))
        edge_feats = edge_feats.type(torch.float32)
    elif d in ['mooc', 'lastfm']:
        edge_feats = torch.randn(g.num_edges(), 128, dtype=torch.float32)
    elif d in ['wiki-talk', 'stackoverflow']:
        edge_feats = torch.randn(g.num_edges(), 172, dtype=torch.float32)

    if Path(os.path.join(data_path, f'/home/volume/{d}/node_features.pt')).exists():
        node_feats = torch.load(os.path.join(data_path, f'/home/volume/{d}/node_features.pt'))
        node_feats = node_feats.type(torch.float32)
    elif d in ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk', 'stackoverflow']:
        node_feats = torch.randn(g.num_nodes(), edge_feats.shape[1], dtype=torch.float32)

    print('edge feat:', None if edge_feats is None else edge_feats.shape)
    print('node feat:', None if node_feats is None else node_feats.shape)
    g.efeat = edge_feats
    g.nfeat = node_feats

def data_split(num_samples: int, train_percent: float, val_percent: float) -> Tuple[int, int]:
    train_end = int(np.ceil(num_samples * train_percent))
    val_end = int(np.ceil(num_samples * (train_percent + val_percent)))
    return train_end, val_end


class EdgePredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.src_fc = nn.Linear(dim, dim)
        self.dst_fc = nn.Linear(dim, dim)
        self.out_fc = nn.Linear(dim, 1)
        self.act = nn.ReLU()

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        h_src = self.src_fc(src)
        # torch.cuda.empty_cache() # empty cache开销很大(会长60s+)
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        h_dst = self.dst_fc(dst)
        # torch.cuda.empty_cache() # 这里的写法对显存释放很不友好
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        h_out = self.act(h_src + h_dst)
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        ans = self.out_fc(h_out)
        return ans


class LinkPredTrainer(object):
    def __init__(self, ctx: tg.TContext, model: nn.Module,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 neg_sampler: Callable, epochs: int, bsize: int,
                 train_end: int, val_end: int,
                 model_path: str, model_mem_path: Optional[str]):
        self.ctx = ctx
        self.g = ctx.graph
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.neg_sampler = neg_sampler
        self.epochs = epochs
        self.bsize = bsize
        self.train_end = train_end
        self.val_end = val_end
        self.model_path = model_path
        self.model_mem_path = model_mem_path
        self.warmup_epochs = 1

    def train(self):
        warmup_epochs = self.warmup_epochs

        tt.csv_open('out-stats.csv')
        tt.csv_write_header()
        best_epoch = 0
        best_ap = 0
        for e in range(self.epochs + warmup_epochs):
            print(f'epoch {e}:')

            if e == 1 + warmup_epochs:
                print("[TEST] torch.cuda.nvtx.range_push")
                cuda.start_profiler()

            torch.cuda.synchronize()
            print("[epoch start]")
            t_epoch = tt.start()

            self.ctx.train()
            self.model.train()
            self.model.sampling_thread = None
            self.model.curr_data = None
            self.model.next_data = None
            if self.g.mem is not None:
                self.g.mem.reset() # TODO 存储需要处理mem mailbox reset
            if self.g.mailbox is not None:
                self.g.mailbox.reset()

            epoch_loss = 0.0
            t_loop = tt.start()
            print("[start] time")

            # 迭代器
            edge_iter = tg.iter_edges(self.g, size=self.bsize, end=self.train_end)
            try:
                batch = next(edge_iter)
                batch.neg_nodes = self.neg_sampler(len(batch))

                while True:
                    with nvtx.annotate(f"Batch {batch._b_id}", color="green"):
                        # print(f"Batch {batch._b_id} - len(batch) {len(batch)}")
                        try:
                            # import pdb;pdb.set_trace()
                            next_batch = next(edge_iter)
                            next_batch.neg_nodes = self.neg_sampler(len(next_batch)) # 这里要用next batch 才是三倍

                        except StopIteration:
                            print("[epoch end]")
                            break

                        batch._nxt_idx = next_batch._end_idx
                        if batch._b_id == 0:
                            if tglite.config.PERF_CEIL: # TODO only TGN
                                self.model._init_samples0_2_perfCeil()
                            elif tglite.config.TEST_BLKM: # ! TODO 
                                self.model._init_samples0_online_TEST_BLKM(batch)

                        # test with nsys WYQ TODO
                        # if (True and tglite.config.ON_HETER): # TODO
                        if (True): # TODO
                        # if (False): # TODO
                            # print(f"Batch {batch._b_id}")
                            # if (e > 2 and batch._b_id > 5):
                            if (batch._b_id > 5):

                                # 训练代码
                                if batch._b_id == 6 and e == 1 + warmup_epochs:
                                    print("[TEST] torch.cuda.nvtx.range_pop")
                                    cuda.stop_profiler()

                                if e > 2 + warmup_epochs:
                                    tt.t_batch_num_5 = tt.elapsed(t_loop)
                                    tt.print_batch_num_5()
                                    prefix='  '
                                    print(f"{prefix}batch5 | max memory {torch.cuda.max_memory_allocated()/(2**20)} MB")
                                    print(f"{prefix}batch5 | cur memory {torch.cuda.memory_allocated()/(2**20)} MB")
                                    print("[end] time")
                                    exit()

                        t_start = tt.start()
                        # batch.neg_nodes = self.neg_sampler(len(batch))

                        tt.t_prep_batch += tt.elapsed(t_start)

                        t_start = tt.start()

                        self.optimizer.zero_grad()

                        # ! pre-sampling logic
                        if tglite.config.PERF_CEIL_BASE: # TODO only TGN 
                            if batch._b_id == 0:
                                self.model._load_new_perfCeilBase(batch) # 在函数内还是改的next
                            else: 
                                assert self.ctx.perfCeilBase_thread is not None
                                self.ctx.perfCeilBase_thread.join()
                            self.ctx.curr_blk_head = self.ctx.next_blk_head
                            self.ctx.curr_blk_tail = self.ctx.next_blk_tail
                            self.ctx.curr_mem = self.ctx.next_mem
                        
                            def perfCeilBase_preloading(self, batch):
                                # print(f"preloading and sampling batch {batch._b_id} - len(batch) {len(batch)}")
                                head = batch.block(self.ctx)
                                for i in range(self.model.num_layers):
                                    tail = head if i == 0 \
                                        else tail.next_block(include_dst=True, use_dst_times=False)
                                    # print(f"batch {batch._b_id} tail._dstnodes {tail._dstnodes.shape}")
                                    tail = tg.op.dedup(tail) if self.model.dedup else tail
                                    with nvtx.annotate("sample", color="purple"):
                                        tail = self.model.sampler.sample(tail) # 先去重再采样 # 为什么这里有通信?
                                with torch.cuda.StreamContext(torch.cuda.Stream()):
                                    
                                    tg.op.preload(head, use_pin=True)

                                    self.ctx.next_blk_tail = tail
                                    self.ctx.next_blk_head = head
                                    self.ctx.next_mem = tail.mem_data()
                                    perfCeilBase_preloadGPUData_event = torch.cuda.Event()
                                    perfCeilBase_preloadGPUData_event.record()
                                perfCeilBase_preloadGPUData_event.synchronize()

                            # 提前取下一个batch add preload logic TODO
                            with nvtx.annotate("threading preload nxt", color="purple"):
                                # Sampling and preloading for next batch
                                self.ctx.perfCeilBase_thread = threading.Thread(target=perfCeilBase_preloading, args=(self, next_batch))
                                self.ctx.perfCeilBase_thread.start()
                    
                        # self.model.is_train = Trues
                        pred_pos, pred_neg = self.model(batch)
                        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                        tt.t_forward += tt.elapsed(t_start)

                        t_start = tt.start()
                        with nvtx.annotate("TRAIN-cal_loss", color="green"):
                            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                            targets = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0)
                            preds = torch.cat([pred_pos, pred_neg], dim=0)
                            loss = self.criterion(preds, targets)
                            # wyq_refine
                            # loss0 = self.criterion(pred_pos, torch.ones_like(pred_pos))
                            # loss1 = self.criterion(pred_neg, torch.zeros_like(pred_neg))
                            # combined_batch_size = pred_pos.size(0) + pred_neg.size(0)
                            # loss_separated = (loss0 * pred_pos.size(0) + loss1 * pred_neg.size(0)) / combined_batch_size
                            # print(f"loss {loss}")
                            # print(f"loss {loss_separated}, loss0 {loss0}, loss1 {loss1}")
                            epoch_loss += float(loss)
                        with nvtx.annotate("TRAIN-backward-optimizer", color="green"):
                            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                            
                            if tglite.config.PERF_CEIL: # TODO only TGN
                                def preloading(self, _nids_cpu, _eids_cpu):
                                    # preload nfeat
                                    ## 1 layer: self.layer == 0 TODO 不过preload 一般只preload 1层就可以
                                    ## self.ctx._nxt_nfeat_pins = self.ctx._get_nfeat_pin(self.layer, len(_nids_cpu), self._g.nfeat.shape[1])
                                    self.ctx._cur_nfeat_pins = self.ctx._get_nfeat_pin(0, len(_nids_cpu), self.ctx._g.nfeat.shape[1])
                                    with nvtx.annotate("index_select", color="red"):
                                        torch.index_select(self.ctx._g.nfeat, 0, _nids_cpu, out=self.ctx._cur_nfeat_pins)
                                    
                                    # preload efeat
                                    self.ctx._cur_efeat_pins = self.ctx._get_efeat_pin(0, len(_eids_cpu), self.ctx._g.efeat.shape[1])
                                    torch.index_select(self.ctx._g.efeat, 0, _eids_cpu, out=self.ctx._cur_efeat_pins)
                                
                                # 提前取下一个batch add preload logic TODO
                                with nvtx.annotate("threading preload nxt", color="purple"): # 取的是next batch
                                    if self.model.sampling_thread is not None:
                                        self.model.sampling_thread.join()
                                        assert self.model.curr_data == None # 表示第一轮预采样/后来next batch 采样的结果已经被使用了
                                        self.model.curr_data = self.model.next_data
                                        self.model.next_data = None # 指示当前batch 可以开始下一轮预采样了
                                        if self.model.curr_data is not None: # 如果还有下一个batch
                                            _inv_idx, b_dstnodes, b_dsttimes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta, \
                                            prev_eids, next_eids, \
                                            prev_nodes, next_nodes, \
                                            unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets, \
                                            _eids_pre, _idx_eids_pre, _eids_cpu, _idx_eids_cpu, \
                                            _eids_nxt, _idx_eids_nxt, \
                                            _nids_pre, _idx_nids_pre, _nids_cpu, _idx_nids_cpu, \
                                            _nids_nxt, _idx_nids_nxt = self.model.curr_data

                                            # Sampling for curr batch(本质是在上一个batch 采样cur)
                                            self.ctx.preload_thread = threading.Thread(target=preloading, args=(self, _nids_cpu, _eids_cpu))
                                            self.ctx.preload_thread.start()

                            loss.backward()
                            # torch.cuda.empty_cache()
                            # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                            self.optimizer.step()
                            tt.t_backward += tt.elapsed(t_start)
                            # 更新下一轮batch
                            batch = next_batch
                        # print(f"cur-batch memory {torch.cuda.memory_allocated()/(2**20)}")
                        # print(f"max-batch memory {torch.cuda.max_memory_allocated()/(2**20)}")
            except StopIteration:
                pass
            tt.t_loop = tt.elapsed(t_loop)

            with nvtx.annotate("TRAIN-eval", color="green"):
                # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
                t_eval = tt.start()
                ap, auc = self.eval(start_idx=self.train_end, end_idx=self.val_end)
                tt.t_eval = tt.elapsed(t_eval)

            torch.cuda.synchronize()
            tt.t_epoch = tt.elapsed(t_epoch)
            if e == 0 or ap > best_ap:
                best_epoch = e
                best_ap = ap
                torch.save(self.model.state_dict(), self.model_path)
                if self.g.mem is not None:
                    torch.save(self.g.mem.backup(), self.model_mem_path)
            print('  loss:{:.4f} val ap:{:.4f} val auc:{:.4f}'.format(epoch_loss, ap, auc))
            tt.csv_write_line(epoch=e)
            tt.print_epoch()
            tt.reset_epoch()
        tt.csv_close()
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        print('best model at epoch {}'.format(best_epoch))

    '''
    @torch.no_grad()
    def eval(self, start_idx: int, end_idx: int = None):
        print("[eval start]")
        self.ctx.eval()
        self.model.eval()
        val_aps = []
        val_auc = []
        for batch in tg.iter_edges(self.g, size=self.bsize, start=start_idx, end=end_idx):
            size = len(batch)
            print(f"size {size} begin_idx {batch._beg_idx} end_idx {batch._end_idx}")
            batch.neg_nodes = self.neg_sampler(size)
            # self.model.is_train = False
            import pdb;pdb.set_trace()
            prob_pos, prob_neg = self.model(batch)
            prob_pos = prob_pos.cpu()
            prob_neg = prob_neg.cpu()
            pred_score = torch.cat([prob_pos, prob_neg], dim=0).sigmoid()
            true_label = torch.cat([torch.ones(size), torch.zeros(size)])
            val_aps.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
        print("[eval end]")
        return np.mean(val_aps), np.mean(val_auc)
    '''
    
    @torch.no_grad()
    def eval(self, start_idx: int, end_idx: int = None):
        # print("[eval start]")
        # self.ctx.eval()
        # self.model.eval()
        # val_aps = []
        # val_auc = []
        # for batch in tg.iter_edges(self.g, size=self.bsize, start=start_idx, end=end_idx):
        #     size = len(batch)
        #     print(f"size {size} begin_idx {batch._beg_idx} end_idx {batch._end_idx}")
        #     batch.neg_nodes = self.neg_sampler(size)
        #     prob_pos, prob_neg = self.model(batch)
        #     prob_pos = prob_pos.cpu()
        #     prob_neg = prob_neg.cpu()
        #     pred_score = torch.cat([prob_pos, prob_neg], dim=0).sigmoid()
        #     true_label = torch.cat([torch.ones(size), torch.zeros(size)])
        #     val_aps.append(average_precision_score(true_label, pred_score))
        #     val_auc.append(roc_auc_score(true_label, pred_score))
        # print("[eval end]")
        # # return np.mean(val_aps), np.mean(val_auc)
    
        print("[eval start]")
        self.ctx.eval()
        self.model.eval()
        self.model.sampling_thread = None
        self.model.curr_data = None
        self.model.next_data = None
        val_aps = []
        val_auc = []
        edge_iter = tg.iter_edges(self.g, size=self.bsize, start=start_idx, end=end_idx)
        try:
            batch = next(edge_iter)
            batch.neg_nodes = self.neg_sampler(len(batch))

            while True:
                with nvtx.annotate(f"Eval-Batch {batch._b_id}", color="green"):
                    try:
                        next_batch = next(edge_iter)
                        next_batch.neg_nodes = self.neg_sampler(len(next_batch)) # 这里要用next batch 才是三倍

                    except StopIteration:
                        print("[eval end]")
                        break

                    batch._nxt_idx = next_batch._end_idx
                    if batch._b_id == 0:
                        if tglite.config.TEST_BLKM:
                            self.model._init_samples0_online_TEST_BLKM(batch)

                    prob_pos, prob_neg = self.model(batch)
                    prob_pos = prob_pos.cpu()
                    prob_neg = prob_neg.cpu()
                    pred_score = torch.cat([prob_pos, prob_neg], dim=0).sigmoid()
                    true_label = torch.cat([torch.ones_like(prob_pos), torch.zeros_like(prob_neg)])
                    val_aps.append(average_precision_score(true_label, pred_score))
                    val_auc.append(roc_auc_score(true_label, pred_score))
                    # pred_score = torch.cat([pred_pos, pred_neg], dim=0).sigmoid()
                    # true_label = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0)
                    # val_aps.append(average_precision_score(true_label, pred_score))
                    # val_auc.append(roc_auc_score(true_label, pred_score))
                    batch = next_batch
        except StopIteration:
            pass
        return np.mean(val_aps), np.mean(val_auc)


    def test(self):
        print('loading saved checkpoint and testing model...')
        self.model.load_state_dict(torch.load(self.model_path))
        if self.g.mem is not None:
            self.g.mem.restore(torch.load(self.model_mem_path))
        t_test = tt.start()
        ap, auc = self.eval(start_idx=self.val_end)
        t_test = tt.elapsed(t_test)
        print('  test time:{:.2f}s AP:{:.4f} AUC:{:.4f}'.format(t_test, ap, auc))
