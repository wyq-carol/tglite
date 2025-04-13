import argparse
import os
import torch
import numpy as np
import tglite as tg

import support
from tgn import TGN
import nvtx
from tglite.gpu_mem_track import *
import tglite.config
from tglite.blockMgrsv4 import *
# from tglite.blockMgrs import *
import pandas as pd
from pathlib import Path
### arguments

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='dataset name')
    parser.add_argument('--data-path', type=str, default='', help='path to data folder')
    parser.add_argument('--prefix', type=str, default='', help='name for saving trained model')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device to use (or -1 for cpu)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 100)')
    parser.add_argument('--bsize', type=int, default=200, help='batch size (default: 200)')
    parser.add_argument('--lr', type=str, default=0.0001, help='learning rate (default: 1e-4)')
    parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate (default: 0.1)')
    parser.add_argument('--n-layers', type=int, default=2, help='number of layers (default: 2)')
    parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads (default: 2)')
    parser.add_argument('--n-nbrs', type=int, default=20, help='number of neighbors to sample (default: 20)')
    parser.add_argument('--dim-time', type=int, default=100, help='dimension of time features (default: 100)')
    parser.add_argument('--dim-embed', type=int, default=100, help='dimension of embeddings (default: 100)')
    parser.add_argument('--seed', type=int, default=-1, help='random seed to use')
    parser.add_argument('--move', action='store_true', help='move data to device')
    parser.add_argument('--n-threads', type=int, default=32, help='number of threads for sampler (default: 32)')
    parser.add_argument('--sampling', type=str, default='recent', choices=['recent', 'uniform'], help='sampling strategy (default: recent)')
    parser.add_argument('--opt-dedup', action='store_true', help='enable dedup optimization')
    parser.add_argument('--opt-time', action='store_true', help='enable precomputing time encodings')
    parser.add_argument('--time-window', type=str, default=1e4, help='time window to precompute (default: 1e4)')
    parser.add_argument('--opt-all', action='store_true', help='enable all available optimizations')
    parser.add_argument('--all-on-gpu', type=int, default=0, help='is node memory all-on-gpu')
    parser.add_argument('--on-heter', type=int, default=0, help='is node memory heterogeneous-aware')
    parser.add_argument('--offline-sample', type=int, default=0, help='using offline sample')
    parser.add_argument('--on-statistic', type=int, default=0, help='is statistic on')
    parser.add_argument('--perf-ceil', type=int, default=0, help='using perf ceil')
    parser.add_argument('--perf-ceil-base', type=int, default=0, help='using perf ceil base')
    parser.add_argument('--block-mem-on', type=int, default=0, help='using blk mem')
    args = parser.parse_args()
    print(args)

    device = support.make_device(args.gpu)
    model_path = support.make_model_path('tgn', args.prefix, args.data)
    model_mem_path = support.make_model_mem_path('tgn', args.prefix, args.data)
    if args.seed >= 0:
        support.set_seed(args.seed)

    DATA: str = args.data
    DATA_PATH: str = args.data_path
    EPOCHS: int = args.epochs
    BATCH_SIZE: int = args.bsize
    LEARN_RATE: float = float(args.lr)
    DROPOUT: float = float(args.dropout)
    N_LAYERS: int = args.n_layers
    N_HEADS: int = args.n_heads
    N_NBRS: int = args.n_nbrs
    DIM_TIME: int = args.dim_time
    DIM_EMBED: int = args.dim_embed
    N_THREADS: int = args.n_threads
    SAMPLING: str = args.sampling
    OPT_DEDUP: bool = args.opt_dedup or args.opt_all
    OPT_TIME: bool = args.opt_time or args.opt_all
    TIME_WINDOW: int = int(args.time_window)
    tglite.config.ALL_ON_GPU = int(args.all_on_gpu)
    tglite.config.ON_HETER = int(args.on_heter)
    tglite.config.ON_STATISTIC = int(args.on_statistic)
    tglite.config.OFFLINE_SAMPLE = int(args.offline_sample)
    tglite.config.PERF_CEIL = int(args.perf_ceil)
    tglite.config.PERF_CEIL_BASE = int(args.perf_ceil_base)
    tglite.config.TEST_BLKM = int(args.block_mem_on)
    lines = f'' \
            f'PERF_CEIL {tglite.config.PERF_CEIL}, PERF_CEIL_BASE {tglite.config.PERF_CEIL_BASE}\n' \
            f'ON_HETER {tglite.config.ON_HETER}, OFFLINE_SAMPLE {tglite.config.OFFLINE_SAMPLE}\n' \
            f'ON_STATISTIC {tglite.config.ON_STATISTIC}, ALL_ON_GPU {tglite.config.ALL_ON_GPU}\n' \
            f'TEST_BLKM {tglite.config.TEST_BLKM}\n'
    print(lines, end='')
    tglite.config.log_name = f"DATA_{DATA}_BS_{BATCH_SIZE}_NLAYER_{N_LAYERS}_NBR_{N_NBRS}_NHEAD_{N_HEADS}"
    tglite.config.log_dir = f"/home/volume/tglake_res/"


    ### load graph

    if tglite.config.TEST_BLKM:
        """Create a TGraph with edges and timestamps loaded from path. Provided data should include
        'src' 'dst' and 'time' columns."""
        df = pd.read_csv(str(os.path.join(DATA_PATH, f'/home/volume/{DATA}/edges.csv')))
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

    else:
        g = support.load_graph(os.path.join(DATA_PATH, f'/home/volume/{DATA}/edges.csv'))


    ### init blk manager

    if tglite.config.TEST_BLKM:
        # TODO A100 40G 目前只支持efeat_dim = 172
        shared_pool = BlockPool(total_mem_gb=30, block_elements=4300)
    
        # manager_nfeat = BlockManager(shared_pool, feature_size=100, max_manager_index=g.num_nodes())
        # manager_efeat = BlockManager(shared_pool, feature_size=172, max_manager_index=g.num_edges())


    ### load data


    if tglite.config.TEST_BLKM:
        # support.load_feats
        edge_feats = None
        node_feats = None
        if Path(os.path.join(DATA_PATH, f'/home/volume/{DATA}/edge_features.pt')).exists():
            edge_feats = torch.load(os.path.join(DATA_PATH, f'/home/volume/{DATA}/edge_features.pt'))
            edge_feats = edge_feats.type(torch.float32)
        elif DATA in ['mooc', 'lastfm']:
            edge_feats = torch.randn(g.num_edges(), 128, dtype=torch.float32)
        elif DATA in ['wiki-talk', 'stackoverflow']:
            edge_feats = torch.randn(g.num_edges(), 172, dtype=torch.float32)

        if DATA in ['wiki-talk']:
            node_feats = torch.randn(g.num_nodes(), 100, dtype=torch.float32)
        elif Path(os.path.join(DATA_PATH, f'/home/volume/{DATA}/node_features.pt')).exists():
            node_feats = torch.load(os.path.join(DATA_PATH, f'/home/volume/{DATA}/node_features.pt'))
            node_feats = node_feats.type(torch.float32)
        elif DATA in ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk', 'stackoverflow']:
            # node_feats = torch.randn(g.num_nodes(), edge_feats.shape[1], dtype=torch.float32)
            node_feats = torch.randn(g.num_nodes(), 100, dtype=torch.float32)

        print('edge feat:', None if edge_feats is None else edge_feats.shape)
        print('node feat:', None if node_feats is None else node_feats.shape)
        # g.efeat = edge_feats
        # g.nfeat = node_feats
        dim_efeat = edge_feats.shape[-1]
        dim_nfeat = node_feats.shape[-1]
        g.dim_edge = dim_efeat
        g.dim_node = dim_nfeat

        import math
        # TODO A100 40G 目前只支持efeat_dim = 172
        manager_nfeat = BlockManager(shared_pool, node_feats, feature_size=100, max_idx=g.num_nodes(), init_blocks=math.ceil(g.num_nodes()/(4300/100)))
        manager_nfeat.copy_from_cpu_batch(indices=torch.arange(g.num_nodes(), dtype=torch.int32, device='cuda'))
        manager_efeat = BlockManager(shared_pool, edge_feats, feature_size=172, max_idx=g.num_edges(), init_blocks=math.ceil(g.num_edges()/(4300/172)))
        manager_efeat.copy_from_cpu_batch(indices=torch.arange(g.num_edges(), dtype=torch.int32, device='cuda'))

        # manager_nfeat.init(torch.arange(g.num_nodes()), node_feats)
        # manager_efeat.init(torch.arange(g.num_edges()), edge_feats)


        g.set_compute(device)

        g.mailbox = tg.Mailbox(g.num_nodes(), 1, 2 * DIM_EMBED + dim_efeat) # mailbox, mem on cpu
        # import pdb;pdb.set_trace()
        g.mem = tg.Memory(g.num_nodes(), DIM_EMBED)
        if args.move:
            g._mem.move_to(device)
            g._mailbox.move_to(device)

        manager_mem_mail = MemMailManager(shared_pool, DIM_EMBED, g.num_nodes(), manager_efeat, math.ceil(g.num_nodes()/(4300/100)), g.num_nodes())

    elif tglite.config.PERF_CEIL: # 结合ON_HETER 和OFFLINE SAMPLE
        support.load_feats(g, "cpu", DATA, DATA_PATH) # feat on cpu
        dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
        dim_nfeat = g.nfeat.shape[1]

        g.set_compute(device)

        g.mailbox = tg.Mailbox(g.num_nodes(), 1, 2 * DIM_EMBED + dim_efeat) # mailbox, mem on cpu
        g.mem = tg.Memory(g.num_nodes(), DIM_EMBED)

    elif tglite.config.ON_HETER:
        support.load_feats(g, "cpu", DATA, DATA_PATH) # feat on cpu
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
        dim_nfeat = g.nfeat.shape[1]

        g.set_compute(device)

        g.mailbox = tg.Mailbox(g.num_nodes(), 1, 2 * DIM_EMBED + dim_efeat) # mailbox, mem on cpu
        g.mem = tg.Memory(g.num_nodes(), DIM_EMBED)

        # TODO memory management
        
    elif tglite.config.OFFLINE_SAMPLE:
        support.load_feats(g, "cpu", DATA, DATA_PATH) # feat on cpu
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
        dim_nfeat = g.nfeat.shape[1]

        g.set_compute(device)

        g.mailbox = tg.Mailbox(g.num_nodes(), 1, 2 * DIM_EMBED + dim_efeat) # mailbox, mem on cpu
        g.mem = tg.Memory(g.num_nodes(), DIM_EMBED)

        # file_path = os.path.join(tglite.config.log_dir, f"new_samples_{tglite.config.log_name}.pt")
        # _new_samples = torch.load(file_path)

        # for sample in _new_samples:
        #     b_dstnodes, b_dstindex, b_srcnodes, b_eids, b_ets, _unique_time_delta, _reverse_time_delta, \
        #     unique_eids, _reverse_eids, _unique_nids, _reverse_nids, _unique_ets, _reverse_ets, \
        #     _eids_pre, _nids_pre, _eids_nxt, _nids_nxt = sample
        #     # print(f"b_dstnodes {b_dstnodes.size()}")
        #     # print(f"b_dstindex {b_dstindex.size()}")
        #     # print(f"b_srcnodes {b_srcnodes.size()}")
        #     # print(f"b_eids {b_eids.size()}")
        #     # print(f"b_ets {b_ets.size()}")
        #     # print("*****")
        #     # print(f"unique_eids {unique_eids.size()}")
        #     # print(f"_reverse_eids {_reverse_eids.size()}")
        #     # print(f"_unique_nids {_unique_nids.size()}")
        #     # print(f"_reverse_nids {_reverse_nids.size()}")
        #     # print(f"_unique_ets {_unique_ets.size()}")
        #     # print(f"_reverse_ets {_reverse_ets.size()}")
        #     # print("*****")
        #     # print(f"_eids_pre {_eids_pre.size()}")
        #     # print(f"_eids_nxt {_eids_nxt.size()}")
        #     # print(f"_nids_pre {_nids_pre.size()}")
        #     # print(f"_nids_nxt {_nids_nxt.size()}")
        #     # print()

    elif tglite.config.ALL_ON_GPU:
        support.load_feats(g, device, DATA, DATA_PATH)
        # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)
        dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
        dim_nfeat = g.nfeat.shape[1]

        g.set_compute(device)
        g.set_storage(device)

        g.mailbox = tg.Mailbox(g.num_nodes(), 1, 2 * DIM_EMBED + dim_efeat, device)
        g.mem = tg.Memory(g.num_nodes(), DIM_EMBED, device)

    else:
        support.load_feats(g, "cpu", DATA, DATA_PATH)
        dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
        dim_nfeat = g.nfeat.shape[1]

        g.set_compute(device)

        g.mailbox = tg.Mailbox(g.num_nodes(), 1, 2 * DIM_EMBED + dim_efeat)
        g.mem = tg.Memory(g.num_nodes(), DIM_EMBED)
        if args.move:
            g.move_data(device)

    tglite.config.num_nodes = g.num_nodes()

    z = None
    # z = torch.zeros(1).float().to(device)

    ctx = tg.TContext(g)
    ctx.set_z(z)
    ctx.need_sampling(True)
    ctx.enable_time_precompute(OPT_TIME)
    ctx.set_time_window(TIME_WINDOW)
    if tglite.config.TEST_BLKM:
        ctx.set_blkm_nfeat(manager_nfeat)
        ctx.set_blkm_efeat(manager_efeat)
        ctx.set_blkm_mem_mail_feat(manager_mem_mail)


    ### model


    sampler = tg.TSampler(N_NBRS, strategy=SAMPLING, num_threads=N_THREADS)
    model = TGN(ctx,
        dim_node=dim_nfeat,
        dim_edge=dim_efeat,
        dim_time=DIM_TIME,
        dim_embed=DIM_EMBED,
        sampler = sampler,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        dropout=DROPOUT)
    model = model.to(device)
    # if tglite.config.TEST_BLKM:
    #     model._load_new_samples_blkm() # 显存占用很小 & all ready on GPU
    if tglite.config.PERF_CEIL: # 结合ON_HETER 和OFFLINE SAMPLE
        model._load_new_samples2()
    if tglite.config.OFFLINE_SAMPLE:
        model._load_new_samples()
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)


    ### training

    train_end, val_end = support.data_split(g.num_edges(), 0.7, 0.15)
    neg_sampler = lambda size: np.random.randint(0, g.num_nodes(), size)

    trainer = support.LinkPredTrainer(
        ctx, model, criterion, optimizer, neg_sampler,
        EPOCHS, BATCH_SIZE, train_end, val_end,
        model_path, model_mem_path)

    with nvtx.annotate("TRAIN", color="green"):
        trainer.train()
    print(f"max memory {torch.cuda.max_memory_allocated()/(2**20)} MB")
    print(f"cur memory {torch.cuda.memory_allocated()/(2**20)} MB")
    d = torch.cuda.memory_stats(device)
    print(f'cur large_pool {sep(d["allocated_bytes.large_pool.peak"]).rjust(20)}')
    print(f'cur small_pool {sep(d["allocated_bytes.small_pool.peak"]).rjust(20)}')

    if tglite.config.ON_STATISTIC:
        get_node_centric_skew(tglite.config.log_dir, tglite.config.log_name)
        draw_node_centric_skew(tglite.config.log_dir, tglite.config.log_name)
        get_samples(tglite.config.log_dir, tglite.config.log_name)

    with nvtx.annotate("TEST", color="green"):
        trainer.test()
