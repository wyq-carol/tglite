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
from tglite.memory_management import *
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
    parser.add_argument('--all-on-gpu', type=int, default=1, help='is node memory all-on-gpu')
    parser.add_argument('--on-heter', type=int, default=0, help='is node memory heterogeneous-aware')
    parser.add_argument('--offline-sample', type=int, default=0, help='using offline sample')
    parser.add_argument('--on-statistic', type=int, default=0, help='is statistic on')
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
    print(f"ON_HETER {tglite.config.ON_HETER}, OFFLINE_SAMPLE {tglite.config.OFFLINE_SAMPLE}, ON_STATISTIC {tglite.config.ON_STATISTIC}, ALL_ON_GPU {tglite.config.ALL_ON_GPU}")
    tglite.config.log_name = f"DATA_{DATA}_BS_{BATCH_SIZE}_NLAYER_{N_LAYERS}_NBR_{N_NBRS}_NHEAD_{N_HEADS}"
    tglite.config.log_dir = f"/home/volume/tglake_res/"

    ### load data

    g = support.load_graph(os.path.join(DATA_PATH, f'/home/volume/{DATA}/edges.csv'))

    if tglite.config.ON_HETER:
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
        #     b_dstnodes, b_dstindex, b_srcnodes, b_eids, b_ets, \
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

    tglite.config.num_nodes = g.nfeat.shape[0]

    z = None
    # z = torch.zeros(1).float().to(device)

    ctx = tg.TContext(g)
    ctx.set_z(z)
    ctx.need_sampling(True)
    ctx.enable_time_precompute(OPT_TIME)
    ctx.set_time_window(TIME_WINDOW)


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
    if tglite.config.OFFLINE_SAMPLE:
        model._load_new_samples()
    
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    # memory_stats(inspect.getfile(inspect.currentframe()), inspect.currentframe().f_lineno)


    ### training

    train_end, val_end = support.data_split(g.num_edges(), 0.7, 0.15)
    neg_sampler = lambda size: np.random.randint(0, g.num_nodes(), size)


    trainer = support.LinkPredTrainer(
        ctx, model, criterion, optimizer, neg_sampler,
        EPOCHS, BATCH_SIZE, train_end, val_end,
        model_path, model_mem_path)

    with nvtx.annotate("TRAIN", color="green"):
        trainer.train()
    print(f"max memory {torch.cuda.max_memory_allocated()/(2**20)}")
    print(f"cur memory {torch.cuda.memory_allocated()/(2**20)}")
    d = torch.cuda.memory_stats(device)
    print(f'cur large_pool {sep(d["allocated_bytes.large_pool.peak"]).rjust(20)}')
    print(f'cur small_pool {sep(d["allocated_bytes.small_pool.peak"]).rjust(20)}')

    if tglite.config.ON_STATISTIC:
        get_node_centric_skew(tglite.config.log_dir, tglite.config.log_name)
        draw_node_centric_skew(tglite.config.log_dir, tglite.config.log_name)
        get_samples(tglite.config.log_dir, tglite.config.log_name)

    with nvtx.annotate("TEST", color="green"):
        trainer.test()
