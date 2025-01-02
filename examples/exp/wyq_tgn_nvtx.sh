#!/usr/bin/env bash
#  ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk', 'gdelt']
examples_dir="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir"

# python tgn/train.py -d wiki --seed 0 --prefix exp \
#     --epochs 1 --bsize 600 --n-threads 64 \
#     --n-layers 2 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@"

# # bs == 600
# nsys profile --force-overwrite true -o tgn_nvtx_origin_bs-600 python tgn/train.py -d wiki --seed 0 --prefix exp \
#     --epochs 1 --bsize 600 --n-threads 64 \
#     --n-layers 2 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@"

# nsys profile --force-overwrite true -o tgn_nvtx_on-gpu_bs-600 python tgn/train.py -d wiki --seed 0 --prefix exp --move \
#     --epochs 1 --bsize 600 --n-threads 64 \
#     --n-layers 2 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@"
# bs == 6000
nsys profile --force-overwrite true -o tgn_nvtx_origin_bs-6000 python tgn/train.py -d wiki --seed 0 --prefix exp \
    --epochs 1 --bsize 6000 --n-threads 64 \
    --n-layers 2 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"

nsys profile --force-overwrite true -o tgn_nvtx_on-gpu_bs-6000 python tgn/train.py -d wiki --seed 0 --prefix exp --move \
    --epochs 1 --bsize 6000 --n-threads 64 \
    --n-layers 2 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"