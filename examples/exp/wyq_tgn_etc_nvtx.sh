#!/usr/bin/env bash
#  ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk', 'gdelt']
examples_dir="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir"

# python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
#     --gpu 0 --epochs 1 --bsize 1500 --n-threads 64 \
#     --n-layers 1 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@"

# all on GPU

nsys profile --force-overwrite true -o tgn_etc_nvtx python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
    --epochs 1 --bsize 6000 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"