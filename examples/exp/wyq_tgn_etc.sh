#!/usr/bin/env bash
#  ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk', 'gdelt']
examples_dir="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir"

# python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
#     --gpu 0 --epochs 1 --bsize 1500 --n-threads 64 \
#     --n-layers 1 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@"

# all-on-GPU
python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
    --epochs 1 --bsize 1500 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@" \
    --on-heter 0 --all-on-gpu 1

# on heter
python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
    --epochs 1 --bsize 1500 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@" \
    --on-heter 1 --all-on-gpu 0

# origin tglite w move
python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
    --epochs 1 --bsize 1500 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@" \
    --on-heter 0 --all-on-gpu 0
    
# origin tglite w/o move
python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
    --epochs 1 --bsize 1500 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@" \
    --on-heter 0 --all-on-gpu 0