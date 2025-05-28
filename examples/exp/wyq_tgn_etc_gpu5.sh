#!/usr/bin/env bash

examples_dir="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir"

# python tgn/train.py --seed 0 --prefix exp \
#     --epochs 10 --bsize 600 --n-threads 64 \
#     --n-layers 2 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@"

# python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
#     --epochs 1 --bsize 1500 --n-threads 8 \
#     --n-layers 1 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@"

python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
    --epochs 3 --bsize 2000 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"

python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
    --epochs 3 --bsize 2000 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"

python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
    --epochs 3 --bsize 6000 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"

python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
    --epochs 3 --bsize 6000 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"

python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
    --epochs 3 --bsize 2000 --n-threads 8 \
    --n-layers 2 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"

python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
    --epochs 3 --bsize 2000 --n-threads 8 \
    --n-layers 2 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"

python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
    --epochs 3 --bsize 6000 --n-threads 8 \
    --n-layers 2 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"

python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
    --epochs 3 --bsize 6000 --n-threads 8 \
    --n-layers 2 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"