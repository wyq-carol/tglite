#!/usr/bin/env bash

examples_dir="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir"

# nsys profile --force-overwrite true -c cudaProfilerApi -o tgn_test_blkm python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
#     --epochs 3 --bsize 6000 --n-threads 8 \
#     --n-layers 1 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@" \
#     --on-heter 0 --all-on-gpu 0 --offline-sample 0 --perf-ceil 0 --block-mem-on 1

python tgn/train.py -d wiki-talk --seed 0 --prefix exp --move \
    --epochs 3 --bsize 6000 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@" \
    --on-heter 0 --all-on-gpu 0 --offline-sample 0 --perf-ceil 0 --block-mem-on 1

# python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
#     --epochs 3 --bsize 6000 --n-threads 8 \
#     --n-layers 1 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@"