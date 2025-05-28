#!/usr/bin/env bash
#  ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk', 'gdelt']
examples_dir="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir"

# python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
#     --gpu 0 --epochs 1 --bsize 1500 --n-threads 64 \
#     --n-layers 1 --n-heads 2 --n-nbrs 10 \
#     --sampling recent "$@"


nsys profile --force-overwrite true -o tgn_etc_nvtx_6000 python tgn/train.py -d wiki-talk --seed 0 --prefix exp \
    --epochs 1 --bsize 6000 --n-threads 8 \
    --n-layers 1 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@" \
    --on-heter 1 --all-on-gpu 0