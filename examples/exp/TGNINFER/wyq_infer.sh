#!/bin/bash
examples_dir="$(cd "$(dirname "$0")"; cd ..;cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir":"$examples_dir/tgn"
echo "PYTHONPATH: $PYTHONPATH"

# python inference.py -d wiki-talk --seed 0 --prefix exp --sampling recent --n-threads 8 --time-window 10000 \
#      --save-bsize 200 --bsize 200 --n-layers 1 \
#     --n-heads 2 --n-nbrs 10 --dim-time 100 --dim-embed 100 \
#     --opt-all
# Run inference
python inference.py -d wiki-talk \
    --prefix exp --save-bsize 6000 --bsize 60000 --n-layers 1 \
    --n-heads 2 --n-nbrs 10 --dim-time 100 --dim-embed 100 --n-threads 8 --sampling recent \
    --opt-all --time-window 10000 --seed 0 --move
