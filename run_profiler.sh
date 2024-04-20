#! /usr/bin/env bash

N="$1"
if [[ -z "$N" ]]; then
	N=10
fi
TIME=$(date +%+4Y-%m-%d_%H:%M:%S)

echo "Profiling BLOCK at N = " "$N"
python -m cProfile -o "profiles/$TIME""_""$N""block.prof" test_block.py "$N"
snakeviz "profiles/$TIME""_""$N""block.prof" &
echo "Profiling FULL at N = " "$N"
python -m cProfile -o "profiles/$TIME""_""$N""full.prof" test_full.py "$N"
snakeviz "profiles/$TIME""_""$N""full.prof" &
# N.B. creates backgorund snakeviz processes; kill with e.g. killall snakeviz
