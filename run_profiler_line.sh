#! /usr/bin/env bash

N="$1"
if [[ -z "$N" ]]; then
	N=10
fi
TIME=$(date +%+4Y-%m-%d_%H:%M:%S)

echo "Profiling BLOCK at N = " "$N"
python -m cProfile -o "profiles/$TIME""_""$N""block_line.prof" test_calc_L_line.py "$N"
snakeviz "profiles/$TIME""_""$N""block_line.prof" &
echo "Profiling BLOCK optimized at N = " "$N"
python -m cProfile -o "profiles/$TIME""_""$N""block_line1.prof" test_calc_L_line1.py "$N"
snakeviz "profiles/$TIME""_""$N""block_line1.prof" &
# N.B. creates backgorund snakeviz processes; kill with e.g. killall snakeviz
