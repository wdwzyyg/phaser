#!/bin/bash

module load anaconda/Python-ML-2025a
eval "$(conda 'shell.bash' hook)"
# load conda env
conda activate phaser

module prepend-path LD_LIBRARY_PATH "/usr/local/pkg/cuda/cuda-11.8/nsight-systems-2022.4.2/target-linux-x64"
module load cuda/11.8

python_exec="python"

url="$1"
echo "Running worker, connecting to '$url'"

while true; do
    "$python_exec" -m phaser worker "$url"
    sig=$(($? - 128))
    if [ $sig -gt 0 ] && [ "$(kill -l $sig)" == "HUP" ]; then
        echo "Restarting worker (SIGHUP)"
        continue
    fi
    break
done
