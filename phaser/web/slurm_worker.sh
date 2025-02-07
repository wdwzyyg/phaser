#!/bin/sh

python_exec="$HOME/code/phaser/venv/bin/python"

url="$1"
echo "Running worker, connecting to '$url'"

while true; do
    "$python_exec" -m phaser worker "$url"
    sig=$(($? - 128))
    if [ $sig -gt 0 ] && [ $(kill -l $sig) == "HUP" ]; then
        echo "Restarting worker (SIGHUP)"
        continue
    fi
    break
done