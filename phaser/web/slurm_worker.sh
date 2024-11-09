#!/bin/sh

python_exec="$HOME/code/phaser/venv/bin/python"

url="$1"
echo "Running worker, connecting to '$url'"
"$python_exec" -m phaser worker "$url"