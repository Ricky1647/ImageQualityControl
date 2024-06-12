#!/bin/bash
#| xargs -i python train.py {}
while true; do
    output=$(gpustat | python runner.py)
    if [ -z "$output" ]; then
        echo "no gpu available"
    else
        python train.py "$output"
    fi
    sleep 30
done