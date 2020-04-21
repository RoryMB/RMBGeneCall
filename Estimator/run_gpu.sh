#!/usr/bin/env bash

rm model_dir/*

python run_gpu.py --model_dir model_dir/ --data_dir ../data/ --mode train --epochs 1 --batch_size 4096 --inpfx micro --gpu 0