#!/usr/bin/env bash

rm model_dir/*

python run_ipu.py --model_dir model_dir/ --data_dir /mnt/data/rmbutler/gene_call/ --mode train --epochs 1 --batch_size 32 --inpfx micro