#!/usr/bin/env bash

NUM_WORKER_NODES=2 srun_train python ./run_cs1.py --model_dir model_dir/ --data_dir data/ --mode train --epochs 1 --batch_size 1 --inpfx micro --cs_ip 10.80.0.100
