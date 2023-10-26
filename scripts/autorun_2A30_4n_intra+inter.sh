#!/bin/bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 --master_addr="172.21.167.14" --master_port=2330 VGG19_2A30_4n_intra+inter.py
