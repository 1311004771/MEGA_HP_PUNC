#!/bin/bash

cmd="./slurm.pl --quiet"

$cmd --num-threads 2 --gpu 1 train_deberta.log \
	python train.py
