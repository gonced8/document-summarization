#!/bin/bash

args=(

#Model
--model_name bart
--from_checkpoint checkpoints/bart_maia/version_0/checkpoints/best.ckpt
--max_input_length 512
--max_output_length 256
--lr 1e-3
--no_repeat_ngram_size 5

# Data
--data_name maia
--use_context
--batch_size 8
--test_batch_size 32
--val_percentage 0.05
--test_percentage 0.05
--num_workers 8

# Trainer
--mode test
--max_epochs 10
--accumulate_grad_batches 8
--val_check_interval 0.2
--monitor rougeL
--results_filename "$(basename $0 .sh)"
#--fast_dev_run

)

echo "${args[@]}"

python main.py "${args[@]}" "$@"