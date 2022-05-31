#!/bin/bash

args=(

#Model
--model_name dialogpt
#--from_checkpoint checkpoints/dialogpt_personachat/version_0/checkpoints/best.ckpt
--max_input_length 1024
--max_output_length 1024
--lr 1e-4
--no_repeat_ngram_size 5

# Data
--data_name personachat
--use_context
--batch_size 4
--test_batch_size 1
--val_percentage 0.05
--test_percentage 0.05
--num_workers 8

# Trainer
--mode test
--max_epochs 10
--accumulate_grad_batches 16
--val_check_interval 0.2
--monitor rougeL
--results_filename "$(basename $0 .sh)"
#--fast_dev_run

)

echo "${args[@]}"

python main.py "${args[@]}" "$@"
