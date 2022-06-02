#!/bin/bash

DATASET_PATH="data/arxiv"

args=(
    #Model
    --model_name retrosum
	--from_checkpoint checkpoints/retrosum_arxiv/version_4/checkpoints/best.ckpt
    --max_input_length 512
    --max_output_length 512
	--chunk_size 64
	--n_neighbors 2
    --lr 1e-3
    --no_repeat_ngram_size 5
	#--retrieval

    # Data
    --data_name arxiv
    --batch_size 16
    --test_batch_size 1
    --num_workers 8

    # Trainer
    --mode test
    --max_epochs 20
    --accumulate_grad_batches 4
    --val_check_interval 0.5
    --monitor val_loss
	--results_filename "$(basename $0 .sh)_new"
    #--fast_dev_run
)

python main.py "${args[@]}" "$@"
