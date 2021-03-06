#!/bin/bash

DATASET_PATH="data/arxiv"

args=(
    #Model
    --model_name retrosum
    --max_input_length 512
    --max_output_length 512
	--chunk_size 64
	--n_neighbors 2
    --lr 1e-4
    --no_repeat_ngram_size 5
	--retrieval

    # Data
    --data_name arxiv
    --train_path $DATASET_PATH/train.prot5.pickle
    --val_path $DATASET_PATH/val.prot5.pickle
    --test_path $DATASET_PATH/test.prot5.pickle
    --batch_size 6
    --test_batch_size 1
    --num_workers 8

    # Trainer
    --mode train
    --max_epochs 30
    --accumulate_grad_batches 20
    --val_check_interval 0.2
    --monitor val_loss
    #--fast_dev_run
)

python main.py "${args[@]}" "$@"
