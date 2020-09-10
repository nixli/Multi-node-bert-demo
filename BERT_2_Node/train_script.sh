#!/bin/bash

DATA_DIR=/tmp/books_wiki_en_corpus

BERT_CONFIG=bert_config.json

BATCH_SIZE=16

GRADIENT_ACCUMULATION=1

python3 -m torch.distributed.launch \
    --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank $1 \
    --master_addr $2 \
    --master_port 41682 \
    \
    run_pretraining.py \
    --input_dir=$DATA_DIR \
    --output_dir=./temp \
    --config_file=$BERT_CONFIG \
    --bert_model=bert-large-uncased \
    --train_batch_size=$BATCH_SIZE \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --max_steps=10 \
    --warmup_proportion=0.128 \
    --num_steps_per_checkpoint=100000 \
    --learning_rate=1e-4 \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --do_train \
    --num_train_epochs=1 \
    --disable_progress_bar
