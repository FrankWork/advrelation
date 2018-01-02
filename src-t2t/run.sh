#! /bin/bash

export CUDA_VISIBLE_DEVICES=1
export PATH="${HOME}/bin/anaconda3/bin:$PATH"
export ROOT_DIR=$(pwd)

USR_DIR=${ROOT_DIR}/src2

# t2t-trainer --t2t_usr_dir=${USR_DIR} --registry_help

PROBLEM=sem_eval2010_task8
MODEL=relation_adv
HPARAMS=relation_adv_base

DATA_DIR=${ROOT_DIR}/data/generated
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=${ROOT_DIR}/saved_models/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
# ==============================================================================
# Inspect data
# ==============================================================================
# python src2/inspect_record.py \
#     --t2t_usr_dir=${USR_DIR} \
#     --data_dir=$DATA_DIR \
#     --problems=$PROBLEM \

# python src2/inspect_input_tensor.py \
#     --t2t_usr_dir=${USR_DIR} \
#     --data_dir=$DATA_DIR \
#     --problems=$PROBLEM \
#     --hparams_set=$HPARAMS \
# ==============================================================================
# Train 
# ==============================================================================
t2t-trainer \
  --t2t_usr_dir=${USR_DIR} \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=800 \
  --eval_steps=28 \
  # --local_eval_frequency=80 \
  
  #--generate_data
  # --tfdbg

# tensorboard --logdir=${TRAIN_DIR}

# src1/scorer.pl data/SemEval/test_keys.txt data/generated/results.txt