#!/usr/bin/env bash
PROJ_ROOT=/iccp/aananth2/pandas-owssd
export PYTHONPATH=${PROJ_ROOT}
cd ${PROJ_ROOT}

SAVE_PATH=/iccp/aananth2/pandas-owssd/pandas_experiments_voc_base
SSL_CKPT=/iccp/aananth2/pandas-owssd/moco/moco_v2_800ep_pretrain.pth.tar
DATA_PATH=/iccp/garvita4/pandas/VOC2012/
SPLIT_PATH=/iccp/garvita4/pandas/voc_splits

EXPT_NAME=voc_base_phase
LOG_FILE=${SAVE_PATH}/logs/${EXPT_NAME}.log
OUTPUT_DIR=${SAVE_PATH}/${EXPT_NAME}
python -u train_base_network.py \
        --backbone_ckpt ${SSL_CKPT} \
        --data_path ${DATA_PATH} \
        --split_path ${SPLIT_PATH} \
        --voc_split 10-10 \
        --num_classes 11 \
        --batch-size 16 \
        --epochs 85 \
        --lr 0.02 \
        --warmup_iters 350 \
        --warmup_factor 0.05 \
        --freeze_bn 0 \
        --output_dir ${OUTPUT_DIR} >${LOG_FILE} \
        --resume /iccp/aananth2/pandas-owssd/pandas_experiments_voc_base/voc_base_phase/model_best.pth \
        --test-only 
