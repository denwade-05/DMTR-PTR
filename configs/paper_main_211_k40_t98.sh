#!/usr/bin/env bash
# Same flags as scripts/train_main.sh (sourced documentation / copy-paste).
# Paper main: vit_dmtr_ptr_211_k40_t98_aux10_bg40_split5

EXPERIMENT=vit_dmtr_ptr_211_k40_t98_aux10_bg40_split5
MODEL=vit_dmtr_ptr
EPOCHS=300
BATCH_SIZE=16
LR=0.0001
LOSS=genexp
H_PATCH=4
H_DEPTH=4
H_HEADS=8
H_DIM=64
H_MLP_DIM=128
H_DIM_HEAD=64
SUPER_BYPASS_LAYER=3
BASE_BYPASS_LAYER=4
BASE_BG_THRESHOLD=0.98
BASE_KEEP_MIN_RATIO=0.4
BG_AUX_WEIGHT=0.1
BG_LABEL_THRESH=0.0
BG_MERGE_RATIO=0.4
SPLIT_RATIO=0.05
