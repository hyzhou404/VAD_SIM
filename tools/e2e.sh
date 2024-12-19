#!/usr/bin/zsh

source ~/.zshrc
conda activate uniad

VAD_PATH=/nas/users/hyzhou/PAMI2024/release/VAD_Sim
cd ${VAD_PATH}
echo ${PWD}
CUDA_VISIBLE_DEVICES=${1} python tools/closeloop/e2e.py \
    projects/configs/VAD/VAD_base_e2e.py ckpts/VAD_base.pth \
    --launcher none --eval bbox --tmpdir tmp --output $2
cd -