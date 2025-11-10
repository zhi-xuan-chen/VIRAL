#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export XRCLIP_CKPT=/jhcnas5/chenzhixuan/checkpoints/VIRAL/XR_clip.ckpt

python /home/chenzhixuan/Workspace/VIRAL/inference_batch.py \
    --model-path /jhcnas5/chenzhixuan/checkpoints/VIRAL/outputs/llava-v1.5-7b-instruct-viral-xrclip-mimic/checkpoint-8000 \
    --image-folder /jhcnas4/kyle/Xray/DATA/MIMIC-CXR/files \
    --json-path /jhcnas5/chenzhixuan/checkpoints/VIRAL/mimic_findings_test.json \
    --output-csv /home/chenzhixuan/Workspace/VIRAL/results/xrclip_mimic_viral_outputs.csv \
    --batch-size 1