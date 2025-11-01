#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python /home/chenzhixuan/Workspace/VIRAL/inference_batch.py \
    --model-path /jhcnas5/chenzhixuan/checkpoints/VIRAL/outputs/llava-v1.5-7b-instruct-llava-mimic_new/checkpoint-9000 \
    --image-folder /jhcnas4/kyle/Xray/DATA/MIMIC-CXR/files \
    --json-path /jhcnas5/chenzhixuan/checkpoints/VIRAL/mimic_findings_test.json \
    --output-csv /home/chenzhixuan/Workspace/VIRAL/results/mimic_llava_outputs.csv \
    --batch-size 4