#!/bin/bash

data=$1
model=$2
time=$3

if [ ! -d "./style_output" ]; then
  mkdir -p ./style_output
fi

python src/generation_llm.py --dataset_name ${data} --llm_config_file ./src/config/${model}.yaml \
--input_file ./data_new/${data}.json \
--output_file ./style_output/${model}_${data}_${time}.jsonl \
--tensor_parallel_size 4 --batch_size 200 --resume \
--mode gen_style1 gen_style2 gen_style3 gen_style4 polish_style1 polish_style2 polish_style3 polish_style4 summary_style1 summary_style2 summary_style3 summary_style4 subject_style1 subject_style2 subject_style3 subject_style4