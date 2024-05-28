#!/bin/bash

data=$1
model=$2
time=$3

if [ ! -d "./ada_output" ]; then
  mkdir -p ./ada_output
fi

python src/generation_llm.py --dataset_name ${data} --llm_config_file ./src/config/${model}.yaml \
--input_file ./data_new/${data}.json \
--output_file ./ada_output/${model}_${data}_${time}.jsonl \
--tensor_parallel_size 4 --batch_size 200 --resume \
--mode gen_ada1 gen_ada2 polish_ada1 polish_ada2 summary_ada1 summary_ada2 subject_ada1 subject_ada2