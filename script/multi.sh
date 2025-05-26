
data=$1
model=$2
scenario=$3

if [ ! -d "./output_s${scenario}" ]; then
  mkdir -p ./output_s${scenario}
fi

python src/generation_multi.py --dataset_name ${data} --llm_config_file ./src/config/${model}.yaml \
--input_file ./data_new/${data}.json \
--output_file ./output_s${scenario}/${model}_${data}.jsonl \
--output_file_2 ./output_s${scenario}/${model}_${data}_2.jsonl \
--output_file_3 ./output_s${scenario}/${model}_${data}_3.jsonl \
--scenario ${scenario}