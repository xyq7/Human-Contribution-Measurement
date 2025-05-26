
data=$1
model=$2
scenario=$3

python src/evaluate_multi.py --dataset_name ${data} --llm_config_file ./src/config/${model}.yaml \
--response_file ./output_s${scenario}/${model}_${data}_3.jsonl \
--output_result_dir ./result_s${scenario}/ \