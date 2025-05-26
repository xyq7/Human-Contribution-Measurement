import copy
import jsonlines
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import concatenate_datasets, Dataset
import json
import os
import yaml
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

import fastchat
import fastchat.model

import argparse
from config import USER_PROMPT_DICT


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="news_bbc.jsonl",
        help="The dataset file",
    )
    parser.add_argument(
        "--response_file",
        type=str,
        default="vicuna_13b_news_bbc_1.jsonl",
        help="The response file",
    )
    parser.add_argument(
        "--llm_config_file",
        type=str,
        default="./configs/llm.yaml",
        help="The config file of evaluating large language models (LLMs).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["news", "paper", "patent", "poem"],
        help="The dataset name. (e.g. news)",
    )
    parser.add_argument(
        "--output_result_dir",
        type=str,
        default="./result/",
        help="The output directory of the result.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output",
        help="The output name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Whether to resume the inference process.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="The number of samples to evaluate.",
    )

    args = parser.parse_args()
    return args

def est_individual_score(
    id, outputs1, neglogtau
):
    """
    Estimate human contribution
    Args:
        id: ID of the input
        outputs1: The output to estimate human contribution
        neglogtau: The estimation threshold for the mean conditional information
    Returns:
        Example with added message field containing formatted prompt
    """
    nll1 = outputs1.loss.item()

    import math
    human_percent3 = (nll1-neglogtau) / nll1

    return human_percent3


def construct_prompt(example, mode, dataset_name, model_id):
    """
    Construct a conversation prompt using FastChat template
    Args:
        example: Dictionary containing prompt and response
        mode: Mode to generate response
        dataset_name: the name of the dataset
        model_id: ID of the language model
    Returns:
        Example with added message field containing formatted prompt
    """
    conv_template = fastchat.model.get_conversation_template(model_id)

    system_prompt = "You are a helpful assistant."
    user_prompt_dict = USER_PROMPT_DICT[dataset_name]
    user_prompt_template, key = user_prompt_dict[mode]
    abstract_len = len(example["abstract"].split())

    user_prompt = user_prompt_template.format(example[key], abstract_len)
    conv_template.set_system_message(system_prompt)
    conv_template.append_message(conv_template.roles[0], user_prompt)
    conv_template.append_message(conv_template.roles[1], None)

    example["message"] = conv_template.get_prompt()
    return example


def get_prompt_dict(file_name, model_id):
    """
    Create a dictionary mapping indices to processed prompts
    Args:
        cases: List of prompt-response pairs
        model_id: ID of the language model
    Returns:
        Dictionary mapping indices to formatted prompts
    """
    with open(file_name, "r") as file:
        data = json.load(file)

    data_dict = defaultdict(list)
    for id_name in data:

        data_dict["id"].append(id_name)
        sample = data[id_name]
        for key, value in sample.items():
            data_dict[key].append(value)

    dataset = Dataset.from_dict(data_dict)

    user_prompt_dict = USER_PROMPT_DICT[args.dataset_name]

    datasets = {}
    for key in user_prompt_dict:
        datasets[key] = dataset.map(
            partial(
                construct_prompt,
                mode=key,
                dataset_name=args.dataset_name,
                model_id=model_id,
            ),
            desc="Processing datasets.",
        )

        def set_mode(example, key):
            example["mode"] = key
            return example

        for key in datasets:
            datasets[key] = datasets[key].map(partial(set_mode, key=key))

    processed_dataset = concatenate_datasets(list(datasets.values()))
    prompt_dict = {
        f"{mode}-{index}": prompt
        for prompt, mode, index in zip(
            processed_dataset["message"],
            processed_dataset["mode"],
            processed_dataset["id"],
        )
    }
    return prompt_dict


def cal_loss_wo_rewrite(
    response_file,
    dataset_file,
    model_id,
    output_name,
    num_samples=None,
    output_result_dir="./result/",
    device="cuda",
    resume=False,
    neglogtau=None
):
    """
    Calculate loss and human contribution scores 
    Args:
        response_file: The file with generated responses (generated by generation_llm.py)
        dataset_file: The dataset file
        model_id: ID of the language model
        output_name: Name for output files
        num_samples: Number of samples for evaluation
        output_result_dir: Directory for saving results
        device: Computing device (cuda/cpu)
        resume: Whether to resume from previous results
        neglogtau: The estimation threshold for the mean conditional information
    Saves results to JSON and JSONL files
    """
    prompt_dict = get_prompt_dict(dataset_file, model_id)

    file_path = output_result_dir + output_name + "_origin.json"
    ori_ids = []
    if resume and os.path.exists(file_path):
        with open(file_path, "r") as file:
            ori = json.load(file)
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(dataset_file, "r") as json_file:
            data = json.load(json_file)
        for idx, value in tqdm(data.items()):
            if num_samples is not None and len(ori_ids) >= num_samples:
                break
            ori_ids.append(idx)

    ori_ids = set(ori_ids)

    file_path = output_result_dir + output_name + "_ces.json"
    if resume and os.path.exists(file_path):
        with open(file_path, "r") as file:
            ces = json.load(file)
    else:
        ces = {}
        out = []
        model, tokenizer = load_model(model_id, device)
        with open(response_file, "r") as file:
            for idx, line in tqdm(enumerate(file)):
                sample = json.loads(line)
                if sample["id"] not in ori_ids:
                    continue

                mode = sample["mode"]
                if mode not in ces:
                    ces[mode] = []

                response = sample["response"]

                with torch.no_grad():
                    extracted_text = response + tokenizer.eos_token

                    input_ids = torch.tensor(
                        np.array([tokenizer(extracted_text).input_ids])
                    ).to(model.device)

                    target_ids = input_ids.clone()
                    sep = 1
                    target_ids[:, :sep] = -100
                    outputs1 = model(input_ids, labels=target_ids)
                    # logits1 = outputs.logits.detach().cpu()


                rslt = est_individual_score(
                    sample["id"],
                    outputs1,
                    neglogtau
                )

                ces[mode].append(rslt)
                out.append(rslt)

        with open(output_result_dir + output_name + "_est_ces.json", "w") as json_file:
            json.dump(ces, json_file)
        with jsonlines.open(
            output_result_dir + output_name + "_est_ces_id.json", "w"
        ) as writer:
            writer.write_all(out)


def load_model(model_id, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )

    model.eval()
    print(model.__class__)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()

    with open(args.llm_config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    Path(args.output_result_dir).mkdir(parents=True, exist_ok=True)
    if config["model_name"] == 'mistralai/Mixtral-8x7B-Instruct-v0.1': 
        neglogtau = 0.1979 + 0.2176
    elif config["model_name"] == 'meta-llama/Meta-Llama-3-8B-Instruct': 
        neglogtau = 0.3760 + 0.1736
        
    cal_loss_wo_rewrite(
        args.response_file,
        args.dataset_file,
        config["model_name"],
        args.output_name + "_wo_rewrite",
        num_samples=args.num_samples,
        output_result_dir=args.output_result_dir,
        device="cuda",
        resume=args.resume,
        neglogtau=neglogtau,
    )
