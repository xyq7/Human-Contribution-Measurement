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


def cal_individual_score(
    id, outputs1, outputs2, target_ids, prompt_length, num_words=None
):
    logits1 = outputs1.logits.detach().cpu()
    logits2 = outputs2.logits.detach().cpu()

    nll1 = outputs1.loss.item()
    nll2 = outputs2.loss.item()

    probabilities = F.softmax(logits1[:, :-1], dim=-1)
    probabilities2 = F.softmax(logits2[:, prompt_length:-1], dim=-1)

    collaborative_one_hot = F.one_hot(
        torch.tensor(target_ids), num_classes=probabilities.size(-1)
    ).float()

    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    entropy2 = -torch.sum(probabilities2 * torch.log(probabilities2), dim=-1)

    ce = -torch.sum(collaborative_one_hot * torch.log(probabilities), dim=-1)
    ce2 = -torch.sum(collaborative_one_hot * torch.log(probabilities2), dim=-1)

    human_percent = (
        (torch.sum(entropy) - torch.sum(entropy2)) / torch.sum(entropy)
    ).item()
    human_percent2 = ((torch.sum(ce) - torch.sum(ce2)) / torch.sum(ce)).item()
    human_percent3 = (nll1 - nll2) / nll1

    rslt = {
        "id": id,
        "en": entropy.squeeze().cpu().numpy().tolist(),
        "en2": entropy2.squeeze().cpu().numpy().tolist(),
        "ce": ce.squeeze().cpu().numpy().tolist(),
        "ce2": ce2.squeeze().cpu().numpy().tolist(),
        "nll1": nll1,
        "nll2": nll2,
        "human_percent": human_percent,
        "human_percent2": human_percent2,
        "human_percent3": human_percent3,
        "num_words": num_words,
    }
    return rslt


def construct_prompt(example, mode, dataset_name, model_id):
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
    key,
    output_name,
    num_samples=None,
    output_result_dir="./result/",
    device="cuda",
    resume=False,
):
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
                # start_token_dict = {
                #     "paper": "Abstract:",
                #     "news": "News:",
                #     "patent": "Abstract:",
                #     "poem": "Poem:",
                # }
                # last_start_index = response.rfind(start_token_dict[key])
                # response = response[last_start_index + len(start_token_dict[key]) :].strip()

                with torch.no_grad():
                    extracted_text = response + tokenizer.eos_token

                    input_ids = torch.tensor(
                        np.array([tokenizer(extracted_text).input_ids])
                    ).to(device)

                    target_ids = input_ids.clone()
                    sep = 1
                    target_ids[:, :sep] = -100
                    outputs1 = model(input_ids, labels=target_ids)
                    # logits1 = outputs.logits.detach().cpu()

                with torch.no_grad():
                    prompt = prompt_dict["{}-{}".format(mode, sample["id"])]

                    # prompt = sample["message"]
                    prompt_tokens = tokenizer(
                        prompt, add_special_tokens=False
                    ).input_ids
                    response_tokens = tokenizer(
                        response, add_special_tokens=False
                    ).input_ids

                    extracted_text = (
                        [tokenizer.bos_token_id]
                        + prompt_tokens
                        + response_tokens
                        + [tokenizer.eos_token_id]
                    )

                    input_ids = torch.tensor([extracted_text]).to(device)
                    target_ids = input_ids.clone()
                    sep = len(prompt_tokens) + 1
                    target_ids[:, :sep] = -100
                    outputs2 = model(input_ids, labels=target_ids)

                rslt = cal_individual_score(
                    sample["id"],
                    outputs1,
                    outputs2,
                    response_tokens + [tokenizer.eos_token_id],
                    len(prompt_tokens),
                    num_words=sample.get("num_words", None),
                )

                ces[mode].append(rslt)
                out.append(rslt)

        with open(output_result_dir + output_name + "_ces.json", "w") as json_file:
            json.dump(ces, json_file)
        with jsonlines.open(
            output_result_dir + output_name + "_ces_id.json", "w"
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

    cal_loss_wo_rewrite(
        args.response_file,
        args.dataset_file,
        config["model_name"],
        args.dataset_name,
        args.output_name + "_wo_rewrite",
        num_samples=args.num_samples,
        output_result_dir=args.output_result_dir,
        resume=args.resume,
    )
