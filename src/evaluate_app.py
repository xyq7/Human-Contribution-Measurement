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
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

import fastchat
import fastchat.model

import argparse

from config import USER_PROMPT_DICT, MAX_RETRY, RPREFIX_DICT


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
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
        "--output_result_dir",
        type=str,
        default="./result/",
        help="The output directory of the figure.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output",
        help="The output name of the figure.",
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


def construct_prompt(example, model_id):
    conv_template = fastchat.model.get_conversation_template(model_id)

    system_prompt = "You are a helpful assistant."
    user_prompt = example["prompt"]

    conv_template.set_system_message(system_prompt)
    conv_template.append_message(conv_template.roles[0], user_prompt)
    conv_template.append_message(conv_template.roles[1], None)

    example["message"] = conv_template.get_prompt()
    return example


def get_prompt_dict(cases, model_id):
    data_dict = defaultdict(list)
    for idx, case in enumerate(cases):
        data_dict["id"].append(idx)
        for key, value in case.items():
            data_dict[key].append(value)

    dataset = Dataset.from_dict(data_dict)

    processed_dataset = dataset.map(
        partial(
            construct_prompt,
            model_id=model_id,
        ),
        desc="Processing datasets.",
    )

    prompt_dict = {
        index: prompt
        for prompt, index in zip(
            processed_dataset["message"],
            processed_dataset["id"],
        )
    }
    return prompt_dict


def cal_loss_wo_rewrite(
    cases,
    model_id,
    output_name,
    output_result_dir="./result/",
    device="cuda",
):
    prompt_dict = get_prompt_dict(cases, model_id)

    file_path = output_result_dir + output_name + "_ces.json"

    out = []
    model, tokenizer = load_model(model_id, device)

    for idx, sample in tqdm(enumerate(cases)):
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

        with torch.no_grad():
            prompt = prompt_dict[idx]

            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            response_tokens = tokenizer(response, add_special_tokens=False).input_ids

            extracted_text = (
                [tokenizer.bos_token_id]
                + prompt_tokens
                + response_tokens
                + [tokenizer.eos_token_id]
            )

            input_ids = torch.tensor([extracted_text]).to(model.device)
            target_ids = input_ids.clone()
            sep = len(prompt_tokens) + 1
            target_ids[:, :sep] = -100
            outputs2 = model(input_ids, labels=target_ids)

        rslt = cal_individual_score(
            idx,
            outputs1,
            outputs2,
            response_tokens + [tokenizer.eos_token_id],
            len(prompt_tokens),
            num_words=sample.get("num_words", None),
        )
        out.append(rslt)

    with jsonlines.open(
        output_result_dir + output_name + "_ces_id.jsonl", "w"
    ) as writer:
        writer.write_all(out)


def load_model(model_id="/home/yueqi/LLMana/Llama-2-7b-chat-hf", device="cuda"):
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

    cases = []
    with jsonlines.open(args.response_file, "r") as f:
        for obj in f:
            cases.append(obj)

    cal_loss_wo_rewrite(
        cases,
        config["model_name"],
        args.output_name + "_wo_rewrite",
        output_result_dir=args.output_result_dir,
    )
