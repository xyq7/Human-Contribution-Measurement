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
        "--output_figure_dir",
        type=str,
        default="./figure/",
        help="The output directory of the figure.",
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
    parser.add_argument(
        "--draw_figure",
        action="store_true",
        default=False,
        help="Whether to draw the figure.",
    )
    parser.add_argument(
        "--num_words_list",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500],
    )

    args = parser.parse_args()
    return args


def cal_individual_score(id, outputs1, outputs2, target_ids, prompt_length, num_words=None):
    """
    Calculate the human contribution score for individual responses
    Args:
        outputs1: Model outputs for response-only input
        outputs2: Model outputs for prompt+response input
        target_ids: Target token IDs
        prompt_length: Length of the prompt in tokens
        num_words: Optional word count
    Returns:
        Dictionary containing entropy scores, cross-entropy scores, and human contribution percentages
    """
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
        "num_words": num_words
    }
    return rslt


def construct_prompt(example, mode, dataset_name, model_id, num_words):
    """
    Construct a conversation prompt using FastChat template
    Args:
        example: Dictionary containing prompt and response
        mode: Mode to generate response
        dataset_name: the name of the dataset
        model_id: ID of the language model
        num_words: the number of words required in generation promp
    Returns:
        Example with added message field containing formatted prompt
    """
    conv_template = fastchat.model.get_conversation_template(model_id)

    system_prompt = "You are a helpful assistant."
    user_prompt_dict = USER_PROMPT_DICT[dataset_name]
    user_prompt_template, key = user_prompt_dict[mode]

    user_prompt = user_prompt_template.format(example[key], num_words)
    conv_template.set_system_message(system_prompt)
    conv_template.append_message(conv_template.roles[0], user_prompt)
    conv_template.append_message(conv_template.roles[1], None)

    example["message"] = conv_template.get_prompt()
    return example


def get_prompt_dict(file_name, model_id, num_words_list):
    """
    Create a dictionary mapping indices to processed prompts
    Args:
        cases: List of prompt-response pairs
        model_id: ID of the language model
        num_words_list: List of number of words required for generation and evaluation
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
    for mode in user_prompt_dict:
        for num_words in num_words_list:
            key = f"{mode}-{num_words}"
            datasets[key] = dataset.map(
                partial(
                    construct_prompt,
                    mode=mode,
                    dataset_name=args.dataset_name,
                    model_id=model_id,
                    num_words=num_words
                ),
                desc="Processing datasets.",
            )

            def set_mode(example, key):
                example["mode"] = key
                return example

            def set_num_words(example, num_words):
                example["num_words"] = num_words
                return example

            datasets[key] = datasets[key].map(partial(set_mode, key=mode))
            datasets[key] = datasets[key].map(partial(set_num_words, num_words=num_words))

    processed_dataset = concatenate_datasets(list(datasets.values()))
    prompt_dict = {
        f"{mode}-{index}-{num_words}": prompt
        for prompt, mode, index, num_words in zip(
            processed_dataset["message"],
            processed_dataset["mode"],
            processed_dataset["id"],
            processed_dataset["num_words"],
        )
    }
    return prompt_dict


def cal_loss_wo_rewrite(
    response_file,
    dataset_file,
    model_id,
    output_name,
    num_samples=None,
    output_figure_dir="./figure/",
    output_result_dir="./result/",
    device="cuda",
    resume=False,
    draw_figure=False,
    num_words_list = None
):
    """
    Calculate loss and human contribution scores without rewriting
    Args:
        response_file: The file with generated responses (generated by generation_vary_lens.py)
        dataset_file: The dataset file
        model_id: ID of the language model
        output_name: Name for output files
        num_samples: Number of samples for evaluation
        output_result_dir: Directory for saving results
        device: Computing device (cuda/cpu)
        resume: Whether to resume from previous results
        draw_figure: Whether to draw figures with the results
        num_words_list: List of number of words required for generation and evaluation
    Saves results to JSON and JSONL files
    """
    
    # prompt_dict = get_prompt_dict(dataset_file, model_id)
    prompt_dict = get_prompt_dict(dataset_file, model_id, num_words_list)

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
                    prompt = prompt_dict["{}-{}-{}".format(mode, sample["id"], sample["num_words"])]

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
                    num_words=sample.get("num_words", None)
                )

                ces[mode].append(rslt)
                out.append(rslt)

        with open(output_result_dir + output_name + "_ces.json", "w") as json_file:
            json.dump(ces, json_file)
        with jsonlines.open(
            output_result_dir + output_name + "_ces_id.json", "w"
        ) as writer:
            writer.write_all(out)

    if not draw_figure:
        return

    if "subject" not in ces:
        data = [ori["original"], ces["polish"], ces["summary"], ces["gen"]]
        names_fig = ["Original", "Polish", "Generate\nw/ Summary", "Generate\nw/ Title"]
        names = ["Original", "Polish", "Gen-Summary", "Gen-Title"]
        names_fig2 = [
            "Original",
            "Polish",
            "Polish\nAttack1",
            "Polish\nAttack2",
            "Generate\nw/ Summary",
            "Generate\nw/ Summary\nAttack1",
            "Generate\nw/ Summary\nAttack2",
            "Generate\nw/ Title",
            "Generate\nw/ Title\nAttack1",
            "Generate\nw/ Title\nAttack2",
        ]
    else:
        data = [
            ori["original"],
            ces["polish"],
            ces["summary"],
            ces["gen"],
            ces["subject"],
        ]
        names_fig = [
            "Original",
            "Polish",
            "Generate\nw/ Summary",
            "Generate\nw/ Title",
            "Generate\nw/ Subject",
        ]
        names = ["Original", "Polish", "Gen-Summary", "Gen-Title", "Gen-Subject"]
        names_fig2 = [
            "Original",
            "Polish",
            "Polish\nAttack1",
            "Polish\nAttack2",
            "Generate\nw/ Summary",
            "Generate\nw/ Summary\nAttack1",
            "Generate\nw/ Summary\nAttack2",
            "Generate\nw/ Title",
            "Generate\nw/ Title\nAttack1",
            "Generate\nw/ Title\nAttack2",
            "Generate\nw/ Subject",
            "Generate\nw/ Subject\nAttack1",
            "Generate\nw/ Subject\nAttack2",
        ]

    plt.figure(figsize=(6, 4.5))

    color_palette = sns.color_palette("turbo", len(data))
    ax = sns.violinplot(
        data=data,
        palette=color_palette,
        inner=None,
        scale="width",
        linewidth=0,
        saturation=0.4,
    )
    sns.boxplot(
        data=data,
        palette=color_palette,
        width=0.2,
        flierprops={"marker": "x"},
        boxprops={"zorder": 2, "linewidth": 1},
        whiskerprops={"linewidth": 1, "linestyle": "dotted"},
        medianprops={"linestyle": "--", "linewidth": 1},
        ax=ax,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = [], []

    boxes = [
        plt.Rectangle(
            (0, 0), 1, 1, fc=color_palette[i], label="Box", zorder=2, edgecolor="black"
        )
        for i in range(len(data))
    ]
    centerline = plt.Line2D([0], [0], color="black", linestyle="--", label="Centerline")
    limits = plt.Line2D([0], [0], color="black", linestyle="-", label="Limits")
    whiskers = plt.Line2D([0], [0], color="black", linestyle="dotted", label="Whiskers")
    points = plt.Line2D(
        [0], [0], marker="x", markersize=5, color="black", linestyle="", label="Points"
    )

    # Append to handles and labels lists'Original',
    handles.extend(boxes + [centerline, limits, whiskers, points])
    labels.extend(
        ["{} (n={})".format(name, len(data[i])) for i, name in enumerate(names)]
    )
    # legend = ax.legend(loc='upper center', ncol=2, handles=handles, labels=labels, bbox_to_anchor=(0.5, 1.3))
    plt.xticks(range(len(data)), names_fig)
    # plt.xlabel("ASR", fontsize=13)
    plt.ylim(0, 5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_figure_dir + output_name + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(output_figure_dir + output_name + ".pdf", dpi=300, bbox_inches="tight")
    plt.clf()

    if "subject" not in ces:
        data = [
            ori["original"],
            ces["polish"],
            ces["polish_ada1"],
            ces["polish_ada2"],
            ces["summary"],
            ces["summary_ada1"],
            ces["summary_ada2"],
            ces["gen"],
            ces["gen_ada1"],
            ces["gen_ada2"],
        ]
    else:
        data = [
            ori["original"],
            ces["polish"],
            ces["polish_ada1"],
            ces["polish_ada2"],
            ces["summary"],
            ces["summary_ada1"],
            ces["summary_ada2"],
            ces["gen"],
            ces["gen_ada1"],
            ces["gen_ada2"],
            ces["subject"],
            ces["subject_ada1"],
            ces["subject_ada2"],
        ]

    plt.figure(figsize=(16, 4.5))

    color_palette = sns.color_palette("turbo", len(data))
    ax = sns.violinplot(
        data=data,
        palette=color_palette,
        inner=None,
        scale="width",
        linewidth=0,
        saturation=0.4,
    )
    sns.boxplot(
        data=data,
        palette=color_palette,
        width=0.2,
        flierprops={"marker": "x"},
        boxprops={"zorder": 2, "linewidth": 1},
        whiskerprops={"linewidth": 1, "linestyle": "dotted"},
        medianprops={"linestyle": "--", "linewidth": 1},
        ax=ax,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = [], []

    boxes = [
        plt.Rectangle(
            (0, 0), 1, 1, fc=color_palette[i], label="Box", zorder=2, edgecolor="black"
        )
        for i in range(len(data))
    ]
    centerline = plt.Line2D([0], [0], color="black", linestyle="--", label="Centerline")
    limits = plt.Line2D([0], [0], color="black", linestyle="-", label="Limits")
    whiskers = plt.Line2D([0], [0], color="black", linestyle="dotted", label="Whiskers")
    points = plt.Line2D(
        [0], [0], marker="x", markersize=5, color="black", linestyle="", label="Points"
    )

    # Append to handles and labels lists'Original',
    handles.extend(boxes + [centerline, limits, whiskers, points])
    # names = ['Original', 'Polish', 'Gen-Summary', 'Gen-Title', 'Gen-Title-Ada1','Gen-Title-Ada2', 'Gen-Subject','Gen-Subject-Ada1','Gen-Subject-Ada2']
    # labels.extend(["{} (n={})".format(name,len(data[i])) for i, name in enumerate(names)] )
    # legend = ax.legend(loc='upper center', ncol=2, handles=handles, labels=labels, bbox_to_anchor=(0.5, 1.3))
    plt.xticks(range(len(data)), names_fig2)
    # plt.xlabel("ASR", fontsize=13)
    plt.ylim(0, 5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(
        output_figure_dir + output_name + "ada.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        output_figure_dir + output_name + "ada.pdf", dpi=300, bbox_inches="tight"
    )
    plt.clf()


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

    Path(args.output_figure_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_result_dir).mkdir(parents=True, exist_ok=True)

    cal_loss_wo_rewrite(
        args.response_file,
        args.dataset_file,
        config["model_name"],
        args.dataset_name,
        args.output_name + "_wo_rewrite",
        num_samples=args.num_samples,
        output_figure_dir=args.output_figure_dir,
        output_result_dir=args.output_result_dir,
        resume=args.resume,
        num_words_list = args.num_words_list
    )