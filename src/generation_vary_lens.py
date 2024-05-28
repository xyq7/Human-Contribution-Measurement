from typing import List, Dict
import json
import jsonlines
from collections import defaultdict

import argparse

from collections import defaultdict
from functools import partial
from tqdm import tqdm
import copy
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader

from datasets import Dataset, concatenate_datasets
from accelerate import Accelerator

from model import AutoLLM
from config import USER_PROMPT_DICT, MAX_RETRY, RPREFIX_DICT


def construct_prompt(example, mode, dataset_name, nums_words):
    system_prompt = "You are a helpful assistant."
    user_prompt_dict = USER_PROMPT_DICT[dataset_name]
    user_prompt_template, key = user_prompt_dict[mode]
    return system_prompt, user_prompt_template.format(example[key], nums_words)


def process_dataset(file_name, num_examples):
    with open(file_name, "r") as file:
        data = json.load(file)

    data_dict = defaultdict(list)
    for id_name in data:
        if num_examples is not None and len(data_dict["id"]) >= num_examples:
            break
        data_dict["id"].append(id_name)
        sample = data[id_name]
        for key, value in sample.items():
            data_dict[key].append(value)

    dataset = Dataset.from_dict(data_dict)
    return dataset


def check_success(response, rprefix):
    return response.rfind(rprefix) != -1


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=[
            "summary",
            "polish",
            "gen",
            "subject",
            "gen_ada1",
            "gen_ada2",
            "polish_ada1",
            "polish_ada2",
            "summary_ada1",
            "summary_ada2",
        ],
        default=["gen"],
    )
    parser.add_argument(
        "--num_words_list",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["news", "paper", "patent", "poem"],
        help="The dataset name. (e.g. news)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="../data/news_bbc.json",
        help="The input data file (a json file).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../output/a.json",
        help="The input data file (a json file).",
    )
    parser.add_argument(
        "--llm_config_file",
        type=str,
        default="./configs/llm.yaml",
        help="The config file of testing large language models (LLMs).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="The size of the tensor parallelism.",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="The download directory of the LLMs for vllm.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Whether to resume the inference process.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="The number of examples to process.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="The batch size for inference."
    )
    args = parser.parse_args()
    return args


class DefaultDataCollator:
    def __call__(self, batch_examples: List) -> Dict:
        batch_rslt = defaultdict(list)

        for example in batch_examples:
            for key in example:
                batch_rslt[key].append(example[key])

        return batch_rslt


if __name__ == "__main__":
    args = parse_args()
    print(args)

    accelerator = Accelerator()

    dataset = process_dataset(args.input_file, args.num_examples)

    user_prompt_dict = USER_PROMPT_DICT[args.dataset_name]
    rprefix = RPREFIX_DICT[args.dataset_name]

    llm = AutoLLM.from_name(args.llm_config_file)(
        config=args.llm_config_file,
        accelerator=accelerator,
        tensor_parallel_size=args.tensor_parallel_size,
        download_dir=args.download_dir,
        temperature=0.7,
        max_new_tokens=2048,
    )

    datasets = defaultdict(dict)
    with accelerator.main_process_first():
        for key in args.modes:
            for num_words in args.num_words_list:
                if key not in user_prompt_dict:
                    print(
                        f"Mode {key} is not supported. Please choose from {user_prompt_dict.keys()}."
                    )
                    continue

                datasets[key][num_words] = dataset.map(
                    partial(
                        llm.process_fn,
                        prompt_construct_fn=partial(
                            construct_prompt,
                            mode=key,
                            dataset_name=args.dataset_name,
                            nums_words=num_words,
                        ),
                    ),
                    # remove_columns=DATA_INFO[args.dataset_name],
                    desc="Processing datasets.",
                )

                def set_mode_len(example, key, num_words):
                    example["mode"] = key
                    example["num_words"] = num_words
                    return example

                for key in datasets:
                    datasets[key][num_words] = datasets[key][num_words].map(partial(set_mode_len, key=key, num_words=num_words))

        datasets[key] = concatenate_datasets(list(datasets[key].values()))

    processed_dataset = concatenate_datasets(list(datasets.values()))

    if args.output_file:
        output_file = Path(args.output_file)
        out = []
        if output_file.exists() and args.resume:
            needed_messages = Counter(
                [
                    f"{i}_{mode}_{num_words}"
                    for i, mode, num_words in zip(
                        processed_dataset["id"], processed_dataset["mode"], processed_dataset["num_words"]
                    )
                ]
            )

            # read existing results and filter them from datasets
            exist_messages = set()
            with jsonlines.open(output_file, "r") as reader:
                for obj in reader:
                    msg = f"{obj['id']}_{obj['mode']}_{obj['num_words']}"

                    if msg in needed_messages and msg not in exist_messages:
                        out.extend([obj] * needed_messages[msg])
                        exist_messages.add(msg)

            def filter_fn(example):
                return f"{example['id']}_{example['mode']}_{example['num_words']}" not in exist_messages

            with accelerator.main_process_first():
                processed_dataset = processed_dataset.filter(
                    filter_fn, desc="Filter pre-requested messages."
                )

            if len(processed_dataset) == 0:
                print("Already Finished. No need to resume.")

                if args.output_file:
                    with jsonlines.open(args.output_file, "w") as writer:
                        writer.write_all(out)
                exit(0)

            print(f"Need to process {len(processed_dataset)} samples.")
        else:
            output_file.parent.mkdir(exist_ok=True, parents=True)

    print("----------- Prompt example --------------")
    print(processed_dataset[0]["message"])
    print("----------- Prompt example --------------")

    dataloader = DataLoader(
        processed_dataset, batch_size=args.batch_size, collate_fn=DefaultDataCollator()
    )

    with torch.no_grad():
        for data in tqdm(dataloader):
            retry_time = 0
            last_dataset = data
            while retry_time < MAX_RETRY:
                dataset = last_dataset
                responses = llm.generate(dataset)

                last_dataset = defaultdict(list)
                for id_name, response, message, mode, num_words in zip(
                    dataset["id"],
                    responses,
                    dataset["message"],
                    dataset["mode"],
                    dataset["num_words"]
                ):
                    if check_success(response, rprefix):
                        out.append(
                            {
                                "id": id_name,
                                "response": response,
                                "message": message,
                                "mode": mode,
                                "num_words": num_words
                            }
                        )
                    else:
                        last_dataset["id"].append(id_name)
                        last_dataset["message"].append(message)
                        last_dataset["mode"].append(mode)
                        last_dataset["num_words"].append(num_words)
                if len(last_dataset["id"]) == 0:
                    break
                else:
                    retry_time += 1
                    print(f"Retry time: {retry_time}, Size: {len(last_dataset['id'])}")

            if args.output_file:
                with jsonlines.open(args.output_file, "w") as writer:
                    writer.write_all(out)

    if args.output_file:
        with jsonlines.open(args.output_file, "w") as writer:
            writer.write_all(out)
