import os
from typing import List, Dict
import json
import jsonlines

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

def construct_prompt_mode(example, dataset_name):
    system_prompt = "You are a helpful assistant."
    user_prompt_dict = USER_PROMPT_DICT[dataset_name]
    mode = example["mode"]
    user_prompt_template, key = user_prompt_dict[mode]
    abstract_len = len(example["abstract"].split())
    return system_prompt, user_prompt_template.format(example[key], abstract_len)


def construct_prompt(example, mode, dataset_name):
    system_prompt = "You are a helpful assistant."
    user_prompt_dict = USER_PROMPT_DICT[dataset_name]
    user_prompt_template, key = user_prompt_dict[mode]
    abstract_len = len(example["abstract"].split())
    return system_prompt, user_prompt_template.format(example[key], abstract_len)

def process_dataset_jsonlines(file_name, num_examples=None):
    data_dict = defaultdict(list)
    
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            if num_examples is not None and len(data_dict["id"]) >= num_examples:
                break
            for key, value in obj.items():
                data_dict[key].append(value)
                
    dataset = Dataset.from_dict(data_dict)
    return dataset



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

    if "subject" in data_dict:
        subjects = sorted(list(set(dataset["subject"])))
        subject_dataset = Dataset.from_dict(
            {
                "id": subjects,
                "subject": subjects,
            }
        )
    else:
        subject_dataset = None
    return dataset, subject_dataset


def check_success(response, rprefix):
    return (
        len(response) > 100 and
        response.rfind(rprefix) != -1 and
        response.rfind("[End]") != -1
    )

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
            "subject_ada1",
            "subject_ada2"
        ],
        default=["summary", "polish", "gen", "subject"],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default= "news",
        choices=["news", "paper", "patent", "poem"],
        help="The dataset name. (e.g. paper)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/yueqi/Human-Contribution-Measurement/data_new/news.json",
        help="The input data file (a json file).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/yueqi/Human-Contribution-Measurement/output_s3/llama3_8b_news_1_2.jsonl",
        help="The input data file (a json file).",
    )
    parser.add_argument(
        "--output_file_2",
        type=str,
        default="/home/yueqi/Human-Contribution-Measurement/output_s4/llama3_8b_news_1_2.jsonl",
        help="The input data file (a json file).",
    )
    parser.add_argument(
        "--output_file_3",
        type=str,
        default="/home/yueqi/Human-Contribution-Measurement/output_s4/llama3_8b_news_1_3.jsonl",
        help="The input data file (a json file).",
    )
    parser.add_argument(
        "--llm_config_file",
        type=str,
        default="./config/llama3_8b.yaml",
        help="The config file of testing large language models (LLMs).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
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
        default=1000,
        help="The number of examples to process.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="The batch size for inference."
    )
    parser.add_argument(
        "--scenario", type=int, default=4, help="The generation scenario."
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

    dataset, subject_dataset = process_dataset(args.input_file, args.num_examples)

    datasets = {}
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

    def replace_last_start_with_news(user_prompt):
        import re
        pattern = r'Start with (["\'])News:\1\.'
        matches = list(re.finditer(pattern, user_prompt))
        if not matches:
            return user_prompt

        last_match = matches[-1]
        quote = last_match.group(1)
        start, end = last_match.span()

        replacement = f"Start with {quote}News:{quote} and end with {quote}[End]{quote}."

        return user_prompt[:start] + replacement + user_prompt[end:]
    
    # Define the prompt for each operation. 
    
    def process_fn_r1(
        example,
        prompt_construct_fn
    ):
        conv_template = llm.get_conv_template()
        if llm.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            conv_template.set_system_message(system_prompt)
        else:
            user_prompt = prompt_construct_fn(example)
        user_prompt = replace_last_start_with_news(user_prompt)
        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], None)

        example["message"] = conv_template.get_prompt()
        return example
    
    def process_fn_r2_grammar(
        example,
        prompt_construct_fn
    ) :
        conv_template = llm.get_conv_template()
        if llm.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            conv_template.set_system_message(system_prompt)
        else:
            user_prompt = prompt_construct_fn(example)

        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], example["response1"])

        result = f"Please only correct grammar errors (if any) and regenerate a version. Start with 'News:' and end with '[End]'."
        conv_template.append_message(conv_template.roles[0], result )
        conv_template.append_message(conv_template.roles[1], None)
        example["prompt2"] = result
        example["message"] = conv_template.get_prompt()
        return example

    def process_fn_r2_detail(
        example,
        prompt_construct_fn
    ) :
        conv_template = llm.get_conv_template()
        if llm.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            conv_template.set_system_message(system_prompt)
        else:
            user_prompt = prompt_construct_fn(example)

        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], example["response1"])

        result = f"Please provide more additional details and regenerate an expanded version. Start with 'News:' and end with '[End]'."
        # print(result)
        conv_template.append_message(conv_template.roles[0], result )
        conv_template.append_message(conv_template.roles[1], None)
        example["prompt2"] = result
        example["message"] = conv_template.get_prompt()
        # import pdb; pdb.set_trace()
        return example
    
    def process_fn_r2_hall(
        example,
        prompt_construct_fn
    ) :
        conv_template = llm.get_conv_template()
        if llm.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            conv_template.set_system_message(system_prompt)
        else:
            user_prompt = prompt_construct_fn(example)

        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], example["response1"])

        result = f"Please only correct hallucinations (if any) and regenerate a version. Start with 'News:' and end with '[End]'."
        # print(result)
        conv_template.append_message(conv_template.roles[0], result )
        conv_template.append_message(conv_template.roles[1], None)
        example["prompt2"] = result
        example["message"] = conv_template.get_prompt()
        # import pdb; pdb.set_trace()
        return example
    
    def process_fn_r3_detail(
        example,
        prompt_construct_fn,
    ):
        conv_template = llm.get_conv_template()
        if llm.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            conv_template.set_system_message(system_prompt)
        else:
            user_prompt = prompt_construct_fn(example)

        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], example["response1"])
        conv_template.append_message(conv_template.roles[0], example["prompt2"])
        conv_template.append_message(conv_template.roles[1], example["response2"])
        
        result = f"Please provide more additional details and regenerate an expanded version. Start with 'News:' and end with '[End]'."
        # print(result)
        conv_template.append_message(conv_template.roles[0], result )
        conv_template.append_message(conv_template.roles[1], None)
        example["prompt3"] = result
        example["message"] = conv_template.get_prompt()
        # import pdb; pdb.set_trace()
        return example
      
    def process_fn_r3_grammar(
        example,
        prompt_construct_fn,
    ):
        conv_template = llm.get_conv_template()
        if llm.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            conv_template.set_system_message(system_prompt)
        else:
            user_prompt = prompt_construct_fn(example)

        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], example["response1"])
        conv_template.append_message(conv_template.roles[0], example["prompt2"])
        conv_template.append_message(conv_template.roles[1], example["response2"])
        
        result = f"Please only correct grammar errors (if any) and regenerate a version. Start with 'News:' and end with '[End]'."
        # print(result) 
        conv_template.append_message(conv_template.roles[0], result )
        conv_template.append_message(conv_template.roles[1], None)
        example["prompt3"] = result
        example["message"] = conv_template.get_prompt()
        # import pdb; pdb.set_trace()
        return example
    
    if args.scenario == 1:
        process_fn_r2 = process_fn_r2_detail
        process_fn_r3 = process_fn_r3_detail
    elif args.scenario == 2:
        process_fn_r2 = process_fn_r2_hall
        process_fn_r3 = process_fn_r3_detail
    elif args.scenario == 3:
        process_fn_r2 = process_fn_r2_hall
        process_fn_r3 = process_fn_r3_grammar
    elif args.scenario == 4:
        process_fn_r2 = process_fn_r2_grammar
        process_fn_r3 = process_fn_r3_detail
        
    # Round 1

    with accelerator.main_process_first():
        for key in args.modes:
            if key not in user_prompt_dict:
                print(
                    f"Mode {key} is not supported. Please choose from {user_prompt_dict.keys()}."
                )
                continue
            datasets[key] = dataset.map(
                partial(
                    process_fn_r1,
                    prompt_construct_fn=partial(
                        construct_prompt,
                        mode=key,
                        dataset_name=args.dataset_name,
                    ),
                ),
                # remove_columns=DATA_INFO[args.dataset_name],
                desc="Processing datasets.",
            )

        def set_mode(example, key):
            example["mode"] = key
            return example

        for key in datasets:
            datasets[key] = datasets[key].map(partial(set_mode, key=key))

    processed_dataset = concatenate_datasets(list(datasets.values()))

    if args.output_file:
        output_file = Path(args.output_file)
        out = []
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
                for id_name, response, message, mode, title, abstract, summary, subject in zip(
                    dataset["id"],
                    responses,
                    dataset["message"],
                    dataset["mode"],
                    dataset["title"],
                    dataset["abstract"],
                    dataset["summary"],
                    dataset["subject"]
                ):
                    if check_success(response, rprefix):
                        out.append(
                            {
                                "id": id_name,
                                "response1": response,
                                "message1": message,
                                "mode": mode,
                                "title": title,
                                "abstract": abstract,
                                "summary": summary,
                                "subject": subject
                            }
                        )
                    else:
                        last_dataset["id"].append(id_name)
                        last_dataset["message"].append(message)
                        last_dataset["mode"].append(mode)
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

    # Round 2

    dataset_round2 = process_dataset_jsonlines(args.output_file, args.num_examples)
    with accelerator.main_process_first():

        processed_dataset_2 = dataset_round2.map(                
            partial(
            process_fn_r2,
            prompt_construct_fn=partial(
                construct_prompt_mode,
                dataset_name=args.dataset_name,
            ),
        ),
        # remove_columns=DATA_INFO[args.dataset_name],
        desc="Processing datasets.",
    )
        def set_mode(example, key):
            example["mode"] = key
            return example


    if args.output_file_2:
        output_file_2 = Path(args.output_file_2)
        out = []
        output_file_2.parent.mkdir(exist_ok=True, parents=True)

    print("----------- Prompt example --------------")
    print(processed_dataset_2[0]["message"])
    print("----------- Prompt example --------------")
    # import pdb; pdb.set_trace()
    dataloader = DataLoader(
        processed_dataset_2, batch_size=args.batch_size, collate_fn=DefaultDataCollator()
    )

    with torch.no_grad():
        for data in tqdm(dataloader):
            retry_time = 0
            last_dataset = data
            while retry_time < MAX_RETRY:
                dataset = last_dataset
                responses = llm.generate(dataset)

                last_dataset = defaultdict(list)
                for id_name, response, message, mode, prompt2, response1, message1, title, abstract, summary, subject in zip(
                    dataset["id"],
                    responses,
                    dataset["message"],
                    dataset["mode"],
                    dataset["prompt2"],
                    dataset["response1"],
                    dataset["message1"],
                    dataset["title"],
                    dataset["abstract"],
                    dataset["summary"],
                    dataset["subject"]
                ):
                    if check_success(response, rprefix):
                        out.append(
                            {
                                "id": id_name,
                                "response2": response,
                                "message2": message,
                                "prompt2": prompt2,
                                "response1": response1,
                                "message1": message1,
                                "mode": mode,
                                "title": title,
                                "abstract": abstract,
                                "summary": summary,
                                "subject": subject
                            }
                        )
                    else:
                        last_dataset["id"].append(id_name)
                        last_dataset["message"].append(message)
                        last_dataset["mode"].append(mode)
                if len(last_dataset["id"]) == 0:
                    break
                else:
                    retry_time += 1
                    print(f"Retry time: {retry_time}, Size: {len(last_dataset['id'])}")

            if args.output_file_2:
                with jsonlines.open(args.output_file_2, "w") as writer:
                    writer.write_all(out)

    if args.output_file_2:
        with jsonlines.open(args.output_file_2, "w") as writer:
            writer.write_all(out)



    #Round 3

    dataset_round3 = process_dataset_jsonlines(args.output_file_2, args.num_examples)
    with accelerator.main_process_first():
        processed_dataset_3 = dataset_round3.map(                
            partial(
            process_fn_r3,
            prompt_construct_fn=partial(
                construct_prompt_mode,
                dataset_name=args.dataset_name,
            ),
        ),
        # remove_columns=DATA_INFO[args.dataset_name],
        desc="Processing datasets.",
    )

    if args.output_file_3:
        output_file_3 = Path(args.output_file_3)
        out = []
        output_file_3.parent.mkdir(exist_ok=True, parents=True)

    print("----------- Prompt example --------------")
    print(processed_dataset_3[0]["message"])
    print("----------- Prompt example --------------")

    dataloader = DataLoader(
        processed_dataset_3, batch_size=args.batch_size, collate_fn=DefaultDataCollator()
    )

    with torch.no_grad():
        for data in tqdm(dataloader):
            retry_time = 0
            last_dataset = data
            while retry_time < MAX_RETRY:
                dataset = last_dataset
                responses = llm.generate(dataset)

                last_dataset = defaultdict(list)
                for id_name, response, message, mode, prompt3,response2, message2, prompt2, response1, message1, title, abstract, summary, subject in zip(
                    dataset["id"],
                    responses,
                    dataset["message"],
                    dataset["mode"],
                    dataset["prompt3"],
                    dataset["response2"],
                    dataset["message2"],
                    dataset["prompt2"],
                    dataset["response1"],
                    dataset["message1"],
                    dataset["title"],
                    dataset["abstract"],
                    dataset["summary"],
                    dataset["subject"]
                ):
                    if check_success(response, rprefix):
                        out.append(
                            {
                                "id": id_name,
                                "response3": response,
                                "message3": message,
                                "prompt3": prompt3,
                                "response2": response2,
                                "message2": message2,
                                "prompt2": prompt2,
                                "response1": response1,
                                "message1": message1,
                                "mode": mode,
                                "title": title,
                                "abstract": abstract,
                                "summary": summary,
                                "subject": subject
                            }
                        )
                    else:
                        last_dataset["id"].append(id_name)
                        last_dataset["message"].append(message)
                        last_dataset["mode"].append(mode)
                if len(last_dataset["id"]) == 0:
                    break
                else:
                    retry_time += 1
                    print(f"Retry time: {retry_time}, Size: {len(last_dataset['id'])}")

            if args.output_file_3:
                with jsonlines.open(args.output_file_3, "w") as writer:
                    writer.write_all(out)

    if args.output_file_3:
        with jsonlines.open(args.output_file_3, "w") as writer:
            writer.write_all(out)
