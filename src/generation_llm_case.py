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

from datasets import Dataset, concatenate_datasets
from accelerate import Accelerator

from model import AutoLLM
from config import MAX_RETRY


def construct_prompt(example):
    system_prompt = "You are a helpful assistant."
    return system_prompt, example["prompt"]


def check_success(response, rprefix):
    return response.rfind(rprefix) != -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
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
    args = parser.parse_args()

    accelerator = Accelerator()
   

    llm = AutoLLM.from_name(args.llm_config_file)(
        config=args.llm_config_file,
        accelerator=accelerator,
        tensor_parallel_size=args.tensor_parallel_size,
        download_dir=args.download_dir,
        temperature=0.7,
        max_new_tokens=2048,
    )

    dataset = Dataset.from_dict(
        {
            "prompt": [
                "Write a poem for Doug Burger",
                "Doug Burger is a beloved technology leader in Microsoft Research. He owns deep insight in many AI problems. He firmly supports MSRA. MSRA members all respect him very much. Please write a poem for Doug Burger based on above information.",
            ]
        }
    )

    with accelerator.main_process_first():
        
        processed_dataset = dataset.map(
            partial(
                llm.process_fn,
                prompt_construct_fn=construct_prompt,
            ),
            # remove_columns=DATA_INFO[args.dataset_name],
            desc="Processing datasets.",
        )


    if args.output_file:
        output_file = Path(args.output_file)
        out = []

        output_file.parent.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        retry_time = 0
        last_dataset = processed_dataset
       
        dataset = last_dataset
        responses = llm.generate(dataset)

        last_dataset = defaultdict(list)
        for response, message in zip(
            responses,
            dataset["message"],
        ):
            out.append(
                {
                    "response": response,
                    "message": message,
                }
            )
                
           
    if args.output_file:
        with jsonlines.open(args.output_file, "w") as writer:
            writer.write_all(out)
