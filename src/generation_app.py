from typing import List, Dict
import json
import jsonlines

import argparse

from collections import defaultdict
from functools import partial
from tqdm import tqdm
from pathlib import Path

import torch

from datasets import Dataset
from accelerate import Accelerator

from model import AutoLLM


def construct_prompt(example, mode, dataset_name):
    """
    Constructs the prompt for LLM by combining system prompt and user prompt
    Args:
        example: The input example containing text data
        mode: The generation mode (summary/polish/gen/subject)
        dataset_name: Name of the dataset being processed
    Returns:
        system_prompt and formatted user prompt
    """

    system_prompt = "You are a helpful assistant."
    return system_prompt, example["prompt"]


def check_success(response, rprefix):
    """
    Checks if the LLM response contains the expected prefix
    Args:
        response: The LLM generated response
        rprefix: Expected prefix to check for
    Returns:
        Boolean indicating if the response contains the prefix
    """
    if not isinstance(response, str):
        return False
    return response.rfind(rprefix) != -1


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["assisting_creative", "editing_rewriting"],
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
        "--temperature", type=float, default=0.7, help="The temperature for generation"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    accelerator = Accelerator()

    # Initialize the LLM model with specified configuration
    llm = AutoLLM.from_name(args.llm_config_file)(
        config=args.llm_config_file,
        accelerator=accelerator,
        tensor_parallel_size=args.tensor_parallel_size,
        download_dir=args.download_dir,
        temperature=args.temperature,
        max_new_tokens=2048,
    )

    with jsonlines.open(args.input_file) as reader:
        data = [example for example in reader]

    dataset = Dataset.from_list(data)

    # Process datasets for each specified mode
    with accelerator.main_process_first():

        processed_dataset = dataset.map(
            partial(llm.process_fn, prompt_construct_fn=construct_prompt),
            desc="Processing datasets.",
        )

    if args.num_examples is not None and len(processed_dataset) >= args.num_examples:
        processed_dataset = processed_dataset.select(range(args.num_examples))

    if args.output_file:
        output_file = Path(args.output_file)
        out = []

        output_file.parent.mkdir(exist_ok=True, parents=True)

    print("----------- Prompt example --------------")
    print(processed_dataset[0]["message"])
    print("----------- Prompt example --------------")

    # Main generation loop
    with torch.no_grad():
        responses = llm.generate(processed_dataset)

        for response, data in zip(responses, processed_dataset):
            out.append(
                {
                    "prompt": data["prompt"],
                    "response": response,
                    "message": data["message"],
                    "label": data["label"],
                    "model": llm.config["model_name"],
                }
            )

    if args.output_file:
        with jsonlines.open(args.output_file, "w") as writer:
            writer.write_all(out)
