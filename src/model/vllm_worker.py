from typing import Any, Callable, Tuple, List, Optional, Union

import torch
from transformers import AutoTokenizer

import fastchat.model
from vllm import LLM, SamplingParams
from .base import BaseModel
from .utils import check_bf16_support

__all__ = ["vLLMModel", "Mixtral"]


class vLLMModel(BaseModel):
    def __init__(self, *, config: str = None, **kwargs):
        self.kwargs = kwargs
        self.config = self.load_config(config)

        # self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

        self.generation_config = self.load_generation_config()

    def get_conv_template(self):
        conv_template = fastchat.model.get_conversation_template(
            self.config.get("template_name", self.config["model_name"])
        )
        return conv_template

    def load_model(self):
        tensor_parallel_size = self.kwargs.get("tensor_parallel_size", 1)
        download_dir = self.kwargs.get("download_dir", None)
        if check_bf16_support():
            dtype = "bfloat16"
        else:
            dtype = "float16"

        self.model = LLM(
            model=self.config["model_name"],
            trust_remote_code=self.config.get("trust_remote_code", False),
            tensor_parallel_size=tensor_parallel_size,
            download_dir=download_dir,
            dtype=dtype,
        )
        return self.model

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=False,
            token=self.config.get("auth_token", None),
            trust_remote_code=self.config.get("trust_remote_code", False),
        )
        self.tokenizer.padding_side = "left"
        return self.tokenizer

    def load_generation_config(self):
        # do_sample is set to False for gready non-probablistic sampling
        conv_template = self.get_conv_template()
        max_new_tokens = self.kwargs.get("max_new_tokens", 2048)
        temperature = self.kwargs.get("temperature", 0)
        self.generation_config = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=conv_template.stop_str,
            stop_token_ids=conv_template.stop_token_ids,
        )
        return self.generation_config

    def generate(self, data):
        sampling_params = self.load_generation_config()
        responses = self.model.generate(data["message"], sampling_params)
        responses = [output.outputs[0].text for output in responses]
        responses = self.post_process(responses)
        return responses

    def post_process(self, responses: List[str]):
        return [i.strip() for i in responses]

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        conv_template = self.get_conv_template()
        if self.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            conv_template.set_system_message(system_prompt)
        else:
            user_prompt = prompt_construct_fn(example)

        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], None)

        example["message"] = conv_template.get_prompt()
        return example


class Mixtral(vLLMModel):
    require_system_prompt = True

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], use_fast=False
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<unk>"
            self.tokenizer.pad_token_id = 0

        self.tokenizer.padding_side = "left"  # Allow batched inference

        return self.tokenizer
