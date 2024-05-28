import yaml
from pathlib import Path
from collections import OrderedDict
from accelerate.logging import get_logger

from .gpt import GPT35, GPT4, GPT35WOSystem, GPT4WOSystem
from .llama import (
    Vicuna,
    Llama2,
)
from .vllm_worker import Mixtral

logger = get_logger(__name__)

LLM_NAME_TO_CLASS = OrderedDict(
    [
        ("vicuna", Vicuna),
        ("llama2", Llama2),
        ("mixtral", Mixtral),
        ("gpt35", GPT35),
        ("gpt35_wosys", GPT35WOSystem),
    ]
)


class AutoLLM:
    @classmethod
    def from_name(cls, name: str):
        if name in LLM_NAME_TO_CLASS:
            name = name
        elif Path(name).exists():
            with open(name, "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
            if "llm_name" not in config:
                raise ValueError("llm_name not in config.")
            name = config["llm_name"]
        else:
            raise ValueError(
                f"Invalid name {name}. AutoLLM.from_name needs llm name or llm config as inputs."
            )

        logger.info(f"Load {name} from name.")

        llm_cls = LLM_NAME_TO_CLASS[name]
        return llm_cls
