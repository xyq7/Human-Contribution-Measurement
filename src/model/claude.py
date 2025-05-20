from typing import Any, Callable, Tuple

import time
import anthropic

from tqdm import tqdm
from anthropic import (
    RateLimitError,
    BadRequestError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    APIError,
)

from accelerate.logging import get_logger

from .base import BaseModel

__all__ = ["CLAUDEModel", "CLAUDE"]

logger = get_logger(__name__)


class CLAUDEModel(BaseModel):
    def __init__(self, *, config: str | dict = None, **kwargs):
        self.config = self.load_config(config)
        self.client = anthropic.Anthropic(
            api_key=self.config.get("api_key", None),
        )
        self.system_prompt = "You are a helpful assistant."
        self.max_retry = self.config.get("max_retry", 10)

    def chat_completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        success = False
        retry_time = 0
        while not success and retry_time < self.max_retry:
            try:
                response = self.client.messages.create(
                    model=self.config.get("model", None),
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=self.system_prompt,
                )
                success = True
            except RateLimitError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except APITimeoutError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except APIConnectionError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except InternalServerError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except BadRequestError as e:
                logger.warning(e, exc_info=True)
                success = True
                response = {"content": []}
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True
                response = {"content": []}
            retry_time += 1
        try:
            rslts = [t.text for t in response.content]
        except Exception as e:
            logger.warning(e, exc_info=True)
            rslts = []

        return rslts

    def completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"],
    ):
        success = False
        while not success:
            try:
                response = self.client.messages.create(
                    model=self.config.get("model", None),
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop,
                )
                success = True
            except RateLimitError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except APITimeoutError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except APIConnectionError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except InternalServerError as e:
                logger.warning(e, exc_info=True)
                time.sleep(1)
            except BadRequestError as e:
                logger.warning(e, exc_info=True)
                success = True
                response = {"content": []}
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True
                response = {"content": []}
        try:
            rslts = [i.text for i in response.content]
        except Exception as e:
            logger.warning(e, exc_info=True)
            rslts = []

        return rslts

    def generate(self, data: Any, **kwargs):
        temperature = kwargs.pop("temperature", 0)
        rslts = []
        for message in tqdm(data["message"]):
            if self.config["chat"]:
                rslt = self.chat_completion(message, temperature=temperature)
            else:
                rslt = self.completion(message, temperature=temperature)
            rslts.extend(rslt)
        return rslts


class CLAUDE(CLAUDEModel):
    require_system_prompt = True

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
        system_prompt, user_prompt = prompt_construct_fn(example)
        self.system_prompt = system_prompt

        if self.config["chat"]:
            message = [
                {"role": "user", "content": user_prompt},
            ]
            example["message"] = message
        else:
            system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
            user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format(
                "user", user_prompt
            )

            message = system_message + user_message + "\n<|im_start|>assistant\n"
            example["message"] = message
        return example
