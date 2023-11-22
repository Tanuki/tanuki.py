import abc
import json

import boto3 as boto3
import botocore
import time

from monkey_patch.exception import MonkeyPatchException
from monkey_patch.language_models.llm_api_abc import LLM_Api
import os

import requests


class PromptTemplate:

    @abc.abstractmethod
    def prompt_gen(self, system_prompt, user_prompt):
        pass


class AnthropicClaudePromptTemplate(PromptTemplate):

    def prompt_gen(self, system_prompt, user_prompt):
        return f"\n\nHuman: {system_prompt}\n\n {user_prompt}\n\nAssistant:\n"


class Bedrock_API(LLM_Api):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )

    def generate(self, model, system_message, prompt, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type,
         generate a response and check if the datapoint can be saved to the finetune dataset
        """

        # check if api key is not none
        content_type = 'application/json'

        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        frequency_penalty = kwargs.get("frequency_penalty", 0)
        presence_penalty = kwargs.get("presence_penalty", 0)
        body = json.dumps({
            "prompt": AnthropicClaudePromptTemplate().prompt_gen(system_prompt=system_message,
                                                                 user_prompt=prompt),
            "max_tokens_to_sample": 4096,
            "temperature": temperature,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [
                "\\n\\nHuman:"
            ],
            "anthropic_version": "bedrock-2023-05-31"
        })

        counter = 0
        while counter < 5:
            try:
                response = self.bedrock_runtime.invoke_model(body=body,
                                                             modelId=model,
                                                             contentType="application/json",
                                                             accept="application/json")
                return json.loads(response.get('body').read().decode())['completion']

            except botocore.exceptions.ClientError as error:
                raise MonkeyPatchException("boto3 ")
            except Exception:
                time.sleep(1 + 3 * counter)
                counter += 1
                continue

        raise MonkeyPatchException(f"Bedrock: Model {model} API failed to generate a response")
