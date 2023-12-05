from typing import List
import abc
import botocore
import openai
import time
# import abstract base class
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.fine_tuning import FineTuningJob
import boto3 as boto3
from tanuki.language_models.llm_finetune_api_abc import LLM_Finetune_API
from tanuki.models.embedding import Embedding
from tanuki.language_models.embedding_api_abc import Embedding_API
from tanuki.language_models.llm_api_abc import LLM_API
import os
import json
from tanuki.models.finetune_job import FinetuneJob
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
import requests


class Bedrock_API(LLM_API, LLM_Finetune_API):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )

    def generate(self, model: BaseModelConfig, system_message: str, prompt: str, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type,
         generate a response and check if the datapoint can be saved to the finetune dataset
        """

        # this needs to be done generally better, introduce the LLM_gen params class
        # so you can config it at the start
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        top_k = kwargs.get("top_k", 250)
        max_tokens_to_sample = kwargs.get("max_tokens_to_sample", 4096)
        chat_prompt = model.chat_template
        if chat_prompt is None:
            raise Exception("Chat prompt is not defined for this model"\
                            "Please define it in the model config")
        final_prompt = chat_prompt.format(system_message=system_message, user_prompt=prompt)
        body = json.dumps({
            "prompt": final_prompt,
            "max_tokens_to_sample": max_tokens_to_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": [
                "\\n\\nHuman:"
            ],
            "anthropic_version": "bedrock-2023-05-31"
        })

        counter = 0
        choice = None
        # initiate response so exception logic doesnt error out when checking for error in response
        response = {}
        while counter <= 5:
            try:
                response = self.bedrock_runtime.invoke_model(body=body,
                                                             modelId=model.model_name,
                                                             contentType="application/json",
                                                             accept="application/json")
                choice = json.loads(response.get('body').read().decode())['completion']
                break
            except botocore.exceptions.ClientError as error:
                raise Exception("boto3 had an error: " + error.response['Error']['Message'])
            except Exception as e:
                if counter == 5:
                    raise Exception(f"AWS Bedrock API failed to generate a response: {e}")
                counter += 1
                time.sleep(2 ** counter)
                continue

        if not choice:
            raise Exception("AWS Bedrock API failed to generate a response")

        return choice

