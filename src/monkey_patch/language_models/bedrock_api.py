import abc
import json

import boto3 as boto3
import openai
import time
# import abstract base class
from monkey_patch.language_models.llm_api_abc import LLM_Api
import os

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
import requests

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
bedrock = boto3.client(
    service_name='bedrock',
    region_name=AWS_REGION,
)
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,

)


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

    def generate(self, model, system_message, prompt, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset
        """

        # check if api key is not none
        content_type = 'application/json'

        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        frequency_penalty = kwargs.get("frequency_penalty", 0)
        presence_penalty = kwargs.get("presence_penalty", 0)
        body = json.dumps({
            "prompt": AnthropicClaudePromptTemplate().prompt_gen(system_prompt=system_message, user_prompt=prompt),
            "max_tokens_to_sample": 4096,
            "temperature": temperature,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [
                "\\n\\nHuman:"
            ],
            "anthropic_version": "bedrock-2023-05-31"
        })

        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model,
            contentType="application/json",
            accept="application/json")
        return json.loads(response.get('body').read().decode())['completion']

        params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 512,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        params["messages"] = messages

        counter = 0
        choice = None
        # initiate response so exception logic doesnt error out when checking for error in response
        response = {}
        while counter < 5:
            try:
                openai_headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                response = requests.post(
                    OPENAI_URL, headers=openai_headers, json=params, timeout=50
                )
                response = response.json()
                choice = response["choices"][0]["message"]["content"].strip("'")
                break
            except Exception:
                if ("error" in response and
                        "code" in response["error"] and
                        response["error"]["code"] == 'invalid_api_key'):
                    raise Exception(f"The supplied OpenAI API key {self.api_key} is invalid")

                time.sleep(1 + 3 * counter)
                counter += 1
                continue

        if not choice:
            raise Exception("OpenAI API failed to generate a response")

        return choice
