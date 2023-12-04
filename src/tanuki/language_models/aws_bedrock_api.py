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

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
import requests


class PromptTemplate:

    @abc.abstractmethod
    def prompt_gen(self, system_prompt, user_prompt):
        pass


class AnthropicClaudePromptTemplate(PromptTemplate):

    def prompt_gen(self, system_prompt, user_prompt):
        return f"\n\nHuman: {system_prompt}\n\n {user_prompt}\n\nAssistant:\n"


class Bedrock_API(LLM_API, LLM_Finetune_API):
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

        # this needs to be done generally better, introduce the LLM_gen params class
        # so you can config it at the start
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        top_k = kwargs.get("top_k", 250)
        max_tokens_to_sample = kwargs.get("max_tokens_to_sample", 4096)
        body = json.dumps({
            "prompt": AnthropicClaudePromptTemplate().prompt_gen(system_prompt=system_message,
                                                                 user_prompt=prompt),
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
                                                             modelId=model,
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



class OpenAI_API(LLM_API, Embedding_API, LLM_Finetune_API):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()

        self.api_key = os.environ.get("OPENAI_API_KEY")

        self.client = None


    def list_finetuned(self, limit=100, **kwargs) -> List[FinetuneJob]:
        self.check_api_key()
        response = self.client.fine_tuning.jobs.list(limit=limit)
        jobs = []
        for job in response.data:
            jobs.append(FinetuneJob(job.id, job.status, job.fine_tuned_model))

        return jobs

    def get_finetuned(self, job_id):
        self.check_api_key()
        return self.client.fine_tuning.jobs.retrieve(job_id)

    def finetune(self, file, suffix, **kwargs) -> FinetuneJob:
        self.check_api_key()
        # Use the stream as a file
        try:
            response = self.client.files.create(file=file, purpose='fine-tune')
        except Exception as e:
            return

        training_file_id = response.id
        # submit the finetuning job
        try:
            finetuning_response: FineTuningJob = self.client.fine_tuning.jobs.create(training_file=training_file_id,
                                                                      model="gpt-3.5-turbo",
                                                                      suffix=suffix)
        except Exception as e:
            return

        finetune_job = FinetuneJob(finetuning_response.id, finetuning_response.status, finetuning_response.fine_tuned_model)

        return finetune_job

    def check_api_key(self):
        # check if api key is not none
        if self.api_key is None:
            # try to get the api key from the environment, maybe it has been set later
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError("OpenAI API key is not set")

        if not self.client:
            self.client = OpenAI(api_key=self.api_key)
