from typing import List

import logging
import time
# import abstract base class
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.fine_tuning import FineTuningJob

from tanuki.language_models.llm_finetune_api_abc import LLM_Finetune_API
from tanuki.models.embedding import Embedding
from tanuki.language_models.embedding_api_abc import Embedding_API
from tanuki.language_models.llm_api_abc import LLM_API
import os
from tanuki.language_models.llm_configs import DEFAULT_STUDENT_MODELS
from tanuki.constants import DEFAULT_DISTILLED_MODEL_NAME
from tanuki.language_models.llm_configs.openai_config import OpenAIConfig
from tanuki.models.finetune_job import FinetuneJob
import copy
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
import requests
LLM_GENERATION_PARAMETERS = ["temperature", "top_p", "max_new_tokens", "frequency_penalty", "presence_penalty"]

class OpenAI_API(LLM_API, Embedding_API, LLM_Finetune_API):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()

        self.api_key = os.environ.get("OPENAI_API_KEY")

        self.client = None

    def embed(self, texts: List[str], model: OpenAIConfig, **kwargs) -> List[Embedding]:
        """
        Generate embeddings for the provided texts using the specified OpenAI model.
        Lightweight wrapper over the OpenAI client.

        :param texts: A list of texts to embed.
        :param model: The model to use for embeddings.
        :return: A list of embeddings.
        """
        self.check_api_key()

        try:
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                input=texts,
                model=model.model_name,
                **kwargs
            )
            assert response.object == "list"
            assert len(response.data) == len(texts)
            embeddings = []
            for embedding_response in response.data:
                assert embedding_response.object == "embedding"
                embeddings.append(Embedding(embedding_response.embedding))
            return embeddings
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def generate(self, model, system_message, prompt, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response
        Args
            model (OpenAIConfig): The model to use for generation.
            system_message (str): The system message to use for generation.
            prompt (str): The prompt to use for generation.
            kwargs (dict): Additional generation parameters.
        """

        self.check_api_key()

        temperature = kwargs.get("temperature", 0.1)
        top_p = kwargs.get("top_p", 1)
        frequency_penalty = kwargs.get("frequency_penalty", 0)
        presence_penalty = kwargs.get("presence_penalty", 0)
        max_new_tokens = kwargs.get("max_new_tokens")
        # check if there are any generation parameters that are not supported
        unsupported_params = [param for param in kwargs.keys() if param not in LLM_GENERATION_PARAMETERS]
        if len(unsupported_params) > 0:
            # log warning
            logging.warning(f"Unused generation parameters sent as input: {unsupported_params}."\
                             f"For OpenAI, only the following parameters are supported: {LLM_GENERATION_PARAMETERS}")
        params = {
            "model": model.model_name,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if model.parsing_helper_tokens["start_token"]:
            prompt += model.parsing_helper_tokens["start_token"]
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
        while counter <= 5:
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
            except Exception as e:
                if ("error" in response and
                        "code" in response["error"] and
                        response["error"]["code"] == 'invalid_api_key'):
                    raise Exception(f"The supplied OpenAI API key {self.api_key} is invalid")
                if counter == 5:
                    raise Exception(f"OpenAI API failed to generate a response: {e}")
                counter += 1
                time.sleep(2 ** counter)
                continue

        if not choice:
            raise Exception("OpenAI API failed to generate a response")
        
        if model.parsing_helper_tokens["end_token"]:
            # remove the end token from the choice
            choice = choice.split(model.parsing_helper_tokens["end_token"])[0]
            # check if starting token is in choice
            if model.parsing_helper_tokens["start_token"] in choice:
                # remove the starting token from the choice
                choice = choice.split(model.parsing_helper_tokens["start_token"])[-1]
        return choice

    def list_finetuned(self, limit=100, **kwargs) -> List[FinetuneJob]:
        self.check_api_key()
        response = self.client.fine_tuning.jobs.list(limit=limit)
        jobs = []
        for job in response.data:
            finetune_job = self.create_finetune_job(job)
            jobs.append(finetune_job)

        return jobs

    def get_finetuned(self, job_id) -> FinetuneJob:
        self.check_api_key()
        response = self.client.fine_tuning.jobs.retrieve(job_id)
        finetune_job = self.create_finetune_job(response)
        return finetune_job

    def finetune(self, file, suffix, **kwargs) -> FinetuneJob:
        self.check_api_key()
        # Use the stream as a file
        response = self.client.files.create(file=file, purpose='fine-tune')

        training_file_id = response.id
        # submit the finetuning job
        finetuning_response: FineTuningJob = self.client.fine_tuning.jobs.create(training_file=training_file_id,
                                                                      model=DEFAULT_DISTILLED_MODEL_NAME,
                                                                      suffix=suffix)
        finetune_job = self.create_finetune_job(finetuning_response)
        return finetune_job

    def create_finetune_job(self, response: FineTuningJob) -> FinetuneJob:
        finetuned_model_config = copy.deepcopy(DEFAULT_STUDENT_MODELS[DEFAULT_DISTILLED_MODEL_NAME])
        finetuned_model_config.model_name = response.fine_tuned_model
        finetune_job = FinetuneJob(response.id, response.status, finetuned_model_config)
        return finetune_job
    
    def check_api_key(self):
        # check if api key is not none
        if not self.api_key:
            # try to get the api key from the environment, maybe it has been set later
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is not set")

        if not self.client:
            self.client = OpenAI(api_key=self.api_key)
