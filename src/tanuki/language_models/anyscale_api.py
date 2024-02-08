from typing import List

import logging
import time
# import abstract base class
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.fine_tuning import FineTuningJob

from tanuki.language_models.llm_finetune_api_abc import LLM_Finetune_API
from tanuki.language_models.llm_api_abc import LLM_API
import os
from tanuki.constants import DEFAULT_DISTILLED_MODEL_NAME
from tanuki.language_models.llm_configs.anyscale_config import Anyscaleconfig
from tanuki.models.finetune_job import FinetuneJob
import copy
ANYSCALE_URL = "https://api.endpoints.anyscale.com/v1"
import requests
LLM_GENERATION_PARAMETERS = ["temperature", "top_p", "max_new_tokens", "frequency_penalty", "presence_penalty"]

class Anyscale_API(LLM_API, LLM_Finetune_API):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()

        self.api_key = os.environ.get("ANYSCALE_API_KEY")

        self.client = None

    
    def generate(self, model, system_message, prompt, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response
        Args
            model (Anyscaleconfig): The model to use for generation.
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
                             f"For Anyscale, only the following parameters are supported: {LLM_GENERATION_PARAMETERS}")
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
                anyscale_headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                response = requests.post(
                    f"{ANYSCALE_URL}/chat/completions", 
                    headers=anyscale_headers, 
                    json=params, timeout=50
                )
                response = response.json()
                choice = response["choices"][0]["message"]["content"].strip("'")
                break
            except Exception as e:
                if ("error" in response and
                        "code" in response["error"] and
                        response["error"]["code"] == 'invalid_api_key'):
                    raise Exception(f"The supplied Anyscale API key {self.api_key} is invalid")
                if counter == 5:
                    raise Exception(f"Anyscale API failed to generate a response: {e}")
                counter += 1
                time.sleep(2 ** counter)
                continue

        if not choice:
            raise Exception("Anyscale API failed to generate a response")
        
        if model.parsing_helper_tokens["end_token"]:
            # remove the end token from the choice
            choice = choice.split(model.parsing_helper_tokens["end_token"])[0]
            # check if starting token is in choice
            if model.parsing_helper_tokens["start_token"] in choice:
                # remove the starting token from the choice
                choice = choice.split(model.parsing_helper_tokens["start_token"])[-1]
        return choice.strip()

    def list_finetuned(self, model_config, limit=100, **kwargs) -> List[FinetuneJob]:
        self.check_api_key()
        response = self.client.fine_tuning.jobs.list(limit=limit)
        jobs = []
        for job in response.data:
            finetune_job = self.create_finetune_job(job, model_config)
            jobs.append(finetune_job)

        return jobs

    def get_finetuned(self, job_id, model_config: Anyscaleconfig) -> FinetuneJob:
        self.check_api_key()
        response = self.client.fine_tuning.jobs.retrieve(job_id)
        finetune_job = self.create_finetune_job(response, model_config= model_config)
        return finetune_job

    def finetune(self, file, suffix, model_config, **kwargs) -> FinetuneJob:
        self.check_api_key()
        # Use the stream as a file
        response = self.client.files.create(file=file, purpose='fine-tune')

        training_file_id = response.id
        if not model_config.base_model_for_sft:
            model_config.base_model_for_sft = DEFAULT_DISTILLED_MODEL_NAME
        # submit the finetuning job
        finetuning_response: FineTuningJob = self.client.fine_tuning.jobs.create(training_file=training_file_id,
                                                                      model=model_config.base_model_for_sft,
                                                                      suffix=suffix)
        finetune_job = self.create_finetune_job(finetuning_response, model_config)
        return finetune_job

    def create_finetune_job(self, response: FineTuningJob, model_config: Anyscaleconfig) -> FinetuneJob:
        finetuned_model_config = copy.deepcopy(model_config)
        finetuned_model_config.model_name = response.fine_tuned_model
        finetune_job = FinetuneJob(response.id, response.status, finetuned_model_config)
        return finetune_job
    
    def check_api_key(self):
        # check if api key is not none
        if not self.api_key:
            # try to get the api key from the environment, maybe it has been set later
            self.api_key = os.getenv("ANYSCALE_API_KEY")
            if not self.api_key:
                raise ValueError("Anyscale API key is not set")

        if not self.client:
            self.client = OpenAI(base_url= ANYSCALE_URL,
                                 api_key=self.api_key)
