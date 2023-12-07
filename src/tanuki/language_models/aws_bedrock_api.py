from typing import List
import abc
import botocore
import time
# import abstract base class
import boto3 as boto3
from tanuki.language_models.llm_api_abc import LLM_API
import os
import json
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig

class Bedrock_API(LLM_API):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()
        self.bedrock_runtime = None

    def send_api_request(self, model: BaseModelConfig, body: str):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type,
         generate a response and check if the datapoint can be saved to the finetune dataset
        """

        # this needs to be done generally better, introduce the LLM_gen params class
        # so you can config it at the start
        self.check_runtime()
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
                response_body = json.loads(response.get('body').read())
                choice = response_body.get('generation')
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

    def check_runtime(self):
        # check if api key is not none
        if self.bedrock_runtime is None:
            self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get("AWS_DEFAULT_REGION")
        )