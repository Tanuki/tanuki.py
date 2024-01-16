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
        Send a API request to the Bedrock API.

        Args:
            model: The model to use for generation.
            body: The body of the request to send to the Bedrock API.
        
        Returns:
            The generated response.
        """

        # check the runtime access
        self.check_runtime()
        counter = 0
        response_body = None
        # initiate response so exception logic doesnt error out when checking for error in response
        response = {}
        while counter <= 5:
            try:
                response = self.bedrock_runtime.invoke_model(body=body,
                                                             modelId=model.model_name,
                                                             contentType="application/json",
                                                             accept="application/json")
                response_body = json.loads(response.get('body').read())
                break
            except botocore.exceptions.ClientError as error:
                raise Exception("boto3 had an error: " + error.response['Error']['Message'])
            except Exception as e:
                if counter == 5:
                    raise Exception(f"AWS Bedrock API failed to generate a response: {e}")
                counter += 1
                time.sleep(2 ** counter)
                continue

        if not response_body:
            raise Exception("AWS Bedrock API failed to generate a response")

        return response_body

    def check_runtime(self):
        # check if the runtime is configured
        if self.bedrock_runtime is None:
            self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get("AWS_DEFAULT_REGION")
        )
