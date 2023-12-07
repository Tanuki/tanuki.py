# import abstract base class
import boto3 as boto3
from tanuki.language_models.llm_api_abc import LLM_API
from tanuki.language_models.aws_bedrock_api import Bedrock_API
import json
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
import logging
LLM_GENERATION_PARAMETERS = ["temperature", "top_p", "max_new_tokens"]


class LLama_Bedrock_API(Bedrock_API):
    def __init__(self) -> None:
        # initialise the base class
        super().__init__()

    def generate(self, model: BaseModelConfig, system_message: str, prompt: str, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type,
         generate a response and check if the datapoint can be saved to the finetune dataset
        """

        # this needs to be done generally better, introduce the LLM_gen params class
        # so you can config it at the start
        temperature = kwargs.get("temperature", 0.1)
        top_p = kwargs.get("top_p", 1)
        max_tokens_to_sample = kwargs.get("max_new_tokens")
        # check if there are any generation parameters that are not supported
        unsupported_params = [param for param in kwargs.keys() if param not in LLM_GENERATION_PARAMETERS]
        if len(unsupported_params) > 0:
            # log warning
            logging.warning(f"Unused generation parameters sent as input: {unsupported_params}."\
                            "For Llama Bedrock, only the following parameters are supported: {LLM_GENERATION_PARAMETERS}")
        chat_prompt = model.chat_template
        if chat_prompt is None:
            raise Exception("Chat prompt is not defined for this model"\
                            "Please define it in the model config")
        final_prompt = chat_prompt.format(system_message=system_message, user_prompt=prompt)
        body = json.dumps({
            "prompt": final_prompt,
            "max_gen_len": max_tokens_to_sample,
            "temperature": temperature,
            "top_p": top_p,
        })

        choice = self.send_api_request(model, body)
        choice = choice.split(model.parsing_helper_tokens["end_token"])[0]

        return choice
