import logging
import time
# import abstract base class
from tanuki.language_models.llm_api_abc import LLM_API
import os
import together

TOGETHER_AI_URL = "https://api.together.xyz/inference"
import requests
LLM_GENERATION_PARAMETERS = ["temperature", "top_p", "max_new_tokens", "frequency_penalty", "presence_penalty"]

class TogetherAI_API(LLM_API):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()

        self.api_key = os.environ.get("TOGETHER_API_KEY")
        self.model_configs = {}


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
        if model.model_name not in self.model_configs:
            self.model_configs[model.model_name] = together.Models.info(model.model_name)['config']
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
            "presence_penalty": presence_penalty
        }
        if "stop" in self.model_configs[model.model_name]:
            params["stop"] = list(self.model_configs[model.model_name]["stop"])
        if model.parsing_helper_tokens["end_token"]:
            params["stop"] = model.parsing_helper_tokens["end_token"]
        chat_prompt = model.chat_template
        if chat_prompt is None:
            try:
                prompt_format = str(self.model_configs[model.model_name]['prompt_format'])
                final_prompt = prompt_format.format(system_message=system_message, prompt=prompt)
            except:
                logging.warning("Chat prompt is not defined for this model. "\
                                "Please define it in the model config. Using default chat prompt")
                chat_prompt = "[INST]{system_message}[/INST]\n{user_prompt}"
                final_prompt = chat_prompt.format(system_message=system_message, user_prompt=prompt)
        else:
            final_prompt = chat_prompt.format(system_message=system_message, user_prompt=prompt)
        if model.parsing_helper_tokens["start_token"]:
            final_prompt += model.parsing_helper_tokens["start_token"]
        params["prompt"] = final_prompt

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
                    TOGETHER_AI_URL, headers=openai_headers, json=params, timeout=50
                )
                response = response.json()
                choice = response["output"]["choices"][0]["text"].strip("'")
                break
            except Exception as e:
                if ("error" in response and
                        "code" in response["error"] and
                        response["error"]["code"] == 'invalid_api_key'):
                    raise Exception(f"The supplied Together AI API key {self.api_key} is invalid")
                if counter == 5:
                    raise Exception(f"Together AI API failed to generate a response: {e}")
                counter += 1
                time.sleep(2 ** counter)
                continue

        if not choice:
            raise Exception("TogetherAI API failed to generate a response")
        
        if model.parsing_helper_tokens["end_token"]:
            # remove the end token from the choice
            choice = choice.split(model.parsing_helper_tokens["end_token"])[0]
            # check if starting token is in choice
            if model.parsing_helper_tokens["start_token"] in choice:
                # remove the starting token from the choice
                choice = choice.split(model.parsing_helper_tokens["start_token"])[-1]
        return choice.strip()
    
    def check_api_key(self):
        # check if api key is not none
        if not self.api_key:
            # try to get the api key from the environment, maybe it has been set later
            self.api_key = os.getenv("TOGETHER_API_KEY")
            if not self.api_key:
                raise ValueError("TogetherAI API key is not set")
