from utils import approximate_token_count
import openai
import datetime
from models.language_model_output import LanguageModelOutput
# import abstract base class
from abc import ABC, abstractmethod
from language_models.llm_api_abc import LLM_Api


class Openai_API(LLM_Api):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()
        
    def generate(self, model, system_message, prompt, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset
        """
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        frequency_penalty = kwargs.get("frequency_penalty", 0)
        presence_penalty = kwargs.get("presence_penalty", 0)
        response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=512,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
        choice = response.choices[0].message.content.strip("'")
        return choice