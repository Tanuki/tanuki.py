import openai
import time
# import abstract base class
from monkey_patch.language_models.llm_api_abc import LLM_Api
import os
API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
import requests


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
        params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 512,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        messages=[
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
        while counter < 5:
            try:
                openai_headers = {
                    "Authorization": f"Bearer {API_KEY}",
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
                    raise Exception(f"The supplied OpenAI API key {API_KEY} is invalid")
                
                time.sleep(1 + 3 * counter)
                counter += 1
                continue
        
        if not choice:
            raise Exception("OpenAI API failed to generate a response")
            
        return choice