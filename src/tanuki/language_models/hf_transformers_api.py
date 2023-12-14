
from tanuki.language_models.llm_api_abc import LLM_API
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM
from tanuki.language_models.jsonformer.jsonformer import Jsonformer


LLM_GENERATION_PARAMETERS = ["temperature", "top_p", "max_new_tokens", "frequency_penalty", "presence_penalty"]

class HF_Transformers_API(LLM_API):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()
        self.models = {}
        self.tokenizers = {}



    def generate(self, model, system_message, prompt, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset
        """
        # Chat template??? vs No chat template
        ## Optionals, literals, lists, dicts, enums, sets, tuples, unions, default values, Annotated[int, Field(gt=0, lt=10)], any cosntraints
        # # int str float bool
        # Bool and str done in Pydantic classes

        max_new_tokens = kwargs.get("max_new_tokens")
        temperature = kwargs.get("temperature", 0.1)
        if model.model_name not in self.models:
            self.models[model.model_name] = AutoModelForCausalLM.from_pretrained(model.model_name)
            self.tokenizers[model.model_name] = AutoTokenizer.from_pretrained(model.model_name)

        jsonformer = Jsonformer(self.models[model.model_name],
                                self.tokenizers[model.model_name], 
                                model.generator_code,
                                prompt,
                                max_number_tokens=max_new_tokens, 
                                temperature=temperature)
        generated_data = jsonformer()
        
        return generated_data
        #temperature = kwargs.get("temperature", 0.1)
        #top_p = kwargs.get("top_p", 1)
        #frequency_penalty = kwargs.get("frequency_penalty", 0)
        #presence_penalty = kwargs.get("presence_penalty", 0)
        
        ## check if there are any generation parameters that are not supported
        #unsupported_params = [param for param in kwargs.keys() if param not in LLM_GENERATION_PARAMETERS]
        #if len(unsupported_params) > 0:
        #    # log warning
        #    logging.warning(f"Unused generation parameters sent as input: {unsupported_params}."\
        #                     "For OpenAI, only the following parameters are supported: {LLM_GENERATION_PARAMETERS}")
        #params = {
        #    "model": model.model_name,
        #    "temperature": temperature,
        #    "max_tokens": max_new_tokens,
        #    "top_p": top_p,
        #    "frequency_penalty": frequency_penalty,
        #    "presence_penalty": presence_penalty,
        #}
        #messages = [
        #    {
        #        "role": "system",
        #        "content": system_message
        #    },
        #    {
        #        "role": "user",
        #        "content": prompt
        #    }
        #]
        #params["messages"] = messages
#
        #counter = 0
        #choice = None
        ## initiate response so exception logic doesnt error out when checking for error in response
        #response = {}
        #while counter <= 5:
        #    try:
        #        openai_headers = {
        #            "Authorization": f"Bearer {self.api_key}",
        #            "Content-Type": "application/json",
        #        }
        #        response = requests.post(
        #            OPENAI_URL, headers=openai_headers, json=params, timeout=50
        #        )
        #        response = response.json()
        #        choice = response["choices"][0]["message"]["content"].strip("'")
        #        break
        #    except Exception as e:
        #        if ("error" in response and
        #                "code" in response["error"] and
        #                response["error"]["code"] == 'invalid_api_key'):
        #            raise Exception(f"The supplied OpenAI API key {self.api_key} is invalid")
        #        if counter == 5:
        #            raise Exception(f"OpenAI API failed to generate a response: {e}")
        #        counter += 1
        #        time.sleep(2 ** counter)
        #        continue
#
        #if not choice:
        #    raise Exception("OpenAI API failed to generate a response")
#
        #return choice
