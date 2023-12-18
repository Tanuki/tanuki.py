
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
        self.default_temperature = 0.1



    def generate(self, model, system_message, prompt, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset
        """
        # Chat template??? vs No chat template
        ## Optionals, literals, lists, dicts, enums, sets, tuples, unions, default values, Annotated[int, Field(gt=0, lt=10)], any cosntraints
        # # int str float bool
        # Bool and str done in Pydantic classes
        generation_params = kwargs
        if "temperature" not in generation_params:
            generation_params["temperature"] = self.default_temperature
        
        if "custom_decoding" not in generation_params:
          custom_decoding = False
        else:
          custom_decoding = generation_params["custom_decoding"]
          del generation_params["custom_decoding"]
        
        if model.model_name not in self.models:
            self.models[model.model_name] = AutoModelForCausalLM.from_pretrained(model.model_name, **model.model_kwargs)
            self.tokenizers[model.model_name] = AutoTokenizer.from_pretrained(model.model_name)

        prompt = self.get_prompt(system_message, prompt, model.chat_template, self.tokenizers[model.model_name])
        if custom_decoding:
          jsonformer = Jsonformer(self.models[model.model_name],
                                  self.tokenizers[model.model_name], 
                                  model.generator_code,
                                  prompt,
                                  generation_params)
          generated_data = jsonformer()
        else:
           generated_data = self.generate_with_default_transformers(model.model_name, prompt, generation_params)
        
        return generated_data
    
    def get_prompt(system_message, prompt, chat_template, tokenizer):
        try:
          model_specific_prompt_messages = [{"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
          ]
          model_specific_prompt = tokenizer.apply_chat_template(model_specific_prompt_messages,
                                                                tokenize = False,
                                                                add_generation_prompt=True)
        except:
          try:
            model_specific_prompt_messages = [{"role": "user", "content": prompt}]
            model_specific_prompt = tokenizer.apply_chat_template(model_specific_prompt_messages,
                                                                  tokenize = False,
                                                                  add_generation_prompt=True)
          except:
            if not chat_template:
              return prompt
            else:
              model_specific_prompt = chat_template.format(system_message=system_message, user_prompt=prompt)
        return model_specific_prompt
    
    def generate_with_default_transformers(self, model_name, prompt, generation_params):
      input_tokens = self.tokenizers[model_name](prompt, return_tensors="pt")
      input_ids = input_tokens.input_ids
      generated_ids = self.models[model_name].generate(input_ids, **generation_params)
      generated_data = self.tokenizers[model_name].batch_decode(generated_ids, skip_special_tokens=True)
      return generated_data