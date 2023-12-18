
from tanuki.language_models.llm_api_abc import LLM_API
from transformers import AutoTokenizer, AutoModelForCausalLM
from tanuki.language_models.jsonformer.jsonformer import Jsonformer


class HF_Transformers_API(LLM_API):
    def __init__(self) -> None:
        # initialise the abstract base class
        super().__init__()
        self.models = {}
        self.tokenizers = {}
        self.default_temperature = 0.1



    def generate(self, model, system_message, prompt, **kwargs):
        """
        Generate a response using the HF transformers library for the specified model.
        Can either use the jsonformers custom decoding or the default decoding.
        Args:
            model: The model to use for generation.
            system_message: The system message to use for generation.
            prompt: The prompt to use for generation.
            kwargs: Additional generation parameters.
        
        Returns:
            The generated response.
        """
        generation_params = kwargs
        if "temperature" not in generation_params:
            generation_params["temperature"] = self.default_temperature
        
        if "json_decoding" not in generation_params:
          custom_decoding = False
        else:
          custom_decoding = generation_params["json_decoding"]
          del generation_params["json_decoding"]
        
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
    
    def get_prompt(system_message, prompt, chat_template, tokenizer) -> str:
        """
        Create a model specific prompt from the system message and user prompt.
        Tries multiple different chat template ways to create the prompt.

        Args:
            system_message (str): The system message to use for generation.
            prompt (str): The prompt to use for generation.
            chat_template (str): The chat template to use for generation.
            tokenizer (transformers.tokenizer): The tokenizer to use for generation.
        """
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
    
    def generate_with_default_transformers(self, model_name, prompt, generation_params) -> str:
      """
      Generate a response using the default decoding of the HF transformers library for the specified model.
      Args:
          model_name (str): The model to use for generation.
          prompt (str): The prompt to use for generation.
          generation_params (dict): Additional generation parameters.
      
      Returns:
          The generated response.
      """
      input_tokens = self.tokenizers[model_name](prompt, return_tensors="pt")
      input_ids = input_tokens.input_ids
      generated_ids = self.models[model_name].generate(input_ids, **generation_params)
      generated_data = self.tokenizers[model_name].batch_decode(generated_ids, skip_special_tokens=True)
      return generated_data