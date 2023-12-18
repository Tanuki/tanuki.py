
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
           generated_data = self.generate_with_default_transformers(model, prompt, generation_params)
        
        return generated_data
    
    def get_prompt(self, system_message, prompt, chat_template, tokenizer) -> str:
        """
        Create a model specific prompt from the system message and user prompt.
        Tries multiple different chat template ways to create the prompt.

        Args:
            system_message (str): The system message to use for generation.
            prompt (str): The prompt to use for generation.
            chat_template (str): The chat template to use for generation.
            tokenizer (transformers.tokenizer): The tokenizer to use for generation.
        """
        if not chat_template:
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
                return prompt
        else:
          model_specific_prompt = chat_template.format(system_message=system_message, user_prompt=prompt)
        return model_specific_prompt
    
    def generate_with_default_transformers(self, model, prompt, generation_params) -> str:
      """
      Generate a response using the default decoding of the HF transformers library for the specified model.
      Args:
          model (BaseModelConfig): The model to use for generation.
          prompt (str): The prompt to use for generation.
          generation_params (dict): Additional generation parameters.
      
      Returns:
          The generated response.
      """
      input_tokens = self.tokenizers[model.model_name](prompt, return_tensors="pt").to(
            self.models[model.model_name].device
        )
      response = self.models[model.model_name].generate(input_ids = input_tokens.input_ids,
                                                        attention_mask=input_tokens.attention_mask,
                                                        **generation_params)
      # Some models output the prompt as part of the response
      # This removes the prompt from the response if it is present
      if (
          len(response[0]) >= len(input_tokens.input_ids[0])
          and (response[0][: len(input_tokens.input_ids[0])] == input_tokens.input_ids).all()
      ):
          response = response[0][len(input_tokens[0]) :]
      if response.shape[0] == 1:
          response = response[0]
      
      generated_data = self.tokenizers[model.model_name].decode(response, skip_special_tokens=True).strip()
      
      if model.parsing_helper_tokens["end_token"]:
            # remove the end token from the choice
            generated_data = generated_data.split(model.parsing_helper_tokens["end_token"])[0]
      return generated_data