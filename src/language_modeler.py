from utils import approximate_token_count
import openai
import datetime
from models.language_model_output import LanguageModelOutput

class LanguageModel(object):
    def __init__(self) -> None:
        self.instruction = "You are given below a function description and input data. The function description of what the function must carry out can be found in the Function section, with input and output type hints. The input data can be found in Input section. Using the function description, apply the function to the Input and return a valid output type, that is acceptable by the output_class_definition and output_class_hint. Return None if you can't apply the function to the input or if the output is optional and the correct output is None.\nINCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format."
        self.system_message = f"You are a skillful and accurate language model, who applies a described function on input data. Make sure the function is applied accurately and correctly and the outputs follow the output type hints and are valid outputs given the output types."

        self.instruction_token_count = approximate_token_count(self.instruction)
        self.system_message_token_count = approximate_token_count(self.system_message)
        self.models = {"openai": {"finetuned":4096, "gpt-4": 8192, "gpt-32k": 32768}} # models and token counts

        self.repair_instruction = "Below are an outputs of a function applied to inputs, which failed type validation. The input to the function is brought out in the INPUT section and function description is brought out in the FUNCTION DESCRIPTION section. Your task is to apply the function to the input and return a correct output in the right type. The FAILED EXAMPLES section will show previous outputs of this function applied to the data, which failed type validation and hence are wrong outputs. Using the input and function description output the accurate output following the output_class_definition and output_type_hint attributes of the function description, which define the output type. Make sure the output is an accurate function output and in the correct type. Return None if you can't apply the function to the input or if the output is optional and the correct output is None."

    def generate(self, args, kwargs, function_modeler, function_description, model_type = "openai"):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset
        """

        prompt, model, save_to_finetune, is_distilled_model = self.get_generation_case(args, kwargs, function_modeler, function_description, model_type)
        if model_type == "openai":
            choice = self.openai_generate(model, prompt)
        
        output = LanguageModelOutput(choice, save_to_finetune,is_distilled_model)
        return output

    
    def openai_generate(self, model, prompt,  temperature = 0, top_p = 1, frequency_penalty = 0, presence_penalty = 0):
        """
        Generate a response from the openai api with the parameters
        """
        response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_message
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

    def get_models(self, function_modeler, func_hash):
        """
        Get the loggers models given the function hash
        """
        current_model, teacher_models = function_modeler.get_models(func_hash)
        return current_model, teacher_models


    def get_generation_case(self, args, kwargs, function_modeler
    , function_description, model_type):
        """
        Get the generation case with the correct prompt and model
        First get the current model, then if distilled model, do zero-shot prompt and return False as suitable_for_finetune
        If not distilled model, check if suitable for finetuning, create the prompt and return the correct model given the token count
        """
        f = str(function_description.__dict__.__repr__())
        
        current_model, teacher_models = self.get_models(function_modeler, function_description.__hash__())
        is_distilled_model = current_model not in teacher_models
        # no examples needed, using a finetuned model. Dont save to finetune dataset
        if is_distilled_model:
            prompt = self.construct_prompt(f, args, kwargs, None)
            return prompt, current_model, False, is_distilled_model
        
        else:
            suitable_for_finetune, input_prompt_token_count = self.suitable_for_finetuning_token_check(args, kwargs, f, model_type)
            aligns = function_modeler.get_alignments(function_description.__hash__(), max=5)
            examples = "\n".join([f"Inputs:\nArgs: {align['args']}\nKwargs: {align['kwargs']}\nOutput: {align['output']}" for align in aligns])
            prompt = self.construct_prompt(f, args, kwargs, examples)
            examples_token_count = approximate_token_count(examples)
            total_token_count = examples_token_count + input_prompt_token_count + self.instruction_token_count + self.system_message_token_count
            model = self.choose_model_from_tokens(teacher_models, model_type, total_token_count)
            if model:
                return prompt, model, suitable_for_finetune, is_distilled_model
            else:
                raise ValueError("The input content and align statements combined are too long, please shorten it. The maximum currently allowed token limit is 32000")
        
    def suitable_for_finetuning_token_check(self, args, kwargs, f, model_type):
        """
        Check if the inputs are suitable for finetuning, i.e are below the finetuning token count
        """
        # check if finetunable
        finetuning_prompt = f"Function: {f}\n---\nInputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:"
        input_prompt_token_count = approximate_token_count(finetuning_prompt)
        suitable_for_finetune =  input_prompt_token_count + self.instruction_token_count + self.system_message_token_count < self.models[model_type]["finetuned"]
        return suitable_for_finetune, input_prompt_token_count


    def construct_prompt(self, f, args, kwargs, examples):
        """
        Cosntruct a prompt given the function description, args, kwargs and examples
        """
        example_input = f"Examples:{examples}\n" if examples else ""
        content = f"{self.instruction}\nFunction: {f}\n{example_input}---\nInputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:"
        return content

    def repair_generate(self, input, f, failed_outputs_list, examples, models, model_type = "openai"):
        prompt = self.generate_repair_prompt(input, f, failed_outputs_list, examples)
        prompt_token_count = approximate_token_count(prompt)
        model = self.choose_model_from_tokens(models, model_type, prompt_token_count)
        if model:
            choice = self.openai_generate(model, prompt)
            return choice
        else:
            return None


    def generate_repair_prompt(self, args, kwargs, f, failed_outputs_list, examples):
        failed_examples = ""
        for failed_output in failed_outputs_list:
                failed_examples += f"Output: {failed_output[0]}\nError: {failed_output[1]}\n\n"
        successful_examples = f"Successful Examples:{examples}\n" if examples else ""
        prompt =  f"{self.repair_instruction}\nFUNCTION DESCRIPTION: {f}\n{successful_examples}---Inputs:\nArgs: {args}\nKwargs: {kwargs}\nFAILED EXAMPLES: {failed_examples}Correct output:"
        return prompt
    
    def choose_model_from_tokens(self, models, model_type, token_count):
        for model in models:
            if token_count < self.models[model_type][model]:
                return model
        return None