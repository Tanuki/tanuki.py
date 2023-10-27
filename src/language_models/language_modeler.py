from utils import approximate_token_count
import openai
import datetime
from models.language_model_output import LanguageModelOutput
from language_models.openai_api import Openai_API


class LanguageModel(object):
    def __init__(self, generation_token_limit = 512) -> None:
        self.instruction = "You are given below a function description and input data. The function description of what the function must carry out can be found in the Function section, with input and output type hints. The input data can be found in Input section. Using the function description, apply the function to the Input and return a valid output type, that is acceptable by the output_class_definition and output_class_hint. Return None if you can't apply the function to the input or if the output is optional and the correct output is None.\nINCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format."
        self.system_message = f"You are a skillful and accurate language model, who applies a described function on input data. Make sure the function is applied accurately and correctly and the outputs follow the output type hints and are valid outputs given the output types."

        self.instruction_token_count = approximate_token_count(self.instruction)
        self.system_message_token_count = approximate_token_count(self.system_message)
        self.api_models = {"openai": Openai_API()}
        self.repair_instruction = "Below are an outputs of a function applied to inputs, which failed type validation. The input to the function is brought out in the INPUT section and function description is brought out in the FUNCTION DESCRIPTION section. Your task is to apply the function to the input and return a correct output in the right type. The FAILED EXAMPLES section will show previous outputs of this function applied to the data, which failed type validation and hence are wrong outputs. Using the input and function description output the accurate output following the output_class_definition and output_type_hint attributes of the function description, which define the output type. Make sure the output is an accurate function output and in the correct type. Return None if you can't apply the function to the input or if the output is optional and the correct output is None."
        self.generation_length = generation_token_limit
        self.models = {"gpt-4":{"token_limit": 8192 - self.generation_length, "type": "openai"},
                        "gpt-4-32k": {"token_limit": 32768 - self.generation_length, "type": "openai"}
                        } # models and token counts


    def generate(self, args, kwargs, function_modeler, function_description, llm_parameters = {}):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset
        """

        prompt, model, save_to_finetune, is_distilled_model = self.get_generation_case(args, kwargs, function_modeler, function_description)
        if is_distilled_model:
            model_type = self.get_distillation_model_type(model)
        else:
            model_type = self.get_teacher_model_type(model)
        choice = self.synthesise_answer(prompt, model, model_type, llm_parameters)

        output = LanguageModelOutput(choice, save_to_finetune,is_distilled_model)
        return output

    def synthesise_answer(self, prompt, model, model_type, llm_parameters):
        """
        Synthesise an answer given the prompt, model, model_type and llm_parameters
        """
        if model_type == "openai":
            return self.api_models[model_type].generate(model, self.system_message, prompt, **llm_parameters)


    def get_distillation_model_type(self, model):
        """
        Get the distilled model type given the model
        """
        # currently only openai is supported
        return "openai"
    
    def get_teacher_model_type(self, model):
        """
        Get the teacher model type given the model
        """
        # check if model is in the models
        if model in self.models.keys():
            return self.models[model]["type"]
        else:
            raise ValueError("This teacher model is not supported")


    def get_models(self, function_modeler, func_hash):
        """
        Get the loggers models given the function hash
        """
        distilled_model, teacher_models = function_modeler.get_models(func_hash)
        return distilled_model, teacher_models


    def get_generation_case(self, args, kwargs, function_modeler, function_description):
        """
        Get the generation case with the correct prompt and model
        First get the current model, then if distilled model, do zero-shot prompt and return False as suitable_for_finetune
        If not distilled model, check if suitable for finetuning, create the prompt and return the correct model given the token count
        """
        f = str(function_description.__dict__.__repr__())
        
        distilled_model, teacher_models = self.get_models(function_modeler, function_description.__hash__())
        is_distilled_model = distilled_model != ""
        suitable_for_distillation, input_prompt_token_count = self.suitable_for_finetuning_token_check(args, kwargs, f, function_modeler.distillation_token_limit)
        # no examples needed, using a finetuned model. Dont save to finetune dataset
        if is_distilled_model and suitable_for_distillation:
            prompt = self.construct_prompt(f, args, kwargs, None)
            return prompt, distilled_model, suitable_for_distillation, True
        
        else:
            aligns = function_modeler.get_alignments(function_description.__hash__(), max=5)
            examples = "\n".join([f"Inputs:\nArgs: {align['args']}\nKwargs: {align['kwargs']}\nOutput: {align['output']}" for align in aligns])
            prompt = self.construct_prompt(f, args, kwargs, examples)
            examples_token_count = approximate_token_count(examples)
            total_token_count = examples_token_count + input_prompt_token_count + self.instruction_token_count + self.system_message_token_count
            model = self.choose_model_from_tokens(teacher_models, total_token_count)
            if model:
                return prompt, model, suitable_for_distillation, False
            else:
                raise ValueError("The input content and align statements combined are too long, please shorten it. The maximum currently allowed token limit is 32000")
        
    def suitable_for_finetuning_token_check(self, args, kwargs, f, distillation_token_count):
        """
        Check if the inputs are suitable for finetuning, i.e are below the finetuning token count
        """
        # check if finetunable
        finetuning_prompt = f"Function: {f}\n---\nInputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:"
        input_prompt_token_count = approximate_token_count(finetuning_prompt)
        suitable_for_finetune =  input_prompt_token_count + self.instruction_token_count + self.system_message_token_count < distillation_token_count
        return suitable_for_finetune, input_prompt_token_count


    def construct_prompt(self, f, args, kwargs, examples):
        """
        Cosntruct a prompt given the function description, args, kwargs and examples
        """
        example_input = f"Examples:{examples}\n" if examples else ""
        content = f"{self.instruction}\nFunction: {f}\n{example_input}---\nInputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:"
        return content

    def repair_generate(self, args, kwargs, f, failed_outputs_list, examples, models):
        """
        Repair the output given the input, function description, failed outputs list, examples and models
        """
        prompt = self.generate_repair_prompt(args, kwargs, f, failed_outputs_list, examples)
        prompt_token_count = approximate_token_count(prompt)
        model = self.choose_model_from_tokens(models, prompt_token_count)
        if model:
            model_type = self.get_teacher_model_type(model)
            choice = self.synthesise_answer(prompt, model, model_type, {})
            return choice
        else:
            return None


    def generate_repair_prompt(self, args, kwargs, f, failed_outputs_list, examples):
        """
        Generate a repair prompt given the args, kwargs, function description, failed outputs list and examples
        """

        failed_examples = ""
        for failed_output in failed_outputs_list:
                failed_examples += f"Output: {failed_output[0]}\nError: {failed_output[1]}\n\n"
        successful_examples = f"Successful Examples:{examples}\n" if examples else ""
        prompt =  f"{self.repair_instruction}\nFUNCTION DESCRIPTION: {f}\n{successful_examples}---Inputs:\nArgs: {args}\nKwargs: {kwargs}\nFAILED EXAMPLES: {failed_examples}Correct output:"
        return prompt
    
    def choose_model_from_tokens(self, models, token_count):
        """
        Choose a model from the models given the token count
        """
        
        for model in models:
            # check if model is in the models
            if model in self.models.keys():
                if token_count < self.models[model]["token_limit"]:
                    return model
        return None