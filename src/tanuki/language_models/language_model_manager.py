import json
from typing import Any, Dict

from tanuki.function_modeler import FunctionModeler
from tanuki.language_models.llm_api_abc import LLM_API
from tanuki.models.function_description import FunctionDescription
from tanuki.models.function_example import FunctionExample
from tanuki.models.language_model_output import LanguageModelOutput
from tanuki.utils import approximate_token_count
from tanuki.validator import Validator
from tanuki.models.api_manager import APIManager
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
import logging
class LanguageModelManager(object):
    """
    The LanguageModelManager is responsible for managing the language models and their outputs operationally,
    this includes:
    - Generating outputs from the language models
    - Repairing outputs from the language models
    - Saving outputs from the language models
    - Finetuning the language models from the saved outputs
    """

    def __init__(self,
                 function_modeler: FunctionModeler,
                 api_provider: APIManager,
                 generation_token_limit=512,) -> None:
        self.api_provider = api_provider
        self.function_modeler = function_modeler
        self.default_generation_length = generation_token_limit
        self.initialized_functions = {}
        self.token_counts = {}

    def __call__(self,
                 args,
                 function_description: FunctionDescription,
                 kwargs,
                 validator: Validator,
                 generation_parameters: dict) -> Any:
        
        # add the generation length if not there
        if "max_new_tokens" not in generation_parameters:
            generation_parameters["max_new_tokens"] = self.default_generation_length

        output = self.generate(args, kwargs, function_description, generation_parameters)
        # start parsing the object, very hacky way for the time being
        choice_parsed = self._parse_choice(output)
        valid = validator.check_type(choice_parsed, function_description.output_type_hint)
        if not valid:
            choice, choice_parsed, successful_repair = self.repair_output(args,
                                                                          kwargs,
                                                                          function_description,
                                                                          output.generated_response,
                                                                          validator,
                                                                          generation_parameters)

            if not successful_repair:
                raise TypeError(
                    f"Output type was not valid. Expected an object of type {function_description.output_type_hint}, got '{output.generated_response}'")
            output.generated_response = choice
            output.distilled_model = False
        datapoint = FunctionExample(args, kwargs, output.generated_response)
        if output.suitable_for_finetuning and not output.distilled_model:
            self.function_modeler.postprocess_symbolic_datapoint(function_description.__hash__(), function_description,
                                                                 datapoint, repaired=not valid)
        instantiated = validator.instantiate(choice_parsed, function_description.output_type_hint)
        return instantiated

    def _parse_choice(self, output):
        try:
            # json load
            choice_parsed = json.loads(output.generated_response)
        except:
            # if it fails, it's not a json object, try eval
            try:
                choice_parsed = eval(output.generated_response)
            except:
                choice_parsed = output.generated_response
        return choice_parsed

    def generate(self, args, kwargs, function_description, llm_parameters={}):
        """
        The main generation function, given the args, kwargs, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset
        """

        func_hash = function_description.__hash__()
        prompt, model, save_to_finetune, is_distilled_model = self.get_generation_case(args, kwargs,
                                                                                       function_description,
                                                                                       llm_parameters, 
                                                                                       func_hash)
        # loggings
        current_function_setup = self.initialized_functions.get(func_hash, None) # getting the current function setup - model and align statements
        if current_function_setup:
            generator_model = current_function_setup["model"]
            if is_distilled_model:
                logging.info(f"Generating function outputs for {function_description.name} with a finetuned model: {model.model_name}.")
                self.initialized_functions[func_hash]["model"] = model.model_name
            elif generator_model == "":
                logging.info(f"Found {len(current_function_setup['examples'])} align statements for {function_description.name}. Generating function outputs with {model.model_name}.")
                self.initialized_functions[func_hash]["model"] = model.model_name
            elif generator_model != model.model_name:
                logging.info(f"Switching output generation from {generator_model} to {model.model_name} for function {function_description.name}.")
                self.initialized_functions[func_hash]["model"] = model.model_name

        choice = self._synthesise_answer(prompt, model, llm_parameters)
        output = LanguageModelOutput(choice, save_to_finetune, is_distilled_model)
        return output

    def _synthesise_answer(self, prompt, model, llm_parameters):
        """
        Synthesise an answer given the prompt, model, model_type and llm_parameters
        Args:
            prompt (str): The prompt to send to the model
            model (BaseModelConfig): The model to use for generation
            llm_parameters (dict): The parameters to use for generation
        return:
            choice (str): The generated response

        """
        system_message = model.system_message
        return self.api_provider[model.provider].generate(model, system_message, prompt, **llm_parameters)


    def get_generation_case(self, args, kwargs, function_description, llm_parameters, func_hash):
        """
        Get the generation case with the correct prompt and model
        First get the current model, then if distilled model, do zero-shot prompt and return False as suitable_for_finetune
        If not distilled model, check if suitable for finetuning, create the prompt and return the correct model given the token count
        """
        f = str(function_description.__dict__.__repr__())

        distilled_model, teacher_models = self.function_modeler.get_models(function_description)
        is_distilled_model = distilled_model.model_name != ""
        suitable_for_distillation, input_prompt_token_count = self.suitable_for_finetuning_token_check(args, kwargs, f,
                                                                                                       distilled_model)
        if func_hash not in self.initialized_functions:
            # initialise the initialized_functions dict
            self.initialized_functions[func_hash] = {"model": "", "examples": []}
        # no examples needed, using a finetuned model. Dont save to finetune dataset
        if is_distilled_model and suitable_for_distillation:
            prompt = self.construct_prompt(f, args, kwargs, [], distilled_model)
            return prompt, distilled_model, suitable_for_distillation, True

        else:
            aligns = self.function_modeler.get_symbolic_alignments(function_description.__hash__(), max=16)
            examples = [f"Inputs:\nArgs: {align['args']}\nKwargs: {align['kwargs']}\nOutput: {align['output']}" for align in
                 aligns]
            
            # update the examples in the initialized_functions dict
            self.initialized_functions[func_hash]["examples"] = examples

            examples_token_count = sum([approximate_token_count(example) for example in examples])
            generation_tokens = llm_parameters.get("max_new_tokens", self.default_generation_length)
            model = self.choose_model_from_tokens(teacher_models,
                                                  examples_token_count + input_prompt_token_count + generation_tokens,
                                                  len(examples))
            if model:
                examples_with_parsing_tokens = [f"Inputs:\nArgs: {align['args']}\nKwargs: {align['kwargs']}\nOutput:{model.parsing_helper_tokens['start_token']}{align['output']}{model.parsing_helper_tokens['end_token']}" for align in
                 aligns]
                prompt = self.construct_prompt(f, args, kwargs, examples_with_parsing_tokens, model)
                return prompt, model, suitable_for_distillation, False
            else:
                raise ValueError(
                    "The input content and align statements combined are too long, please shorten it. The maximum currently allowed token limit is 32000")

    def suitable_for_finetuning_token_check(self, args, kwargs, f, distilled_model: BaseModelConfig):
        """
        Check if the inputs are suitable for finetuning, i.e are below the finetuning token count
        """
        # check if finetunable
        finetuning_prompt = f"Function: {f}\n---\nInputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:"
        input_prompt_token_count = approximate_token_count(finetuning_prompt)
        if distilled_model.system_message_token_count < 0:
            distilled_model.system_message_token_count = approximate_token_count(distilled_model.system_message)
        if distilled_model.instruction_token_count < 0:
            distilled_model.instruction_token_count = approximate_token_count(distilled_model.instructions)

        suitable_for_finetune = input_prompt_token_count + distilled_model.instruction_token_count + distilled_model.system_message_token_count < distilled_model.context_length
        return suitable_for_finetune, input_prompt_token_count

    def construct_prompt(self, f, args, kwargs, examples, model):
        """
        Construct a prompt given the model, function description, args, kwargs and examples

        Args:
            model (BaseModelConfig): The model to use for generation
            f (str): The function description
            args (tuple): The args of the function
            kwargs (tuple): The kwargs of the function
            examples (list): The examples of the function
        
        Returns:
            content (str): The prompt to send to the model
        """
        if examples:
            final_examples = "\n".join(
                    [f"{align}" for align in
                     examples])
            example_input = f"Examples:{final_examples}\n"
        else:
            example_input = ""

        instruction_prompt = model.instructions
        content = f"{instruction_prompt}\nFunction: {f}\n{example_input}---\nInputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:"
        return content

    def repair_generate(self, args, kwargs, f, failed_outputs_list, aligns, models, llm_parameters):
        """
        Repair the output given the input, function description, failed outputs list, examples and models
        """
        # get the token counts
        examples = [f"Inputs:\nArgs: {align['args']}\nKwargs: {align['kwargs']}\nOutput: {align['output']}" for align in
                 aligns]
        examples_token_count = sum([approximate_token_count(example) for example in examples])
        failed_examples_token_count = sum([approximate_token_count(failed_output[0]) + approximate_token_count(failed_output[1]) for failed_output in failed_outputs_list])
        input_prompt_token_count = approximate_token_count(f"Function: {f}\n---\nInputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:")
        generation_tokens = llm_parameters.get("max_new_tokens", self.default_generation_length)
        model = self.choose_model_from_tokens(models, 
                                              examples_token_count+input_prompt_token_count+generation_tokens+failed_examples_token_count,
                                              len(examples))
        if model:
            prompt = self.generate_repair_prompt(args, kwargs, f, failed_outputs_list, examples, model)
            logging.info(f"Previous output failed type validation, attempting to repair with {model.model_name}")
            choice = self._synthesise_answer(prompt, model, llm_parameters)
            return choice
        else:
            return None

    def generate_repair_prompt(self, args, kwargs, f, failed_outputs_list, examples, model):
        """
        Generate a repair prompt given the args, kwargs, function description, failed outputs list and examples
        """

        if examples:
            final_examples = "\n".join(
                    [f"{model.parsing_helper_tokens['start_token']}{align}{model.parsing_helper_tokens['end_token']}" for align in
                     examples])
            successful_examples = f"Examples:{final_examples}\n"
        else:
            successful_examples = ""

        failed_examples = ""
        for failed_output in failed_outputs_list:
            failed_examples += f"Output: {failed_output[0]}\nError: {failed_output[1]}\n\n"
        end_token_addition = ""
        if model.parsing_helper_tokens["end_token"]:
            end_token_addition = f"Make sure to add the {model.parsing_helper_tokens['end_token']} token at the end of the output."
        prompt = f"{model.repair_instruction}{end_token_addition}\nFUNCTION DESCRIPTION: {f}\n{successful_examples}---{model.parsing_helper_tokens['start_token']}Inputs:\nArgs: {args}\nKwargs: {kwargs}\nFAILED EXAMPLES: {failed_examples}Correct output:"
        return prompt

    def choose_model_from_tokens(self, models, input_token_count, nr_of_examples=0):
        """
        Choose a model from the models given the token count and number of examples

        Args:
            models (list): The models to choose from
            input_token_count (int): The token count of the input
            nr_of_examples (int): The number of examples
        
        Returns:
            model (BaseModelConfig): The chosen model
        """

        for model in models:
            # check if input token count is less than the context length
            # If the model config has custom messages, then use those, otherwise use the default ones
            if model.system_message_token_count < 0:
                model.system_message_token_count = approximate_token_count(model.system_message)
            if model.instruction_token_count < 0:
                model.instruction_token_count = approximate_token_count(model.instructions)
            if model.parsing_helper_tokens["start_token"]:
                input_token_count += 2*nr_of_examples
            if model.parsing_helper_tokens["end_token"]:
                input_token_count += 2*nr_of_examples
            total_token_count = input_token_count + model.instruction_token_count + model.system_message_token_count
            if total_token_count < model.context_length:
                return model
        return None

    def repair_output(self,
                      args: tuple,
                      kwargs: dict,
                      function_description: FunctionDescription,
                      choice,
                      validator: Validator,
                      generation_parameters: dict) -> tuple:
        """
        Repair an output, that failed type validation by generating a new output using the teacher model and the error
        Args:
            args (tuple): The args of the function
            kwargs (dict): The kwargs of the function
            function_description (FunctionDescription): The function description
            choice: The output that failed type validation, type is arbitrary
            validator (Validator): The validator object

        Returns:
            choice (str): The choice that was generated by the language model
            choice_parsed: The parsed choice, type is arbitrary
            valid (bool): Whether the output was correctly repaired was valid
        """

        # get the teacher models
        teacher_models = self.function_modeler.get_models(function_description)[1]
        valid = False
        retry_index = 5
        f = str(function_description.__dict__.__repr__() + "\n")
        error = f"Output type was not valid. Expected an valid object of type {function_description.output_type_hint}, got '{choice}'"
        # instantiate the failed outputs list
        failed_outputs_list = [(choice, error)]
        while retry_index > 0 and not valid:
            # get the alignments
            aligns = self.function_modeler.get_symbolic_alignments(function_description.__hash__(), max=5)
            # Generate the reparied LLM output
            choice = self.repair_generate(args, 
                                          kwargs, 
                                          f, 
                                          failed_outputs_list, 
                                          aligns, 
                                          teacher_models,
                                          generation_parameters)
            if not choice:
                # if no choice then the input was too long for the model
                # no specific error but the retry index goes down
                retry_index -= 1
                continue

            # start parsing the object
            try:
                # json load
                choice_parsed = json.loads(choice)
            except:
                # if it fails, it's not a json object, try eval
                try:
                    choice_parsed = eval(choice)
                except:
                    choice_parsed = choice

            valid = validator.check_type(choice_parsed, function_description.output_type_hint)
            if not valid:
                # if it's not valid, add it to the failed outputs list
                error = f"Output type was not valid. Expected an object of type {function_description.output_type_hint}, got '{choice}'"
                failed_outputs_list.append((choice, error))
                retry_index -= 1
            if valid:
                logging.info(f"Successfully repaired output.")

        return choice, choice_parsed, valid
