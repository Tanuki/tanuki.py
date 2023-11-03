import io
import json
from typing import get_args, Any

import ijson as ijson

from monkey_patch.language_models.openai_api import Openai_API
from monkey_patch.models.function_description import FunctionDescription
from monkey_patch.models.language_model_output import LanguageModelOutput
from monkey_patch.utils import approximate_token_count
from monkey_patch.validator import Validator

INSTRUCTION = "You are given below a function description and input data. The function description of what the " \
              "function must carry out can be found in the Function section, with input and output type hints. The " \
              "input data can be found in Input section. Using the function description, apply the function to the " \
              "Input and return a valid output type, that is acceptable by the output_class_definition and " \
              "output_class_hint. Return None if you can't apply the function to the input or if the output is " \
              "optional and the correct output is None.\nINCREDIBLY IMPORTANT: Only output a JSON-compatible string " \
              "in the correct response format. If there are no inputs, but a defined output, you must follow the " \
              "instructions of the docstring and generate an output. "

SYSTEM = f"You are a skillful and accurate language model, who applies a described function on input data. Make sure " \
         f"the function is applied accurately and correctly and the outputs follow the output type hints and are " \
         f"valid outputs given the output types. "

REPAIR = "Below are an outputs of a function applied to inputs, which failed type validation. The input to the " \
         "function is brought out in the INPUT section and function description is brought out in the FUNCTION " \
         "DESCRIPTION section. Your task is to apply the function to the input and return a correct output in the " \
         "right type. The FAILED EXAMPLES section will show previous outputs of this function applied to the data, " \
         "which failed type validation and hence are wrong outputs. Using the input and function description output " \
         "the accurate output following the output_class_definition and output_type_hint attributes of the function " \
         "description, which define the output type. Make sure the output is an accurate function output and in the " \
         "correct type. Return None if you can't apply the function to the input or if the output is optional and the " \
         "correct output is None. "


class LanguageModel(object):
    def __init__(self, generation_token_limit = 512) -> None:
        self.instruction = INSTRUCTION
        self.system_message = SYSTEM
        self.instruction_token_count = approximate_token_count(self.instruction)
        self.system_message_token_count = approximate_token_count(self.system_message)
        self.api_models = {"openai": Openai_API()}
        self.repair_instruction = REPAIR
        self.generation_length = generation_token_limit
        self.models = {"gpt-4":{"token_limit": 8192 - self.generation_length, "type": "openai"},
                        "gpt-4-32k": {"token_limit": 32768 - self.generation_length, "type": "openai"}
                        } # models and token counts
        self.validator = Validator()


    def generate(self, args, kwargs, function_modeler, function_description, llm_parameters = {}) -> LanguageModelOutput:
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset
        """

        prompt, model, save_to_finetune, is_distilled_model = self.get_generation_case(args, kwargs, function_modeler, function_description)
        if is_distilled_model:
            model_type = self.get_distillation_model_type(model)
        else:
            model_type = self.get_teacher_model_type(model)
        choice = self.synthesise_answer(prompt, model, model_type, llm_parameters)

        output = LanguageModelOutput(choice, save_to_finetune, is_distilled_model)

        # Create the object from the output of the language model
        instantiated = args.get_object_from_output(function_description,
                                                   args,
                                                   kwargs,
                                                   output,
                                                   self.validator)


        return instantiated

    def synthesise_answer(self, prompt, model, model_type, llm_parameters):
        """
        Synthesise an answer given the prompt, model, model_type and llm_parameters
        """
        if model_type == "openai":
            return self.api_models[model_type].generate(model, self.system_message, prompt, **llm_parameters)

    async def generate_async(self, args, kwargs,
                             function_modeler,
                             function_description: FunctionDescription,
                             llm_parameters={}) -> LanguageModelOutput:
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type,
        generate a response and check if the datapoint can be saved to the finetune dataset :return:
        """
        prompt, model, save_to_finetune, is_distilled_model = self.get_generation_case(args,
                                                                                       kwargs,
                                                                                       function_modeler,
                                                                                       function_description)
        if is_distilled_model:
            model_type = self.get_distillation_model_type(model)
        else:
            model_type = self.get_teacher_model_type(model)

        buffer = ""
        async for choice in self.synthesise_answer_async(prompt, model, model_type, llm_parameters):
            delta = choice.get('choices', [{}])[0].get('delta', {})
            content_chunk = delta.get('content', '')
            buffer += content_chunk

            if not buffer:
                continue

            # Convert set representation to JSON-compatible list
            #if buffer.startswith("{'") and "', '" in buffer or buffer.startswith('{"') and '", "' in buffer:
            #    buffer = '[' + buffer[1:]

            # Use ijson to parse buffer as a stream
            try:
                parser = ijson.parse(io.StringIO(buffer))

                stack = []
                key = None
                for prefix, event, value in parser:
                    if event == 'map_key':
                        key = value
                    elif event in ('start_map', 'start_array'):
                        new_obj = [] if event == 'start_array' else {}
                        if stack:
                            parent_key, parent_obj = stack[-1]
                            if isinstance(parent_obj, list):
                                parent_obj.append(new_obj)
                            elif parent_key is not None:
                                parent_obj[parent_key] = new_obj
                        stack.append((key, new_obj))  # Initially set key as None
                    elif event in ('end_map', 'end_array'):
                        key, obj = stack.pop()
                        # Handle the case where obj is a list of strings and we are at the top level
                        if not stack and isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                            for item in obj:
                                is_instantiable = self.validator.check_type(item,
                                                                            function_description.output_class_definition)
                                if is_instantiable:
                                    output = LanguageModelOutput(item, save_to_finetune, is_distilled_model)
                                    yield output
                            buffer = ""  # Reset buffer for next object
                    elif prefix:
                        parent_key, current_obj = stack[-1]
                        if isinstance(current_obj, list):
                            # Check if we are at the top level and handling a list of strings
                            if len(stack) == 1 and isinstance(value, str):
                                output_type_args = get_args(function_description.output_type_hint)
                                if output_type_args:
                                    output_type_arg = output_type_args[0]
                                else:
                                    output_type_arg = Any
                                is_instantiable = self.validator.check_type(value, output_type_arg)
                                if is_instantiable:
                                    output = LanguageModelOutput(value, save_to_finetune, is_distilled_model)
                                    yield output
                                    buffer = "[" + buffer[len(json.dumps(value))+2:].lstrip(', ')
                            else:
                                current_obj.append(value)
                        else:
                            current_obj[key] = value

            except ijson.JSONError as e:
                # Not enough data to constitute a complete JSON object, continue reading more data
                pass



    async def synthesise_answer_async(self, prompt, model, model_type, llm_parameters):
        if model_type == "openai":
            async for chunk in self.api_models[model_type].generate_async(model, self.system_message, prompt, **llm_parameters):
                yield chunk

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