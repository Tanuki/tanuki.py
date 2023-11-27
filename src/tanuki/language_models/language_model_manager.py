import json
from typing import Any

from tanuki.function_modeler import FunctionModeler
from tanuki.language_models.openai_api import OpenAI_API
from tanuki.models.function_description import FunctionDescription
from tanuki.models.function_example import FunctionExample
from tanuki.models.language_model_output import LanguageModelOutput
from tanuki.utils import approximate_token_count
from tanuki.validator import Validator


class LanguageModelManager(object):
    """
    The LanguageModelManager is responsible for managing the language models and their outputs operationally,
    this includes:
    - Generating outputs from the language models
    - Repairing outputs from the language models
    - Saving outputs from the language models
    - Finetuning the language models from the saved outputs
    """

    def __init__(self, function_modeler: FunctionModeler, generation_token_limit=512) -> None:
        self.function_modeler = function_modeler
        self.instruction = "You are given below a function description and input data. The function description of what the function must carry out can be found in the Function section, with input and output type hints. The input data can be found in Input section. Using the function description, apply the function to the Input and return a valid output type, that is acceptable by the output_class_definition and output_class_hint. Return None if you can't apply the function to the input or if the output is optional and the correct output is None.\nINCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format."
        self.system_message = f"You are a skillful and accurate language model, who applies a described function on input data. Make sure the function is applied accurately and correctly and the outputs follow the output type hints and are valid outputs given the output types."

        self.instruction_token_count = approximate_token_count(self.instruction)
        self.system_message_token_count = approximate_token_count(self.system_message)
        self.api_models = {"openai": OpenAI_API()}
        self.repair_instruction = "Below are an outputs of a function applied to inputs, which failed type validation. The input to the function is brought out in the INPUT section and function description is brought out in the FUNCTION DESCRIPTION section. Your task is to apply the function to the input and return a correct output in the right type. The FAILED EXAMPLES section will show previous outputs of this function applied to the data, which failed type validation and hence are wrong outputs. Using the input and function description output the accurate output following the output_class_definition and output_type_hint attributes of the function description, which define the output type. Make sure the output is an accurate function output and in the correct type. Return None if you can't apply the function to the input or if the output is optional and the correct output is None."
        self.generation_length = generation_token_limit
        self.models = {
            "gpt-4-1106-preview": {
                "token_limit": 128000 - self.generation_length, "type": "openai"
            },
            "gpt-4": {
                "token_limit": 8192 - self.generation_length, "type": "openai"
            },
            "gpt-4-32k": {
                "token_limit": 32768 - self.generation_length, "type": "openai"}
        }  # models and token counts

    def __call__(self,
                 args,
                 function_description: FunctionDescription,
                 kwargs,
                 validator: Validator) -> Any:
        output = self.generate(args, kwargs, function_description)
        # start parsing the object, very hacky way for the time being
        choice_parsed = self._parse_choice(output)
        valid = validator.check_type(choice_parsed, function_description.output_type_hint)
        if not valid:
            choice, choice_parsed, successful_repair = self.repair_output(self,
                                                                          args,
                                                                          kwargs,
                                                                          function_description,
                                                                          output.generated_response,
                                                                          validator)

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

        prompt, model, save_to_finetune, is_distilled_model = self.get_generation_case(args, kwargs,
                                                                                       function_description)
        if is_distilled_model:
            model_type = self._get_distillation_model_type(model)
        else:
            model_type = self._get_teacher_model_type(model)
        choice = self._synthesise_answer(prompt, model, model_type, llm_parameters)

        output = LanguageModelOutput(choice, save_to_finetune, is_distilled_model)
        return output

    def _synthesise_answer(self, prompt, model, model_type, llm_parameters):
        """
        Synthesise an answer given the prompt, model, model_type and llm_parameters
        """
        if model_type == "openai":
            return self.api_models[model_type].generate(model, self.system_message, prompt, **llm_parameters)
        else:
            raise NotImplementedError("Only OpenAI is supported currently. " + \
                                      "Please feel free to raise a PR to support development")

    def _get_distillation_model_type(self, model):
        """
        Get the distilled model type given the model
        """
        # currently only openai is supported
        return "openai"

    def _get_teacher_model_type(self, model):
        """
        Get the teacher model type given the model
        """
        # check if model is in the models
        if model in self.models.keys():
            return self.models[model]["type"]
        else:
            raise ValueError("This teacher model is not supported")

    def get_generation_case(self, args, kwargs, function_description):
        """
        Get the generation case with the correct prompt and model
        First get the current model, then if distilled model, do zero-shot prompt and return False as suitable_for_finetune
        If not distilled model, check if suitable for finetuning, create the prompt and return the correct model given the token count
        """
        f = str(function_description.__dict__.__repr__())

        distilled_model, teacher_models = self.function_modeler.get_models(function_description)
        is_distilled_model = distilled_model != ""
        suitable_for_distillation, input_prompt_token_count = self.suitable_for_finetuning_token_check(args, kwargs, f,
                                                                                                       self.function_modeler.distillation_token_limit)
        # no examples needed, using a finetuned model. Dont save to finetune dataset
        if is_distilled_model and suitable_for_distillation:
            prompt = self.construct_prompt(f, args, kwargs, None)
            return prompt, distilled_model, suitable_for_distillation, True

        else:
            aligns = self.function_modeler.get_symbolic_alignments(function_description.__hash__(), max=16)
            examples = "\n".join(
                [f"Inputs:\nArgs: {align['args']}\nKwargs: {align['kwargs']}\nOutput: {align['output']}" for align in
                 aligns])
            prompt = self.construct_prompt(f, args, kwargs, examples)
            examples_token_count = approximate_token_count(examples)
            total_token_count = examples_token_count + input_prompt_token_count + self.instruction_token_count + self.system_message_token_count
            model = self.choose_model_from_tokens(teacher_models, total_token_count)
            if model:
                return prompt, model, suitable_for_distillation, False
            else:
                raise ValueError(
                    "The input content and align statements combined are too long, please shorten it. The maximum currently allowed token limit is 32000")

    def suitable_for_finetuning_token_check(self, args, kwargs, f, distillation_token_count):
        """
        Check if the inputs are suitable for finetuning, i.e are below the finetuning token count
        """
        # check if finetunable
        finetuning_prompt = f"Function: {f}\n---\nInputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:"
        input_prompt_token_count = approximate_token_count(finetuning_prompt)
        suitable_for_finetune = input_prompt_token_count + self.instruction_token_count + self.system_message_token_count < distillation_token_count
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
            model_type = self._get_teacher_model_type(model)
            choice = self._synthesise_answer(prompt, model, model_type, {})
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
        prompt = f"{self.repair_instruction}\nFUNCTION DESCRIPTION: {f}\n{successful_examples}---Inputs:\nArgs: {args}\nKwargs: {kwargs}\nFAILED EXAMPLES: {failed_examples}Correct output:"
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

    def repair_output(self,
                      args: tuple,
                      kwargs: dict,
                      function_description: FunctionDescription,
                      choice,
                      validator: Validator) -> tuple:
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
            examples = "\n".join(
                [f"Inputs:\nArgs: {align['args']}\nKwargs {align['kwargs']}\nOutput: {align['output']}" for align in
                 aligns])
            # Generate the reparied LLM output
            choice = self.repair_generate(args, kwargs, f, failed_outputs_list, examples, teacher_models)
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

        return choice, choice_parsed, valid
