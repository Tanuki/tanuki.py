import dis
import inspect
import logging
import os
import sys
from functools import wraps
from typing import Optional
from unittest.mock import patch
import openai
from models.function_description import FunctionDescription
from models.function_example import FunctionExample
from register import Register
from trackers.buffered_logger import BufferedLogger
from validator import Validator


# Define a new level
def _log_align(self, message, *args, **kws):
    if self.isEnabledFor(ALIGN_LEVEL_NUM):
        args, kwargs, output = args
        example = FunctionExample(args, kwargs, output)

        # Define a safe directory within the project for logs
        # (You can make this configurable if needed)
        log_directory = os.path.join(os.getcwd(), ALIGN_FILE_NAME)

        # Ensure the directory exists
        if not os.path.exists(log_directory):
            try:
                os.makedirs(log_directory)
            except OSError as e:
                self.error(f"Failed to create log directory: {e}")
                return

        # Write to the file
        log_file_path = os.path.join(log_directory, message)
        try:
            with open(log_file_path, "a") as f:
                f.write(str(example.__dict__) + "\n")
        except IOError as e:
            self.error(f"Failed to write to log file: {e}")


# Set up logging with custom logger
def logger_factory(name):
    return BufferedLogger(name)


ALIGN_LEVEL_NUM = 15
PATCH_LEVEL_NUM = 14
logging.addLevelName(ALIGN_LEVEL_NUM, "ALIGN")
logging.addLevelName(PATCH_LEVEL_NUM, "PATCH")

ALIGN_FILE_NAME = ".align"

# Set up basic configuration
logging.setLoggerClass(BufferedLogger)
logging.basicConfig(level=ALIGN_LEVEL_NUM)

logger = logger_factory(__name__)

alignable_functions = {}

class Monkey:

    @staticmethod
    def _load_alignments():
        pass

    @staticmethod
    def align(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            instructions = list(dis.Bytecode(test_func))

            if args:
                instance = args[0]
                args = args[1:]
            else:
                instance = None

            mock_behaviors = {}
            func_args = []
            stack_variables = {}
            mockable_functions = []

            # Iterate through the instructions in the monkey-patched function
            for idx, instruction in enumerate(instructions):

                # Identify method calls on the instance
                if instruction.opname in ('LOAD_METHOD', 'LOAD_ATTR'):
                    func_name = instruction.argval
                    module_name = instruction.argval.split('.')[0]
                    mockable_functions.append(func_name)
                # If we are loading a reference to a monkey function, register it to the list of mockable functions
                elif instruction.opname == 'LOAD_GLOBAL':
                    func_name = instruction.argval
                    mockable_functions.append(func_name)
                # If we are assigning a variable, add it to the stack
                elif instruction.opname == 'STORE_FAST':
                    stack_variables[instruction.argval] = func_args.pop()
                # If we are loading a variable, add it to the stack
                elif instruction.opname.startswith('LOAD_'):
                    if instruction.argval in stack_variables:
                        func_args.append(stack_variables[instruction.argval])
                    else:
                        func_args.append(instruction.argval)
                # If we are calling a function we need to mock (e.g preceded by an assert),
                # pop the arguments off the stack and register the expected output
                elif instruction.opname.startswith(('CALL_FUNCTION', 'CALL_METHOD')):
                    num_args = instruction.arg
                    args_for_call = func_args[-num_args:]
                    func_args = func_args[:-num_args]

                    # Search for the next COMPARE_OP with '==' after CALL_FUNCTION
                    for next_idx in range(idx + 1, len(instructions)):
                        next_instruction = instructions[next_idx]
                        if next_instruction.opname == 'POP_TOP':
                            break
                        if next_instruction.opname == 'COMPARE_OP':
                            if next_instruction.argval == '==' or next_instruction.argval == 'is':
                                expected_value = instructions[next_idx - 1].argval
                                mock_behaviors[tuple(args_for_call)] = expected_value
                                break
                        # elif next_instruction.argval == '!=' or next_instruction.argval == 'is not':
                        #     expected_value = instructions[next_idx - 1].argval
                        #     mock_behaviors[tuple(args_for_call)] = expected_value
                        #     break

            def extract_attributes(result):
                attributes = {}

                # If the result is a list, get its length
                if isinstance(result, list):
                    attributes['length'] = len(result)

                # If the result is a dictionary, get its keys (or any other attributes)
                elif isinstance(result, dict):
                    attributes['keys'] = list(result.keys())

                return attributes

            def create_mock_func(instance: Optional,
                                 func_name: str,
                                 description: FunctionDescription):

                def mock_func(*args, **kwargs):
                    hashed_description = description.__hash__()

                    if not instance:
                        result = Register.get(func_name)(*args, **kwargs)
                    else:
                        result = Register.get(func_name)(instance, *args, **kwargs)

                    # Extract attributes from the result
                    attributes = extract_attributes(result)
                    for attr_name, attr_value in attributes.items():
                        # If the attribute is a list, get its length
                        if isinstance(attr_value, list):
                            attributes[attr_name] = len(attr_value)

                    mocked_behaviour = mock_behaviors.get(args, None)
                    logger.log_align(hashed_description, args, kwargs, mocked_behaviour)
                    return mocked_behaviour

                return mock_func

            function_names_to_patch = Register.function_names_to_patch()

            # Identify all functions that need to be patched based on mock_behaviors
            if instance:
                functions_descriptions = [Register.load_function_description_from_name(instance, func_name)
                                          for func_name in function_names_to_patch]

            else:
                functions_descriptions = [Register.load_function_description_from_name(func_name)
                                          for func_name in function_names_to_patch]

            patched_func = test_func
            for desc, func in zip(functions_descriptions, function_names_to_patch):
                mock_function = create_mock_func(instance, func, desc)
                module_name = sys.modules[test_func.__module__].__name__

                if instance:
                    patched_func = patch.object(instance, func, new=mock_function)(patched_func)
                else:
                    patched_func = patch(f'{module_name}.{func}', new=mock_function)(patched_func)

            # Get the signature of the function
            sig = inspect.signature(test_func)

            if sig.parameters:
                first_param_name = next(iter(sig.parameters))

                # If the instance is the "self" or the name of the first parameter,
                # then pass it as the first argument
                if first_param_name in ['self', 'cls'] or first_param_name == instance:
                    return patched_func(instance, *args, **kwargs)
                else:
                    return patched_func(*args, **kwargs)
            else:
                return patched_func(*args, **kwargs)

        return wrapper

    @staticmethod
    def patch(test_func):
        Monkey._load_alignments()

        @wraps(test_func)
        def wrapper(*args, **kwargs):
            function_description = Register.load_function_description(test_func)

            # f = json_dumps(function_description.__dict__)
            f = str(function_description.__dict__.__repr__() + "\n")
            instruction = "Optionally convert the input into the output type, using the docstring as a guide. Return None if you can't."
            warning = "INCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format."
            content = f"{instruction}\n{warning}\nFunction: {f}\nInput: {args}\nOutput:"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            choice = response.choices[0].message.content.strip("'")
            validator = Validator()

            valid = validator.check_type(choice, function_description.output_type_hint)

            if not valid:
                raise TypeError(
                    f"Output type was not valid. Expected an object of type {function_description.output_type_hint}, got '{choice}'")

            logger.log_patch(function_description.__hash__(), args, kwargs, choice)

            instantiated = validator.instantiate(choice, function_description.output_type_hint)

            return instantiated  # test_func(*args, **kwargs)

        wrapper._is_alignable = True
        Register.add_function(test_func, wrapper)
        return wrapper
