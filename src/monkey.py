import dis
import inspect
import logging
import os
import sys
from functools import wraps
from importlib import import_module
from typing import Optional
from unittest.mock import patch
import openai
from pydantic import BaseModel

from models.function_description import FunctionDescription
from models.function_example import FunctionExample
from register import Register
from todo_item import TodoItem
from trackers.buffered_logger import BufferedLogger
from validator import Validator
from repair import repair_output
import json


# Define a new level
def _log_align(self, func_hash, *args, **kws):
    if self.isEnabledFor(ALIGN_LEVEL_NUM):
        args, kwargs, output = args
        kwargs['align'] = True
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
        log_file_path = os.path.join(log_directory, func_hash)
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
ALIGN_FILE_NAME = ".align"

alignable_functions = {}

class Monkey:
    # Set up basic configuration
    logging.setLoggerClass(BufferedLogger)
    logging.addLevelName(ALIGN_LEVEL_NUM, "ALIGN")
    logging.addLevelName(PATCH_LEVEL_NUM, "PATCH")
    logging.basicConfig(level=ALIGN_LEVEL_NUM)
    logger = logger_factory(__name__)


    @staticmethod
    def _load_alignments():
        Monkey.logger.load_alignments()

    @staticmethod
    def align(test_func):

        def _deep_tuple(obj):
            """
            Convert a list or dict to a tuple recursively to allow for hashing and becoming a key for mock_behaviors
            :param obj:
            :return:
            """
            if isinstance(obj, list):
                return tuple(_deep_tuple(e) for e in obj)
            elif isinstance(obj, dict):
                return tuple((k, _deep_tuple(v)) for k, v in sorted(obj.items()))
            else:
                return obj

        def get_key(args, kwargs) -> tuple:
            args_tuple = _deep_tuple(args)
            kwargs_tuple = _deep_tuple(kwargs)
            return args_tuple, kwargs_tuple

        def construct(current_class, *args, **kwargs):
            if isinstance(current_class, BaseModel):
                try:
                    constructed_object = current_class.model_validate(kwargs)
                except TypeError:
                    raise TypeError(f"Fatal: TypeError when constructing object of type {current_class}")
                    constructed_object = None
            else:
                try:
                    constructed_object = current_class(*args, **kwargs)  # Construct the object
                except TypeError as e:
                    raise TypeError(f"Fatal: TypeError when constructing object of type {current_class}", e)
                    constructed_object = None
            return constructed_object

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
            co_consts = test_func.__code__.co_consts
            kwarg_names = []
            current_class = None
            constructor_stack = []
            current_class_stack = []

            # Iterate through the instructions in the monkey-patched function
            for idx, instruction in enumerate(instructions):

                # Identify method calls on the instance
                if instruction.opname in ('LOAD_METHOD', 'LOAD_ATTR'):
                    func_name = instruction.argval
                    mockable_functions.append(func_name)

                # If we are loading a reference to a monkey function, register it to the list of mockable functions
                elif instruction.opname == 'LOAD_GLOBAL':
                    func_name = instruction.argval
                    mockable_functions.append(func_name)

                    if instruction.argrepr.startswith('NULL +'):
                        _globals = globals()
                        # This is either a class or a function
                        _mounted_class = _globals.get(func_name, None) # This will be None if the function is not a class

                        if _mounted_class:
                            current_class_stack.append(_mounted_class)
                        pass

                # If we are assigning a variable, add it to the stack
                elif instruction.opname == 'STORE_FAST':
                    stack_variables[instruction.argval] = func_args.pop()

                # Build lists
               # elif instruction.opname == 'LIST_EXTEND':
               #     num_elements = instruction.arg
               #     list_to_extend = func_args[-1]  # assuming the list to extend is on top of the stack
                #    elements = func_args[-num_elements - 1:-1]  # elements to extend the list with
                #    list_to_extend.extend(elements)
                #    func_args = func_args[:-num_elements]  # remove the elements from func_args

                # If we are loading a variable, add it to the stack
                elif instruction.opname.startswith('LOAD_'):
                    if instruction.argval in stack_variables:
                        func_args.append(stack_variables[instruction.argval])
                    else:
                        func_args.append(instruction.argval)

                elif instruction.opname == 'KW_NAMES':
                    kwarg_names = co_consts[instruction.arg]
                    print(f"Debug: kwarg_names = {kwarg_names}")


                # Handle 'BUILD_LIST', 'BUILD_SET', 'BUILD_TUPLE'
                elif instruction.opname in ('BUILD_LIST', 'BUILD_SET', 'BUILD_TUPLE'):
                    num_elements = instruction.arg
                    if num_elements == 0:
                        continue
                    elements = func_args[-num_elements:]
                    func_args = func_args[:-num_elements]
                    constructed_data = elements if instruction.opname == 'BUILD_LIST' else set(
                        elements) if instruction.opname == 'BUILD_SET' else tuple(elements)
                    func_args.append(constructed_data)

                # Handle 'BUILD_MAP', 'BUILD_CONST_KEY_MAP'
                elif instruction.opname in ('BUILD_MAP', 'BUILD_CONST_KEY_MAP'):
                    num_elements = instruction.arg
                    if instruction.opname == 'BUILD_MAP':
                        key_values = dict(zip(func_args[-2 * num_elements::2], func_args[-2 * num_elements + 1::2]))
                    else:  # 'BUILD_CONST_KEY_MAP'
                        keys = func_args.pop()  # assuming keys are on top of the stack as a tuple
                        values = func_args[-num_elements:]
                        key_values = dict(zip(keys, values))
                    func_args = func_args[:-num_elements]
                    func_args.append(key_values)

                # If we are calling a function we need to mock (e.g preceded by an assert),
                # pop the arguments off the stack and register the expected output
                elif instruction.opname.startswith('CALL'):
                    num_args = instruction.arg
                    args_for_call, func_args, kwargs_for_call = _get_args(func_args, kwarg_names, num_args)
                    constructed_object = None
                    func_args = func_args[:-num_args]  # Remove the arguments from func_args
                    if current_class_stack:
                        current_class = current_class_stack[-1]
                        constructed_object = construct(current_class, *args_for_call, **kwargs_for_call)
                        constructor_stack.append(constructed_object)  # Push the newly constructed object to stack
                        current_class_stack.pop()
                        #func_args.append(constructed_object)
                        #current_class = None
                    #elif constructor_stack:
                    #    constructed_object = constructor_stack.pop()  # Pop the last object
                        #func_args.append(constructed_object)

                    # Search for the next COMPARE_OP with '==' after CALL
                    for next_idx in range(idx + 1, len(instructions)):
                        next_instruction = instructions[next_idx]
                        opname = next_instruction.opname
                        if opname == 'POP_TOP':
                            break
                        elif opname == 'KW_NAMES':
                            kwarg_names = co_consts[next_instruction.arg]
                        elif opname == 'COMPARE_OP':
                            if next_instruction.argval == '==' or next_instruction.argval == 'is':
                                #expected_value = func_args[-1]
                                #args_for_call2, func_args2, kwargs_for_call2 = _get_args(func_args, kwarg_names, num_args)
                                # this would be the last element on the stack
                                key = get_key(args_for_call, kwargs_for_call)
                                try:
                                    mock_behaviors[key] = func_args[-1] if not constructed_object else constructed_object
                                    #func_args.pop()
                                except TypeError as e:
                                    print("Debug: TypeError when trying to get key", e)
                                break
                        elif opname == 'LOAD_CONST':
                            func_args.append(next_instruction.argval)
                        elif opname in ('BUILD_LIST', 'BUILD_SET', 'BUILD_TUPLE'):
                            num_elements = next_instruction.arg
                            elements = func_args[-num_elements:]
                            func_args = func_args[:-num_elements]
                            constructed_data = elements if opname == 'BUILD_LIST' else set(
                                elements) if opname == 'BUILD_SET' else tuple(elements)
                            func_args.append(constructed_data)

                        elif instruction.opname == 'BUILD_MAP':
                            num_pairs = instruction.arg
                            pairs = [(func_args.pop(), func_args.pop()) for _ in range(num_pairs)]
                            func_args.append(dict(pairs[::-1]))  # Reversed because we popped from the stack

                        elif instruction.opname == 'BUILD_CONST_KEY_MAP':
                            num_values = instruction.arg
                            keys = func_args.pop()  # the top of the stack should contain a tuple of keys
                            values = [func_args.pop() for _ in range(num_values)]
                            func_args.append(dict(zip(keys, values[::-1])))  # Reversed because we popped from the stack

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

                    func = Register.get(func_name)
                    if not instance:
                        result = func(*args, **kwargs)
                    else:
                        result = func(instance, *args, **kwargs)

                    # Extract attributes from the result
                    attributes = extract_attributes(result)
                    for attr_name, attr_value in attributes.items():
                        # If the attribute is a list, get its length
                        if isinstance(attr_value, list):
                            attributes[attr_name] = len(attr_value)

                    key = get_key(args, kwargs)
                    mocked_behaviour = mock_behaviors.get(key, None)
                    Monkey.logger.log_align(hashed_description, args, kwargs, mocked_behaviour)
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

        def _get_args(func_args, kwarg_names, num_args):
            num_pos_args = num_args - len(kwarg_names)  # Calculate number of positional arguments
            args_for_call = func_args[:num_pos_args]
            # Pop keyword arguments off the stack
            kwargs_for_call = {}  # New dictionary to hold keyword arguments for the call
            for name in reversed(kwarg_names):  # Reverse to match the order on the stack
                try:
                    kwargs_for_call[name] = func_args.pop()  # Pop the value off the stack
                except IndexError:
                    print(f"Debug: func_args is empty, can't pop for {name}")
            func_args = func_args[:-num_pos_args]  # Remove the positional arguments from func_args
            return args_for_call, func_args, kwargs_for_call

        return wrapper

    @staticmethod
    def patch(test_func):
        Monkey._load_alignments()

        @wraps(test_func)
        def wrapper(*args, **kwargs):
            function_description = Register.load_function_description(test_func)
            model = Monkey.logger.get_model(function_description.__hash__())
            aligns = Monkey.logger.get_alignments(function_description.__hash__(), max=5)
            examples = "\n".join([f"Input: {align['args']}\nOutput: {align['output']}" for align in aligns])
            # f = json_dumps(function_description.__dict__)
            f = str(function_description.__dict__.__repr__() + "\n")
            instruction = "Optionally convert the input into the output type, using the docstring as a guide. Return None if you can't."
            warning = "INCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format."
            content = f"{instruction}\n{warning}\nFunction: {f}\nExamples:{examples}\n---\nInput: {args}\nOutput:"
            response = openai.ChatCompletion.create(
                model=model,
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
            # start parsing the object, WILL NEED TO BE CHANGED, VERY HACKY
            try:
                # json load
                choice_parsed = json.loads(choice)
            except:
                # if it fails, it's not a json object, try eval
                try:
                    choice_parsed = eval(choice)
                except: 
                    choice_parsed = choice

            validator = Validator()

            valid = validator.check_type(choice_parsed, function_description.output_type_hint)

            if not valid:
                error = f"Output type was not valid. Expected an object of type {function_description.output_type_hint}, got '{choice}'"
                choice, successful_repair = repair_output(args, function_description, choice, error, validator)

                if not successful_repair:
                    raise TypeError(f"Output type was not valid. Expected an object of type {function_description.output_type_hint}, got '{choice}'")

            datapoint = FunctionExample(args, kwargs, choice)
            Monkey.logger.postprocess_datapoint(function_description.__hash__(), f, datapoint, log = not valid)

            instantiated = validator.instantiate(choice, function_description.output_type_hint)

            return instantiated  # test_func(*args, **kwargs)

        wrapper._is_alignable = True
        Register.add_function(test_func, wrapper)
        return wrapper
