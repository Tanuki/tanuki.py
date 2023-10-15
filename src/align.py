import dataclasses
import datetime
import hashlib
import json
import os
import typing
from dataclasses import dataclass
import logging
import sys
from typing import Literal, Optional, get_type_hints, get_origin, get_args, _UnionGenericAlias
from unittest.mock import patch
from functools import wraps
import dis
import inspect
import openai
from dotenv import load_dotenv

load_dotenv()

# Define a new level
ALIGN_LEVEL_NUM = 15
logging.addLevelName(ALIGN_LEVEL_NUM, "ALIGN")


def json_default(thing):
    try:
        return dataclasses.asdict(thing)
    except TypeError:
        pass
    if isinstance(thing, datetime.datetime):
        return thing.isoformat(timespec='microseconds')
    if isinstance(thing, type):
        return thing.__name__
    if isinstance(thing, _UnionGenericAlias):
        return {
            "Union": [json_default(arg) for arg in get_args(thing)]
        }
    if thing == Literal[...]:
        return {
            "Literal": thing.__args__
        }
    if isinstance(thing, type(None)):
        return "None"
    if isinstance(thing, typing._SpecialForm):
        return thing._name
    if isinstance(thing, typing._GenericAlias):
        return {
            "GenericAlias": [json_default(arg) for arg in get_args(thing)]
        }
    if isinstance(thing, str):
        return thing
    if isinstance(thing, list) or isinstance(thing, tuple) or isinstance(thing, set):
        return [json_default(item) for item in thing]
    if isinstance(thing, dict):
        return {json_default(key): json_default(value) for key, value in thing.items()}

    raise TypeError(f"object of type {type(thing).__name__} not serializable")


def json_dumps(thing):
    return json.dumps(
        thing,
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(',', ':'),
    )


@dataclass(frozen=True)
class FunctionDescription:
    name: str
    docstring: str
    input_type_hints: dict[str, type]
    input_class_definitions: dict[str, str]
    output_type_hint: type
    output_class_definition: str

    def __hash__(self):
        json_encoded = json_dumps(self).encode('utf-8')
        h = hashlib.md5(json_encoded).hexdigest()
        return str(h)


@dataclass(frozen=True)
class FunctionExample:
    args: tuple
    kwargs: dict
    output: typing.Any


ALIGN_FILE_NAME = ".align"


def log_align(self, message, *args, **kws):
    if self.isEnabledFor(ALIGN_LEVEL_NUM):
        log_msg = str(message) + str(args) + str(kws)
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


# Add the new method to Logger class
logging.Logger.align = log_align

# Set up basic configuration
logging.basicConfig(level=ALIGN_LEVEL_NUM)

# Use the new level
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


def load_function_description_from_name(func_name) -> FunctionDescription:
    """
    Load a function description from a function name from the global scope.
    :param func_name:
    :return:
    """
    func_object = globals().get(func_name)
    return load_function_description(func_object)


def load_function_description(func_object) -> FunctionDescription:
    """
    Create a function description from a function object that can be used to register the function.
    :param func_object:
    :return:
    """
    sig = inspect.signature(func_object)
    type_hints = get_type_hints(func_object)

    # Extract input type hints and output type hint
    input_type_hints = {k: v for k, v in type_hints.items() if k in sig.parameters}
    output_type_hint = type_hints.get('return')

    # Fetch the docstring
    docstring = func_object.__doc__.strip() if func_object.__doc__ else ""

    def get_class_definition(class_type):
        """Helper function to get class definition source if not a built-in type"""
        if hasattr(class_type, "__origin__"):  # Check if it's a generic type
            origin_type = class_type.__origin__
            if hasattr(class_type, "__args__"):  # Access inner types
                return [get_class_definition(arg) for arg in class_type.__args__ if arg is not None]
        elif inspect.isclass(class_type) and class_type.__module__ != "builtins":
            return inspect.getsource(class_type)
        return class_type.__name__

    # Extract class definitions for input and output types
    input_class_definitions = {
        param_name: get_class_definition(param_type)
        for param_name, param_type in input_type_hints.items()
    }

    output_class_definition = get_class_definition(output_type_hint)

    return FunctionDescription(
        name=func_object.__name__,
        docstring=docstring,
        input_type_hints=input_type_hints,
        output_type_hint=output_type_hint,
        input_class_definitions=input_class_definitions,
        output_class_definition=output_class_definition
    )


def func(test_func):
    def validate_output(output: str, type_definition) -> bool:
        if output is None:
            if type(None) in get_args(type_definition):
                return True
            return False

        origin = get_origin(type_definition)
        args = get_args(type_definition)

        if origin == Literal:
            return output in args

        # Handling list types
        if origin == List:
            try:
                deserialized_output = json.loads(output)
                if not isinstance(deserialized_output, list):
                    return False
                return all(validate_output(json.dumps(item), args[0]) for item in deserialized_output)
            except json.JSONDecodeError:
                return False


        # For nested types, recursively validate each arg
        if args:
            return any(validate_output(output, arg) for arg in args)

        return False

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        function_description = load_function_description(test_func)
        f = str(function_description.__dict__)
        instruction = "Optionally convert the input into the output type, using the docstring as a guide. Return None if you can't."
        warning = "INCREDIBLY IMPORTANT: Only output a JSON string in the correct response format."
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

        valid = validate_output(choice, function_description.output_type_hint)


        return choice  # test_func(*args, **kwargs)

    wrapper._is_alignable = True
    return wrapper


def align(test_func):
    @wraps(test_func)
    def wrapper(instance, *args, **kwargs):
        instructions = list(dis.Bytecode(test_func))
        mock_behaviors = {}
        func_args = []
        stack_variables = {}
        mockable_functions = []

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

            # ... add more conditions as needed ...

            return attributes

        def create_mock_func(func_name, description: FunctionDescription):
            def mock_func(*args, **kwargs):
                hashed_description = description.__hash__()

                result = globals()[func_name](*args, **kwargs)

                # Extract attributes from the result
                attributes = extract_attributes(result)
                for attr_name, attr_value in attributes.items():
                    # If the attribute is a list, get its length
                    if isinstance(attr_value, list):
                        attributes[attr_name] = len(attr_value)

                # logger.align(
                #    f"'{func_name}({hashed_description})' - args: {args}, kwargs: {kwargs}, expects: {mock_behaviors.get(args, None)}")
                logger.align(hashed_description, args, kwargs, mock_behaviors.get(args, None))
                return mock_behaviors.get(args, None)

            return mock_func

        # Identify all functions that need to be patched based on mock_behaviors
        function_names_to_patch = set(
            [func for func in mockable_functions if getattr(globals().get(func, None), '_is_alignable', False)])
        # Identify all functions that need to be patched based on mock_behaviors
        funcs_to_patch = [func for func in mockable_functions if
                          hasattr(instance, func) and getattr(getattr(instance, func), '_is_alignable', False)]

        functions_descriptions = [load_function_description_from_name(func_name) for func_name in
                                  function_names_to_patch]

        patched_func = test_func
        for desc, func in zip(functions_descriptions, function_names_to_patch):
            mock_function = create_mock_func(func, desc)
            module_name = sys.modules[test_func.__module__].__name__
            # patched_func = patch(f'{module_name}.{func}', new=mock_function)(patched_func)
            patched_func = patch.object(instance, func, new=mock_function)(patched_func)

        # Get the signature of the function
        sig = inspect.signature(test_func)
        first_param_name = next(iter(sig.parameters))

        # If the instance is the "self" or the name of the first parameter,
        # then pass it as the first argument
        if first_param_name in ['self', 'cls'] or first_param_name == instance:
            return patched_func(instance, *args, **kwargs)
        else:
            return patched_func(*args, **kwargs)

    return wrapper


@func
def classify_sentiment_2(input: str, input_2: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the inputs are positive or negative sentiment, or None
    """


@func
def classify_sentiment(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """


@align
def align_classify_sentiment():
    """We can test the function as normal using Pytest or Unittest"""

    i_love_you = "I love you"
    print(classify_sentiment_2(i_love_you, "I love woo"))
    assert classify_sentiment_2(i_love_you, "I love woo") == 'Good'

    print(classify_sentiment("I love you"))
    assert classify_sentiment("I love you") == 'Good'

    assert classify_sentiment("I hate you") == 'Bad'
    assert classify_sentiment("I hate you") != 'Good'
    assert not classify_sentiment("Wednesdays are in the middle of the week")


if __name__ == '__main__':
    align_classify_sentiment()
