import dataclasses
import datetime
import inspect
import json
import typing
from typing import get_args, Literal
import string
import types

def json_default(thing):
    try:
        return dataclasses.asdict(thing)
    except TypeError:
        pass
    if isinstance(thing, datetime.datetime):
        return thing.isoformat(timespec='microseconds')
    if isinstance(thing, type):
        return thing.__name__
    #if hasattr(typing, "_GenericAlias") and isinstance(thing, typing._GenericAlias):
    if hasattr(typing, "_UnionGenericAlias"):
        if isinstance(thing, typing._UnionGenericAlias):
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
    if isinstance(thing, typing._GenericAlias) or isinstance(thing, types.GenericAlias):
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


def get_model(content, logger, func_hash):
    """
    Get the model from the content and the logger.
    Decide on model depending on the length of the content. if is finetunable, return model, true, otherwise return model, false
    Args:
        content (str): the content to be aligned
        logger (buffered logger): the logger
        func_hash (str): the function hash
    Returns:
        model (str): the model to be used
        finetunable (bool): whether the model is finetunable
    """
    num_tokens = approximate_token_count(content)
    finetune_limit = logger.finetune_token_limit
    finetune_model, teacher_models = logger.get_models(func_hash)
    if num_tokens < finetune_limit:
        return finetune_model, True
    else:
        # this is just for backwards compatibility currently
        if len(teacher_models) == 0 or isinstance(teacher_models[0], str):
            teacher_models = [("gpt-4", 7000),("gpt-4-32k", 31000)]
        
        for model, token_limit in teacher_models:
            if num_tokens < token_limit:
                return model, False
        raise ValueError("The input content and align statements combined are too long, please shorten it. The maximum currently allowed token limit is 32000")
    

def approximate_token_count(content):
    """
    Approximate the token count of input
    Number of tokens is word tokens (nr of words * 1.33) + nr of special characters (which are usually their own tokens)
    Args:
        content (str, bytes): the content to be approximated
    Returns:
        number_of_tokens (int): the number of tokens
    """
    common_special_characters = r"\/(){}[]<>|`~@#$%^&*+=-_:;\""
    # check if input type is string
    if isinstance(content, str):
        number_of_word_tokens = int(len(content.split(" "))*1.333)
        nr_of_special_characters = sum([content.count(char) for char in common_special_characters])
        return number_of_word_tokens + nr_of_special_characters
    # check if input is a byte string
    if isinstance(content, bytes):
        number_of_word_tokens = int(len(content.split(b" "))*1.333)
        nr_of_special_characters = sum([content.count(char.encode("utf-8")) for char in common_special_characters])
        return number_of_word_tokens + nr_of_special_characters


def _deep_tuple(obj):
    """
    Convert a list or dict to a tuple recursively to allow for hashing and becoming a key for mock_behaviors
    :param obj:
    :return:
    """
    # transform pydantic objects into dicts
    if hasattr(obj, "__dict__"):
        obj = obj.__dict__
    if isinstance(obj, list) or isinstance(obj, tuple):
        return tuple(_deep_tuple(e) for e in obj)
    elif isinstance(obj, dict):
        return tuple((k, _deep_tuple(v)) for k, v in sorted(obj.items()))
    else:
        return obj


def get_key(args, kwargs) -> tuple:
    args_tuple = _deep_tuple(args)
    kwargs_tuple = _deep_tuple(kwargs)
    return args_tuple, kwargs_tuple


def prepare_object_for_saving(input_object):
    """
    Get a dictionary representation of the object
    """
    # check if list
    if isinstance(input_object, list):
        return [prepare_object_for_saving(item) for item in input_object]
    # check if tuple
    elif isinstance(input_object, tuple):
        return tuple([prepare_object_for_saving(item) for item in input_object])
    # check if dict
    elif isinstance(input_object, dict):
        return {key: prepare_object_for_saving(value) for key, value in input_object.items()}
    # check if pydantic object
    if hasattr(input_object, "__dict__"):
        attributes = input_object.__dict__
        for key, value in attributes.items():
            attributes[key] = prepare_object_for_saving(value)
        return attributes
        # 
    # check if datetime for custom logic
    elif isinstance(input_object, datetime.datetime):
        attrs = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'tzinfo']
        attributes = {attr: getattr(input_object, attr, None) for attr in attrs}
        return attributes

    return input_object
def encode_int(n):
    # Define the character set for encoding
    charset = string.ascii_lowercase + string.digits + "_"
    return charset[n]


def decode_int(s):
    # Define the character set for encoding
    charset = string.ascii_lowercase + string.digits + "_"
    return charset.index(s)


def _get_source_ipython(func) -> str:
    """
    Get the source code of a function from IPython (to support Colab and Jupyter notebooks)
    :param func: The function to get the source code from
    :return: The source code of the function
    """
    # Get the IPython instance
    from IPython import get_ipython
    ipython = get_ipython()

    # Get the input history
    input_cells = ipython.history_manager.input_hist_parsed

    class_name = func.__name__
    source_code = None

    for cell in input_cells:
        if f"class {class_name}" in cell:
            source_code = cell
            break

    # If found, print the source code
    return source_code


def get_source(func) -> str:
    """
    Get the source code of a function
    Args:
        func (function): the function to get the source code from
    Returns:
        source (str): the source code of the function
    """
    try:
        return inspect.getsource(func)
    except Exception:
        return _get_source_ipython(func)