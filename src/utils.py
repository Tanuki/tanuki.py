import dataclasses
import datetime
import json
import typing
from typing import get_args, Literal


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
