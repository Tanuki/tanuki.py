import dataclasses
import datetime
import json
import math
import typing
from typing import _UnionGenericAlias, get_args, Literal


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


def optimal_bloom_filter_params(n, p):
    """
    Calculate the optimal bit array size (m) and number of hash functions (k)
    for a Bloom filter.

    n: expected number of items to be stored
    p: acceptable false positive probability

    Returns a tuple (m, k)
    """
    m = - (n * math.log(p)) / (math.log(2) ** 2)
    k = (m / n) * math.log(2)
    return int(math.ceil(m)), int(math.ceil(k))
