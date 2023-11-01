import hashlib
from dataclasses import dataclass
from typing import Dict

from monkey_patch.utils import json_dumps


@dataclass(frozen=True)
class FunctionDescription:
    name: str
    docstring: str
    input_type_hints: Dict[str, type]
    input_class_definitions: Dict[str, str]
    output_type_hint: type
    output_class_definition: str

    def __hash__(self):
        json_encoded = json_dumps(self).encode('utf-8')
        h = hashlib.md5(json_encoded).hexdigest()
        return str(h)