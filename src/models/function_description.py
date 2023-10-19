import hashlib
from dataclasses import dataclass
from typing import Dict

import utils


@dataclass(frozen=True)
class FunctionDescription:
    name: str
    docstring: str
    input_type_hints: Dict[str, type]
    input_class_definitions: Dict[str, str]
    output_type_hint: type
    output_class_definition: str

    def __hash__(self):
        json_encoded = utils.json_dumps(self).encode('utf-8')
        h = hashlib.md5(json_encoded).hexdigest()
        return str(h)