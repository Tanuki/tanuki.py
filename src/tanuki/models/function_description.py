import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Literal

from tanuki.models.function_type import FunctionType
from tanuki.utils import json_dumps

@dataclass(frozen=True)
class FunctionDescription:
    name: str
    docstring: str
    input_type_hints: Dict[str, type]
    input_class_definitions: Dict[str, str]
    output_type_hint: type
    output_class_definition: Optional[str]
    type: FunctionType = FunctionType.SYMBOLIC

    def __hash__(self, purpose: str = "general"):
        if purpose == "general": 
            json_encoded = json_dumps(self).encode('utf-8')
            h = hashlib.md5(json_encoded).hexdigest()
            return str(h)
        if purpose == "finetune": 
            json_encoded = json_dumps(self).encode('utf-8')
            h = hashlib.shake_256(json_encoded).hexdigest(8)
            return str(h)