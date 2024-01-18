from typing import List, Union, Dict, Any

from tanuki.language_models.jsonformer.logits_processors import (
    NumberStoppingCriteria,
    OutputNumbersTokens,
    StringStoppingCriteria,
    LiteralStoppingCriteria,
    OutputLiteralsTokens,
)
from termcolor import cprint
from transformers import PreTrainedModel, PreTrainedTokenizer
import json
import copy
GENERATION_MARKER = "|GENERATION|"
"""
This was forked and developed upon 
https://github.com/1rgs/jsonformer
All credit for the great idea and implementation goes to the original author
"""

class Jsonformer:
    value: Dict[str, Any] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: Dict[str, Any],
        prompt: str,
        *,
        debug: bool = False,
        generation_params: dict,
        max_array_length: int = 100,
    ):
    
        """
        Initialize the Jsonformer class
        """
        self.model = model
        self.tokenizer = tokenizer
        # add a padding token to the tokenizer if it doesn't already have one
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.json_schema = json_schema
        self.prompt = prompt

        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)

        self.generation_marker = "|GENERATION|"
        self.supported_schema_types = ["int", "float", "bool", "literal", "str", "list", "tuple", "set", "pydantic_object"]
        self.debug_on = debug
        self.generation_params = generation_params

        self.max_array_length = max_array_length
        self.max_number_tokens = generation_params.get("max_new_tokens")
        self.generation_params.pop("max_new_tokens")
        self.temperature = generation_params["temperature"]
        self.generation_params.pop("temperature")
        self.locked_generation_params = ["temperature", 
                                         "max_new_tokens", 
                                         "do_sample", 
                                         "num_return_sequences", 
                                         "logits_processor", 
                                         "stopping_criteria",
                                         "pad_token_id"]
        self._remove_duplicate_generation_params()

    def _remove_duplicate_generation_params(self):
        """
        Remove the locked generation params from the generation params
        """
        for key in self.locked_generation_params:
            if key in self.generation_params:
                self.generation_params.pop(key)
    
    def debug(self, caller: str, value: str, is_prompt: bool = False):
        if self.debug_on:
            if is_prompt:
                cprint(caller, "green", end=" ")
                cprint(value, "yellow")
            else:
                cprint(caller, "green", end=" ")
                cprint(value, "blue")

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        """
        Customer decoding for number (int of float) generation
        """
        prompt = self.get_prompt()
        self.debug("[generate_number]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        temperature = temperature or self.temperature
        do_sample = True if temperature > 0 else False
        response = self.model.generate(
            input_tokens,
            do_sample = do_sample,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[
                NumberStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.generation_params
        )
        # Some models output the prompt as part of the response
        # This removes the prompt from the response if it is present
        if (
            len(response[0]) >= len(input_tokens[0])
            and (response[0][: len(input_tokens[0])] == input_tokens).all()
        ):
            response = response[0][len(input_tokens[0]) :]
        if response.shape[0] == 1:
            response = response[0]

        response = self.tokenizer.decode(response, skip_special_tokens=True)
        response = response.strip().rstrip(".")
        self.debug("[generate_number]", response)
        try:
            return float(response)
        except ValueError:
            if iterations > 3:
                raise ValueError("Failed to generate a valid number")

            return self.generate_number(temperature=temperature * 1.3, iterations=iterations+1)



    def generate_literal(self, literal: list,  temperature: Union[float, None] = None, iterations=0):
        """
        Customer decoding for literal (list of allowed options) generation
        """
        prompt = self.get_prompt()
        self.debug("[generate_literal]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        stopping_criteria = LiteralStoppingCriteria(self.tokenizer,
                                                    OutputLiteralsTokens(),
                                                    len(input_tokens[0]),
                                                    literal)
        temperature = temperature or self.temperature
        do_sample = True if temperature > 0 else False
        response = self.model.generate(
            input_tokens,
            do_sample = do_sample,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[stopping_criteria.logits_processor],
            stopping_criteria=[
                stopping_criteria
            ],
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.generation_params
        )
        
        response = self.tokenizer.decode(response[0][len(input_tokens[0]):], skip_special_tokens=True)
        self.debug("[generate_literal]", response)
        # postprocessing to make sure the literal is generated correctly

        options = [idx for idx, x in enumerate(literal) if x.startswith(response)]
        if len(options) != 1:
            if iterations > 3:
                raise ValueError("Failed to generate a valid literal")
            return self.generate_literal(literal, temperature=temperature * 1.3, iterations=iterations+1)
        else:
            return options[0]

    def generate_string(self, temperature: Union[float, None] = None) -> str:
        """
        Customer decoding for string generation
        """
        prompt = self.get_prompt() + '"'
        self.debug("[generate_string]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        temperature = temperature or self.temperature
        do_sample = True if temperature > 0 else False
        response = self.model.generate(
            input_tokens,
            do_sample = do_sample,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            temperature=temperature,
            stopping_criteria=[
                StringStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            pad_token_id=self.tokenizer.eos_token_id,
            **self.generation_params
        )

        # Some models output the prompt as part of the response
        # This removes the prompt from the response if it is present
        if (
            len(response[0]) >= len(input_tokens[0])
            and (response[0][: len(input_tokens[0])] == input_tokens).all()
        ):
            response = response[0][len(input_tokens[0]) :]
        if response.shape[0] == 1:
            response = response[0]

        response = self.tokenizer.decode(response, skip_special_tokens=True)

        self.debug("[generate_string]", "|" + response + "|")

        if response.count('"') < 1:
            return response

        return response.split('"')[0].strip()

    def generate_object(
        self, properties: Dict[str, Any], obj: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate the object with object generation schema
        """
        for key, schema in properties.items():
            self.debug("[generate_object] generating value for", key)
            #if len(schema["properties"]) != 1:
            #    raise ValueError(
            #        "We currently only support objects where each attribute has only one typehint"
            #    )
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Union[str, None] = None,
    ) -> Any:
        """
        Main generation function, given the schema, generate a value
        """
        try:
            schema_type = schema["type"]
            if schema_type not in self.supported_schema_types:
                raise ValueError(f"The schema type {schema_type} is not supported yet for custom decoding. "\
                                  f"The following schemas are supported {self.supported_schema_types}. "\
                                    f"For {schema_type} please use the normal decoding or a different model.")
        except KeyError:
            raise ValueError(f"The schema type {schema} is not supported yet for custom decoding. "\
                                  f"A combination of following schemas are supported {self.supported_schema_types}. "\
                                    f"For {schema} please use the normal decoding or a different model.")
        

        if schema_type in ["int", "float"]:
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            if schema_type == "int":
              return int(self.generate_number())
            else:
              return float(self.generate_number())
        elif schema_type == "bool":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            response_idx = self.generate_literal(literal=["true", "false"])

            return [True, False][response_idx]

        elif schema_type == "literal":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            response_idx = self.generate_literal(literal=schema["str_options"])
            return schema["original_options"][response_idx]

        elif schema_type == "str":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_string()
        elif schema_type in ["list", "tuple", "set"]:
            if key:
                obj[key] = [self.generation_marker]
            else:
                obj.append(self.generation_marker)
            if len(schema["properties"]) != 1:
                raise ValueError(
                    "We currently only support arrays with 1 typehint for items. "\
                    f"Attribute {key} has {len(schema['properties'])} typehints set."
                )
            output = self.generate_array(item_schema= schema["properties"][0], 
                                         obj = obj[key],
                                         collection=schema_type)
            return output

        elif schema_type == "pydantic_object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)
        else:
            raise ValueError(f"The schema type {schema_type} is not supported yet for custom decoding. Please use the normal decoding or a different model.")



    def generate_array(self, item_schema: Dict[str, Any], obj: List, collection: str) -> Union[list, tuple, set]:
    
        end_token = "]"
        self.max_number_tokens -= 2 # for the start and end token
        for _ in range(self.max_array_length):
            prompt = self.get_prompt()
            # check if the second 2 last elements are ", "
            prompt = prompt.rstrip(", ")
            # first check if need to close and return the sequence
            input_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
            output = self.model.forward(input_tensor.to(self.model.device))
            logits = output.logits[0, -1]
            top_indices = logits.topk(30).indices
            sorted_token_ids = top_indices[logits[top_indices].argsort(descending=True)]

            found_comma = -1
            found_close_bracket = -1
            # get the , token id
            for idx, token_id in enumerate(sorted_token_ids):
                decoded_token = self.tokenizer.decode(token_id)
                if ',' in decoded_token and found_comma == -1:
                    found_comma = idx
                    
                if end_token in decoded_token and found_close_bracket == -1:
                    found_close_bracket = idx
            
            if len(obj) == 1: # special case for the first case generation
                # if the frist token was the end token, return the empty array
                if found_close_bracket == 0:
                    break
            else:
                if found_comma == -1 and found_close_bracket == -1:
                    break
                # if found close bracket had a smaller index, then we should stop
                if found_comma >= found_close_bracket:
                    break

            # generate an element into the array if the stopping conditions werent met
            element = self.generate_value(item_schema, [])
            element_tokens = self.tokenizer.encode(element) + 1 # for the comma
            if element_tokens > self.max_number_tokens:
                break
            self.max_number_tokens -= element_tokens
            # add the element as the second to last element
            obj.insert(-1, element)

        # remove the generation marker
        obj.pop(-1)
        return obj




    def generate_array_object(self, schema: dict, obj) -> Union[list, tuple, set]:
        # currently support schemas with only 1 type
        if len(schema["properties"]) != 1:
                raise ValueError(
                    "We currently only support custom decoding with arrays with exactly 1 typehint for items"
                )
        generated_data = self.generate_array(
            self.json_schema["properties"][0], obj, self.json_schema["type"]
        )
        if schema["type"] == "tuple":
            generated_data = tuple(generated_data)
        elif schema["type"] == "set":
            generated_data = set(generated_data)
        return generated_data

    def get_prompt(self):
        """
        Format the prompt for generation
        """
        template = """{prompt} {progress}"""
        progress = json.dumps(self.value)
        gen_marker_index = progress.find(f"{self.generation_marker}")-1
        if gen_marker_index >= 0:
            progress = progress[:gen_marker_index]
        elif progress != "{}":
            raise ValueError("Failed to find generation marker")

        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=progress,
        )

        return prompt
    
    def __call__(self) -> Dict[str, Any]:
        if self.json_schema["type"] == "pydantic_object":
            self.value = {}
            generated_data = self.generate_object(
                self.json_schema["properties"], self.value
            )
        elif self.json_schema["type"] in ["list", "tuple", "set"]:
            self.value = [self.generation_marker]
            generated_data = self.generate_array_object(
                self.json_schema, self.value
            )
        
        else:
            generated_data = self.generate_value(self.json_schema, [])
        return generated_data
