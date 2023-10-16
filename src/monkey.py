import json
from functools import wraps

import openai

from register import Register
from utils import json_dumps
from validator import Validator


class monkey:

    @staticmethod
    def _load_alignments():
        pass

    @staticmethod
    def patch(test_func):

        monkey._load_alignments()

        @wraps(test_func)
        def wrapper(*args, **kwargs):
            function_description = Register.load_function_description(test_func)

            #f = json_dumps(function_description.__dict__)
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
                raise TypeError(f"Output type was not valid. Expected an object of type {function_description.output_type_hint}, got '{choice}'")

            instantiated = validator.instantiate(choice, function_description.output_type_hint)

            return instantiated  # test_func(*args, **kwargs)

        wrapper._is_alignable = True
        return wrapper