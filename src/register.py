import inspect
from typing import get_type_hints, Literal, Optional

from models.function_description import FunctionDescription

alignable_functions = {}


class Register:

    def __init__(self):
        pass

    @staticmethod
    def get(func_name):
        if func_name not in alignable_functions:
            pass
        return alignable_functions[func_name]

    @staticmethod
    def function_names_to_patch():
        return list(alignable_functions.keys())

    @staticmethod
    def add_function(func, wrapper):
        alignable_functions[func.__name__] = func #wrapper

    @staticmethod
    def load_function_description_from_name(*args) -> FunctionDescription:
        """
        Load a function description from a function name from the global scope.
        :param func_name:
        :return:
        """
        if len(args) == 1:
            instance = None
            func_name = args[0]
        elif len(args) == 2:
            instance = args[0]
            func_name = args[1]
        else:
            raise ValueError("Invalid number of arguments")

        if not instance:
            func_object = alignable_functions[func_name]
            return Register.load_function_description(func_object)
        else:
            func_object = getattr(instance, func_name)
            return Register.load_function_description(func_object)

    @staticmethod
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
                if origin_type is Literal:  # Handle Literal case
                    return [literal for literal in class_type.__args__]
                elif hasattr(class_type, "__args__"):  # Access inner types
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
