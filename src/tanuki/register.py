import inspect
from typing import get_type_hints, Literal, get_origin, Tuple, Callable, Optional, Dict, Union
import json
from tanuki.models.embedding import Embedding
from tanuki.models.function_description import FunctionDescription
from tanuki.models.function_type import FunctionType
from tanuki.utils import get_source

alignable_symbolic_functions = {}
alignable_embedding_functions = {}


class Register:

    def __init__(self):
        pass

    @staticmethod
    def get(func_name) -> Tuple[FunctionType, Callable]:
        if func_name not in alignable_symbolic_functions and func_name not in alignable_embedding_functions:
            pass

        if func_name in alignable_symbolic_functions:
            return FunctionType.SYMBOLIC, alignable_symbolic_functions[func_name]
        elif func_name in alignable_embedding_functions:
            return FunctionType.EMBEDDABLE, alignable_embedding_functions[func_name]

    @staticmethod
    def function_names_to_patch(*args, type: Optional[FunctionType] = None):
        """
        Get the registered function names that should be patched, either globally (if len(args)==0) or as members of
        an instance
        :param args: Optional instance to check
        :return:
        """
        function_names = []
        if len(args) == 1:
            instance = args[0]

            if type == FunctionType.SYMBOLIC:
                for key in alignable_symbolic_functions.keys():
                    if hasattr(instance, key):
                        function_names.append(key)
                return function_names
            elif type == FunctionType.EMBEDDABLE:
                for key in alignable_embedding_functions.keys():
                    if hasattr(instance, key):
                        function_names.append(key)
                return function_names
            else:
                for key in alignable_symbolic_functions.keys():
                    if hasattr(instance, key):
                        function_names.append(key)
                for key in alignable_embedding_functions.keys():
                    if hasattr(instance, key):
                        function_names.append(key)
                return function_names
        else:
            if type == FunctionType.SYMBOLIC:
                return list(alignable_symbolic_functions.keys())
            elif type == FunctionType.EMBEDDABLE:
                return list(alignable_embedding_functions.keys())
            else:
                return list(alignable_symbolic_functions.keys()) + list(alignable_embedding_functions.keys())

    @staticmethod
    def functions_to_patch(*args, type: Optional[FunctionType] = None) -> Dict[str, Callable]:
        function_names = Register.function_names_to_patch(*args, type=type)
        if type == FunctionType.SYMBOLIC:
            return {key: alignable_symbolic_functions[key] for key in function_names}
        elif type == FunctionType.EMBEDDABLE:
            return {key: alignable_embedding_functions[key] for key in function_names}
        else:
            return {key: alignable_symbolic_functions[key] for key in function_names} + \
                   {key: alignable_embedding_functions[key] for key in function_names}

    @staticmethod
    def add_function(func, function_description: FunctionDescription):
        if function_description.type == FunctionType.SYMBOLIC:
            alignable_symbolic_functions[func.__name__] = func
        elif function_description.type == FunctionType.EMBEDDABLE:
            alignable_embedding_functions[func.__name__] = func

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
            if func_name in alignable_symbolic_functions:
                func_object = alignable_symbolic_functions[func_name]
            elif func_name in alignable_embedding_functions:
                func_object = alignable_embedding_functions[func_name]
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
                return get_source(class_type)
            return class_type.__name__

        # Extract class definitions for input and output types
        input_class_definitions = {
            param_name: get_class_definition(param_type)
            for param_name, param_type in input_type_hints.items()
        }
        # if inspect.isclass(output_type_hint) and issubclass(output_type_hint, Embedding):
        #     output_class_definition = None
        # else:
        #     output_class_definition = get_class_definition(output_type_hint)
        output_class_definition = None
        function_type = FunctionType.SYMBOLIC
        # check if the output type hint is a class or a subclass of a Union
        if inspect.isclass(output_type_hint) or (hasattr(output_type_hint, "__origin__") and
                                                 output_type_hint.__origin__ == Union):
            if (hasattr(output_type_hint, "__origin__") and output_type_hint.__origin__ == Union): # it's a union
                # get all the types in the union
                union_types = output_type_hint.__args__
                output_type_descriptions = {}
                for output_type in union_types:
                    # check if it is a class Nonetype
                    if output_type is type(None):
                        output_type_descriptions["NoneType"] = "None"
                    elif inspect.isclass(output_type):
                        # Check if the base class of the output type hint is Embedding
                        base_class = get_origin(output_type) or output_type
                        if issubclass(base_class, Embedding):
                            output_class_definition = None
                            function_type = FunctionType.EMBEDDABLE
                            break
                        else:
                            class_type_description = get_class_definition(output_type)
                            if isinstance(class_type_description,str):
                                class_type_description = class_type_description.replace('"', "'") # less horrible prompt formatting when dump to json
                            output_type_descriptions[output_type.__name__] = class_type_description
                output_class_definition = f"Union of following classes {json.dumps(output_type_descriptions)}"

            else: # it's a class
                # Check if the base class of the output type hint is Embedding
                base_class = get_origin(output_type_hint) or output_type_hint
                if issubclass(base_class, Embedding):
                    output_class_definition = None
                    function_type = FunctionType.EMBEDDABLE
                else:
                    output_class_definition = get_class_definition(output_type_hint)

        return FunctionDescription(
            name=func_object.__name__,
            docstring=docstring,
            input_type_hints=input_type_hints,
            output_type_hint=output_type_hint,
            input_class_definitions=input_class_definitions,
            output_class_definition=output_class_definition,
            type=function_type
        )
