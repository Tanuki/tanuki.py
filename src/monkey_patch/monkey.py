import ast
import inspect
import json
import logging
import os
import sys
import textwrap
from functools import wraps
from typing import Optional, Union, Any
from unittest.mock import patch

import requests

from monkey_patch.assertion_visitor import AssertionVisitor
from monkey_patch.function_modeler import FunctionModeler
from monkey_patch.language_models.embedding_model_manager import EmbeddingModelManager
from monkey_patch.language_models.language_model_manager import LanguageModelManager
from monkey_patch.models.embedding import Embedding
from monkey_patch.models.function_description import FunctionDescription
from monkey_patch.models.function_example import FunctionExample
from monkey_patch.models.function_type import FunctionType
from monkey_patch.register import Register
from monkey_patch.trackers.buffered_logger import BufferedLogger
from monkey_patch.utils import get_key
from monkey_patch.validator import Validator


# Define a new level
def _log_align(self, func_hash, *args, **kws):
    if self.isEnabledFor(ALIGN_LEVEL_NUM):
        args, kwargs, output = args
        kwargs['align'] = True
        example = FunctionExample(args, kwargs, output)

        # Define a safe directory within the project for logs
        # (You can make this configurable if needed)
        log_directory = os.path.join(os.getcwd(), ALIGN_FILE_NAME)

        # Ensure the directory exists
        if not os.path.exists(log_directory):
            try:
                os.makedirs(log_directory)
            except OSError as e:
                self.error(f"Failed to create log directory: {e}")
                return

        # Write to the file
        log_file_path = os.path.join(log_directory, func_hash)
        try:
            with open(log_file_path, "a") as f:
                f.write(str(example.__dict__) + "\n")
        except IOError as e:
            self.error(f"Failed to write to log file: {e}")


# Set up logging with custom logger
def logger_factory(name):
    return BufferedLogger(name)


ALIGN_LEVEL_NUM = 15
PATCH_LEVEL_NUM = 14
ALIGN_FILE_NAME = ".align"

alignable_functions = {}


class Monkey:
    # Set up basic configuration
    logging.setLoggerClass(BufferedLogger)
    logging.addLevelName(ALIGN_LEVEL_NUM, "ALIGN")
    logging.addLevelName(PATCH_LEVEL_NUM, "PATCH")
    logging.basicConfig(level=ALIGN_LEVEL_NUM)
    logger = logger_factory(__name__)
    # currently only use buffered logger as default
    function_modeler = FunctionModeler(data_worker=logger)
    language_modeler = LanguageModelManager(function_modeler)
    embedding_modeler = EmbeddingModelManager(function_modeler)

    @staticmethod
    def _load_alignments(func_hash: str):
        Monkey.function_modeler.load_align_statements(func_hash)

    @staticmethod
    def _anonymous_usage(*args, **kwargs):
        """
        Post anonymously to the usage server so we know what configs are commonly used in the project.
        :return:
        """
        requests.post('https://idhhnusnhkkjkpwkm1fr.monkeypatch.ai/telemetry', data=json.dumps(kwargs))

    @staticmethod
    def align(test_func):
        """
        Decorator to align a function.

        By adding the @align decorator to a function, we can declare the desired input-output
        behaviour of the patched functions using assertions.

        :param test_func:
        :return:
        """

        @wraps(test_func)
        def wrapper(*args, **kwargs):
            source = textwrap.dedent(inspect.getsource(test_func))
            tree = ast.parse(source)
            _locals = locals()

            # We are handling symbolic and embeddable functions differently, as they have different semantics during
            # the alignment process.
            patch_symbolic_funcs = Register.functions_to_patch(type=FunctionType.SYMBOLIC)
            patch_embeddable_funcs = Register.functions_to_patch(type=FunctionType.EMBEDDABLE)
            visitor = AssertionVisitor(_locals,
                                       patch_symbolic_funcs=patch_symbolic_funcs,
                                       patch_embeddable_funcs=patch_embeddable_funcs)
            visitor.visit(tree)
            mock_behaviors = visitor.mocks

            if args:
                instance = args[0]
                args = args[1:]
            else:
                instance = None

            def extract_attributes(result):
                attributes = {}

                # If the result is a list, get its length
                if isinstance(result, list):
                    attributes['length'] = len(result)

                # If the result is a dictionary, get its keys (or any other attributes)
                elif isinstance(result, dict):
                    attributes['keys'] = list(result.keys())

                return attributes

            def create_mock_func(instance: Optional,
                                 func_name: str,
                                 description: FunctionDescription):

                def mock_func(*args, **kwargs):
                    hashed_description = description.__hash__()

                    function_type, func = Register.get(func_name)

                    # If we are aligning a function that returns an embedding,
                    # we need to ensure both sides of the equality are future embeddings,
                    # as it is nonsensical to declare that an embedding should 'be' an object or a string, etc.
                    if function_type == FunctionType.EMBEDDABLE:
                        key = get_key(args, kwargs)
                        mocked_behaviour = mock_behaviors.get(key, None)
                        return mocked_behaviour
                    else:
                        # If we are aligning a function that returns an object
                        if not instance:
                            result = func(*args, **kwargs)
                        else:
                            result = func(instance, *args, **kwargs)

                        # Extract attributes from the result
                        attributes = extract_attributes(result)
                        for attr_name, attr_value in attributes.items():
                            # If the attribute is a list, get its length
                            if isinstance(attr_value, list):
                                attributes[attr_name] = len(attr_value)

                        key = get_key(args, kwargs)
                        mocked_behaviour = mock_behaviors.get(key, None)
                        Monkey.function_modeler.save_align_statements(hashed_description, args, kwargs, mocked_behaviour)
                        return mocked_behaviour

                return mock_func

            function_names_to_patch = Register.function_names_to_patch()

            # Identify all functions that need to be patched based on mock_behaviors
            if instance:
                functions_descriptions = [Register.load_function_description_from_name(instance, func_name)
                                          for func_name in function_names_to_patch]
            else:
                functions_descriptions = [Register.load_function_description_from_name(func_name)
                                          for func_name in function_names_to_patch]

            patched_func = test_func
            for desc, func in zip(functions_descriptions, function_names_to_patch):
                mock_function = create_mock_func(instance, func, desc)
                module_name = sys.modules[test_func.__module__].__name__

                if instance:
                    patched_func = patch.object(instance, func, new=mock_function)(patched_func)
                else:
                    patched_func = patch(f'{module_name}.{func}', new=mock_function)(patched_func)

            # Get the signature of the function
            sig = inspect.signature(test_func)

            if sig.parameters:
                first_param_name = next(iter(sig.parameters))

                # If the instance is the "self" or the name of the first parameter,
                # then pass it as the first argument
                if first_param_name in ['self', 'cls'] or first_param_name == instance:
                    return patched_func(instance, *args, **kwargs)
                else:
                    return patched_func(*args, **kwargs)
            else:
                return patched_func(*args, **kwargs)

        return wrapper

    @staticmethod
    def generate_from_embedding_model_manager(function_description):
        choice_parsed = []
        instantiated = function_description.output_type_hint(choice_parsed)
        return instantiated

    @staticmethod
    def patch(patchable_func=None,
              environment_id: int = 0,
              ignore_finetune_fetching: bool = False,
              ignore_finetuning: bool = False,
              ignore_data_storage: bool = False
              ):
        """
        The main decorator for patching a function.
        args:
            patchable_func: The function to be patched, should be always set to none. This is used here to allow for keyword arguments or no arguments to be passed to the decorator
            environment_id (int): The environment id. Used for fetching correct finetuned models
            ignore_finetune_fetching (bool): Whether to ignore fetching finetuned models.
                If set to False, during the first call openai will not be queried for finetuned models, which reduces initial startup latency
            ignore_finetuning (bool): Whether to ignore finetuning the models altogether. If set to True the teacher model will always be used.
                The data is still saved however if in future would need to use finetuning
            ignore_data_storage (bool): Whether to ignore storing the data.
                If set to True, the data will not be stored in the finetune dataset and the align statements will not be saved
                This improves latency as communications with data storage is minimised
        """

        def wrap(test_func):
            @wraps(test_func)
            def wrapper(*args, **kwargs) -> Union[Embedding, Any]:
                validator = Validator()
                function_description: FunctionDescription = Register.load_function_description(test_func)

                # If the function is expected to return an embedding, we choose the embedding API, rather than an LLM.
                if inspect.isclass(function_description.output_type_hint) and \
                        issubclass(function_description.output_type_hint, Embedding):
                    instantiated: Embedding = Monkey.embedding_modeler(args, function_description, kwargs)
                else:
                    # If the function is expected to return a choice, we choose the LLM API.
                    instantiated: Any = Monkey.language_modeler(args, function_description, kwargs, validator)

                return instantiated  # test_func(*args, **kwargs)

            Monkey._anonymous_usage(logger=Monkey.logger.name)
            function_description = Register.load_function_description(test_func)
            func_hash = function_description.__hash__()
            Monkey.function_modeler.environment_id = environment_id
            if ignore_finetuning:
                Monkey.function_modeler.execute_finetune_blacklist.append(func_hash)
            if ignore_finetune_fetching:
                Monkey.function_modeler.check_finetune_blacklist.append(func_hash)
            if ignore_data_storage:
                Monkey.function_modeler.store_data_blacklist.append(func_hash)
            Monkey._load_alignments(func_hash)

            wrapper._is_alignable = True
            Register.add_function(test_func, function_description)
            return wrapper

        if callable(patchable_func):
            func = patchable_func
            return wrap(func)
        if patchable_func is not None:
            raise TypeError(
                "The first argument to patch must not be specified. Please use keyword arguments or specify the first argument as None")
        return wrap
