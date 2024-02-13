import ast
import importlib
import inspect
import json
import logging
import os
import sys
import textwrap
from functools import wraps
from typing import Optional, Union, Any
from unittest.mock import patch as mock_patch

import astor as astor
import requests

import tanuki
from tanuki.models.api_manager import APIManager
from tanuki.runtime_assertion_visitor import RuntimeAssertionVisitor
from tanuki.static_assertion_visitor import StaticAssertionVisitor
from tanuki.function_modeler import FunctionModeler
from tanuki.language_models.embedding_model_manager import EmbeddingModelManager
from tanuki.language_models.language_model_manager import LanguageModelManager
from tanuki.models.embedding import Embedding
from tanuki.models.function_description import FunctionDescription
from tanuki.models.function_example import FunctionExample
from tanuki.models.function_type import FunctionType
from tanuki.register import Register
from tanuki.trackers.filesystem_buffered_logger import FilesystemBufferedLogger
from tanuki.utils import get_key
from tanuki.validator import Validator


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

class DisableRuntimeAlign:
    def __enter__(self):
        globals()[DISABLE_RUNTIME_ALIGN] = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        globals()[DISABLE_RUNTIME_ALIGN] = False

# Set up logging with custom logger
def logger_factory(name):
    return FilesystemBufferedLogger(name)


ALIGN_LEVEL_NUM = 15
PATCH_LEVEL_NUM = 14
ALIGN_FILE_NAME = ".align"

alignable_functions = {}

# Set up basic configuration
logging.setLoggerClass(FilesystemBufferedLogger)
logging.addLevelName(ALIGN_LEVEL_NUM, "ALIGN")
logging.addLevelName(PATCH_LEVEL_NUM, "PATCH")
logging.basicConfig(level=ALIGN_LEVEL_NUM)
logger = logger_factory(__name__)


api_provider = APIManager()
function_modeler = FunctionModeler(data_worker=logger, api_provider=api_provider)
language_modeler = LanguageModelManager(function_modeler, api_provider=api_provider)
embedding_modeler = EmbeddingModelManager(function_modeler, api_provider=api_provider)
telemetry_enabled: bool = True

DISABLE_RUNTIME_ALIGN = '__disable_runtime_align__'
__disable_runtime_align__ = False

@staticmethod
def _load_alignments(func_hash: str):
    function_modeler.load_symbolic_align_statements(func_hash)

@staticmethod
def _anonymous_usage(*args, **kwargs):
    """
    Post anonymously to the usage server so we know what configs are commonly used in the project.
    :return:
    """
    if not telemetry_enabled:
        return
    try:
        requests.post('https://idhhnusnhkkjkpwkm1fr.monkeypatch.ai/telemetry', data=json.dumps(kwargs))
    except:
        pass


@staticmethod
def align(test_func):
    """
    Decorator to align a function.

    This version modifies the AST to replace assert statements with runtime
    register calls, then compiles and executes the modified function.
    """

    def get_instance_from_args(args):
        # Check if there are any arguments
        if args:
            first_arg = args[0]

            # Check if the first argument is named "self" or "cls" (or any other specific name)
            if isinstance(first_arg, ast.Name) and first_arg.id in ("self", "cls"):
                instance = first_arg
                args = args[1:]  # Remove the first argument
            else:
                instance = None
        else:
            instance = None

        return instance, args

    def register(func_name, *args, expected_output, positive=False, instance=None, **kwargs):
        if instance:
            function_names_to_patch = Register.function_names_to_patch(instance)  # , type=FunctionType.SYMBOLIC)
            functions_descriptions = [Register.load_function_description_from_name(instance, func_name)
                                      for func_name in function_names_to_patch]
        else:
            function_names_to_patch = Register.function_names_to_patch()  # type=FunctionType.SYMBOLIC)
            functions_descriptions = [Register.load_function_description_from_name(func_name)
                                      for func_name in function_names_to_patch]

        for desc, fn_name in zip(functions_descriptions, function_names_to_patch):
            if fn_name == func_name:
                hashed_description = desc.__hash__()
                if desc.type == FunctionType.EMBEDDABLE:
                    if positive:
                        # Implement logic for handling positive (equality) assertions
                        positive_pairs = []
                        logger.log(ALIGN_LEVEL_NUM, f"Registering positive embedding pairs for {fn_name}: {positive_pairs}")
                        function_modeler.save_embeddable_align_statements(hashed_description, args, kwargs, positive_pairs=positive_pairs)
                    else:
                        negative_pairs = []
                        logger.info(f"Registering negative embedding pairs for {fn_name}: {negative_pairs}")
                        function_modeler.save_embeddable_align_statements(hashed_description, args, kwargs, negative_pairs=negative_pairs)
                elif desc.type == FunctionType.SYMBOLIC:
                    if positive:
                        logger.info(f"Registering symbolic align for {fn_name}{args}{dict(**kwargs)}: {expected_output}")
                        print(f"Registering symbolic align for {fn_name}{args}{dict(**kwargs)}: {expected_output}")
                        function_modeler.save_symbolic_align_statements(hashed_description, args, kwargs, expected_output)
                    else:
                        raise NotImplementedError("Negative assertions for symbolic functions are not supported yet because most LLM providers dont offer a negative finetuning API")
                break

    def make_dynamic_call_with_namespace(namespace):
        def dynamic_call(func_name, *args, __expected_output: any = None, __align_direction: bool = True, **kwargs):
            if 'self' in globals():
                # If 'self' is defined globally (unlikely in normal use),
                # this will throw an error as this approach relies on runtime context
                raise NameError("'self' found in globals; ambiguous context.")
            try:
                # Attempt to call as a method on an instance
                instance = kwargs.pop('instance', None)
                if instance is not None:
                    method = getattr(instance, func_name)
                    register(func_name, *args, **kwargs, positive=__align_direction, expected_output=__expected_output, instance=instance)
                    return method(*args, **kwargs)
            except AttributeError:
                # 'instance' does not have the method; fall back to a function call
                pass

            _namespace = namespace
            # Possibly we should register the module of the function being called too.
            register(func_name, *args, **kwargs, positive=__align_direction, expected_output=__expected_output, instance=None)

        return dynamic_call

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # If the global variable is set, run the function without aligning. This is to prevent infinite recursion.
        _globals = test_func.__globals__

        if _globals.get(DISABLE_RUNTIME_ALIGN, False):
            #_globals = globals()
            _locals = locals().copy()
            return test_func(*args, **kwargs)
        else:
            source = textwrap.dedent(inspect.getsource(test_func))
            # Extract imports from the file where test_func is defined
            #imported_modules = extract_file_imports(test_func)

            tree = ast.parse(source)
            _locals = locals().copy()

            instance, args = get_instance_from_args(args)

            # Registered functions to patch
            patch_symbolic_funcs = Register.functions_to_patch(type=FunctionType.SYMBOLIC)
            patch_embeddable_funcs = Register.functions_to_patch(type=FunctionType.EMBEDDABLE)

            # Initialize the visitor with function_modeler
            visitor = RuntimeAssertionVisitor(#function_modeler=function_modeler,
                                              instance=instance,
                                              patch_symbolic_funcs=patch_symbolic_funcs,
                                              patch_embeddable_funcs=patch_embeddable_funcs)
            tree = ast.parse(source)
            modified_tree = visitor.visit(tree)

            modified_tree = ast.fix_missing_locations(modified_tree)

            # Dump the modified AST to string for debug purposes
            dump = ast.dump(modified_tree, indent=2, include_attributes=True)
            # Get the modified source code for debug purposes
            source_code = astor.to_source(modified_tree)

            # Compile the modified AST
            try:
                compiled_code = compile(modified_tree, filename="<ast>", mode="exec")
            except Exception as e:
                raise SyntaxError(f"Syntax error in modified code: {e}")

            # Execute the compiled code in a new namespace to avoid overwriting
            parent_module = inspect.getmodule(test_func).__dict__
            namespace = {
                DISABLE_RUNTIME_ALIGN: True,
                'dynamic_call': make_dynamic_call_with_namespace(parent_module),
                **parent_module
            }

            with DisableRuntimeAlign():
                exec(compiled_code, namespace)

            # Retrieve the modified function from the namespace and call it
            modified_func_name = test_func.__name__
            modified_func = namespace.get(modified_func_name)

            if modified_func:
                return modified_func(*args, **kwargs)
            else:
                raise RuntimeError("Modified function not found after AST modification.")
    return wrapper


@staticmethod
def align_static(test_func):
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
        visitor = StaticAssertionVisitor(_locals,
                                         patch_symbolic_funcs=patch_symbolic_funcs,
                                         patch_embeddable_funcs=patch_embeddable_funcs)
        visitor.visit(tree)

        # Get the mocked behaviours from analyzing the AST of the aligned function
        mock_behaviors = visitor.mocks

        # Negative examples (i.e. embeddable function examples that should have maximum distance in the embedding space)
        mock_negatives = visitor.negative_mocks

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
                    mocked_embedding = mock_behaviors.get(key, None)

                    # Find positive examples by matching the mocked embedding with identical embeddings in the values
                    # of the mock_behaviors dictionary
                    mock_positives_list = []
                    for k, v in mock_behaviors.items():
                        if v == mocked_embedding and k != key:
                            mock_positives_list.append(k)
                    equivalent_mocks = mock_positives_list
                    negative_mocks = list(mock_negatives.values())
                    function_modeler.save_embeddable_align_statements(hashed_description,
                                                                             args,
                                                                             kwargs,
                                                                             equivalent_mocks,
                                                                             negative_mocks)
                    return mocked_embedding
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
                    function_modeler.save_symbolic_align_statements(hashed_description, args, kwargs,
                                                                           mocked_behaviour)
                    return mocked_behaviour

            return mock_func


        # Identify all functions that need to be patched based on mock_behaviors
        if instance:
            function_names_to_patch = Register.function_names_to_patch(instance)#, type=FunctionType.SYMBOLIC)
            functions_descriptions = [Register.load_function_description_from_name(instance, func_name)
                                      for func_name in function_names_to_patch]
        else:
            function_names_to_patch = Register.function_names_to_patch()#type=FunctionType.SYMBOLIC)
            functions_descriptions = [Register.load_function_description_from_name(func_name)
                                      for func_name in function_names_to_patch]

        patched_func = test_func
        for desc, func in zip(functions_descriptions, function_names_to_patch):
            mock_function = create_mock_func(instance, func, desc)
            module_name = sys.modules[test_func.__module__].__name__

            if instance:
                patched_func = mock_patch.object(instance, func, new=mock_function)(patched_func)
            else:
                patched_func = mock_patch(f'{module_name}.{func}', new=mock_function)(patched_func)

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
          ignore_data_storage: bool = False,
          teacher_models : list = [],
          student_model : str = "",
          generation_params : dict = {}
          ):
    """
    The main decorator for patching a function.
    args:
        patchable_func: The function to be patched, should be always set to none. This is used here to allow for keyword arguments or no arguments to be passed to the decorator
        environment_id (int): The environment id. Used for fetching correct finetuned models
        ignore_finetune_fetching (bool): Whether to ignore fetching finetuned models.
            If set to True, during the first call openai will not be queried for finetuned models, which reduces initial startup latency
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
                instantiated: Embedding = embedding_modeler(args, function_description, kwargs)
            else:
                # If the function is expected to return a choice, we choose the LLM API.
                instantiated: Any = language_modeler(args, 
                                                     function_description, 
                                                     kwargs, 
                                                     validator, 
                                                     generation_params)

            return instantiated  # test_func(*args, **kwargs)

        _anonymous_usage(logger=logger.name)
        function_description = Register.load_function_description(test_func)
        func_hash = function_description.__hash__()
        # Configure the function modeler using incoming parameters
        function_modeler.environment_id = environment_id
        if ignore_finetuning:
            logging.info(f"The flag for ignoring finetuning has been set True for {test_func.__name__}. No model distillation will be performed.")
            function_modeler.execute_finetune_blacklist.append(func_hash)
        if ignore_finetune_fetching:
            logging.info(f"The flag for ignoring searching for finetuned models has been set True for {test_func.__name__}. No already finetuned models will be looked for.")
            function_modeler.check_finetune_blacklist.append(func_hash)
        if ignore_data_storage:
            logging.info(f"The flag for ignoring data storage has been set True for {test_func.__name__}. No data will be read or saved and model distillation will not be performed.")
            function_modeler.store_data_blacklist.append(func_hash)
        task_type = function_description.type
        function_modeler._configure_function_models(teacher_models, 
                                                    student_model,
                                                    func_hash = func_hash,
                                                    task_type=task_type)
        _load_alignments(func_hash)

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
