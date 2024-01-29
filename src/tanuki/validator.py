import abc
from collections import defaultdict
import collections
import typing
from collections import deque
import dataclasses
import inspect
import json
from dataclasses import is_dataclass
from typing import get_origin, get_args, Any, Mapping, MutableMapping, OrderedDict, Literal, Union, get_type_hints, \
    Type, Sequence, Tuple, Optional

from pydantic import BaseModel, create_model
import datetime

class Validator:

    def __init__(self):
        # Extract types from collections and collections.abc
        collection_types = {cls for name, cls in collections.__dict__.items() if isinstance(cls, type)}
        abc_collection_types = {cls for name, cls in collections.abc.__dict__.items() if isinstance(cls, type)}
        # Filter out types that have dictionary-like methods
        self.dict_like_types = {
            cls for cls in collection_types.union(abc_collection_types)
            if hasattr(cls, 'keys') and hasattr(cls, 'items')
        }

        self.list_like_types = {
            cls for cls in collection_types.union(abc_collection_types)
            if hasattr(cls, 'append') and hasattr(cls, 'pop')
        }

        self.set_like_types = {
            cls for cls in collection_types.union(abc_collection_types)
            if hasattr(cls, 'add') and hasattr(cls, 'discard')
        }
        # Add the general Sequence to list-like types
        # if python version is 3.9 or above, use collections.abc.Sequence
        if hasattr(collections.abc, 'Sequence'):
            self.list_like_types.add(collections.abc.Sequence)
        else:
            self.list_like_types.add(collections.Sequence)

        self.list_like_types.add(typing.List)

        # Add the general Mapping to dict-like types
        if hasattr(collections.abc, 'Mapping'):
            self.dict_like_types.add(collections.abc.Mapping)
        else:
            self.dict_like_types.add(collections.Mapping)

        self.dict_like_types.add(typing.Dict)

        # Add the general Set to set-like types
        if hasattr(collections.abc, 'Set'):
            self.set_like_types.add(collections.abc.Set)
        else:
            self.set_like_types.add(collections.Set)

        self.set_like_types.add(typing.Set)

        # Add the general Tuple to tuple-like types
        self.tuple_like_types = {
            cls for cls in collection_types.union(abc_collection_types)
            if hasattr(cls, '__getitem__') and hasattr(cls, '__len__')
        }

        self.tuple_like_types.add(typing.Tuple)

    def is_base_type(self, _type: Any) -> bool:
        """Determine if a type is a base type."""
        return _type in {int, float, str, bool, None}

    def validate_base_type(self, value: Any, typ: Any) -> bool:
        """Validate base types."""
        if typ is None:
            return value is None
        return isinstance(value, typ)

    def validate_output(self, output: str, type_definition: Any) -> bool:
        try:
            deserialized_output = json.loads(output)
        except json.JSONDecodeError:
            return False

        return self.check_type(deserialized_output, type_definition)

    def check_type(self, value: Any, type_definition: Any) -> bool:
        """
        Validate a value against a type definition.

        Args:
            value: Any object or primitive value
            type_definition: The type definition to validate against

        Returns:
            Whether the value is valid for the type definition
        """
        if type_definition is Any:
            return True

        if self.is_base_type(type_definition):
            return self.validate_base_type(value, type_definition)

        origin = get_origin(type_definition) or type_definition
        args = get_args(type_definition)

        # Handle base types
        if self.is_base_type(origin):
            return self.validate_base_type(value, origin)

        if origin == Literal:
            return value in args

        if origin == Union:
            return any(self.check_type(value, union_type) for union_type in args)

        # Handle tuples
        if origin == tuple:
            if not isinstance(value, tuple):
                return False
            item_type = args[0] if args else Any
            return all(self.check_type(v, item_type) for v in value)

        # Handle lists
        if origin == list:
            if not isinstance(value, list):
                return False
            item_type = args[0] if args else Any
            return all(self.check_type(v, item_type) for v in value)

        # Handle more complex types that are collections and list-like
        if origin is list or issubclass(origin, tuple(self.list_like_types)):
            if not any(isinstance(value, t) for t in self.list_like_types):
                return False
            item_type = args[0] if args else Any
            return all(self.check_type(v, item_type) for v in value)

        # Handle sets
        if origin == set:
            if not isinstance(value, set):
                return False
            item_type = args[0] if args else Any
            return all(self.check_type(v, item_type) for v in value)
        
        # Handle datetime
        if origin == datetime.datetime:
            # try to instantiate datetime
            try:
                obj = datetime.datetime(**value)
                return True
            except:
                return False
        # Handle dictionaries
        if origin is dict or issubclass(origin, tuple(self.dict_like_types)):
            if not isinstance(value, (dict, Mapping)):#, MutableMapping, OrderedDict)):
                return False

            if args:
                if len(args) == 1:
                    key_type = args[0]
                    value_type = Any  # General assumption; specific dict-like types might differ
                elif len(args) == 2:
                    key_type, value_type = args
                else:
                    key_type = value_type = Any
            else:
                key_type = value_type = Any

            return all(
                self.check_type(k, key_type) and self.check_type(v, value_type)
                for k, v in value.items()
            )

        # Handle pydantic models
        if self.is_pydantic_model(origin):
            try:
                #temp_model = create_model('TempModel', **value)
                if isinstance(value, origin):
                    return True
                #return isinstance(temp_model, origin)
                # check if value is dict
                if not isinstance(value, dict):
                    return False
                # get all required init arguments for origin
                # required arguments are the ones withouyt default values
                required_fields = [field for field, field_type in origin.__annotations__.items() if not (typing.get_origin(field_type) is Union and type(None) in typing.get_args(field_type))]
                # check that all required arguments are in value and do type checking
                for arg in required_fields:
                    # check if it is in value
                    if arg not in value:
                        return False
                    # get the type of the argument
                    arg_type = origin.__annotations__[arg]
                    if not self.check_type(value[arg], arg_type):
                        return False
                # check that all arguments in value are correct type
                # this is additional check, because the above check only checks required arguments
                for arg, obj in value.items():
                    if arg in required_fields:
                        continue
                    arg_type = origin.__annotations__[arg]
                    if not self.check_type(value[arg], arg_type):
                        return False
                
                #origin.parse_obj(value)
                return True
            except Exception as e:
                print(e)
                return False

        # Handle dataclasses
        if self.is_dataclass_instance(origin):
            try:
                # for field in dataclasses.fields(origin):
                #     field_name = field.name
                #     field_type = field.type
                #     if field_name not in value or not self.check_type(value[field_name], field_type):
                #         return False
                # return True
                obj = origin(**value)
                return dataclasses.asdict(obj) == value
            except:
                return False

        # Handle dataclasses and arbitrary class types
        if inspect.isclass(origin) and not self.is_base_type(origin):
            # Ensure the value is an instance of the class
            if not isinstance(value, origin):
                return False

            # Gather type hints from the class and its bases
            type_hints = {}
            for cls in reversed(origin.__mro__):
                type_hints.update(get_type_hints(cls))

            # Validate each attribute of the class
            for attr, attr_type in type_hints.items():
                attr_value = getattr(value, attr, None)
                if not self.check_type(attr_value, attr_type):
                    return False

            return True

        return False

    @staticmethod
    def is_pydantic_model(cls):
        return hasattr(cls, 'parse_obj')

    @staticmethod
    def is_dataclass_instance(cls):
        return hasattr(cls, '__annotations__') and hasattr(cls, '__dataclass_fields__')


    @staticmethod
    def _is_subclass_of_generic(cls: Type, generic: Type) -> bool:
        """Determine if the class is a subclass of a generic type."""
        try:
            return issubclass(cls, generic) and cls is not generic
        except TypeError:
            if not hasattr(cls, '__origin__'):
                return False
        return cls.__origin__ is generic

    @staticmethod
    def _is_generic(cls: Type) -> bool:
        """Check if the provided type is a generic."""
        return hasattr(cls, "__origin__")

    def _get_recursive_args(self, target_type: Type) -> Tuple[Type, ...]:
        """
        Recursively check the base classes (i.e., the superclass chain) of the target type until we find one that
        retains the type arguments.
        :return: Type chain
        """
        if get_args(target_type):
            return get_args(target_type)
        for base in target_type.__bases__:
            args = self._get_recursive_args(base)
            if args:
                return args
        return ()

    def _find_generic_base_and_args(self, target_type: Type) -> Tuple[Type, Tuple[Type, ...]]:
        """
        Navigate up the MRO to find the first generic base and its arguments.
        """
        # First, check if target_type is a type annotation.
        # If so, directly return its origin and arguments.
        origin = get_origin(target_type)
        args = get_args(target_type)
        if origin and args:
            return origin, args

        # If target_type is a real class, then navigate its MRO.
        if hasattr(target_type, '__mro__'):
            if hasattr(target_type, '__orig_bases__'):
                for base in target_type.__orig_bases__:
                    if get_args(base):
                        return base, get_args(base)

            for base in target_type.__mro__:
                if get_args(base):
                    return base, get_args(base)

        return None, ()

    def _is_list_like(self, target_type: Type) -> bool:
        """Determine if the target type is list-like."""
        if target_type in {list, typing.List}:
            return True
        if hasattr(target_type, "__origin__") and target_type.__origin__ in {list, typing.List}:
            return True
        return False

    def _is_tuple_like(self, target_type: Type) -> bool:
        """Determine if the target type is tuple-like."""
        if target_type in {tuple, typing.Tuple}:
            return True
        if hasattr(target_type, "__origin__") and target_type.__origin__ in {tuple, typing.Tuple}:
            return True
        return False

    def _is_dict_like(self, target_type: Type) -> bool:
        """Determine if the target type is dict-like."""
        if target_type in {dict, typing.Dict}:
            return True
        if hasattr(target_type, "__origin__") and target_type.__origin__ in {dict, typing.Dict}:
            return True
        return False

    def _is_set_like(self, target_type: Type) -> bool:
        """Determine if the target type is set-like."""
        if target_type in {set, typing.Set}:
            return True
        if hasattr(target_type, "__origin__") and target_type.__origin__ in {set, typing.Set}:
            return True
        return False

    def instantiate(self, data: Any, target_type: Type) -> Any:
        """
        Attempts to convert a JSON-compatible data structure into an instance of the specified type.

        Args:
            data: JSON-compatible data structure to instantiate the target type.
            target_type: The type to instantiate from the given data.

        Returns:
            An instance of the target type initialized with the data.
        """

        # Handle None type
        if data is None:
            return None

        origin = get_origin(target_type) or target_type

        # If the target type is a built-in, attempt to instantiate and return
        if self.is_base_type(target_type) or target_type is Any:
            # If the parsed data is a string and target type is str, return it directly
            if isinstance(data, str) and target_type is str:
                return data

            # If any, return the data directly
            if target_type is Any:
                return data

            try:
                return target_type(data)
            except (ValueError, TypeError):
                # Handle the special case where the string represents a float but we want an integer
                if target_type is int:
                    try:
                        return int(float(data))
                    except (ValueError, TypeError):
                        pass

                if target_type is float:
                    try:
                        return int(float(data))
                    except (ValueError, TypeError):
                        pass
                raise TypeError(f"Failed to instantiate {target_type} from provided data.")
        # special handling for datetime
        if origin == datetime.datetime:
            # try to instantiate datetime
            try:
                return datetime.datetime(**data)
            except:
                raise TypeError(f"Failed to instantiate {target_type} from provided data.")

        # check if origin is Union, if so, instantiate the first type that works
        if origin == Union:
            for arg in get_args(target_type):
                try:
                    return self.instantiate(data, arg)
                except:
                    continue
            raise TypeError(f"Failed to instantiate {target_type} from provided data.")

        # If the data is a dictionary and the target is a custom class that can be instantiated from a dictionary.
        if isinstance(data, dict):
            if inspect.isclass(target_type) and not self.is_base_type(target_type):
                # Special handling for dataclasses
                if is_dataclass(target_type):
                    fields = [f.name for f in dataclasses.fields(target_type)]
                    type_hints = get_type_hints(target_type)
                    filtered_data = {k: self.instantiate(v, type_hints.get(k, Any)) for k, v in data.items() if
                                     k in fields}
                    return target_type(**filtered_data)

                # Special handling for Pydantic models
                if issubclass(target_type, BaseModel):
                    # instantiate the sub attributes
                    for attr, attr_type in target_type.__annotations__.items():
                        if attr in data:
                            data[attr] = self.instantiate(data[attr], attr_type)
                    try:
                        return target_type.model_validate(data)
                    except AttributeError as e:
                        # backwards compatibility with pydantic < 2
                        return target_type.parse_obj(data)


                # For general classes, attempt instantiation
                try:
                    return target_type(**data)
                except TypeError:
                    raise TypeError(f"Failed to instantiate {target_type.__name__} from dictionary.")
            # Handle dictionary-like types
            # Check if the target type is or inherits from defaultdict
            if origin is defaultdict or (isinstance(origin, type) and issubclass(origin, defaultdict)):
                key_type, value_type = get_args(target_type) if get_args(target_type) else (Any, Any)
                instantiated_items = {self.instantiate(k, key_type): self.instantiate(v, value_type) for k, v in
                                      data.items()}

                # For defaultdict, you'll need a default factory. Here, I'm using `int` for simplicity,
                # but you might want to adapt this based on your needs.
                return defaultdict(int, instantiated_items)

            # Handle set-like dict types like OrderedDict
            # the first check needs to be done to ensure origin has the __mro__ attribute
            elif inspect.isclass(origin)and any(issubclass(base, dict) for base in origin.__mro__):
                key_type, value_type = get_args(target_type) if get_args(target_type) else (Any, Any)
                instantiated_items = {self.instantiate(k, key_type): self.instantiate(v, value_type) for k, v in data.items()}
                return origin(instantiated_items)
            
            # Handle other dictionary-like types
            elif origin is dict or self._is_subclass_of_generic(origin, dict):
                key_type, value_type = get_args(target_type) if get_args(target_type) else (Any, Any)
                instantiated_dict = {self.instantiate(k, key_type): self.instantiate(v, value_type) for k, v in
                                     data.items()}

                # If the target_type is a subclass of dict, return an instance of target_type
                if self._is_subclass_of_generic(target_type, dict) and not self._is_generic(target_type):
                    return target_type(instantiated_dict)
                else:
                    return dict(instantiated_dict)

        # Tuples aren't supported in JSONable types, so we look for lists instead
        if isinstance(data, list):
            try:
                # If the origin or target type is a list-like type, or if it implements a list-like collections type
                # e.g Sequence[int]
                if origin is list or self._is_subclass_of_generic(origin, list):
                    base, item_types = self._find_generic_base_and_args(target_type)

                    item_type = item_types[0] if item_types else Any

                    instantiated_items = []

                    for item in data:
                        # For each item, validate and instantiate it
                        try:
                            instantiated_item = self.instantiate(item, item_type)
                        except ValueError:
                            raise TypeError(
                                f"Item of type {type(item).__name__} does not match expected type {item_type[0].__name__}.")

                        safe = self.check_type(instantiated_item, item_type)
                        if not safe:
                            raise TypeError(
                                f"Item of type {type(item).__name__} does not match expected type {item_type[0].__name__}.")
                        instantiated_items.append(instantiated_item)

                    # If target_type is a subclass of list, return an instance of target_type
                    if self._is_subclass_of_generic(target_type, list) and not self._is_generic(target_type):
                        return target_type(instantiated_items)

                    return instantiated_items
            
                # Handle tuples
                if self._is_tuple_like(target_type) or (isinstance(origin, type) and issubclass(origin, tuple)):
                    base, item_types = self._find_generic_base_and_args(target_type)

                    instantiated_items = []

                    # If there are no subscripted types, assume Any
                    if not item_types:
                        item_types = (Any,) * len(data)

                    for i, item in enumerate(data):
                        # For each item, validate and instantiate it
                        instantiated_item = self.instantiate(item, item_types[i])
                        instantiated_items.append(instantiated_item)

                        # If the instantiated item does not match the expected type, raise an exception
                        _type = item_types[i]
                        if not isinstance(instantiated_item, _type):
                            raise TypeError(
                                f"Item {i} of type {type(item).__name__} does not match expected type {item_types[i].__name__}.")

                    # Convert the list of instantiated items to a tuple
                    instantiated_tuple = tuple(instantiated_items)
                    # If target_type is a subclass of tuple, return an instance of target_type
                    if self._is_subclass_of_generic(target_type, tuple):
                        return target_type(instantiated_tuple)

                    return instantiated_tuple

                # Handle sets
                if self._is_set_like(target_type) or (isinstance(origin, type) and issubclass(origin, set)):
                    base, item_type = self._find_generic_base_and_args(target_type)

                    if not item_type:
                        item_type = Any

                    instantiated_items = set()

                    for item in data:
                        # For each item, validate and instantiate it
                        instantiated_item = self.instantiate(item, item_type[0])
                        instantiated_items.add(instantiated_item)

                        # If the instantiated item does not match the expected type, raise an exception
                        if not isinstance(instantiated_item, item_type[0]):
                            raise TypeError(
                                f"Item of type {type(item).__name__} does not match expected type {item_type[0].__name__}.")

                    # If target_type is a subclass of set, return an instance of target_type
                    if self._is_subclass_of_generic(target_type, set):
                        return target_type(instantiated_items)

                    return instantiated_items

                # Handle deques
                if origin is deque or (isinstance(origin, type) and issubclass(origin, set)):
                   item_type = get_args(target_type)[0] if get_args(target_type) else Any
                   return deque(self.instantiate(item, item_type) for item in data)

                if origin is frozenset or (isinstance(origin, type) and issubclass(origin, frozenset)):
                   item_type = get_args(target_type)[0] if get_args(target_type) else Any
                   return frozenset(self.instantiate(item, item_type) for item in data)

            except TypeError as e:
                print(e)
                raise TypeError(f"Failed to instantiate {target_type} from list. {e}")

        # If none of the above, return the data as-is
        return data
