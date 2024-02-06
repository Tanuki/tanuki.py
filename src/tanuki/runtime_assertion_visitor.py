import ast
import sys
from typing import Optional, Dict, Callable

from tanuki.models.function_type import FunctionType
from tanuki.register import Register


class Or(list):
    pass


class RuntimeAssertionVisitor(ast.NodeTransformer):
    def __init__(self,
                 instance: Optional[object] = None,
                 patch_symbolic_funcs: Dict[str, Callable] = {},
                 patch_embeddable_funcs: Dict[str, Callable] = {}):

        self.instance = instance
        self.patch_symbolic_names = list(patch_symbolic_funcs.keys())  # names of symbolic functions to patch
        self.patch_symbolic_funcs = patch_symbolic_funcs  # symbolic functions to patch
        self.patch_embeddable_names = list(patch_embeddable_funcs.keys())  # names of embeddable functions to patch
        self.patch_embeddable_funcs = patch_embeddable_funcs  # embeddable functions to patch

    def extract_func_name(self, call_node):
        """
        Extracts the function name from a call node.

        Args:
            call_node (ast.Call): The AST node representing a function call.

        Returns:
            str: The extracted function name.
        """
        if isinstance(call_node.func, ast.Name):
            # Direct function call, e.g., func_name(...)
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            # Method or namespaced function call, e.g., module.func_name(...)
            # For simplicity, this returns just the attribute name (func_name) but could be
            # extended to include the full path (module.func_name) if needed.
            return call_node.func.attr
        else:
            # If the function call doesn't match expected patterns (Name or Attribute),
            # raise an error or handle it according to your needs.
            raise NotImplementedError(f"Unsupported function call type: {type(call_node.func)}")

    def visit_Assert(self, node):
        # Check if either side of the assert involves a patched function
        left_func_name = self.extract_func_name(node.test.left) if isinstance(node.test.left, ast.Call) else None
        right_func_name = self.extract_func_name(node.test.comparators[0]) if isinstance(node.test.comparators[0],
                                                                                         ast.Call) else None

        left_is_patchable = self.is_function_patchable(left_func_name, self.instance) if left_func_name else False
        right_is_patchable = self.is_function_patchable(right_func_name,
                                                        self.instance) if right_func_name else False

        # Proceed only if at least one side is a patched function call
        if left_is_patchable or right_is_patchable:
            if isinstance(node.test.ops[0], ast.Eq):
                return self.create_register_call(node, _align_direction=True)
            elif isinstance(node.test.ops[0], ast.NotEq):
                return self.create_register_call(node, _align_direction=False)
            else:
                return node  # Non-supported comparison operator
        else:
            return node  # Neither side involves a patched function

    def visit_FunctionDef(self, node):
        # Remove the decorator if it matches decorator_name
        node.decorator_list = [dec for dec in node.decorator_list
                               if not (isinstance(dec, ast.Name) and dec.id == self.decorator_name)]
        self.generic_visit(node)
        return node

    def is_function_patchable(self, func_name, instance=None):
        """
        Checks if a function is listed as a patchable function in the Register.
        """
        if instance:
            function_names_to_patch = Register.function_names_to_patch(instance)
        else:
            function_names_to_patch = Register.function_names_to_patch()

        return func_name in function_names_to_patch

    def transform_arg(self, node):
        """
        Transforms an AST node into a form suitable for inclusion in the args list of an ast.Call node.
        """
        if isinstance(node, ast.Str):
            return node  # Strings can be directly included
        elif isinstance(node, ast.Num):
            return node  # Numbers can be directly included
        elif isinstance(node, ast.Name):
            return ast.Name(id=node.id, ctx=ast.Load())  # Variables are included by name
        elif isinstance(node, (ast.List, ast.Dict, ast.Tuple)):
            return node  # Directly include compound types
        elif isinstance(node, ast.Constant):
            return node
        else:
            raise NotImplementedError(f"Unsupported argument type: {type(node)}")

    def create_register_call(self, assert_node, _align_direction: bool = True):
        # Extract the function call and the expected output from the assert statement
        func_call = assert_node.test.left
        expected_output = assert_node.test.comparators[0]

        # Assuming the function call is directly a call to a function (not nested in other expressions)
        if not isinstance(func_call, ast.Call):
            raise ValueError("Assertion left side is not a function call")

        # Extract function name
        func_name_to_patch = self.extract_func_name(func_call)

        # Extract arguments and keyword arguments directly from the func_call node
        args = [ast.Str(s=func_name_to_patch)]  # Start with the function name as the first argument

        # Add arguments from the function call
        for arg in func_call.args:
            args.append(self.transform_arg(arg))

        # Transform keyword arguments if present
        kwargs = []
        for kw in func_call.keywords:
            kwargs.append(ast.keyword(arg=kw.arg, value=self.transform_arg(kw.value)))

        kwargs.append(ast.keyword(arg='expected_output', value=self.transform_arg(expected_output)))

        # Prepare 'expected_output' as a keyword argument
        expected_output_kwarg = ast.keyword(
            arg='__expected_output',
            value=self.transform_arg(expected_output)
        )
        align_direction_kwarg = ast.keyword(
            arg='__align_direction',
            value=ast.Constant(_align_direction)
        )

        # Construct the call to dynamic_call
        dynamic_call = ast.Call(
            func=ast.Name(id='dynamic_call', ctx=ast.Load()),
            args=args,
            keywords=[expected_output_kwarg, align_direction_kwarg]
        )

        return ast.Expr(value=dynamic_call)
