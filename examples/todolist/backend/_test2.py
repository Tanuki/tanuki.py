import ast
import builtins
import datetime


class TodoItem:
    def __init__(self, goal, people, deadline=None):
        self.goal = goal
        self.people = people
        self.deadline = deadline


class AssertionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.mocks = {}  # {args: output}
        self.scopes = [{}]  # Stack of scopes to mimic variable scope in code
        self.imported_modules = {}  # keys are module names, values are the actual modules

    def load_variable_values(self, var_name):
        for scope in reversed(self.scopes):
            if var_name in scope:
                return scope[var_name]
        return None

    def visit_Import(self, node):
        for name in node.names:
            self.imported_modules[name.name.split('.')[0]] = __import__(name.name)

    def visit_ImportFrom(self, node):
        module_name = node.module
        self.imported_modules[module_name.split('.')[0]] = __import__(module_name)

    def visit_Assign(self, node):
        target = node.targets[0]
        value = node.value
        if isinstance(target, ast.Name):
            if isinstance(value, ast.List):
                self.scopes[-1][target.id] = [self.get_value(elt) for elt in value.elts]
            elif isinstance(value, ast.Name):
                self.scopes[-1][target.id] = self.load_variable_values(value.id)
            else:
                self.scopes[-1][target.id] = self.get_value(value)

    def get_value(self, node):
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.List):
            return [self.get_value(e) for e in node.elts]
        elif isinstance(node, ast.Dict):
            keys = [self.get_value(k) for k in node.keys]
            values = [self.get_value(v) for v in node.values]
            return dict(zip(keys, values))
        elif isinstance(node, ast.Call):
            return self.extract_output(node)
        else:
            # This is a simplification; you might want to add more types
            raise NotImplementedError(f"Node type {type(node)} not supported")

    def process_assert(self, node, iter_name=None):
        left = node.test.left
        right = node.test.comparators[0]

        if iter_name:
            inputs = self.load_variable_values(iter_name)
            for input_val in inputs:
                self.scopes.append({iter_name: input_val})
                self.process_assert_helper(left, right, iter_name)
                self.scopes.pop()
        else:
            self.process_assert_helper(left, right)

    def process_assert_with_tuple(self, node, iter_names, evaluated_expr):
        left = node.test.left
        right = node.test.comparators[0]

        if evaluated_expr is not None:
            # Loop through the evaluated expression (which should be an iterable)
            for input_tuple in evaluated_expr:
                # Create a new scope for these variables
                self.scopes.append(dict(zip(iter_names, input_tuple)))

                # Process the assert statement
                self.process_assert_helper(left, right)

                # Remove the temporary scope
                self.scopes.pop()

    def process_assert_helper(self, left, right, iter_name=None):
        if isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "create_todolist_item":
            self.process_assert_helper_lr(left, right, iter_name)

        if isinstance(right, ast.Call) and isinstance(right.func, ast.Name) and right.func.id == "create_todolist_item":
            self.process_assert_helper_lr(right, left, iter_name)

    def process_assert_helper_lr(self, left, right, iter_name=None):
        input_args = self.extract_args(left, iter_name)
        output = self.extract_output(right, iter_name)
        if len(input_args) == 1:
            self.mocks[input_args[0]] = output  # Use the value directly if only one argument
        else:
            self.mocks[tuple(input_args)] = output  # Use a tuple otherwise

    def extract_args(self, node, iter_name=None):
        # Assuming all arguments are positional
        args = []
        current_scope = self.scopes[-1]
        for arg in node.args:
            value = self.eval_expr(arg)
            if callable(value):
                if iter_name is not None:
                    arg_values = [current_scope.get(arg_name, None) for arg_name in iter_name]
                    value = value(*arg_values)
                else:
                    value = value()
            elif value is None:
                # If evaluation returns None, fall back to scope
                value = current_scope.get(arg.id, None)
            args.append(value)
        return args

    def eval_expr(self, node):
        current_scope = self.scopes[-1]
        if isinstance(node, ast.Name):
            # Prioritize variables in the current scope over built-in functions
            if node.id in current_scope:
                return current_scope[node.id]
            elif hasattr(builtins, node.id):
                func = getattr(builtins, node.id)
                return func

        elif isinstance(node, ast.Call):
            # Function call, could be 'zip' or other
            func = self.eval_expr(node.func)  # Recursively evaluate the function expression
            if func is None:
                raise ValueError(f"Function {node.func.id if isinstance(node.func, ast.Name) else ''} not found in the current scope or built-ins.")

            args = [self.eval_expr(arg) for arg in node.args]  # Recursively evaluate the arguments
            return func(*args)  # Assume func_name is a Python built-in, e.g., zip
        elif isinstance(node, ast.Num):
            return node.n  # Assume it's a number for simplicity
        elif isinstance(node, ast.Str):
            return node.s  # Assume it's a string for simplicity
        # ... handle other types of AST nodes
        else:
            raise NotImplementedError(f"Node type {type(node).__name__} not handled yet")

    def extract_output(self, node, scope=None):
        current_scope = self.scopes[-1]

        def eval_args(args, scope):
            return [self.extract_output(arg, scope) for arg in args]

        def eval_kwargs(keywords, scope):
            return {kw.arg: self.extract_output(kw.value, scope) for kw in keywords}

        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.List):
            return [self.extract_output(elt, scope) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {self.extract_output(k, scope): self.extract_output(v, scope) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Name):
            return current_scope.get(node.id, None)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                args = eval_args(node.args, scope)
                kwargs = eval_kwargs(node.keywords, scope)

                # if func_name == "TodoItem":
                #     return TodoItem(*args, **kwargs)
                # elif func_name == "datetime":
                #     return datetime.datetime(*args, **kwargs)
                # Generalized object instantiation
                if func_name in self.imported_modules:
                    module = self.imported_modules[func_name]
                    if hasattr(module, func_name):
                        obj = getattr(module, func_name)(*args, **kwargs)
                        return obj

                elif func_name in globals():
                    obj = globals()[func_name](*args, **kwargs)
                    return obj
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
                obj = self.extract_output(node.func.value, scope)
                args = eval_args(node.args, scope)
                kwargs = eval_kwargs(node.keywords, scope)

                if obj == datetime and func_name == "datetime":
                    return datetime.datetime(*args, **kwargs)
                else:
                    raise NotImplementedError(f"Attribute-based call {obj}.{func_name} not handled yet")

            # Add additional function handling here
        elif isinstance(node, ast.Attribute):
            # Handle attributes like datetime.datetime
            obj = self.extract_output(node.value, scope)
            if obj == datetime:
                return datetime
            else:
                return getattr(obj, node.attr)
        elif isinstance(node, ast.Tuple):
            return tuple(self.extract_output(elt, scope) for elt in node.elts)
        elif isinstance(node, ast.Constant):
            return node.value

        raise NotImplementedError(f"Node type {type(node).__name__} not handled yet")

    def visit_For(self, node):
        if isinstance(node.target, ast.Name):
            iter_name = self.eval_expr(node.iter)
            if iter_name:
                for stmt in node.body:
                    if isinstance(stmt, ast.Assert):
                        self.process_assert(stmt, iter_name)
        # If loop variable is a tuple
        elif isinstance(node.target, ast.Tuple):
            iter_names = [elt.id for elt in node.target.elts]  # Extract names from tuple
            #iter_name = self.extract_variable_name(node.iter)
            evaluated_expr = self.eval_expr(node.iter)
            for stmt in node.body:
                if isinstance(stmt, ast.Assert):
                    self.process_assert_with_tuple(stmt, iter_names, evaluated_expr)

        self.scopes.pop()

    def extract_variable_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id  # For now, just return the function name. You may also traverse args.
        return None  # Or raise an exception depending on your need

    def visit_Assert(self, node):
        self.process_assert(node)

    def visit_FunctionDef(self, node):
        self.scopes.append({})
        self.generic_visit(node)
        self.scopes.pop()


if __name__ == "__main__":
    source_code = '''
from datetime import datetime
async def define_behavior():
    assert create_todolist_item("I would like to go to the shop and buy some milk") == TodoItem(goal="Go to the store and buy some milk", people=["Me"])
    assert create_todolist_item("I need to go and visit George at 3pm tomorrow") == TodoItem(goal="Go and visit Jeff", people=["Me"], deadline="datetime")
    
    inputs = ["I would like to go to the store and buy some milk", "I need to go and visit Jeff at 3pm tomorrow"]
    outputs = [TodoItem(goal="Go to the store and buy some milk", people=["Me"]), TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime(2021, 1, 1, 15, 0))]
    for input, output in zip(inputs, outputs):
        assert create_todolist_item(input) == output
    '''

    tree = ast.parse(source_code)
    visitor = AssertionVisitor()
    visitor.visit(tree)

    print(visitor.mocks)
