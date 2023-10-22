import ast
import datetime
from pprint import pprint

from todo_item import TodoItem


class AssertionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.mocks = {}  # {args: output}
        self.variable_mapping = {}  # new
        self.scopes = [{}]

    def load_variable_values(self, iter_name):
        # Look for the variable value in the current scope stack
        for scope in reversed(self.scopes):
            if iter_name in scope:
                return scope[iter_name]
        return None

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            variable_name = node.targets[0].id
            # Evaluate the value and store it in the mapping
            value = eval(compile(ast.Expression(node.value), '', 'eval'))
            self.variable_mapping[variable_name] = value

    def visit_For(self, node):
        if isinstance(node.target, ast.Tuple):
            target_names = [elt.id for elt in node.target.elts]
            iter_name = node.iter.id if isinstance(node.iter, ast.Name) else None  # This will hold the array ('inputs')
            for stmt in node.body:
                if isinstance(stmt, ast.Assert):
                    self.process_assert_with_loop(stmt, target_names, iter_name)
        elif isinstance(node.target, ast.Name):
            target_name = node.target.id
            iter_name = node.iter.id if isinstance(node.iter, ast.Name) else None  # This will hold the array ('inputs')
            for stmt in node.body:
                if isinstance(stmt, ast.Assert):
                    self.process_assert_with_loop_single_target(stmt, target_name, iter_name)

    def process_assert(self, left, right, scoped_variables):
        for side1, side2 in [(left, right), (right, left)]:
            if isinstance(side1, ast.Call) and isinstance(side1.func,
                                                          ast.Name) and side1.func.id == "create_todolist_item":
                input = self.extract_args_kwargs(side1, scoped_variables)
                output = self.extract_output(side2, scoped_variables)
                self.mocks[input] = output

    def process_assert_with_loop_single_target(self, node, iter_name):
        scoped_variables = self.variable_mapping.get(iter_name, {})
        left = node.test.left
        right = node.test.comparators[0]
        self.process_assert(left, right, scoped_variables)

    def process_assert_with_loop(self, node, target_names, iter_name):
        scoped_variables = self.variable_mapping.get(iter_name, {})
        left = node.test.left
        right = node.test.comparators[0]
        self.process_assert(left, right, scoped_variables)

    def visit_Assert(self, node):
        left = node.test.left
        right = node.test.comparators[0]
        self.process_assert(left, right, {})

    # def extract_args_kwargs(self, node):
    #     # For simplicity, assuming only positional arguments
    #     args = [arg.s if isinstance(arg, ast.Str) else None for arg in node.args]
    #     return tuple(args)

    def extract_args_kwargs(self, node, input_var=None):
        # For simplicity, assuming only positional arguments
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Str):
                args.append(arg.s)
            elif isinstance(arg, ast.Name):
                args.append(input_var.get(arg.id, None))
        return tuple(args)

    def extract_output(self, node, input_var=None):
        if isinstance(node, ast.Call):
            class_name = node.func.id if isinstance(node.func, ast.Name) else None
            if not class_name:
                class_name = node.func.attr if isinstance(node.func, ast.Attribute) else None

            kwargs = {}
            for kw in node.keywords:
                kwarg_name = kw.arg
                if isinstance(kw.value, ast.Call):
                    kwarg_value = self.extract_output(kw.value, input_var)
                else:
                    kwarg_value = self.get_value(kw.value, input_var)
                kwargs[kwarg_name] = kwarg_value

            args = [self.get_value(arg, input_var) for arg in node.args]

            return self.construct_object(class_name, *args, **kwargs)

    def construct_object(self, class_name, *args, **kwargs):
        # This is just a mockup. In real code, you'd probably want to dynamically
        # construct the object based on its class name and the available imports.
        if class_name == 'TodoItem':
            return TodoItem(*args, **kwargs)
        elif class_name == 'datetime':
            return datetime.datetime(*args, **kwargs)
        else:
            _globals = globals()
            _class = _globals.get(class_name, None)
            return _class(*args, **kwargs) if _class else None
        # Add more class construction logic here

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


# Parse the source code into an AST
source_code_1 = '''
async def define_behavior():
    assert create_todolist_item("I would like to go to the store and buy some milk") \
           == TodoItem(goal="Go to the store and buy some milk", people=["Me"])
    assert create_todolist_item("I need to go and visit Jeff at 3pm tomorrow") \
           == TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime.datetime(2021, 1, 1, 15, 0))
'''

source_code_2 = '''
async def define_behavior():
    inputs = ["I would like to go to the store and buy some milk",
                "I need to go and visit Jeff at 3pm tomorrow"]
    outputs = [TodoItem(goal="Go to the store and buy some milk", people=["Me"]),
                TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime.datetime(2021, 1, 1, 15, 0))]
    for input, output in zip(inputs, outputs):
        assert create_todolist_item(input) == output
'''
tree = ast.parse(source_code_2)

# Visit the AST nodes and identify assertions
visitor = AssertionVisitor()
visitor.visit(tree)

# The `mocks` attribute now contains the information needed to mock the function
pprint(visitor.mocks)

