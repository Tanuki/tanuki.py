import ast
import datetime
import pprint
from typing import List, Optional

from pydantic import BaseModel

from assertion_visitor import AssertionVisitor


class TodoItem(BaseModel):
    # def __init__(self, goal, people, deadline=None):
    #     self.goal = goal
    #     self.people = people
    #     self.deadline = deadline
    deadline: Optional[datetime.datetime] = None
    goal: str
    people: List[str]




if __name__ == "__main__":
    source_code = '''
from datetime import datetime
async def define_behavior():
    assert create_todolist_item("I would like to go to the shop and buy some milk") == TodoItem(goal="Go to the store and buy some milk", people=["Me"])
    assert create_todolist_item("I need to go and visit George at 3pm tomorrow") == TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime(2021, 1, 1, 15, 0))
    
    inputs = ["I would like to go to the store and buy some milk", "I need to go and visit Jeff at 3pm tomorrow"]
    outputs = [TodoItem(goal="Go to the store and buy some milk", people=["Me"]), TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime(2021, 1, 1, 15, 0))]
    for input, output in zip(inputs, outputs):
        assert create_todolist_item(input) == output
    '''

    source_code_2 = '''
from datetime import datetime
async def define_behavior():
    # Checking Equality
    assert create_todolist_item("I would like to go to the shop and buy some milk") == TodoItem(goal="Go to the store and buy some milk", people=["Me"])
    assert create_todolist_item("I need to go and visit George at 3pm tomorrow") == TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime(2021, 1, 1, 15, 0))
    
    # Checking Non-Equality
    assert create_todolist_item("I need to go and visit George at 3pm tomorrow") != TodoItem(goal="Go and visit George", people=["Me"], deadline=datetime(2021, 1, 1, 14, 0))
    
    # Checking None
    assert create_todolist_item("Invalid input") is None
    
    # Checking Not None
    assert create_todolist_item("I would like to go to the shop and buy some tomatoes") is not None
    
    # Checking Inclusion in List
    valid_outputs = [
        TodoItem(goal="Go to the store and buy some milk", people=["Me"]),
        TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime(2021, 1, 1, 15, 0))
    ]
    assert create_todolist_item("I would like to go to the shop and buy some cheese") in valid_outputs
    
    # Checking Exclusion from List
    invalid_outputs = [
        TodoItem(goal="Go to the gym", people=["Me"]),
        TodoItem(goal="Go and visit Sarah", people=["Me"], deadline=datetime(2021, 1, 1, 16, 0))
    ]
    assert create_todolist_item("I would like to go to the terrace and buy some milk") not in invalid_outputs
    
    # Iterating Through Lists of Inputs and Outputs
    inputs = ["I would like to go to the store and buy some milk", "I need to go and visit Jeff at 3pm tomorrow"]
    outputs = [TodoItem(goal="Go to the store and buy some milk", people=["Me"]), TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime(2021, 1, 1, 15, 0))]
    for input, output in zip(inputs, outputs):
        assert create_todolist_item(input) == output
    '''

    tree = ast.parse(source_code_2)
    visitor = AssertionVisitor(locals())
    visitor.visit(tree)

    pprint.pprint(visitor.mocks)
