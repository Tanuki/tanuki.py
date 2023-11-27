from dataclasses import dataclass
from datetime import datetime

from pydantic import Field
import tanuki
import unittest
from typing import Literal, Optional, List



@dataclass(frozen=True)
class ActionItem:
    goal: str = Field(..., description="The goal of the action item")
    deadline: datetime = Field(..., description="The deadline of the action item")
    completed: bool = Field(False, description="Whether the action item is completed")


class TestActionItems(unittest.TestCase):


    @tanuki.patch
    def create_action_items(self, input: str) -> List[ActionItem]:
        """
        Create action items from a bit of text.
        """
        pass

    @tanuki.align
    def test_align_create_action_items(self):
        """We can test the function as normal using Pytest or Unittest"""

        speech = "Heya, I need you to do three things for me. " \
                 "First, I need you to get me a coffee by 3pm. " \
                 "Second, I need you to get me a cheese by 4pm. " \
                 "Third, I need you to get me a chocolate by 5pm on Tuesday."
        output = self.create_action_items(speech)
        assert len(output) == 3


if __name__ == '__main__':
    unittest.main()
