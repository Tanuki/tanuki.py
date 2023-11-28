import ast
import datetime
import pprint

from typing import Optional, List

from pydantic import BaseModel, Field

import tanuki
from tanuki.assertion_visitor import AssertionVisitor, Or


def _parse(source, func):
    tree = ast.parse(source)
    _locals = locals()

    visitor = AssertionVisitor(locals(), patch_symbolic_funcs={"analyze_article": func})
    visitor.visit(tree)
    return visitor.mocks

def test_positive_contraints():
    class ArticleSummary(BaseModel):
        sentiment: float = Field(..., ge=-10, le=1.0)

    @tanuki.patch
    def analyze_article(html: str, company: str) -> ArticleSummary:
        pass

    source = \
    """
assert analyze_article(html_content, "nvidia") == ArticleSummary(
    sentiment=0.5,
)
    """

    tree = ast.parse(source)
    _locals = locals()

    visitor = AssertionVisitor(locals(), patch_symbolic_funcs={"analyze_article": analyze_article})
    visitor.visit(tree)

    mocks = visitor.mocks

    assert len(mocks) == 1

def test_negative_contraints():
    class ArticleSummary(BaseModel):
        sentiment: float = Field(..., ge=-10, le=1.0)

    @tanuki.patch
    def analyze_article(html: str, company: str) -> ArticleSummary:
        pass

    source = \
    """
assert analyze_article(html_content, "nvidia") == ArticleSummary(
    sentiment=-0.5,
)
    """

    tree = ast.parse(source)
    _locals = locals()

    visitor = AssertionVisitor(locals(), patch_symbolic_funcs={"analyze_article": analyze_article})
    visitor.visit(tree)

    mocks = visitor.mocks

    assert len(mocks) == 1

if __name__ == "__main__":
    test_negative_contraints()
    test_positive_contraints()

