import openai
import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

from monkey_patch.monkey import Monkey as monkey

openai.api_key = os.getenv("OPENAI_API_KEY")


@monkey.patch
def match_email(email: str, names: List[str]) -> Optional[str]:
    """
    Examine the list of names and return the name that is most likely to correspond to the email address.
    The names array includes both first names and last names. Make sure to consider both the first and last name
    as unique values when searching for a match. Be very conservative in what you consider to be a match.
    If there is no clear match, return an empty string. If there are multiple possible matches, return an empty string.
    If there are multiple partial matches, return an empty string.
    """


@monkey.align
def align_match_email() -> None:

    # 1:1 Matching
    assert match_email("john.smith@gmail.com", ["John Smith"]) == "John Smith"
    assert match_email("jsmith@google.com", ["John Smith"]) == "John Smith"
    assert match_email("j.smith@google.com", ["John Smith"]) == "John Smith"
    assert match_email("jasmith@google.com", ["John Smith"]) == "John Smith"

    # 1 from Many Matching
    assert match_email("john.smith@gmail.com", ["John Smith", "Jen Smith"]) == "John Smith"

    # Return None because there are multiple possible matches
    assert match_email("jsmith@gmail.com", ["John Smith", "Jen Smith"]) == ""
    assert match_email("johnsmith@gmail.com", ["John Smith", "John Smith Jr."]) == ""
    assert match_email("johnsmith123@gmail.com", ["John Smith", "John Smith III"]) == ""
    assert match_email("bill@example.net", ["Bill Burr", "Bill"]) == ""
    assert match_email("smith@example.co", ["Smith Sanders", "Jen Smith", "Smit Harris"]) == ""


# Example usage
if __name__ == '__main__':
    print("Aligning...")
    align_match_email()

    print("Matching...")
    match = match_email("ethan.brown@gmail.com", ["Ethan Brown", "John Smith", "Mary Johnson", "Eva Brown"])
    print(match)

    match = match_email("ebrown@gmail.com", ["Ethan Brown", "Eva Brown"])
    print(match)
