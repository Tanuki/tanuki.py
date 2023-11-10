import openai
import os
from dotenv import load_dotenv
from typing import List

from monkey_patch.monkey import Monkey as monkey

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")


@monkey.patch
def match_email(email: str, names: List[str]) -> str:
    """
    Examine the list of names and return the name that is most likely to correspond to the email address.
    Return an empty string if no match is found.
    """

@monkey.align
def align_match_email() -> None:
    assert match_email("john.smith@gmail.com", ["John Smith"]) == "John Smith"


@monkey.patch
def summarize_text_extra(input: str, instructions: str) -> str:
    """
    Use the input argument and apply the instructions argument for how to assemble into response.
    """


# Example usage
if __name__ == '__main__':
    print("Aligning...")
    # align_match_email()

    print("Matching...")
    # match = match_email("ethan.brown@gmail.com", ["Ethan Brown", "John Smith", "Mary Johnson"])
    # print(match)
    print(openai.api_key)
    summ = summarize_text_extra("This is a test", "Summarize the text")