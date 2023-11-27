import openai
import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

import tanuki

openai.api_key = os.getenv("OPENAI_API_KEY")


@tanuki.patch
def match_email(email: str, names: List[str]) -> Optional[List[str]]:
    """
    Examine the list of names and return all the names that are likely to correspond to the email address.
    The names array includes both first names and last names. Make sure to consider both the first and last name
    as unique values when searching for a match. Be very conservative in what you consider to be a match.
    If there is one clear match, return a list with one name. If there are multiple clear matches, return a list with all the possible matches.
    If there are no possible matches, return None.
    """


@tanuki.align
def align_match_email() -> None:

    # 1:1 Matching
    assert match_email("john.smith@gmail.com", ["John Smith"]) == ["John Smith"]
    assert match_email("jsmith@google.com", ["John Smith"]) == ["John Smith"]
    assert match_email("j.smith@google.com", ["John Smith"]) == ["John Smith"]
    assert match_email("jasmith@google.com", ["John Smith"]) == ["John Smith"]
    assert match_email("t.swift@yahoo.com", ["John Smith", "Taylor Swift", "Satia Swift", "Milton Swift"]) == ["Taylor Swift"]
    
    # 1: Many Matching
    assert match_email("john.smith@gmail.com", ["John Smith", "Jen Smith"]) == ["John Smith"]

    # Multiple matches
    assert match_email("jsmith@google.com", ["John Smith", "Ella Oakly", "George Bush", "Jaden Smith"]) == ["John Smith", "Jaden Smith"]
    assert match_email("George@google.com", ["John Smith", "George Oakly", "George Bush", "Jaden Smith"]) == ["George Oakly", "George Bush"]

    # Return None because there are no possible matches
    assert match_email("rsunak@gmail.com", ["John Smith", "Jen Smith", "Emily Johnson"]) == None
    assert match_email("marcopolo@gmail.com", ["John Smith", "John Smith Jr.", "David Anderson"]) == None
    assert match_email("johnsmith123@gmail.com", ["Lando Norrs", "David Singh"]) == None


def wrap_match_email(email: str, names: List[str]) -> Optional[str]:
    """
    Wrapper method to call `match_email` method multiple times.
    For this particular use-case, the filter function (match_email) needs to be called atleast twice as GPT4 has quite a high False-Positive rate when working with longer lists of names.
    Calling the filtering function twice worked well to reduce the final False-Positive rate to a acceptable minimum
    """

    match = match_email(email, names) # first call, may include false-positives

    if match:
        match_revised = match_email(email, match) # second call to remove false positives
        if match_revised and len(match_revised) == 1:
            return match_revised[0]

    return None

# Example usage
if __name__ == '__main__':
    print("Aligning...")
    align_match_email()

    print("Matching...")
    match = wrap_match_email("ethan.brown@gmail.com", ["Ethan Brown", "John Smith", "Mary Johnson", "Eva Brown"])
    print(match)

    match = wrap_match_email("ebrown@gmail.com", ["Ethan Brown", "Eva Brown"])
    print(match)
