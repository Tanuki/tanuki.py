import os

import openai
from dotenv import load_dotenv

from monkey_patch.monkey import Monkey as monkey

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@monkey.patch
def match_email_to_name(email: str, first_name: str, last_name: str) -> bool:
    """
    Uses an LLM to determine if the given email matches the provided first and last name,
    accounting for various edge cases that a regex might miss.

    Parameters:
    email (str): The email address to be matched.
    first_name (str): The first name of the person.
    last_name (str): The last name of the person.

    Returns:
    bool: True if the email matches the name, False otherwise.
    """

@monkey.align
def test_match_email_to_name():
    # Standard format test cases
    assert match_email_to_name("sarah.connor@skynet.com", "Sarah", "Connor") == True
    assert match_email_to_name("john.oconnor@cyberdyne.com", "John", "O'Connor") == True
    assert match_email_to_name("mike.smith@tynet.com", "Michael", "Smith") == False  # Nickname case

    # Middle initial cases
    assert match_email_to_name("david.m.johnson@enterprise.com", "David", "Johnson") == True
    assert match_email_to_name("peter.j.brown@oldtech.com", "Peter", "Brown") == True

    # Middle name cases
    assert match_email_to_name("james.robert.hendrix@rock.com", "James", "Hendrix") == True
    assert match_email_to_name("j.r.hendrix@rock.com", "James", "Hendrix") == True  # Initials for middle name

    # Initials cases
    assert match_email_to_name("e.k.eliot@poetry.com", "Thomas", "Eliot") == False  # Wrong first name
    assert match_email_to_name("t.s.eliot@poetry.com", "Thomas", "Eliot") == True

    # Hyphenated last names
    assert match_email_to_name("anne-marie.smith@books.com", "Anne", "Marie Smith") == False  # Hyphenated last name
    assert match_email_to_name("anne.marie-smith@books.com", "Anne", "Marie-Smith") == True

    # Don't try and guess
    assert match_email_to_name("theboss@dundermifflin.com", "Michael", "Scott") == False
    assert match_email_to_name("number1boss@dundermifflin.com", "Michael", "Scott") == False

    # Common nicknames
    assert match_email_to_name("bill.g@microsoft.com", "William", "Gates") == True
    assert match_email_to_name("peggy.m@commercial.com", "Margaret", "Mulligan") == True

    # Domain variations
    assert match_email_to_name("tim.apple@apple.com", "Tim", "Cook") == True
    assert match_email_to_name("jeff.bezos@amazon.com", "Jeff", "Bezos") == True

    # Non-matching cases
    assert match_email_to_name("steve.jobs@apple.com", "Tim", "Cook") == False
    assert match_email_to_name("m.zuckerberg@facebook.com", "Mark", "Zuck") == False  # Last name abbreviated

    # Additional cases
    # Including underscores, numbers, and common domain prefixes
    assert match_email_to_name("jane_doe123@domain.com", "Jane", "Doe") == True
    assert match_email_to_name("doe.jane@domain.com", "Jane", "Doe") == True
    assert match_email_to_name("info.janedoe@domain.com", "Jane", "Doe") == False  # Prefix not related to name

# Ensure the alignment assertions are registered
test_match_email_to_name()

# Example usage
email_matches = match_email_to_name("jane.doe@company.com", "Jane", "Doe")
print(f"Does the email match the name? {'Yes' if email_matches else 'No'}")

email_matches = match_email_to_name("j-bridgerton@sears", "John", "Bridgy")
print(f"Does the email match the name? {'Yes' if email_matches else 'No'}")