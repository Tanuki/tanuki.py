import openai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

import tanuki
from utils import scrape_url


openai.api_key = os.getenv("OPENAI_API_KEY")


class Quote(BaseModel):
    text: str
    author: str
    tags: List[str] = []


@tanuki.patch
def extract_quote(content: str) -> Optional[Quote]:
    """
    Examine the content string and extract the quote details for the text, author, and tags.
    """


@tanuki.align
def align_extract_quote() -> None:
    print("Aligning...")
    quote = "\nIt takes courage to grow up and become who you really are.\nby E.E. Cummings\n(about)\n\n\n            Tags:\n            \ncourage\n\n"
    assert extract_quote(quote) == Quote(
        text="It takes courage to grow up and become who you really are.",
        author="E.E. Cummings",
        tags=["courage"],
    )


if __name__ == '__main__':

    # Align the function
    align_extract_quote()

    # Web scrape the url and extract the list of quotes
    url = "https://quotes.toscrape.com/page/1/"
    contents = scrape_url(url=url, class_name="quote")

    # Process the quote blocks using Tanuki (only sampling a couple for demo purposes)
    quotes = []
    for content in contents[0:2]:
        c = content.replace('“', '')
        c = c.replace('”', '')
        quotes.append(extract_quote(c))
    print(quotes)
