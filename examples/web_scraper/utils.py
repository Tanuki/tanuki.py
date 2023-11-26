from bs4 import BeautifulSoup
import os
import requests
from typing import List, Optional


# Update request header if user agent is provided
USER_AGENT = os.getenv("USER_AGENT", None)

headers = {}
if USER_AGENT:
    headers['User-Agent'] = USER_AGENT


def scrape_url(url: str, class_name: Optional[str] = None) -> List[str]:
    """Scrape the url and return the raw content(s) of all entities with a given
    div class name."""
    print(f"Scraping {url=} for {class_name=} entities...")

    response = requests.get(url, headers=headers)
    contents = []

    # Parse html content if response is successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Loop through the matched elements to get contents if a class name is provided
        if class_name:
            entities = soup.find_all('div', class_=class_name)
            for entity in entities:
                content = entity.text
                content = content.lstrip().rstrip()
                contents.append(content)

        # Otherwise, just get the text content
        else:
            content = soup.text
            content = content.lstrip().rstrip() 
            contents.append(content)

    # Print error if response is unsuccessful
    else:
        print(response.status_code)
        print(response.text)

    return contents
