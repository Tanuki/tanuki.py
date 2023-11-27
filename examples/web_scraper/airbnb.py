import openai
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from typing import Optional

load_dotenv()

import tanuki


openai.api_key = os.getenv("OPENAI_API_KEY")


class AirBnb(BaseModel):
    city: str
    state: str
    dates: str
    price: float
    stars: float


@tanuki.patch
def extract_airbnb(content: str) -> Optional[AirBnb]:
    """
    Examine the content string and extract the airbnb details for the city, state,
    dates available, nightly price, and stars rating.
    """


@tanuki.align
def align_extract_airbnb() -> None:
    print("Aligning...")
    airbnb1 = "Caroga Lake, New YorkRoyal Mountain Ski ResortDec 3 – 8$200\xa0night$200 per night4.99"
    assert extract_airbnb(airbnb1) == AirBnb(
        city="Caroga Lake",
        state="New York",
        dates="Dec 3 - 8",
        price=200.0,
        stars=4.99,
    )


def selenium_driver() -> str:
    """Use selenium to scrape the airbnb url and return the page source."""

    # configure webdriver
    options = Options()
    # options.add_argument('--headless')  # Enable headless mode
    # options.add_argument('--disable-gpu')  # Disable GPU acceleration

    # launch driver for the page
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.airbnb.com/?tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&search_mode=flex_destinations_search&flexible_trip_lengths%5B%5D=one_week&location_search=MIN_MAP_BOUNDS&monthly_start_date=2023-12-01&monthly_length=3&price_filter_input_type=0&channel=EXPLORE&search_type=category_change&price_filter_num_nights=5&category_tag=Tag%3A5366")
    time.sleep(3)

    # refresh the page to remove the dialog modal
    driver.refresh()
    time.sleep(3)

    # Scroll halfway down page to get rest of listings to load
    scroll_position = driver.execute_script("return (document.body.scrollHeight - window.innerHeight) * 0.4;")
    driver.execute_script(f"window.scrollTo(0, {scroll_position});")
    time.sleep(3)

    # extract the page source and return
    page_source = driver.page_source
    driver.quit()
    return page_source


if __name__ == '__main__':

    # Align the function
    align_extract_airbnb()

    # Selenium driver to scrape the url and extract the airbnb information
    page_source = selenium_driver()

    # Beautiful Soup to parse the page source
    soup = BeautifulSoup(page_source, 'html.parser')
    entities = soup.find_all('div', class_="dir dir-ltr")

    # Remove entries that are not airbnb listings
    contents = [entity.text for entity in entities if entity.text != ""]
    contents = [c for c in contents if "$" in c]
    print(contents)

    # Tanuki to extract the airbnb information
    print("Tanuki Time!")
    airbnbs = []
    for content in contents[1:3]:
        airbnbs.append(extract_airbnb(content))
    print(airbnbs)
