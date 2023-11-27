import openai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

import tanuki
from utils import scrape_url


openai.api_key = os.getenv("OPENAI_API_KEY")


class Car(BaseModel):
    price: float
    mpg: str
    seating: int
    horsepower: int
    weight: int
    fuel_size: float
    warranty_basic: str
    warranty_powertrain: str
    warranty_roadside: str


@tanuki.patch
def extract_car(content: str) -> Optional[Car]:
    """
    Examine the content string and extract the car details for the price, miles per gallon, seating, horsepower,
    weight, fuel tank size, and warranty.
    """


if __name__ == '__main__':

    # Web scrape the url and extract the car information
    # url = "https://www.cars.com/research/ford-mustang-2024/"
    url = "https://www.cars.com/research/mazda-cx_90-2024/"
    contents = scrape_url(url=url)
    print(contents)

    # Process the cocktail block using Tanuki
    car = extract_car(contents[0])
    print(car)
