import os
from typing import List, Annotated
import openai
from dotenv import load_dotenv
from pydantic import Field

from monkey import Monkey

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

reviews = [
    "I had the Grilled Salmon, and it was fantastic, definitely recommended!",
    "The Lobster Bisque was decent but not outstanding.",
    "I couldn't resist trying the Tiramisu, and it was a sweet surprise.",
    "The Spaghetti Carbonara is a definite must-try on the menu.",
    "The Beef Tenderloin is consistently good, a safe choice.",
    "Shrimp Scampi – a bit too heavy on the garlic, not my cup of tea.",
    "I paired a Chardonnay with the Duck Confit, an exquisite combination.",
    "The Beet Salad is a colorful and refreshing option.",
    "Branzino was recommended by the staff, and I wasn't disappointed.",
    "Save room for the Chocolate Lava Cake, it's worth every calorie.",
    "The Chicken Parmesan is a standout, the best in town.",
    "Ratatouille – a flavorful vegetarian choice, quite impressive.",
    "Ribeye Steak – always a winner here, highly recommended.",
    "The Onion Soup Gratinee was cheesy and flavorful, a great starter.",
    "Try the Miso-Glazed Cod – a unique, delicious dish.",
    "The Burrata Caprese was fresh and flavorful, definitely recommended.",
    "The Prime Rib is a crowd-pleaser and rightfully so.",
    "The Creme Brulee was a sweet ending to a fantastic meal.",
    "Indulge in the New York Cheesecake; it's a delightful treat.",
    "The Seafood Paella is a must-avoid; it's lacking in flavor and overpriced.",
]


@Monkey.patch
def recommended_dishes(reviews: List[str]) -> List[str]:
    """
    List the top 5 (or fewer) best dishes based on the given reviews
    """


@Monkey.align
def test_food_rating():
    """We can test the function as normal using Pytest or Unittest"""


if __name__ == '__main__':
    recommended = recommended_dishes(reviews)
    print("Recommended dishes: ", recommended)
