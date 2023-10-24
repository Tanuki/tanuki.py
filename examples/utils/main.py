import requests
from bs4 import BeautifulSoup
from typing import List, Annotated

def get_yelp_reviews(yelp_url: str) -> List[str]:
    response = requests.get(yelp_url)
    soup = BeautifulSoup(response.text, "html.parser")
    # get all "p" tags with class beginning with comment__
    reviews = soup.find_all("p", class_=lambda x: x and x.startswith("comment__"))
    return [review.text for review in reviews]