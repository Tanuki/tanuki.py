from typing import List, Dict
from services.food_service import (
    specific_ratings,
    recommended_dishes,
    test_specific_ratings,
)


def get_ratings(reviews: List[str]) -> Dict[str, int]:
    # test_specific_ratings()
    rating = specific_ratings(reviews)
    return rating.model_dump()


def get_best_dishes(reviews: List[str]) -> List[str]:
    recommended = recommended_dishes(reviews)
    return recommended
