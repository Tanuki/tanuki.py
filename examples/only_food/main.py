import os
from typing import List, Annotated
import openai
from dotenv import load_dotenv
from pydantic import Field

from monkey import Monkey

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

good_everything = [
    "This restaurant is an absolute gem! From the moment we walked in, we were impressed. The food was out of this "
    "world – every dish was a burst of flavor and innovation. The service was impeccable, with attentive staff who "
    "made us feel like royalty. The atmosphere was cozy and inviting, and the location was perfect, right in the "
    "heart of the city. We'll be returning soon, no doubt about it!",
    "What an extraordinary dining experience! This restaurant has it all. The food was a masterpiece, with each dish "
    "carefully crafted and bursting with flavor. The service was outstanding; the staff were knowledgeable and "
    "courteous. The atmosphere was elegant and romantic, setting the perfect mood. And the location couldn't be "
    "better, making it a convenient choice for any occasion. A true culinary haven!",
    "This restaurant is a culinary paradise! The food here is a work of art, with flavors that danced on my taste "
    "buds. The service was impeccable, with a staff that made sure every need was met. The atmosphere was elegant yet "
    "comfortable, and the location was ideal – easy to access. Whether it's a special occasion or a casual dinner, "
    "this place exceeds expectations on all fronts.",
    "If you're a food lover, this restaurant is a must-visit. The food here is top-notch, with a creative and diverse "
    "menu that caters to all tastes. The service is exceptional, with attentive and friendly staff who truly care "
    "about your dining experience. The atmosphere is warm and inviting, and the location is centrally located for "
    "easy access. I can't recommend this place enough!",
    "This restaurant is the epitome of a perfect dining experience. The food is a revelation, with dishes that are "
    "both visually stunning and incredibly delicious. The service is unparalleled, with a staff that goes above and "
    "beyond to ensure your satisfaction. The atmosphere is cozy and intimate, setting the stage for memorable meals. "
    "And the location is just right – easy to find. I can't praise this place enough; it's a culinary delight!"
]

good_food_bad_service = [
    "I visited this restaurant with high hopes for their food, and I must say, it did not disappoint. The dishes were "
    "absolutely delicious, and the flavors were spot on. However, that's where the positives end. The service was "
    "abysmal, with slow and inattentive waitstaff, and it seemed like they were more interested in chatting among "
    "themselves than taking care of the customers. The atmosphere was lackluster, with uncomfortable seating and a "
    "noisy environment. And let's not even get started on the location; it's in the middle of nowhere, making it a "
    "hassle to get to. Great food, but not worth the trouble.",
    "I have mixed feelings about this place. The food is fantastic; there's no denying that. But everything else is a "
    "letdown. The service is unbelievably slow, and the servers seem disinterested in providing even the most basic "
    "customer service. The atmosphere is dull and uninspiring, and the location is in the middle of an industrial "
    "park – definitely not the kind of place you'd expect to find a good restaurant. If you're a true foodie who can "
    "overlook everything else, the dishes might be worth the trek, but I wouldn't recommend it for a special night out.",
    "I had high hopes for this restaurant due to the raving reviews about their food, and it did not disappoint in "
    "that department. The dishes were indeed superb. However, everything else was a letdown. The service was "
    "incredibly slow, and the staff seemed more interested in ignoring us than attending to our needs. The atmosphere "
    "was sterile and uninspiring, and the location in a remote, industrial area is far from ideal. If you're a "
    "die-hard foodie and can tolerate terrible service, a dismal atmosphere, and a terrible location, this might be "
    "the place for you.",
    "I have a love-hate relationship with this restaurant. The food is mind-blowingly good; I can't deny that. "
    "However, the service is painfully slow, and the servers act like they'd rather be anywhere else but here. The "
    "atmosphere is uninspiring, with no effort put into decor or ambiance. And don't even get me started on the "
    "location; it's in a desolate part of town that's inconvenient to reach. If you're only here for the food, "
    "then it might be worth the visit, but be prepared for terrible service, a lackluster atmosphere, and a terrible "
    "location."
    "I had heard so much about this restaurant's culinary delights, so I decided to give it a try. The food was, "
    "without a doubt, exceptional, and I couldn't fault it in any way. However, the service was shockingly slow and "
    "inattentive, making the dining experience less enjoyable. The atmosphere was bland and uninspiring, "
    "and the location is in a remote area that feels like a trek. If you're willing to put up with terrible service, "
    "a lackluster atmosphere, and a lousy location for the sake of excellent food, then this place is for you."
    ]

bad_everything = [
    "Where do I even begin with this restaurant? It's a complete disaster on all fronts. The food was inedible – I've "
    "had better microwave dinners. The service was a joke; it took an eternity to get our orders, and the staff "
    "seemed utterly disinterested. The atmosphere was depressing, with dim lighting and outdated decor that "
    "transported me back to the '70s. And let's not even talk about the location – it's in the middle of nowhere. "
    "Avoid this place at all costs.",
    "I cannot express enough how terrible this restaurant is. The food was not just bad; it was revolting. I wouldn't "
    "serve it to my worst enemy. The service was equally abysmal; the staff were rude, and it felt like they resented "
    "having customers. The atmosphere was gloomy, and the furniture was falling apart. As for the location, "
    "it's a hidden gem in the world of awful dining spots. Save your money and your sanity; don't ever set foot in "
    "this place.",
    "This restaurant is a disaster in every way imaginable. The food was disgusting; I couldn't even finish a single "
    "bite. The service was a nightmare; the waitstaff seemed to have no idea what they were doing, and they were "
    "incredibly slow. The atmosphere was dismal, with a strange smell in the air, and the decor was straight out of a "
    "horror movie. The location is the cherry on top – it's in a sketchy part of town that I wouldn't recommend "
    "anyone venture into. Stay far, far away from this place.",
    "I wouldn't wish this restaurant on my worst enemy. The food was a culinary catastrophe; I've never tasted "
    "anything so terrible. The service was non-existent; it took forever to even get a menu, and the waitstaff seemed "
    "to be on another planet. The atmosphere was gloomy and depressing, and the location was in a dodgy area that "
    "left me feeling unsafe. If you value your taste buds, your time, your mood, and your safety, stay far away from "
    "this restaurant.",
    "I made a grave mistake by dining at this wretched establishment. The food was inedible, and it left me "
    "questioning the culinary skills of the chef. The service was horrendous; the waitstaff were clueless and "
    "apathetic. The atmosphere was dismal, with peeling wallpaper and flickering lights that added to the overall "
    "misery. And the location? Well, it's in the sketchiest part of town. Save yourself the agony and avoid this "
    "restaurant like the plague."]

bad_food_good_service = [
    "I had high hopes for this restaurant after hearing raving reviews about its atmosphere, service, and location. "
    "The ambiance was indeed delightful, with a charming decor and cozy lighting. The service was impeccable, "
    "with attentive and friendly staff. The location was perfect, right in the heart of the city. But the one crucial "
    "aspect, the food, was a major disappointment. It was bland and uninspiring, a stark contrast to the rest of the "
    "experience. I wish the menu lived up to the rest of the restaurant's offerings.",
    "This restaurant has all the right ingredients for a perfect dining experience, except for the most important one "
    "– the food. The atmosphere was exquisite, with a chic and cozy interior that set the mood perfectly. The service "
    "was top-notch, with knowledgeable and attentive staff. The location was convenient, with easy access. But the "
    "food, oh, the food was a letdown. It lacked flavor, creativity, and passion. If only the culinary aspect matched "
    "the rest of the restaurant, it would be a dining paradise.",
    "It pains me to give this restaurant a less than stellar review, but I must be honest. The atmosphere was "
    "enchanting, the service was flawless, and the location was ideal – all of these aspects were beyond reproach. "
    "However, the food was a colossal letdown. It lacked the depth of flavors and creativity I was hoping for. It "
    "felt like the chef was just going through the motions. If only the culinary team could match the excellence of "
    "the rest of the establishment.",
    "This restaurant is a true paradox. The atmosphere was charming, the service was exemplary, and the location was "
    "convenient – all of these were fantastic. But, and it's a big \"but,\" the food was a significant "
    "disappointment. It lacked the taste and innovation I expected. It was as if the culinary team forgot the most "
    "crucial aspect of dining. I truly wish the kitchen could measure up to the excellence of the rest of the "
    "experience.",
    "I wanted to love this restaurant; I really did. The atmosphere was enchanting, the service was exceptional, "
    "and the location was perfect – all signs pointed to an outstanding dining experience. Alas, the food fell flat. "
    "It was mediocre at best, lacking the complexity and flavor that I had anticipated. It's a shame that the "
    "culinary aspect didn't match the excellence of the restaurant's other attributes."
    ]

@Monkey.patch
def rate_food(reviews: List[str]) -> Annotated[int, Field(ge=1, le=5)]:
    """
    Taking into account only remarks made specifically about the food
    and disregarding anything pertaining to atmosphere, location, service, etc. What would the rating be from 1 to 5?
    """


@Monkey.align
def test_food_rating():
    """We can test the function as normal using Pytest or Unittest"""

if __name__ == '__main__':
    rating = rate_food(good_everything)
    print(f"When the EVERYTHING was positive the rating is {rating} out of 5")

    rating = rate_food(bad_everything)
    print(f"When the EVERYTHING was negative the rating is {rating} out of 5")

    rating = rate_food(good_food_bad_service)
    print(f"When the food is good and service is bad the rating is {rating} out of 5")

    rating = rate_food(bad_food_good_service)
    print(f"When the food is bad and service is good the rating is {rating} out of 5")