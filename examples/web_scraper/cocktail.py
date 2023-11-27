import openai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

import tanuki
from utils import scrape_url


openai.api_key = os.getenv("OPENAI_API_KEY")


class Cocktail(BaseModel):
    name: str
    ingredients: List[str] = []
    instructions: str
    similar: List[str] = []


@tanuki.patch
def extract_cocktail(content: str) -> Optional[Cocktail]:
    """
    Examine the content string and extract the cocktail details for the ingredients, instructions, and similar cocktails.
    """


@tanuki.align
def align_extract_cocktail() -> None:
    print("Aligning...")
    cocktail = """Black Rose | Kindred Cocktails\n\n\n\n\n\n      Skip to main content\n    \n\n\n\n\n\nKindred Cocktails\n\n\nToggle navigation\n\n\n\n\n\n\n\n\nMain navigation\n\n\nHome\n\n\nCocktails\n\n\nNew\n\n\nInfo \n\n\nStyle guidelines\n\n\nIngredients\n\n\n\n\n\nMeasurement units\n\n\nHistoric Cocktail Books\n\n\nRecommended Brands\n\n\nAmari & Friends\n\n\nArticles & Reviews\n\n\n\n\n\nAbout us\n\n\nLearn More\n\n\nFAQ\n\n\nTerms of Use\n\n\nContact us\n\n\n\n\nYou \n\n\nLog in\n\n\nSign Up\n\n\nReset your password\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nHome\n\n\nCocktails\n\n\n                  Black Rose\n              \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nCopy\n\n\n\n\nBlack Rose\n \n\n\n\n\n\n\n\n\n\n2 oz Bourbon\n\n1 ds Grenadine\n\n2 ds Peychaud's Bitters\n\n1  Lemon peel (flamed, for garnish)\n\n\n\nInstructions\nFill an old-fashioned glass three-quarters full with ice.  Add the bourbon, grenadine, and bitters, and stir.  Garnish with the lemon peel.\n\n\n\n\n\n\nCocktail summary\n\n\n\nPosted by\nThe Boston Shaker\n on \n4/12/2011\n\n\n\n\nIs of\nunknown authenticity\n\n\nReference\nDale Degroff, The Essential Cocktail, p48\n\n\n\nCurator\nNot yet rated\n\n\nAverage\n3.5 stars (6 ratings)\n\n\n\nYieldsDrink\n\n\nScale\n\n\nBourbon, Peychaud's Bitters, Grenadine, Lemon peel\nPT5M\nPT0M\nCocktail\nCocktail\n1\ncraft, alcoholic\n3.66667\n6\n\n\n\n\n\n\n\n\n\n\nCocktail Book\n\nLog in or sign up to start building your Cocktail Book.\n\n\n\n\nFrom other usersWith a modest grenadine dash, this drink didn't do much for me, but adding a bit more won me over.\nSimilar cocktailsNew Orleans Cocktail — Bourbon, Peychaud's Bitters, Orange Curaçao, Lemon peelOld Fashioned — Bourbon, Bitters, Sugar, Lemon peelBattle of New Orleans — Bourbon, Peychaud's Bitters, Absinthe, Orange bitters, Simple syrupImproved Whiskey Cocktail — Bourbon, Bitters, Maraschino Liqueur, Absinthe, Simple syrup, Lemon peelDerby Cocktail — Bourbon, Bénédictine, BittersMother-In-Law — Bourbon, Orange Curaçao, Maraschino Liqueur, Peychaud's Bitters, Bitters, Torani Amer, Simple syrupMint Julep — Bourbon, Rich demerara syrup 2:1, MintThe Journey — Bourbon, Mezcal, Hazelnut liqueurBenton's Old Fashioned — Bourbon, Bitters, Grade B maple syrup, Orange peelFancy Mint Julep — Bourbon, Simple syrup, Mint, Fine sugar\n\nComments\n\n\n\n\n\nLog in or register to post comments\n\n\n\n\n\n\n\n\n© 2010-2023 Dan Chadwick. Kindred Cocktails™ is a trademark of Dan Chadwick."""
    assert extract_cocktail(cocktail) == Cocktail(
        name="Black Rose",
        ingredients=["2 oz Bourbon", "1 ds Grenadine", "2 ds Peychaud's Bitters", "1  Lemon peel (flamed, for garnish)"],
        instructions="Fill an old-fashioned glass three-quarters full with ice.  Add the bourbon, grenadine, and bitters, and stir.  Garnish with the lemon peel.",
        similar=["New Orleans Cocktail", "Old Fashioned", "Battle of New Orleans", "Improved Whiskey Cocktail", "Derby Cocktail", "Mother-In-Law", "Mint Julep", "The Journey", "Benton's Old Fashioned", "Fancy Mint Julep"],
    )


if __name__ == '__main__':

    # Align the function
    align_extract_cocktail()

    # Web scrape the url and extract the cocktail information
    url = "https://kindredcocktails.com/cocktail/old-fashioned"
    # url = "https://kindredcocktails.com/cocktail/journey"
    contents = scrape_url(url=url)
    print(contents)

    # Process the cocktail block using Tanuki
    cocktail = extract_cocktail(contents[0])
    print(cocktail)
