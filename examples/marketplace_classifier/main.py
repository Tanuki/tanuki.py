import os

# Load in the environment variables
import dotenv
dotenv.load_dotenv()

# Initialize OpenAI
import openai
openai.api_key = os.getenv("OPENAI_API_KEY") # Your OpenAI API key

# import Tanuki
import tanuki
# import Pydantic to define the response type
from pydantic import Field, BaseModel

# Additional imports for the demo that will be used to fetch a random product from the Best Buy API
from random import randint
import requests


# First we define the desired response type using Pydantic
class ProductTags(BaseModel):
    category: str = Field(..., description="What category does this product fit into?")
    brand: str = Field(..., description="What brand is this product?")
    features: str = Field(..., description="What are the features of this product?")


# Next we define the function that will be used to generate the response with the Monkey Patch decorator
@tanuki.patch
def product_tagger(product_story: str) -> ProductTags:
    """
    based on the provided product story, extract the following aspects:
    - what category does this product fit into? (e.g., Sports & outdoors, Home Goods, Mens Clothing, Garden Tools, etc.)
    - what brand is this product? (e.g., Nike, Adidas, Apple, Samsung, etc.)
    - what are the features of this product? (e.g., Bluetooth, 4K, 5G, water-resistant, removable liner, etc.)
    If it is not possible to extract any of these aspects, return "unknown" for that aspect.
    """

# (OPTIONAL!) We use the align decorator to both test the function, and to teach the model about our expected output.
# This can include as many or as few assert statements as you like.
@tanuki.align
def test_product_tagger():
    """We can test the function as normal using Pytest or Unittest"""
    assert product_tagger("Product: Apple iPhone 12 Pro Max 5G 128GB - Pacific Blue (Verizon) Short Description: "
                          "iPhone 12 Pro Max. 5G to download movies on the fly and stream high-quality video.¹ "
                          "Beautifully bright 6.7-inch Super Retina XDR display.² Ceramic Shield with 4x better drop "
                          "performance.³ Incredible low-light photography with the best Pro camera system on an "
                          "iPhone, and 5x optical zoom range. Cinema-grade Dolby Vision video recording, editing, "
                          "and playback. Night mode portraits and next-level AR experiences with the LiDAR Scanner. "
                          "Powerful A14 Bionic chip. And new MagSafe accessories for easy attach and faster wireless "
                          "charging.⁴ For infinitely spectacular possibilities. Features: 6.7-inch Super Retina XDR "
                          "display Ceramic Shield, tougher than any smartphone glass A14 Bionic chip, the fastest "
                          "chip ever in a smartphone Pro camera system with 12MP Ultra Wide, Wide, and Telephoto "
                          "cameras; 5x optical zoom range; Night mode, Deep Fusion, Smart HDR 3, Apple ProRAW, "
                          "4K Dolby Vision HDR recording LiDAR Scanner for improved AR experiences, Night mode "
                          "portraits 12MP TrueDepth front camera with Night mode, 4K Dolby Vision HDR recording "
                          "Industry-leading IP68 water resistance Supports MagSafe accessories for easy attach and "
                          "faster wireless charging iOS with redesigned widgets on the Home screen, all-new App "
                          "Library, App Clips and more") == ProductTags(
        category='Electronics',
        brand='Apple',
        features='5G, 4K, Dolby Vision, LiDAR Scanner, Night mode, 12MP TrueDepth front camera, IP68 water '
                 'resistance, MagSafe accessories, iOS'
    )


# Finally we can run the function on a random product from the Best Buy API
if __name__ == '__main__':

    # get the Best Buy API key from the environment variables (this is set in .env file)
    bestbuy_api_key = os.getenv("BESTBUY_API")

    # get random number between 1 and 160000 (roughly the number of products in the Best Buy API)
    random_number = randint(1, 160000)

    # set the Best Buy API endpoint
    bestbuy_api = "https://api.bestbuy.com/v1/products"

    # fetch a random product from the API
    response = requests.get(f"{bestbuy_api}?apiKey={bestbuy_api_key}&pageSize=1&page={random_number}&format=json")
    response.raise_for_status()

    # get the product from the response (if no errors were raised)
    product = response.json()['products'][0]

    # Let's assemble the input for the product tagger
    product_story = f"""
    Product: {product['name']}
    Short Description: {product['shortDescription']}
    Description: {product['description']}
    Long Description: {product['longDescription']}
    """

    # (Optional) Run the test to ensure the function works as expected and to teach the model about the expected output
    test_product_tagger()

    # Run the function on the product story and show the results
    print("\nOriginal input:")
    print(product_story)
    print("\nGenerated output:")

    tagged_product = product_tagger(product_story)

    # we can safely print the model dump because we know its type is ProductTags
    print(tagged_product.model_dump())