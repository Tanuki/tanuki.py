import json
import os

from apify_client import ApifyClient

# Initialize the ApifyClient with your API token
APIFY_KEY = os.getenv("APIFY_API_KEY")
client = ApifyClient(APIFY_KEY)


def yelp_scraper(url: str):
    # Prepare the Actor input
    run_input = {
        "debugLog": False,
        "directUrls": [url],
        "maxImages": 0,
        "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
        "reviewLimit": 20,
        "reviewsLanguage": "ALL",
        "scrapeReviewerName": False,
        "scrapeReviewerUrl": False,
        "searchLimit": 1,
    }

    # Run the Actor and wait for it to finish
    run = client.actor("yin/yelp-scraper").call(run_input=run_input)

    # Fetch and print Actor results from the run's dataset (if there are any)
    items = client.dataset(run["defaultDatasetId"]).get_items_as_bytes()
    items = json.loads(items)

    return items
