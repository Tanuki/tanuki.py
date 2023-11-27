# Web Scraping

This example shows how Tanuki can be used with web scraping to easily populate the desired values into a structured class.

Six examples for web scraping with BeautifulSoup are provided:
- [Quotes](https://quotes.toscrape.com/)
- [Countries](https://www.scrapethissite.com/pages/simple/)
- [Job Postings](https://realpython.github.io/fake-jobs/)
- [Cocktails](https://kindredcocktails.com/cocktail/old-fashioned)
- [Car Specs](https://www.cars.com/research/mazda-cx_90-2024/)
- [StreetEasy Apartments](https://streeteasy.com/2-bedroom-apartments-for-rent/manhattan)

An additional example has been provided showing how to use Selenium with BeautifulSoup for scraping:
- [AirBnb](https://www.airbnb.com/?tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&search_mode=flex_destinations_search&flexible_trip_lengths%5B%5D=one_week&location_search=MIN_MAP_BOUNDS&monthly_start_date=2023-12-01&monthly_length=3&price_filter_input_type=0&channel=EXPLORE&category_tag=Tag%3A5366&search_type=category_change)

## Configuration

Make sure you have an account with OpenAI to access their underlying models.

Set the following environment variables in your `.env` file:
```
OPENAI_API_KEY=sk-XXX
USER_AGENT=... (Optional: only needed for StreetEasy example)
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

To align and test, run following command for the desired example of interest:
```
python quotes.py

python countries.py

python jobs.py

python cocktail.py

python cars.py

python streeteasy.py   # make sure to update User-Agent!

python airbnb.py
```
