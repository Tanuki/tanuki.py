# Web Scraping

This example shows how MonkeyPatch can be used with web scraping to easily populate the desired values into a structured class.

Six examples for web scraping are provided:
- [Quotes](https://quotes.toscrape.com/)
- [Countries](https://www.scrapethissite.com/pages/simple/)
- [Job Postings](https://realpython.github.io/fake-jobs/)
- [Cocktails](https://kindredcocktails.com/cocktail/old-fashioned)
- [Car Specs](https://www.cars.com/research/mazda-cx_90-2024/)
- [StreetEasy Apartments](https://streeteasy.com/2-bedroom-apartments-for-rent/manhattan)

## Configuration

Ensure you have an account with OpenAI to access their underlying models.

Set the following environment variables in your `.env` file:
```
OPENAI_API_KEY=sk-XXX
USER_AGENT=... (Optional and only needed for StreetEasy example)
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
```
