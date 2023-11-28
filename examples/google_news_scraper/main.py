import logging
import os
from datetime import datetime
from typing import List
from xml.etree import ElementTree as ET

import openai
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from requests_html import HTMLSession
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

from examples.google_news_scraper.utils import send_email
import tanuki

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Define Pydantic model of an article summary
class ArticleSummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    impact: int = Field(..., ge=0, le=10)
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    date: datetime
    companies_involved: List[str]
    people_involved: List[str]
    summary: str

def configure_selenium_user_agent():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=en-US,en")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    return options


def get_absolute_redirect_url_from_google_rss(url):
    """
    This function uses requests-html to get the absolute URL of a give link in a Google RSS feed.
    :param url:
    :return:
    """
    session = HTMLSession()
    response = session.get(url, allow_redirects=True)
    try:
        response.html.render(timeout=20)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        session.close()

    absolute_links = list(response.html.absolute_links)
    redirected_url = absolute_links[0] if absolute_links else None
    return redirected_url


def parse_article_with_selenium(url: str) -> str:
    """
    This function uses Selenium to extract the text of an article from a given URL.
    """

    options = configure_selenium_user_agent()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # Perform your parsing with BeautifulSoup here

        # For example, to get text without tags:
        article_text = soup.get_text(separator=' ', strip=True)
        return article_text
    finally:
        driver.quit()


def scrape_google_news(search_term: str, recipient: str, max=5):
    """
    This function scrapes Google News for articles about a given search term.
    :param search_term:
    :param recipient:
    :param max:
    :return:
    """
    # RSS feed URL with the search term
    url = f"https://news.google.com/rss/search?q={search_term}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)
    root = ET.fromstring(response.content)
    items = root.findall('./channel/item')

    logging.info(f"Found {len(items)} articles for {search_term}")

    relevant_articles = []
    for item in items[:max]:
        # Extract the link to the full article
        link = item.find('link').text

        final_url = get_absolute_redirect_url_from_google_rss(link)
        if final_url:
            article_content = parse_article_with_selenium(final_url)
            try:
                article_summary = analyze_article(article_content, search_term)
                logging.info(article_summary)

                # Check if the article is relevant based on impact and sentiment
                if article_summary.impact > 5 and article_summary.sentiment < 0:
                    relevant_articles.append(article_summary)
            except Exception as e:
                logging.warning(f"An error occurred: {final_url=} {e}")
                continue

    return relevant_articles


def email_if_relevant(relevant_articles: List[ArticleSummary], search_term: str, recipient: str):
    """
    This function sends an email if relevant articles were found relating to the search term.
    :param relevant_articles: A list of relevant articles extracted from a website.
    :param search_term:
    :param recipient:
    :return:
    """
    if relevant_articles:
        subject = f"Summary of Important Articles about {search_term}"
        body = "The following articles about {search_term} have high impact and negative sentiment:\n\n"
        for summary in relevant_articles:
            body += f"- {summary.summary} (Impact: {summary.impact}, Sentiment: {summary.sentiment})\n"

        send_email(subject, body, recipient)


@tanuki.patch
def analyze_article(html_content: str, subject: str) -> ArticleSummary:
    """
    Analyzes the article's HTML content and extracts information relevant to the subject.
    """


@tanuki.align
def align_analyze_article():

    html_content = "<head></head><body><p>Nvidia has made the terrible decision to buy ARM for $40b on 8th November. This promises to "\
                   "be an extremely important decision for the industry, even though it creates a monopoly.</p></body> "
    assert analyze_article(html_content, "nvidia") == ArticleSummary(
        impact=10,
        sentiment=-0.9,
        date=datetime(2023, 11, 8),
        companies_involved=["Nvidia", "ARM"],
        people_involved=[],
        summary="Nvidia is acquiring ARM for $40 billion, which will have a huge impact on the semiconductor industry.",
    )




# Example usage:
if __name__ == '__main__':
    align_analyze_article()

    recipient = 'dummy@example.com'
    search_term = 'nvidia'
    relevant_articles = scrape_google_news(search_term, recipient)
    email_if_relevant(relevant_articles, search_term, recipient)
