# Google News Scraper

This example scrapes the Google News RSS website for the latest news articles on a certain topic.

Each article is parsed using Tanukie, and all relevant articles are emailed to the recipient at the end of the scraping run.

## Configuration

You need to set the following environment variables:
```
OPENAI_API_KEY=sk-XXX
AWS_SECRET_ACCESS_KEY=XXX
AWS_ACCESS_KEY_ID=XXX
```

Ensure you have an account with AWS for their email sending service, and OpenAI to access their underlying models.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Ensure that `run.sh` has sufficient permissions
```bash
chmod +x run_main.sh
```

Open up CronTab
```
crontab -e
```
Get the absolute path to the `run_main.sh` script
```
pwd
```

Add the following line to your crontab file to run the script every 3 hours.

Ensure you include the absolute path of the `run_main.sh`, and a log file path to write to.
```
0 */3 * * * /path/to/your/bash/script/run_main.sh >> /path/to/your/logfile.log 2>&1
```
