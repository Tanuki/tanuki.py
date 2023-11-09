# Email cleaner

This example first classifies email addresses as real or fake, extracts the name and company from the email address and saves them as a excel file. The data folder has the data and is where the excel will be written

Moneypatch is used in this example to first classify the emails and then extract information in a easily parsable way

## Configuration

You need to set the following environment variables:
```
OPENAI_API_KEY=sk-XXX
```

Ensure you have an account with OpenAI to access their underlying models.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Ensure that `run.sh` has sufficient permissions
```bash
chmod +x run_main.sh
```