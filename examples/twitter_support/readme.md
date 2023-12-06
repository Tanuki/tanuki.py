# Twitter Support bot

This example takes in a tweet to a company's support account and first creates an empathetic response to the tweet and classifies the tweet as if it requires further action from the company's support team. If the tweet does require further action, then a support ticket is created that can be saved to the company's internal database for further action.

Tanuki is used in this example to create the response, classify the tweet and create the support ticket if needed

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