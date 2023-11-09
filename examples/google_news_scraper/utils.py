import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

VERIFIED_SOURCE_EMAIL = os.getenv('VERIFIED_SOURCE_EMAIL')

from boto3.session import Session

def send_email(subject, body, recipient):
    session = Session()
    ses_client = session.client('ses')
    try:
        response = ses_client.send_email(
            Source=VERIFIED_SOURCE_EMAIL,  # This email must be verified with Amazon SES.
            Destination={
                'ToAddresses': [
                    recipient,
                ],
            },
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': 'UTF-8'
                },
                'Body': {
                    'Text': {
                        'Data': body,
                        'Charset': 'UTF-8'
                    },
                },
            }
        )
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
