import base64
import os
import pickle

import tensorflow as tf
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

"""
Function get_email_service and first half of function classify_emails is we reference the site
https://www.geeksforgeeks.org/how-to-read-emails-from-gmail-using-gmail-api-in-python/ and
https://codehandbook.org/how-to-read-email-from-gmail-using-python/ So we can learn how to
get the email from the email provider and read them into the string so we can use them in our
model to see if this is spam or not.
"""
KEY_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']


def get_gmail_service():
    """
    This function returns a Gmail service object using user credentials.
    :return: a Gmail API service object that can be used to interact with a user's Gmail account.
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)


def classify_emails(service, user_id='me'):
    """
    This function classifies emails as spam or not using a pre-trained model and marks spam
    emails with the SPAM label.

    :param service: The Gmail API service object used to interact with the user's Gmail account
    :param user_id: The ID of the user whose emails are being classified. By default, it is set to
    'me',
    which refers to the authenticated user, defaults to me (optional)
    :return: the list of messages that were retrieved and processed by the function. However, if an
    error occurs, the function returns None.
    """
    try:
        # Example: mark all emails containing the word 'spam' in the subject as spam
        with open('spam_classifier_model.pkl', 'rb') as f:
            loaded_clf = pickle.load(f)

        with open('count_vectorizer.pkl', 'rb') as f:
            loaded_vectorizer = pickle.load(f)

        results = service.users().messages().list(
            userId=user_id, labelIds=['INBOX']).execute()
        messages = results['messages']

        for message in messages:
            msg = service.users().messages().get(
                userId=user_id, id=message['id']).execute()
            payload = msg['payload']
            headers = payload['headers']

            subject = ''
            from_email = ''
            # Iterating through the headers of an email message and extracting the values
            # of the "subject" and "From" fields.
            for header in headers:
                if header['name'] == 'subject' or header['name'] == 'Subject':
                    subject += header['value']
                if header['name'] == 'From':
                    from_email += header['value']

            body_msg_str = ''
            part_msg_str = ''

            # Extracting the email message body from the payload
            if 'data' in payload['body'].keys():
                body_msg_str += base64.urlsafe_b64decode(payload['body']['data'].encode('ASCII')).decode('utf-8')
            elif 'parts' in payload.keys():
                for part in payload['parts']:
                    if 'data' in part['body'].keys():
                        body_msg_str += base64.urlsafe_b64decode(part['body']['data'].encode('ASCII')).decode('utf-8')

            # Preprocessing the email body and subject
            body_msg_str = body_msg_str.lower()
            subject = subject.lower()

            # Transforming the email body and subject using the pre-trained vectorizer
            body_msg_str_counts = loaded_vectorizer.transform([body_msg_str])
            subject_str_counts = loaded_vectorizer.transform([subject])

            # Predicting the email class (spam or not spam)
            body_msg_prediction = loaded_clf.predict(body_msg_str_counts)
            subject_prediction = loaded_clf.predict(subject_str_counts)

            # Checking if the email is predicted as spam based on body or subject
            if body_msg_prediction[0] == "spam" or subject_prediction[0] == "spam":
                is_spam = True
            else:
                is_spam = False

            # Checking if the email address in the "From" field of the email is a known legitimate email address
            ham_email_addresses = ['accounts.google.com', 'googlecommunityteam-noreply@google.com', 'hotmail', '.edu']
            for ham_email_address in ham_email_addresses:
                if ham_email_address in from_email:
                    is_spam = False

            # Marking the email as spam if it is classified as such
            if is_spam:
                service.users().messages().modify(userId=user_id, id=message['id'],
                                                  body={'addLabelIds': ['SPAM']}).execute()

    except HttpError as error:
        print(f'An error occurred: {error}')
        messages = None

    return messages


def main():
    """
    The main function calls the get_gmail_service function and passes it to the classify_emails
    function.
    """
    service = get_gmail_service()
    classify_emails(service)


if __name__ == '__main__':
    main()
