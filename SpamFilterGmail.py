import base64
import os
import pickle

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

"""
Function get_email_service and first half of function classify_emails is we reference the site https://www.geeksforgeeks.org/how-to-read-emails-from-gmail-using-gmail-api-in-python/ and
https://codehandbook.org/how-to-read-email-from-gmail-using-python/ So we can learn how to get the email from the email provider and read them into the string so we can use them in our model to see if this is spam or not.
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
    This function classifies emails as spam or not using a pre-trained model and marks spam emails with the SPAM label.

    :param service: The Gmail API service object used to interact with the user's Gmail account
    :param user_id: The ID of the user whose emails are being classified. By default, it is set to 'me',
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

            for header in headers:
                if header['name'] == 'subject' or header['name'] == 'Subject':
                    subject += header['value']
                if header['name'] == 'From':
                    from_email += header['value']

            body_msg_str = ''
            part_msg_str = ''

            if 'data' in payload['body'].keys():
                body_msg_str += base64.urlsafe_b64decode(
                    payload['body']['data'].encode('ASCII')).decode('utf-8')
            elif 'parts' in payload.keys():
                for part in payload['parts']:
                    if 'data' in part['body'].keys():
                        body_msg_str += base64.urlsafe_b64decode(
                            part['body']['data'].encode('ASCII')).decode('utf-8')

                if 'parts' in payload.keys():
                    for part in payload['parts']:
                        if 'data' in part['body'].keys():
                            part_msg_str += base64.urlsafe_b64decode(
                                part['body']['data'].encode('ASCII')).decode('utf-8')
                        elif 'parts' in part.keys():
                            for sub_part in part['parts']:
                                if 'data' in sub_part['body'].keys():
                                    part_msg_str += base64.urlsafe_b64decode(
                                        sub_part['body']['data'].encode('ASCII')).decode('utf-8')
                                elif 'parts' in sub_part.keys():
                                    for sub_sub_part in sub_part['parts']:
                                        if 'data' in sub_sub_part['body'].keys():
                                            part_msg_str += base64.urlsafe_b64decode(
                                                sub_sub_part['body']['data'].encode('ASCII')).decode('utf-8')
                                        elif 'parts' in sub_sub_part.keys():
                                            for sub_sub_sub_part in sub_sub_part['parts']:
                                                if 'data' in sub_sub_sub_part['body'].keys():
                                                    part_msg_str += base64.urlsafe_b64decode(
                                                        sub_sub_sub_part['body']['data'].encode('ASCII')).decode('utf-8')

            """
            Start from here is we using our module to filter the spam email. Credit to Jingbo Wang
            """
            is_spam = False


"""
Taking the email body message string, converting it to lowercase,
transforming it using a pre-trained CountVectorizer object (loaded from a saved file), and then
using a pre-trained classifier (loaded from a saved file) to predict whether the email is spam or
not based on the transformed string. The result of the prediction is stored in the
`body_msg_prediction` variable.
"""
            body_msg_str = body_msg_str.lower()
            body_msg_str_counts = loaded_vectorizer.transform([body_msg_str])
            body_msg_prediction = loaded_clf.predict(body_msg_str_counts)

"""
Taking the email body message string from a specific part of the email (if
it exists), converting it to lowercase, transforming it using a pre-trained CountVectorizer object
(loaded from a saved file), and then using a pre-trained classifier (loaded from a saved file) to
predict whether the email is spam or not based on the transformed string. The result of the
prediction is stored in the `part_msg_prediction` variable.
"""
            part_msg_str = part_msg_str.lower()
            part_msg_str_counts = loaded_vectorizer.transform([part_msg_str])
            part_msg_prediction = loaded_clf.predict(part_msg_str_counts)

"""
Taking the subject of an email, converting it to lowercase, transforming it using a pre-trained CountVectorizer object (loaded from a saved file), and then using a pre-trained
 classifier (loaded from a saved file) to predict whether the email is spam or not based on the
 transformed string. The result of the prediction is stored in the `subject_prediction` variable.
 """
            subject = subject.lower()
            subject_str_counts = loaded_vectorizer.transform([subject])
            subject_prediction = loaded_clf.predict(subject_str_counts)
"""
Checking if the email body message string is not empty. If it is not empty, it uses a pre-trained classifier to predict whether the email is spam or not based on the transformed string. If the prediction is "spam", the `is_spam` variable is set to True. If the email body message string is empty, it checks the prediction of the pre-trained classifier on a specific part of the email (if it exists) and the subject of the email. If either of these predictions is "spam", the `is_spam` variable is set to True.
"""
            if len(body_msg_str) > 0:
                if body_msg_prediction[0] == "spam":
                    is_spam = True
            else:
                if part_msg_prediction[0] == "spam":
                    is_spam = True
                if subject_prediction[0] == "spam":
                    is_spam = True

            from_email = from_email.lower()
            ham_email_addresses = [
                'accounts.google.com', 'googlecommunityteam-noreply@google.com', 'hotmail', 'outlook', 'aol', '.edu']
            for ham_email_address in ham_email_addresses:
                if ham_email_address in from_email:
                    is_spam = False

            if is_spam:
                service.users().messages().modify(userId=user_id,
                                                  id=message['id'], body={'addLabelIds': ['SPAM']}).execute()

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
