import base64
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle


# Replace with the path to your credentials file
KEY_FILE = 'credentials.json'

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_gmail_service():
  """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
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
    try:
        results = service.users().messages().list(userId=user_id, labelIds=['INBOX']).execute()
        messages = results['messages']

        for message in messages:
          msg = service.users().messages().get(userId=user_id, id=message['id']).execute()
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
            body_msg_str += base64.urlsafe_b64decode(payload['body']['data'].encode('ASCII')).decode('utf-8')
          elif 'parts' in payload.keys():
            for part in payload['parts']:
              if 'data' in part['body'].keys():
                body_msg_str += base64.urlsafe_b64decode(part['body']['data'].encode('ASCII')).decode('utf-8')

            if 'parts' in payload.keys():
              for part in payload['parts']:
                if 'data' in part['body'].keys():
                  part_msg_str += base64.urlsafe_b64decode(part['body']['data'].encode('ASCII')).decode('utf-8')
                elif 'parts' in part.keys():
                  for sub_part in part['parts']:
                    if 'data' in sub_part['body'].keys():
                      part_msg_str += base64.urlsafe_b64decode(sub_part['body']['data'].encode('ASCII')).decode('utf-8')
                    elif 'parts' in sub_part.keys():
                      for sub_sub_part in sub_part['parts']:
                        if 'data' in sub_sub_part['body'].keys():
                          part_msg_str += base64.urlsafe_b64decode(sub_sub_part['body']['data'].encode('ASCII')).decode('utf-8')
                        elif 'parts' in sub_sub_part.keys():
                          for sub_sub_sub_part in sub_sub_part['parts']:
                            if 'data' in sub_sub_sub_part['body'].keys():
                              part_msg_str += base64.urlsafe_b64decode(sub_sub_sub_part['body']['data'].encode('ASCII')).decode('utf-8')

          is_spam = False
          # Example: mark all emails containing the word 'spam' in the subject as spam
          with open('spam_classifier_model.pkl', 'rb') as f:
            loaded_clf = pickle.load(f)

          with open('count_vectorizer.pkl', 'rb') as f:
            loaded_vectorizer = pickle.load(f)

          body_msg_str = body_msg_str.lower()
          body_msg_str_counts = loaded_vectorizer.transform([body_msg_str])
          body_msg_prediction = loaded_clf.predict(body_msg_str_counts)

          part_msg_str = part_msg_str.lower()
          part_msg_str_counts = loaded_vectorizer.transform([part_msg_str])
          part_msg_prediction = loaded_clf.predict(part_msg_str_counts)

          subject = subject.lower()
          subject_str_counts = loaded_vectorizer.transform([subject])
          subject_prediction = loaded_clf.predict(subject_str_counts)

          if len(body_msg_str) > 0:
            if body_msg_prediction[0] == "spam":
              is_spam = True
          else:
            if part_msg_prediction[0] == "spam":
              is_spam = True
            if subject_prediction[0] == "spam":
              is_spam = True

          from_email = from_email.lower()
          ham_email_addresses = ['accounts.google.com', 'googlecommunityteam-noreply@google.com', 'hotmail', 'outlook', 'aol', '.edu']
          for ham_email_address in ham_email_addresses:
            if  ham_email_address in from_email:
              is_spam = False

          if is_spam:
            service.users().messages().modify(userId=user_id, id=message['id'], body={'addLabelIds': ['SPAM']} ).execute()


    except HttpError as error:
        print(f'An error occurred: {error}')
        messages = None

    return messages

def main():
   service = get_gmail_service()

   classify_emails(service)

if __name__ == '__main__':
    main()
