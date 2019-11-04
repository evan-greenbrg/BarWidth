import pickle
import googleapiclient
from googleapiclient.discovery import build


GDRIVE_CREDS = '/Users/evangreenberg/.config/gdrive/token.pickle'

def google_drive():
    with open(GDRIVE_CREDS, 'rb') as token:
        creds = pickle.load(token)

    # drive_service
    return build('drive', 'v3', credentials=creds)
