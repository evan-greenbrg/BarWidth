import pickle
import googleapiclient
from googleapiclient.discovery import build


OS_GDRIVE_CREDS = '/Users/evangreenberg/.config/gdrive/token.pickle'
WIN_GDRIVE_CREDS = 'C:/Users/grad/.config/gdrive/token.pickle'


def google_drive(os):
    if os.lower() == 'windows':
        cred_path = WIN_GDRIVE_CREDS
    else:
        cred_path = OS_GDRIVE_CREDS
    with open(cred_path, 'rb') as token:
        creds = pickle.load(token)

    # drive_service
    return build('drive', 'v3', credentials=creds)
