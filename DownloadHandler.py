import requests
import tempfile
import io
import os
import shutil

import googleapiclient

import connections


TEST_PATH = 'http://landsat-pds.s3.amazonaws.com/c1/L8/052/122/LC08_L1GT_052122_20191103_20191103_01_RT/LC08_L1GT_052122_20191103_20191103_01_RT_B1.TIF'
TEST_FILENAME = 'LC08_L1GT_052122_20191103_20191103_01_RT_B1.TIF' 

class DownloadError(Exception):
    pass


class DownloadHandler():

    def __init__(self, os):
        self.folder_id = '1kB8PsWY0xRo7pLVHKXlcg4yVf6S0wlDj'
        self.drive_service = connections.google_drive(os)


    def download_from_landsat(self, url):
        print('Beginning file download')
        print('From: {0}'.format(url))

        r = requests.get(url)
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(r.content)
        f.close()
        print('Finished')

        return f

    def upload_to_gdrive(self, filebody, img_name):
        print('Uploading File')
        file_metadata = {
           'name': img_name,
           'parents': [self.folder_id]
        }
        media = googleapiclient.http.MediaFileUpload(
            filebody.name,
            mimetype='image/tiff'
        )
        file = self.drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print ('File ID: %s' % file.get('id'))

    def put(self, url):
        img_name = url.split('/')[-1]
        f = self.download_from_landsat(url)
        self.upload_to_gdrive(f, img_name) 

    def save(self, url, root):
        img_name = url.split('/')[-1]
        path = root + img_name
        f = self.download_from_landsat(url)
        file_name = f.name
        print(file_name)
        shutil.copy(file_name, path)
        os.remove(file_name)

    def get(self, filename):
        files_tot = []
        page_token = None
        while True:
            response = self.drive_service.files().list(
                q="name='{0}' and '{1}' in parents".format(
                    filename, self.folder_id),
                spaces='drive',
                fields='nextPageToken, files(id, name)',
                pageToken=page_token
            ).execute()
            files_tot.append(file for file in response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
        file_list = [[file for file in files] for files in files_tot]
        if len(file_list[0]) > 1:
            print('Found files:')
            for file in file_list[0]:
                print('Name: {0}, ID: {1}'.format(
                    file.get('name'), file.get('id')
                ))
            raise DownloadError('Non-Unique Files')
        elif len(file_list[0]) == 0:
            raise DownloadError('No Matching Files')
        else:
            file_id = file_list[0][0].get('id')
            print('Pulled: {0}'.format(file_id))

        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = googleapiclient.http.MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("Download %d%%." % int(status.progress() * 100))
        
        print(fh)
        return fh.getvalue()


# download = Downloader()
# download.put(TEST_PATH)
# fh = download.get(TEST_FILENAME)
