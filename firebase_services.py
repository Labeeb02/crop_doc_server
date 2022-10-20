import pyrebase
import requests
import time
import os
import json


class FirebaseServices:

    def __init__(self):
        firebase_config = {}
        with open('resources/firebase/config.json') as json_file:
            firebase_config = json.load(json_file)

        self.firebase = pyrebase.initialize_app(firebase_config)

    # UPDATE THE SERVER IP ADDRESS
    def update_server_address(self):
        database = self.firebase.database()

        ip_address = requests.get('https://api.ipify.org').content.decode('utf8')
        print('Server IP address: {}'.format(ip_address))

        server_url = 'http://' + ip_address + ':5000/predict_disease'
        database.child('server_url').set(server_url)

        last_updated = int(round(time.time() * 1000))
        database.child('last_updated').set(last_updated)

    # DOWNLOAD IMAGE
    def download_image(self, image_URL):
        if not os.path.exists('images'):
            os.makedirs('images')

        file_path = 'images/' + 'd' + str(round(time.time() * 1000)) + '.jpg'

        # DOWNLOAD AND SAVE FILE AS JPG
        result = requests.get(image_URL, allow_redirects=True)
        open(file_path, 'wb').write(result.content)

        return file_path

    # UPLOAD PROCESSED IMAGE
    def upload_image(self, processed_image):
        file_name = 'processed_images/' + 'p' + str(round(time.time() * 1000))

        # UPLOAD IMAGE AND GET DOWNLOAD URL
        firebase_storage = self.firebase.storage()
        firebase_storage.child(file_name).put(processed_image)
        image_URL = firebase_storage.child(file_name).get_url(None)

        return image_URL
