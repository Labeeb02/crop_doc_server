from firebase_services import FirebaseServices
import os


class DiseasePredictor:

    def __init__(self, image_url):
        self.image_url = image_url

    def predict_disease(self, model, selected_crop):
        firebase = FirebaseServices()

        # DOWNLOAD IMAGE FROM FIREBASE STORAGE
        file_path = firebase.download_image(self.image_url)

        ##print("FILE PATH: ", file_path)
        # PREDICT DISEASE
        prediction = model.classify(file_path, selected_crop)
        os.remove(file_path)

        # RETURN IMAGE LINK AND PREDICTED RESULT
        
        return {
            "status": "SUCCESS",
            "prediction": prediction,
            "image_URL": self.image_url
        }
