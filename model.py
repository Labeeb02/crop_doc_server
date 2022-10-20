import numpy as np  # version 1.19.5
from skimage.feature import hog  # verison 0.18.3
from tensorflow import keras  # version 2.7.0
from keras import backend as k
from tensorflow.keras.preprocessing.image import load_img

from PIL import Image
import onnxruntime as ort
from torchvision import transforms

IDX_TO_DISEASES_TOMATO = {
    0: "disease 09",
    1: "disease 08",
    2: "disease 07",
    3: "disease 06",
    4: "disease 05",
    5: "disease 04",
    6: "disease 03",
    7: "disease 01",
    8: "disease 02",
    9: "disease 00"
}

IDX_TO_DISEASES_OTHER_PLANTS = {
    0: "disease 10",
    1: "plant 13",
    2: "disease 11",
    3: "plant 12",
    4: "disease 12",
    5: "plant 11",
    6: "plant 10",
    7: "disease 13",
    8: "disease 14",
    9: "disease 15",
    10: "plant 08",
    11: "disease 16",
    12: "disease 17",
    13: "plant 06",
    14: "plant 05",
    15: "plant 04",
    16: "plant 03",
    17: "disease 08",
    18: "disease 05",
    19: "plant 01",
    20: "disease 09",
    21: "disease 07",
    22: "disease 02",
    23: "disease 01",
    24: "disease 06",
    25: "plant 02",
    26: "disease 18"
}

IDX_TO_PLANT = {
    "plant 01": "Tomato",
    "plant 02": "Grape",
    "plant 03": "Strawberry",
    "plant 04": "Squash Powdery Mildew",
    "plant 05": "Soyabean",
    "plant 06": "Raspberry",
    "plant 07": "Potato",
    "plant 08": "Peach",
    "plant 09": "Corn",
    "plant 10": "Cherry",
    "plant 11": "Blueberry",
    "plant 12": "Bell Pepper",
    "plant 13": "Apple"
}

IDX_TO_CLASS_MILLET_EAR = {
    0: "disease 22",
    1: "disease 23"
}

IDX_TO_CLASS_MILLET_LEAF = {
    0: "disease 24",
    1: "disease 25"
}

IDX_TO_CLASS_RICE = {
    0: "disease 19",
    1: "disease 20",
    2: "disease 21"
}


def precision(y_true, y_pred):  # taken from old keras source code

    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    _precision = true_positives / (predicted_positives + k.epsilon())

    return _precision


def recall(y_true, y_pred):  # taken from old keras source code

    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    _recall = true_positives / (possible_positives + k.epsilon())

    return _recall


def predict_hdf5(img_size, image_path, model, shape, idx_to_class):
    img = load_img(image_path, target_size=img_size)
    x = np.zeros((1,) + img_size + (3,), dtype="float32")
    x[0] = img

    hog_in = np.array(img)
    fd, hog_image = hog(hog_in, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                        multichannel=True)
    x1 = np.zeros((1,) + img_size + (3,), dtype="float32")
    x2 = np.zeros((1,) + shape, dtype="float32")
    x1[0] = img
    x2[0] = fd
    prediction = model.predict([x2, x1])
    val = np.argmax(prediction, axis=1)
    output = idx_to_class[int(val)]

    return output


class Model:

    def __init__(self):
        self.img_size = (256, 256)
        self.tomato_model = keras.models.load_model('resources/models/tomato/tomato.hdf5',
                                                    custom_objects={"recall": recall, "precision": precision})
        self.millet_ear_model = keras.models.load_model('resources/models/millet/millet_ear.hdf5',
                                                        custom_objects={"recall": recall, "precision": precision})
        self.millet_leaf_model = keras.models.load_model('resources/models/millet/millet_leaf.hdf5',
                                                         custom_objects={"recall": recall, "precision": precision})
        self.rice_model = keras.models.load_model('resources/models/rice/rice.hdf5',
                                                  custom_objects={"recall": recall, "precision": precision})
        self.ort_sess = ort.InferenceSession('resources/models/other crops/other_crops.onnx')

    def classify(self, image_path, selected_crop):

        if selected_crop == 'tomato':
            # TOMATO MODEL
            shape = (34596,)
            return predict_hdf5(self.img_size, image_path, self.tomato_model, shape, IDX_TO_DISEASES_TOMATO)

        elif selected_crop == 'millet_ear':
            shape = (36288,)
            img_size = (200, 350)
            return predict_hdf5(img_size, image_path, self.millet_ear_model, shape, IDX_TO_CLASS_MILLET_EAR)

        elif selected_crop == 'millet_leaf':
            shape = (36288,)
            img_size = (200, 350)
            return predict_hdf5(img_size, image_path, self.millet_leaf_model, shape, IDX_TO_CLASS_MILLET_LEAF)

        elif selected_crop == 'rice':
            shape = (63936,)
            img_size = (200, 600)
            return predict_hdf5(img_size, image_path, self.rice_model, shape, IDX_TO_CLASS_RICE)

        elif selected_crop == 'other_crops':
            # OTHER PLANTS MODEL
            data_transforms = [transforms.Compose([transforms.Resize((256, 256)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.469, 0.536, 0.369],
                                                                        std=[0.260, 0.244, 0.282])]),

                               transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.469, 0.536, 0.369],
                                                                        std=[0.260, 0.244, 0.282])])]

            image = Image.open(image_path).convert('RGB')

            im1 = data_transforms[0](image).unsqueeze(0)
            im2 = data_transforms[1](image).unsqueeze(0)

            output = self.ort_sess.run(None, {'input1': im1.numpy(), 'input2': im2.numpy()})
            y_pred = np.argmax(output)
            other_plants_output = IDX_TO_DISEASES_OTHER_PLANTS[y_pred]

            return other_plants_output

        else:
            return 'invalid input encountered'


# if __name__ == "__man__":
#     _model = Model()
#     print('Tomato -', _model.classify('images/Tomato/e1693a70-df15-4743-829a-67f593d1e381___PSU_CG 2110.JPG', 'tomato'))
#     print('Millet Ear Sick -', _model.classify('images/Millet/Ear/ear_diseased/ed_1_0.jpg', 'millet_ear'))
#     print('Millet Ear Healthy -', _model.classify('images/Millet/Ear/ear_healthy/eh_1_0.jpg', 'millet_ear'))
#     print('Millet Leaf Sick -', _model.classify('images/Millet/Leaf/leaf_diseased/ld_1_0.jpg', 'millet_leaf'))
#     print('Millet Leaf Healthy -', _model.classify('images/Millet/Leaf/leaf_healthy/lh_7_3.jpg', 'millet_leaf'))
#     print('Rice -', _model.classify('images/Rice/Bacterial leaf blight/DSC_0365.jpg', 'rice'))
#     print('Rice -', _model.classify('images/Rice/Brown spot/DSC_0100.jpg', 'rice'))
#     print('Rice -', _model.classify('images/Rice/Leaf smut/DSC_0293.jpg', 'rice'))
#     print('Other Crops -', _model.classify('images/Other Crops/apple_rust_leaf_1.jpg', 'other_crops'))


# import os

# assign directory
# rice_directory = 'images/Rice/Brown spot'
# millet_leaf_directory = 'images/Millet/Leaf/leaf_diseased'
# _model = Model()
# iterate over files in
# that directory
# for filename in os.scandir(millet_leaf_directory):
#     if filename.is_file():
#         print('Rice -', _model.classify(filename.path, 'rice'))
