from django.apps import AppConfig
import tensorflow as tf
import joblib
import pickle
import os
from tensorflow import keras
from tensorflow.keras.models import load_model

class YourAppConfig(AppConfig):
    name = 'app'

    def ready(self):
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        self.classification_model = load_model(
            os.path.join(models_dir, 'model2.h5'),
            custom_objects={'rgb_to_grayscale': rgb_to_grayscale}
        )
        
        self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        
        with open(os.path.join(models_dir, 'regression_model.pkl'), 'rb') as f:
            self.regression_model = pickle.load(f)


def rgb_to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)
