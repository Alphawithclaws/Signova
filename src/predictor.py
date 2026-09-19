import tensorflow as tf
import numpy as np
import cv2
import os


class Predictor:

    def __init__(self):

        self.model = tf.keras.models.load_model(
            "model/sign_language.keras"
        )

        self.labels = sorted([
            folder
            for folder in os.listdir("dataset")
            if os.path.isdir(os.path.join("dataset", folder))
        ])

        if self.model.output_shape[-1] != len(self.labels):

            raise ValueError(
                f"Model has {self.model.output_shape[-1]} outputs but dataset has {len(self.labels)} folders.\n"
                "Run train.py again."
            )

        print("Loaded labels:")
        print(self.labels)

    def predict(self, crop):

        if crop is None:
            return "-", 0.0

        if crop.size == 0:
            return "-", 0.0

        image = cv2.resize(crop, (224,224))

        image = image.astype(np.float32)

        image /= 255.0

        image = np.expand_dims(image, axis=0)

        prediction = self.model.predict(
            image,
            verbose=0
        )[0]

        index = int(np.argmax(prediction))

        confidence = float(prediction[index])

        if confidence < 0.60:
            return "-", confidence

        return self.labels[index], confidence