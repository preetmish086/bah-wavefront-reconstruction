import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from config import (
    SEQUENCE_LENGTH,
    NUM_ZERNIKE_MODES
)


class ZernikeAR:

    def __init__(self):

        self.model = LinearRegression()

    def train(self, data):

        X = []
        y = []

        for i in range(
            len(data) - SEQUENCE_LENGTH
        ):

            X.append(
                data[
                    i:i+SEQUENCE_LENGTH
                ].flatten()
            )

            y.append(
                data[
                    i+SEQUENCE_LENGTH
                ]
            )

        X = np.array(X)

        y = np.array(y)

        self.model.fit(X, y)

    def predict(self, sequence):

        sequence = sequence.flatten()

        prediction = self.model.predict(
            [sequence]
        )

        return prediction[0]