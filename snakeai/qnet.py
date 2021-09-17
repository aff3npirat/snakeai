from tensorflow import keras
from tensorflow.keras import layers

from snakeai.models import AdaptiveEps, QNetTrainer


class QNet:

    def __init__(self, input_size, hidden_size):
        self.model = keras.Sequential(
            [
                layers.InputLayer(input_shape=(input_size,)),
                layers.Dense(hidden_size, activation="relu", name="layer2"),
                layers.Dense(4, name="layer_out"),
            ]
        )

    def __getitem__(self, state):
        return self.model(state)

    def __setitem__(self, key, value):
        pass


class AdaptiveQnetAgent:

    def __init__(self):
        self.Q = QNet(11, 256)
        self.model = AdaptiveEps(0.5, 10, 7)
        self.trainer = QNetTrainer(self.Q.model, 1.0, 0.1)
