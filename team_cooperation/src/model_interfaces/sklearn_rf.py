from pathlib import Path

import numpy as np

from team_cooperation.src.interface.bases import ModelInterface
from joblib import dump, load


class SKLRFModelInterface(ModelInterface):
    def __init__(self):
        self.model = None
        self.binarizer = None

    def initialize(self, model_parameters: dict):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier

        forest = RandomForestClassifier(**model_parameters)
        self.model = MultiOutputClassifier(forest, n_jobs=2)

    def train(self, x, y, w=None):
        from sklearn.preprocessing import LabelBinarizer
        self.binarizer = LabelBinarizer()
        y = self.binarizer.fit_transform(y)
        return self.model.fit(x, y)

    def predict(self, x):
        return self.binarizer.inverse_transform(self.model.predict(x))

    def save(self, model_interface_dir_path: Path):
        dump(self.model, model_interface_dir_path / 'model.pkl')
        dump(self.binarizer, model_interface_dir_path / 'binarizer.pkl')

    def load(self, model_interface_dir_path: Path):
        self.model = load(model_interface_dir_path / 'model.pkl')
        self.binarizer = load(model_interface_dir_path / 'binarizer.pkl')
