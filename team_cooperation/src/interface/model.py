import json
import os
from collections import deque
from pathlib import Path

from team_cooperation.src.interface.bases import Feature, ModelInterface


class Model:
    def __init__(self, features: list[Feature] = tuple(), model_interface: ModelInterface = None,
                 model_parameters: dict = None):
        model_parameters = model_parameters or {}

        self.features = features
        self.model_interface = model_interface
        self.model_parameters = model_parameters

        if model_interface:
            model_interface.initialize(model_parameters)

    def add_features(self, x):
        for feature in self.features:
            feature.add_feature(x)

    def train(self, x, y, train_parameters=None):
        train_parameters = train_parameters or {}
        self.add_features(x)
        self.model_interface.train(x, y, train_parameters)

    def predict(self, x):
        self.add_features(x)
        return self.model_interface.predict(x)

    def save(self, dir_path: Path):
        features_dir_path = dir_path / 'features'
        os.makedirs(features_dir_path, exist_ok=True)
        for i, feature in enumerate(self.features):
            with open(features_dir_path / f'{i}-{feature.__class__.__name__}.json', 'w+') as f:
                json.dump(feature.serialize(), f)

        model_interface_dir_path = dir_path / 'model_interface'
        os.makedirs(model_interface_dir_path, exist_ok=True)

        with open(model_interface_dir_path / 'model_interface_class.json', 'w+') as f:
            json.dump({'__type': self.model_interface.__class__.__name__}, f)

        self.model_interface.save(model_interface_dir_path)

    @classmethod
    def load(cls, model_dir_path: Path):
        ret_model = cls()

        features = deque()
        for path in sorted((model_dir_path / 'features').glob('*.json'), key=lambda x: int(x.name.split('-')[0])):
            with open(path, 'r') as f:
                features.append(Feature.deserialize(json.load(f)))
        ret_model.features = list(features)

        ret_model.model_interface = ModelInterface.load(model_dir_path / 'model_interface')

        return ret_model
