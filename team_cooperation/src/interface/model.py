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

    def save(self, model_dir_path: Path):
        self.save_features(model_dir_path)
        self.save_model_interface(model_dir_path)

    def save_model_interface(self, model_dir_path: Path):
        model_interface_dir_path = model_dir_path / 'model_interface'
        os.makedirs(model_interface_dir_path, exist_ok=True)

        with open(model_interface_dir_path / 'model_interface_class.json', 'w+') as f:
            json.dump({'__type': self.model_interface.__class__.__name__}, f)

        self.model_interface.save(model_interface_dir_path)

    def save_features(self, dir_path: Path):
        features_dir_path = dir_path / 'features'
        os.makedirs(features_dir_path, exist_ok=True)

        for i, feature in enumerate(self.features):  # Use i to keep feature order and avoid collisions.
            with open(features_dir_path / f'{i}-{feature.__class__.__name__}.json', 'w+') as f:
                json.dump(feature.serialize(), f)

    @classmethod
    def load(cls, model_dir_path: Path):
        ret_model = cls()
        ret_model.features = cls.load_features(model_dir_path)
        ret_model.model_interface = Model.load_model_interface(model_dir_path)

        return ret_model

    @classmethod
    def load_model_interface(cls, model_dir_path):
        return ModelInterface.load(model_dir_path / 'model_interface')

    @classmethod
    def load_features(cls, model_dir_path: Path):
        features = deque()  # Use deque to avoid reallocation of list.
        for path in sorted((model_dir_path / 'features').glob('*.json'),  # Get all feature files
                           key=lambda path_: int(path_.name.split('-')[0])):  # Sort by the same i from `save_features`.
            with open(path, 'r') as f:
                features.append(Feature.deserialize(json.load(f)))
        return list(features)
