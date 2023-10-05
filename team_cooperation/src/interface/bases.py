import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DataClassSerializer:
    def serialize(self):
        return {'__type': self.__class__.__name__, **asdict(self)}

    @classmethod
    def deserialize(cls, params: dict):
        class_name = params.pop('__type')
        return_class = next(filter(lambda x: x.__name__ == class_name, cls.__subclasses__()))
        return return_class(**params)


class Feature(ABC, DataClassSerializer):
    @abstractmethod
    def add_feature(self, data):
        ...


class ModelInterface(ABC):
    @abstractmethod
    def initialize(self, model_parameters: dict):
        ...

    @abstractmethod
    def train(self, x, y, train_parameters: dict):
        ...

    @abstractmethod
    def predict(self, x):
        ...

    @abstractmethod
    def save(self, model_interface_dir_path: Path):
        ...

    @classmethod
    def load(cls, model_interface_dir_path: Path):
        with open(model_interface_dir_path / 'model_interface_class.json', 'r') as f:
            model_interface_class_name = json.load(f).pop('__type')

        model_interface_class = next(filter(lambda x: x.__name__ == model_interface_class_name, cls.__subclasses__()))
        model_interface = model_interface_class()

        model_interface.load(model_interface_dir_path)
        return model_interface
