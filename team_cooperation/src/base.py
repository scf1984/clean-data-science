from abc import ABC, abstractmethod
from functools import reduce


class Transformer(ABC):
    @abstractmethod
    def transform(self, data):
        pass


class ModelInterface(ABC):
    @abstractmethod
    def initialize(self, model_parameters):
        """Initialize and return a model object."""
        pass

    @abstractmethod
    def train(self, x, y, train_parameters):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class Model(ABC):
    def __init__(self, transformers: list[Transformer], model_interface: ModelInterface, model_parameters: dict):
        self.transformers = transformers
        self. model_interface = model_interface
        self.model_parameters = model_parameters

        model_interface.initialize(model_parameters)

    def transform(self, x):
        return reduce(lambda data, transformer: transformer.transform(data), self.transformers, x)

    def train(self, x, y, train_parameters=None, transform=False):
        train_parameters = train_parameters or {}
        if transform:
            x = self.transform(x)
        self.model_interface.train(x, y, train_parameters)

    def predict(self, x, transform=False):
        if transform:
            x = self.transform(x)
        return self.model_interface.predict(x)
