from team_cooperation.src.base import ModelInterface


class SKLRFModelInterface(ModelInterface):
    def __init__(self):
        self.model = None

    def initialize(self, model_parameters):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier

        forest = RandomForestClassifier(**model_parameters)
        self.model = MultiOutputClassifier(forest, n_jobs=2)

    def train(self, model, x, y, w=None):
        return model.fit(x, y, w)

    def predict(self, x):
        pass
