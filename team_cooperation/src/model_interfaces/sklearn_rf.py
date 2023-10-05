from pathlib import Path

from team_cooperation.src.interface.bases import ModelInterface
from joblib import dump, load


class SKLRFModelInterface(ModelInterface):
    def __init__(self):
        self.model = None

    def initialize(self, model_parameters):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier

        forest = RandomForestClassifier(**model_parameters)
        self.model = MultiOutputClassifier(forest, n_jobs=2)

    def train(self, x, y, w=None):
        from sklearn.preprocessing import MultiLabelBinarizer
        y = MultiLabelBinarizer().fit_transform(y)
        return self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, model_interface_dir_path: Path):
        dump(self.model, model_interface_dir_path / 'model.pkl')

    def load(self, model_interface_dir_path: Path):
        self.model = load(model_interface_dir_path / 'model.pkl')
