from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from interface.model import Model
from models import best_model_so_far


if __name__ == '__main__':
    x, y = load_iris(as_frame=True, return_X_y=True)
    y = y.apply(str)

    best_model_so_far.train(x, y)

    with TemporaryDirectory() as dir_path:
        dir_path = Path(dir_path)
        best_model_so_far.save(dir_path)

        model = Model.load(dir_path)

    # We use this model after it was trained, saved to disk and loaded back again.
    print(accuracy_score(y, model.predict(x)))
