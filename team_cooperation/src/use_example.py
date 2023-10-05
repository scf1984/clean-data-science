from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.datasets import load_iris

from interface.model import Model
from models import best_model_so_far

x, y = load_iris(as_frame=True, return_X_y=True)
y = y.apply(str)
best_model_so_far.train(x, y)

with TemporaryDirectory() as dir_path:
    dir_path = Path(dir_path)
    best_model_so_far.save(dir_path)

    model = Model.load(dir_path)

print(model.predict(x))
