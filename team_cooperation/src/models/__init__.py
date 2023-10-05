from team_cooperation.src.interface.model import Model
from team_cooperation.src.features import petal_sum, sepal_diagonal
from team_cooperation.src.model_interfaces.sklearn_rf import SKLRFModelInterface

best_model_so_far = Model([petal_sum, sepal_diagonal], SKLRFModelInterface(), {})
