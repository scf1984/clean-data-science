from team_cooperation.src.base import Model
from team_cooperation.src.model_interfaces.sklearn_rf import SKLRFModelInterface
from team_cooperation.src.transformers.my_transformers import SepalDiagonalTransformer

best_model_so_far = Model([SepalDiagonalTransformer()], SKLRFModelInterface(), {})
