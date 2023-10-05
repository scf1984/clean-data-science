from dataclasses import dataclass

from team_cooperation.src.interface.bases import Feature


class SepalDiagonalFeature(Feature):
    def add_feature(self, data):
        data['sepal diagonal (cm)'] = (data['sepal length (cm)'] ** 2 + data['sepal length (cm)'] ** 2) ** 0.5


@dataclass
class AdditionFeature(Feature):
    col_a: str
    col_b: str
    output_col: str

    def add_feature(self, data):
        data[self.output_col] = data[self.col_a] + data[self.col_b]
