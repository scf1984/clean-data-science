from team_cooperation.src.base import Transformer


class SepalDiagonalTransformer(Transformer):
    def transform(self, data):
        data['SepalDiagonal'] = (data.SepalLengthCm**2 + data.SepalWidthCm**2) ** 0.5

