from typing import Any


class Panel:
    def __init__(self, vector: tuple, centroid: tuple, name: str) -> None:
        self.normal_unit_vector = vector
        self.panel_centroid = centroid
        self.panel_name = name

    def __getattribute__(self, name: str) -> Any:
        pass

    def get_normal_vector(self) -> tuple:
        return self.normal_unit_vector