from Panel import Panel
from typing import List
from utilities import *                 # the * imports all of this functions from utilites


class Array:
    def __init__(self) -> None:
        self.list_of_panels: List[Panel] = [] # Initialize an empty list of type Panel
        
    
    def add_panel(self, panel: Panel):
        self.list_of_panels.append(panel) # Add a Panel instance to the list of Panels

    def compare_two_panels(self, first_panel_to_be_compared: int, second_panel_to_be_compared: int) -> float: # TODO make this do something with two panels
        
        panel_one = self.list_of_panels[first_panel_to_be_compared]
        panel_two = self.list_of_panels[second_panel_to_be_compared]

        vector_one = panel_one.get_normal_vector
        vector_two = panel_two.get_normal_vector

        angle_between_panels = get_angle_between_two_vectors(vector_one, vector_two)

        return angle_between_panels
