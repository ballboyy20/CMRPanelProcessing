from Panel import Panel
from typing import List
class Array:
    def __init__(self) -> None:
        self.list_of_panels: List[Panel] = [] # Initialize an empty list of type Panel
        
    
    def add_panel(self, panel: Panel):
        self.list_of_panels.append(panel) # Add a Panel instance to the list of Panels