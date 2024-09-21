import numpy as np

def get_angle_between_two_vectors(vector_one, vector_two) -> float:
    dot_product = np.dot(vector_one, vector_two)
    
    angle_in_radians = np.arcos(dot_product)

    angle_in_degrees = np.rad2deg(angle_in_radians)

    return angle_in_degrees