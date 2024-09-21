from ..source.utilities import get_angle_between_two_vectors



def test_angle_between_two_vectors():
    vector_one = (0.0, 0.0, 1.0)
    vector_two = (1.0, 0.0, 0.0)
    assert(vector_one, vector_two) == 90.0

    vector_three = (5.5,-6.5,2.0)
    vector_four = (-0.25, .64, .78)
    assert(vector_three,vector_four) == 115.926256
