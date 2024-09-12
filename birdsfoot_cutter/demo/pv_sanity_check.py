import numpy as np
import pyvista as pv
from stl import mesh

def numpy_stl():
    # Example vertices and faces
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [0, 3, 7],
        [0, 7, 4]
    ])
    
    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]
    
    # Write the mesh to file "cube.stl"
    cube.save('cube_np.stl')
    
    print("STL file has been saved as cube.stl")
    
def pyvista_stl():
    
    # Example vertices and faces
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [0, 3, 7],
        [0, 7, 4]
    ])
    
    # Convert faces to the format required by PyVista (with face size at the start)
    faces_with_size = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    
    # Create a PyVista mesh
    mesh = pv.PolyData(vertices, faces_with_size)
    
    # Save the mesh to file "cube.stl"
    mesh.save('cube_pv.stl')
    
    print("STL file has been saved as cube.stl")

if __name__ == '__main__':
    pyvista_stl()
    numpy_stl()
