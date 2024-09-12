import pyvista as pv
import numpy as np

def simple_box_1():
    # Load a mesh
    #file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\Part Clamp V1.stl'
    file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\mesh1.stl'
    # Load a mesh
    mesh = pv.read(file)
    
    # Create a plotter object
    plotter = pv.Plotter()
    
    # Add the mesh to the plotter
    plotter.add_mesh(mesh, show_edges=True)
    
    # Store selected points
    selected_points = []
    
    # Callback function for box selection
    def box_picker(bounds):
        global selected_points
        # Extract vertices within the box bounds
        selected_ids = mesh.extract_points_within_bounds(bounds)
        selected_vertices = mesh.points[selected_ids]
        
        # Add selected vertices to the global list
        selected_points.append(selected_vertices)
        
        # Optionally, add a marker to visualize selected points
        plotter.add_mesh(pv.PolyData(selected_vertices), color='red', point_size=5)
    
    # Enable box picking
    plotter.add_box_widget(callback=box_picker)
    
    # Start the interactive plot
    plotter.show()
    
    # Convert selected points to a numpy array
    selected_points = np.vstack(selected_points)
    print(selected_points)


def box_2():
    # Load a mesh
    #file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\Part Clamp V1.stl'
    file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\mesh1.stl'
    # Load a mesh
    mesh = pv.read(file)
    
    # Create a plotter object
    plotter = pv.Plotter()
    
    # Add the mesh to the plotter
    plotter.add_mesh(mesh, show_edges=True)
    
    # Store selected points
    selected_points = []
    
    # Callback function for plane widget
    def plane_picker(normal, origin):
        global selected_points
        
        # Create a plane from the normal and origin
        plane = pv.Plane(center=origin, direction=normal, i_size=100, j_size=100)
        
        # Extract points within a certain distance from the plane
        distance = mesh.point_normals * normal - (mesh.points - origin).dot(normal)
        selected_ids = np.abs(distance) < 1.0  # Adjust tolerance as needed
        
        selected_vertices = mesh.points[selected_ids]
        
        # Add selected vertices to the global list
        selected_points.append(selected_vertices)
        
        # Optionally, add a marker to visualize selected points
        plotter.add_mesh(pv.PolyData(selected_vertices), color='red', point_size=5)
    
    # Add a plane widget to the plotter
    plotter.add_plane_widget(callback=plane_picker, normal_rotation=True)
    
    # Function to finalize and accept the selection
    def accept_selection():
        global selected_points
        selected_points = np.vstack(selected_points)
        print(selected_points)
        plotter.close()  # Close the plotter to end the selection process
    
    # Add a button to accept the selection
    plotter.add_text("Press 'a' to accept the selection", position='upper_left', font_size=12)
    plotter.add_key_event('a', accept_selection)
    
    # Start the interactive plot
    plotter.show()
    
    # Convert selected points to a numpy array
    print("Selected points coordinates:")
    print(selected_points)
    
def accept_box_3():
    # Load a mesh
    #file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\Part Clamp V1.stl'
    file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\mesh1.stl'
    
    # Load a mesh
    mesh = pv.read(file)
    
    # Create a plotter object
    plotter = pv.Plotter()
    
    # Add the mesh to the plotter
    plotter.add_mesh(mesh, show_edges=True)
    
    # Initialize selected points list
    selected_points = []
    
    # Callback function for box selection
    def box_picker(bounds):
        
        # Ensure bounds are unpacked properly
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]
        
        # Filter points within the box bounds
        selected_ids = np.where(
            (mesh.points[:, 0] >= x_min) & (mesh.points[:, 0] <= x_max) &
            (mesh.points[:, 1] >= y_min) & (mesh.points[:, 1] <= y_max) &
            (mesh.points[:, 2] >= z_min) & (mesh.points[:, 2] <= z_max)
        )[0]
        
        selected_vertices = mesh.points[selected_ids]
        
        # Add selected vertices to the global list
        selected_points.append(selected_vertices)
        
        # Optionally, add a marker to visualize selected points
        plotter.add_mesh(pv.PolyData(selected_vertices), color='red', point_size=5)
        
    # Enable box picking
    plotter.add_box_widget(callback=box_picker)
    
    # Function to finalize and accept the selection
    def accept_selection():
        #global selected_points
        
        if selected_points:
            selected_points_combined = np.vstack(selected_points)
            print("Selected points coordinates:")
            print(selected_points_combined)
        else:
            print("No points selected.")
        
        plotter.close()  # Close the plotter to end the selection process
    
    # Add instructions and bind the accept selection function to the 'a' key
    plotter.add_text("Press 'a' to accept the selection", position='upper_left', font_size=12)
    plotter.add_key_event('a', accept_selection)
    
    # Start the interactive plot
    plotter.show()

if __name__ == '__main__':
    #simple_box_1()
    #box_2()
    accept_box_3()

