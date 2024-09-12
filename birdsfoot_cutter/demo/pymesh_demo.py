import pymeshlab as pml

ms = pml.MeshSet()

bf_filepath = r"mesh01.stl"
# Load a mesh from a given filepath
ms.load_new_mesh(bf_filepath)
# Some random demo filter
ms.generate_convex_hull()
# Saving the current mesh
ms.save_current_mesh('convex_hull.ply')

# Printing every filter
pml.print_filter_list()
# Printing the parameters for a random given filter
pml.print_filter_parameter_list('generate_surface_reconstruction_screened_poisson')