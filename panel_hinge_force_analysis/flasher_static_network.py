"""
Code written by Trevor K. Carter, started on August 21st, 2024

"""
from json import load

from skspatial import objects as skobj
import skspatial.transformation as sktrf
from panel_sectioning import generate_orthonormal_basis

import networkx as nx
from scipy.optimize import minimize

import numpy as np
import pandas as pd

# Optional: Visualize the relative forces in each spring (e.g., using a bar graph)
import matplotlib.pyplot as plt


def main():
	labeled_flasher = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\warped flasher\mesh_cut_labeled_panelsREAL.json'
	names, centers = get_2d_flasher_centers(labeled_flasher, skip_name='ref')
	# A dictionary containing only panel names and their centers in 2d
	node_data = dict(zip(names, centers))
	
	# Define the node_data (panels)
	edge_nodes = [('center', 'a1'), ('a1', 'a2'), ('a2', 'a3'), ('a2', 'a5'), ('a3', 'a4'), ('a5', 'a4'), ('a1', 'e5'), ('a3', 'e5'),
				 ('center', 'b1'), ('b1', 'b2'), ('b2', 'b3'), ('b2', 'b5'), ('b3', 'b4'), ('b5', 'b4'), ('b1', 'a5'), ('b3', 'a5'),
				 ('center', 'c1'), ('c1', 'c2'), ('c2', 'c3'), ('c2', 'c5'), ('c3', 'c4'), ('c5', 'c4'), ('c1', 'b5'), ('c3', 'b5'),
				 ('center', 'd1'), ('d1', 'd2'), ('d2', 'd3'), ('d2', 'd5'), ('d3', 'd4'), ('d5', 'd4'), ('d1', 'c5'), ('d3', 'c5'),
				 ('center', 'e1'), ('e1', 'e2'), ('e2', 'e3'), ('e2', 'e5'), ('e3', 'e4'), ('e5', 'e4'), ('e1', 'd5'), ('e3', 'd5')
				 ]
	# The x and y attributes will each contain dictionaries where keys are variable names, like
	# <edge1_name>-<edge1_name>, and values are the coefficients for that variable for that equation
	
	# Initialize a NetworkX graph
	G = nx.Graph()
	
	for new_node, new_pos in node_data.items():
		G.add_node(new_node, pos=new_pos)
	
	G.add_edges_from(edge_nodes)

	edge_forces = []
	for this_node, this_data in G.nodes(data=True):
		print(this_node, this_data)
		# In building these formulas, assume that all forces are in the direction of +x, +y. Each force variable (key) will have a corresponding coefficient (cos or sin) of an angle
		x_components = {}
		y_components = {}
		for _, end_node in G.edges([this_node]):
			if end_node+'-'+this_node in edge_forces:
				edge_var = end_node+'-'+this_node
			else:
				edge_var = this_node+'-'+end_node
				edge_forces.append(edge_var)
			vec_x, vec_y = node_data[end_node] - node_data[this_node]
			node_to_end_angle = np.arctan2(vec_y, vec_x)
			x_components[edge_var] = np.cos(node_to_end_angle)
			y_components[edge_var] = np.sin(node_to_end_angle)
		
		G.add_node(this_node, x=x_components, y=y_components)
		print(this_node, this_data)
	A_empty = np.zeros((2*len(G.nodes), len(edge_forces)))
	A_indices = []
	for some_node in G.nodes:
		A_indices.append(some_node+'_x')
		A_indices.append(some_node+'_y')
	A = pd.DataFrame(data=A_empty, columns=edge_forces, index=A_indices)
	print(A)

	for next_node, equations in G.nodes(data=True):
		print(next_node)
		print(equations)
		for x_force, coeff in equations['x'].items():
			print(x_force)
			print(coeff)
			A.loc[next_node+'_x', x_force] = coeff
	print(A)
	

	external_force_nodes = ['a4', 'b4', 'c4', 'd4', 'e4']
	force_source_pos = G.nodes['center']['pos']
	external_force = 0.35
	b = pd.Series(data=0, index=A_indices, dtype=np.float64)
	print(b)
	for ext_node in external_force_nodes:
			ext_vec_x, ext_vec_y = G.nodes[ext_node]['pos'] - force_source_pos
			ext_force_angle = np.arctan2(ext_vec_y, ext_vec_x)
			print(ext_force_angle)
			b[ext_node+'_x'] = -external_force * np.cos(ext_force_angle)
			b[ext_node+'_y'] = -external_force * np.sin(ext_force_angle)
	print(b)
	
	print(A.shape)
	#x = np.linalg.solve(A.values, b)
	
	x = pd.Series(data=A.values.T @ b, index=edge_forces)
	print(x)
		
	
	
	

# This takes a .json of labeled panels, extracts their centers, projects them onto their overall
# plane of best fit, and returns normalized 2d coordinates of all panel centers with their original labels
def get_2d_flasher_centers(filepath, skip_name=None):
	with open(filepath, 'r') as json_file:
		flasher_dict = load(json_file)
	panel_names_list = []
	panel_centers_list = []
	for name, data in flasher_dict.items():
		if skip_name is not None:
			if name == skip_name:
				continue
		panel_names_list.append(name)
		panel_centers_list.append(data['center'])
	
	panel_names = np.array(panel_names_list)
	panel_centers = np.array(panel_centers_list)
	
	best_plane = skobj.Plane.best_fit(panel_centers)
	
	b1, b2 = generate_orthonormal_basis(best_plane)
	
	panel_centers_trf = sktrf.transform_coordinates(panel_centers, (0, 0, 0), (b1, b2, best_plane.normal.unit()))
	panel_centers_2d = np.array(panel_centers_trf[:, 0:2])
	
	return panel_names, panel_centers_2d


if __name__ == '__main__':
	main()
