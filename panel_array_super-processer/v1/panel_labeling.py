"""
Code written by Trevor Kent Carter, August 2024

The purpose of this module is to provide an interface for manually assigning labels to panel arrays.
It visualizes the existing names (in their respective positions), allowing the user to apply whatever
naming convention they so choose to use for the panel arrays. This is the foundation of the auto-labeler
code, which will take these manually labeled templates and extend their names to other panel arrays.

"""

from json import load as json_load
from json import dumps as json_dumps
import numpy as np

import skspatial.objects as skobj
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


def main():
	panel_data_path = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\warped flasher\mesh_cut_panelData.json'
	labeled_panel_data = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\warped flasher\mesh_cut_labeled_panels.json'

	with open(panel_data_path, 'r') as panel_data_obj:
		panel_data = json_load(panel_data_obj)
	overwrite_labels(panel_data, labeled_panel_data)


def overwrite_labels(panel_data_dict, save_as=None):
	"""
	This function takes an existing collection of nested dictionaries representing panel data
	and uses a graphical interface to let the user reassign each panel's name. The new collection
	is returned (and saved as a .json file if a full filepath was passed as a parameter).

	:param panel_data_dict: a collection of nested Python dictionaries and lists representing panel data as follows:
		{
			"<cluster_label>": {
				"center": [
					-437.89662057836836,
					221.23474330596954,
					565.891259105087
				],
				"normal": [
					-0.10980522428272566,
					-0.993527439362412,
					-0.029086765963050078
				]
			},
			...
		}
	:param save_as: None OR a complete filepath (including the .json ending) to save the return value as a .json file

	:return: panel_data_dict: the modified version of the original dictionary.
	"""
	center_list = []
	norm_list = []
	# Making seperate lists of all the centers, then all the vectors
	for center_norm in panel_data_dict.values():
		center_list.append(center_norm["center"])
		norm_list.append(center_norm["normal"])
	# Converting the lists to NumPy arrays
	centers = np.array(center_list)
	normals = np.array(norm_list)

	# Making a list to keep track of the sizes of each dot representing panels
	default_dot_size = 50
	big_dot_size = 250
	size_key = [default_dot_size] * centers.shape[0]
	# Making a list of the previous panel names
	panel_keys = list(panel_data_dict.keys())
	# Making a list to store the new panel names. Names cannot be repeated, so
	# a negative numbering scheme (unlikely to be manually assigned) was chosen.
	key_conversions = list(range(-len(panel_keys), 0))
	# Loop through the indices of the panels in this array, letting the GraphInputWindow panel take user input
	pos = 0
	while pos < len(panel_keys):
		fig = display_graph(centers, normals, pos, size_key, text_key=key_conversions)
		window = GraphInputWindow(fig)
		this_new_label = window.run()
		# If a duplicate label is given, the program will repeat this iteration until a unique names is given
		if this_new_label in key_conversions:
			print("Label", this_new_label, "already exists in this dictionary. Choose a new name.")
			continue
		key_conversions[pos] = this_new_label
		size_key[pos] = big_dot_size
		pos += 1
	# Remove the old panel dictionary elements, adding new newly named elements with the same data as the originals
	for old_key, new_key in zip(panel_keys, key_conversions):
		panel_data_dict[new_key] = panel_data_dict.pop(old_key)
	
	# Save the dictionary as a .json file if "save_as" was provided
	if save_as is not None:
		with open(save_as, "w") as outfile:
			outfile.write(json_dumps(panel_data_dict, indent=2))
		
	return panel_data_dict
	

# Written with the assistance of ChatGPT
class GraphInputWindow:
	"""
	A GraphInputWindow object takes a figure from display_graph, displays it using the Tkinter library, allows the
	user to enter a text value for the name of a highlighted panel, then returns that new user chosen name.
	"""
	def __init__(self, fig):
		self.root = tk.Tk()
		self.root.title("Graph and Input")
		self.user_input = None
		self.fig = fig

		# Set the window position
		window_width = 800
		window_height = 600
		x_offset = 100
		y_offset = 100
		self.root.geometry(f"{window_width}x{window_height}+{x_offset}+{y_offset}")


		# Embed the plot in the tkinter window
		self.canvas = FigureCanvasTkAgg(fig, master=self.root)
		self.canvas.draw()
		self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

		# Create a text entry widget
		self.entry = tk.Entry(self.root)
		self.entry.pack(side=tk.LEFT, padx=10)

		# Create a submit button
		self.submit_button = tk.Button(self.root, text="Submit", command=self.on_submit)
		self.submit_button.pack(side=tk.LEFT, padx=10)

	def on_submit(self):
		self.user_input = self.entry.get()
		self.root.destroy()  # Close the tkinter window
		plt.close(self.fig)

	def run(self):
		self.root.mainloop()
		return self.user_input


def display_graph(coords, norms, highlight_point_index, size_key=None, text_key=None):
	"""
	This function takes panel center and normal information, including the sizes of	points
	and a point to be highlighted, and graphs them into a MatPlotLib plot, such that the
	user can clearly see where any highlighted point is relative to all others.

	:param coords: an m*3 array, where m is some number of points representing panel centers, each in 3D coordinates
	:param norms: an m*3 array, where m is some number of vectors representing panel normals, each in 3D coordinates
	:param highlight_point_index: an integer that is > 0 and < m, representing
		the row index of a point from "coords" to be highlighted in a different color
	:param size_key: None to use default point sizes, or an array of shape m containing sizes for each point that will be graphed
	:param size_key: None to ommit labels for points, or an array of shape m containing text labels for each point that will be graphed

	:return: fig: a matplotlib.pyplot.figure object, to which all of these points, vectors and labels have been graphed
	"""
	# Create a single 3D figure
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# TODO: Align the view relative to the average normal of the panels, such that the user can clearly see which panels are which
	ax.view_init(elev=15, azim=105)
	# If a size key wasn't provided, plot them all the same. Include the size key if it was provided.
	if size_key is None:
		skobj.Points(coords).plot_3d(ax, s=50, alpha=1, c='c', zorder=1)
	else:
		skobj.Points(coords).plot_3d(ax, s=size_key, alpha=1, c='c', zorder=1)
	# Plot the highlighted point bigger than any of the others
	skobj.Points([coords[highlight_point_index, :]]).plot_3d(ax, s=250, alpha=1, c='m', zorder=2)
	# This scalar value will be used to scale the lengths of the normal vectors for each panel when they are graphed
	norm_len = np.linalg.norm(np.std(coords, axis=0))/2
	for i, norm_vec in enumerate(norms):
		skobj.Vector(norm_vec).plot_3d(ax, coords[i], scalar=norm_len, c='k', zorder=3)
	# If a list of text labels for panels was provided, graph them at the already included panel center positions
	if text_key is not None:
		for i, label in enumerate(text_key):
			if label is not None:
				ax.text(*coords[i].T, label, zorder=4)
	ax.set_title('Example Graph')
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')
	ax.set_aspect('equal', adjustable='box')
	
	return fig


	
if __name__ == '__main__':
	main()
