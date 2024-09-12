from json import load as json_load
from json import dumps as json_dumps
import numpy as np

import skspatial.objects as skobj
# jake wrote this on his machine
# Trevor wrote this to test merging!
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
	center_list = []
	norm_list = []
	# Making an array of all the centers and their vectors
	for center_norm in panel_data_dict.values():
		center_list.append(center_norm["center"])
		norm_list.append(center_norm["normal"])
	centers = np.array(center_list)
	normals = np.array(norm_list)
	size_key = [50] * centers.shape[0]

	print(panel_data_dict)
	panel_keys = list(panel_data_dict.keys())
	pos = 0
	key_conversions = list(range(-len(panel_keys), 0))
	print(panel_keys)
	while pos < len(panel_keys):
		fig = display_graph(centers, normals, pos, size_key, text_key=key_conversions)
		window = GraphInputWindow(fig)
		this_new_label = window.run()
		if this_new_label in key_conversions:
			print("Label", this_new_label, "already exists in this dictionary. Choose a new name.")
			continue
		key_conversions[pos] = this_new_label
		size_key[pos] = 250
		pos += 1
	print(key_conversions)
	#key_conversions = [chr(val) for val in range(97, 97+len(panel_keys))]
	for old_key, new_key in zip(panel_keys, key_conversions):
		panel_data_dict[new_key] = panel_data_dict.pop(old_key)
	print(panel_data_dict)
	
	if save_as is not None:
		# Writing to a .json
		with open(save_as, "w") as outfile:
			outfile.write(json_dumps(panel_data_dict, indent=2))
		
	return panel_data_dict
	

class GraphInputWindow:
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
	# Create a figure and plot the data
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev=15, azim=105)
	if size_key is None:
		skobj.Points(coords).plot_3d(ax, s=50, alpha=1, c='c', zorder=1)
	else:
		skobj.Points(coords).plot_3d(ax, s=size_key, alpha=1, c='c', zorder=1)
	norm_len = np.linalg.norm(np.std(coords, axis=0))/2
	skobj.Points([coords[highlight_point_index, :]]).plot_3d(ax, s=250, alpha=1, c='m', zorder=2)
	for i, norm_vec in enumerate(norms):
		skobj.Vector(norm_vec).plot_3d(ax, coords[i], scalar=norm_len, c='k', zorder=3)
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
