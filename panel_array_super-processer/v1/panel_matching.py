"""
Code written by Trevor K. Carter for Python 3.11 starting August 1st, 2024


Pre-Scan Procedure:
	1) Create panel groups with universal panel pre-processing
	2) Create a visualization of internal labels by panel group
	3) Let the user label each of the panel surface groups manually
	4) In a .csv file, save the labels and central positions of each of the groups


New-Scan Procedures (BEFORE CODE WAS WRITTEN):
	1) Create panel groups with universal panel pre-processing
	2) Compare the new scan to a pre-existing scan whose individual panel center positions (and the normal vectors
		to the planes of best fit of those panels) were automatically recorded, then manually labeled by the user.
		2.1) Normalize the new scan scale by its average distance between all groups,
			such that that value matches the corresponding value for the old scan
		2.2) Transform the new scan such that the average position of all panels
			matches the average position of the old scan's panels
		2.3) Test all 360 degrees of rotations (by 10 degree increments?) about the normal of
			that plane of best fit from our average panel position. At each position, loop
			through the old panel groups and find each group's nearest match from amongst the
			new panels.	No new panel can be counted as the closest panel to multiple old panels.
			Use those paired distances to evaluate a mean squared error.
		2.4) Repeat step 3.3 with progressively smaller increments until converging (or getting close enough)
		2.5) With the new groups positioned such that they are as aligned as possible with the old groups,
			label the new groups by the labels for its closest paired old group
	3) IF an origin panel was specified in the template, save all panel positions and
	rotations as relative to the corresponding origin panel of the new scan. Otherwise,
	save them as absolute positions and rotations in their original world coordinates.


"""
from os import scandir
from json import load, dumps

import re
import numpy as np
import skspatial.objects as skobj
import skspatial.transformation as sktrf
from scipy.spatial.distance import cdist as scipy_cdist

from matplotlib import pyplot as plt

import panel_sectioning
import panel_labeling




def main():
	# The user may choose to make a template, auto-label based on a template or assign labels before auto-labeling the rest
	print("Welcome! This code was written by Trevor K. Carter in August of 2024."
	      "\nIt's purpose was to aid in the process of processing 3D scans of panel"
	      "\narrays to ultimately evaluate the precision and repeatability of their"
	      "\nhinges. This code takes panel arrays with uniformly sized, preferably"
		  "\ncircular sections from a flat_scans face on each of the array's panels."
		  "\nIt groups these points, finding centers and planes of best fit by panel."
		  "\nIt allows the user to label each panel manually, then to process other"
		  "\n3D scans (already cut) of a similar shape, to automatically label the"
		  "\npanels of each new array. For each panel array, its centers and the"
		  "\nnormal vector for its plane of best fit are saved to a .json.")
	choice = input("\nChoose which of the following you would like to do:"
				   "\n\t1) Make and save a template"
				   "\n\t2) Provide a template and auto-label cuts from a directory"
				   "\n\t3) Make a temporary template and auto-label cuts from a directory\n")
	if choice == '1':
		print("\n\nYou will now make and save a template.")
		cut_filepath = input("Please provide below the filepath to the panel array"
							 "\nthat will be used in the creation of this template:\n")
		save_directory = input("Please provide below the directory to which the labels"
							   "\nand positions of this template should be saved:\n")
		use_posttext = input("By default, the template is saved by the same name as"
							   "\nthe mesh cut used in its creation, with '_labeled'"
							   "\nappended afterwards. Press enter to accept, or type"
							   "\n'<text>' to instead append <text>\n")
		num_panels = input("The process for creating templates requires that"
						   "\nyou know how many panel sections are present first."
						   "\nHow many panels does this template model include?\n")
		if use_posttext.strip():
			custom_posttext = use_posttext
			make_template(cut_filepath, save_directory, int(num_panels), labeled_posttext=custom_posttext)
		else:
			make_template(cut_filepath, save_directory, int(num_panels))
	elif choice == '2':
		print("\n\nYou will now use an existing template to auto-label other panels. Note that"
			  "\nit will assume that the other panels being auto labeled have the exact same"
			  "\nnumber of panel cuts, in relatively close  positions, cut such that there is"
			  "\nonly one unique matching orientation.")
		template_filepath = input("Please provide below the filepath to the template file"
								  "\nthat will be used to auto-label all other files:\n")
		autolabel_directory = input("Please provide below the directory containing the"
									"\ncut, unlabeled panels that will be auto-labeled:\n")
		save_directory = input("Please provide below the directory to which the newly"
							   "\nlabeled panels should be saved:\n")
		use_posttext = input("By default, labeled cuts are saved by the name of"
							   "\nthe mesh used in its creation, with '_labeled'"
							   "\nappended afterwards. Press enter to accept, or type"
							   "\n'<text>' to instead append <text>\n")
		if not use_posttext.strip():
			posttext = use_posttext
		else:
			posttext = '_labeled'
		# FIXME: TESTING OVERRIDES
		#template_filepath = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\warped flasher\mesh_cut_labeled_panelsREAL.json'
		#autolabel_directory = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\autolabel_demo_cuts'
		#save_directory = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\autolabel_demo'
		with open(template_filepath, 'r') as panel_data_obj:
			template_labels = load(panel_data_obj)
		num_panels = len(template_labels)
		saved_labeled_cuts = []
		autolabel_dir_item_paths = [file_obj.path for file_obj in scandir(autolabel_directory)]
		for new_cut_filepath in autolabel_dir_item_paths:
			new_cut_random_labels = panel_sectioning.find_panel_data(new_cut_filepath, None, num_panels, graph_clusters=False)
			new_cut_labels = match_new_cuts(template_labels, new_cut_random_labels)
			saved_labeled_cuts.append(new_cut_labels)
			this_cut_name = get_file_name(new_cut_filepath)
			save_file = save_directory+'\\'+this_cut_name+posttext+'.json'
			with open(save_file, 'w') as labeled_panel:
				labeled_panel.write(dumps(new_cut_labels, indent=2))
		graph_choice = input('\n\nThe files have been saved! Would you like to plot'
							 '\nall of the panels in comparison with each other? Type'
							 '\n"y" and press enter to graph, other wise just press enter.\n')
		if graph_choice == 'y':
			fig = graph_all_panels((template_labels, *saved_labeled_cuts), (template_filepath, *autolabel_dir_item_paths))
			plt.show()
	elif choice == '3':
		print("\n\nYou will now create a template to auto-label other panels.")
		autolabel_directory = input("Please provide below the directory containing the"
									"\ncut, unlabeled panels that will be auto-labeled:\n")
		template_special = input("A template will be created from the first panel in the directory."
							  "\nWould you like to give the panel used as the template a special name?"
							  "\nPress enter to skip this, or type '<text>' to instead append <text>\n")
		save_directory = input("Please provide below the directory to which the newly"
							   "\nlabeled panels should be saved:\n")
		use_posttext = input("By default, labeled cuts are saved by the name of"
							   "\nthe mesh used in its creation, with '_labeled'"
							   "\nappended afterwards. Press enter to accept, or type"
							   "\n'<text>' to instead append <text>\n")
		num_panels = input("The process for creating templates requires that"
						   "\nyou know how many panel sections are present first."
						   "\nHow many panels does this template model include?\n")
		if not use_posttext.strip():
			posttext = use_posttext
		else:
			posttext = '_labeled'
		# FIXME: TESTING OVERRIDES
		#autolabel_directory = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\autolabel_demo_cuts'
		#save_directory = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\autolabel_demo_manual'
		#with open(template_filepath, 'r') as panel_data_obj:
		#	template_labels = load(panel_data_obj)
		saved_labeled_cuts = []
		autolabel_dir_item_paths = [file_obj.path for file_obj in scandir(autolabel_directory)]
		template_labels = None
		for new_cut_filepath in autolabel_dir_item_paths:
			new_cut_random_labels = panel_sectioning.find_panel_data(new_cut_filepath, None, int(num_panels), graph_clusters=False)
			if template_labels is None:
				# TODO: Modify the "make_template" function such that it works here, removing the
				#  overhead of processing filepaths and just working directly with dictionaries
				new_cut_labels = panel_labeling.overwrite_labels(new_cut_random_labels)
				template_labels = new_cut_labels
				this_cut_name = get_file_name(new_cut_filepath)+template_special
			else:
				new_cut_labels = match_new_cuts(template_labels, new_cut_random_labels)
				this_cut_name = get_file_name(new_cut_filepath)
			saved_labeled_cuts.append(new_cut_labels)
			save_file = save_directory+'\\'+this_cut_name+posttext+'.json'
			with open(save_file, 'w') as labeled_panel:
				labeled_panel.write(dumps(new_cut_labels, indent=2))
		graph_choice = input('\n\nThe files have been saved! Would you like to plot'
							 '\nall of the panels in comparison with each other? Type'
							 '\n"y" and press enter to graph, other wise just press enter.\n')
		if graph_choice == 'y':
			fig = graph_all_panels(saved_labeled_cuts, autolabel_dir_item_paths)
			plt.show()
	else:
		raise TypeError("Respond to this prompt with either 1, 2, or 3 and nothing more")


# This function will be used to label template scans with various panels,
# for the purpose of automatically labeling future panels that are scanned.
def make_template(template_cut_file, save_dir, num_panels, labeled_posttext='_labeled'):
	# If output_or_continue is a filepath, make a dictionary grouping the panel sections, let the
	# user label the dictionary panel elements, then save it to the provided filepath
	if isinstance(save_dir, str):
		random_labels = panel_sectioning.find_panel_data(template_cut_file, None, num_panels)
		cut_mesh_name = get_file_name(template_cut_file)
		# This will be the path of the saved .json with panel array data
		template_filepath = save_dir + '\\' + cut_mesh_name + labeled_posttext + '.json'
		panel_labeling.overwrite_labels(random_labels, save_as=template_filepath)
	# If output_or_continue is None, do the same as above, but rather than saving, return the labeled dictionary.
	elif save_dir is None:
		random_labels = panel_sectioning.find_panel_data(template_cut_file, None, num_panels)
		panel_labels = panel_labeling.overwrite_labels(random_labels)
		return panel_labels
	else:
		raise TypeError("The second parameter passed to 'make_template' was neither a"
						"\nfilepath nor 'None'. Please try again and provide either.")


# This function will be used to automatically orient and label the
# panels of a given new scan relative to the specified old scan.
# It will take the template dictionary of positions, then transform,
# scale and rotate the positions of the new_cut dictionary of positions
# until it is as aligned as possible, at which point it will make a new
# dictionary matching the template labels to the original cut data
def match_new_cuts(template_dict, new_cut_dict, rot_steps=720, verbose=False):
	template_labels = np.array([label for label in template_dict.keys()])
	template_points = np.array([panel['center'] for panel in template_dict.values()])
	template_normals = np.array([panel['normal'] for panel in template_dict.values()])
	
	new_cut_labels = np.array([label for label in new_cut_dict.keys()])
	new_cut_points = np.array([panel['center'] for panel in new_cut_dict.values()])
	new_cut_normals = np.array([panel['normal'] for panel in new_cut_dict.values()])
	
	# Here, we scale the new_cut points such that their standard deviation matches the template
	template_scale = np.linalg.norm(np.std(template_points, axis=0))
	new_cut_scale = np.linalg.norm(np.std(new_cut_points, axis=0))
	new_cut_scaled_points = new_cut_points * template_scale / new_cut_scale
	
	# This set of points is small enough that we can use the native skspatial best fit plane method
	template_plane = skobj.Plane.best_fit(template_points)
	new_cut_plane = skobj.Plane.best_fit(new_cut_scaled_points)
	
	# Here, both sets of points are projected into coordinate systems using the same x and y bases, but using
	# the normals of each of their respective bases for z (such that both panels are facing towards z)
	b1, b2 = panel_sectioning.generate_orthonormal_basis(template_plane)
	template_points_trf = sktrf.transform_coordinates(template_points, template_plane.point, (b1, b2, template_plane.normal.unit()))
	template_normals_trf = sktrf.transform_coordinates(template_normals, (0, 0, 0), (b1, b2, template_plane.normal.unit()))
	
	b1_new, b2_new = panel_sectioning.generate_orthonormal_basis(new_cut_plane)
	new_cut_points_trf = sktrf.transform_coordinates(new_cut_scaled_points, new_cut_plane.point, (b1_new, b2_new, new_cut_plane.normal.unit()))
	new_cut_normals_trf = sktrf.transform_coordinates(new_cut_normals, (0, 0, 0), (b1_new, b2_new, new_cut_plane.normal.unit()))
	
	# We will also compare our template with the new_cut where it was rotated 180 deg
	# around the x axis, in case the normal of one of our planes is backwards
	flip_rotation_matrix = np.multiply(np.identity(3).T, (1, -1, -1)).T
	#flip_rotation_matrix = np.einsum('i,ij->ij', (1, -1, -1), np.identity(3))
	new_cut_points_flipped_trf = np.matmul(flip_rotation_matrix, new_cut_points_trf.T).T
	
	rot_error = np.full(rot_steps, -1.0, dtype=np.float64)
	rot_error_flipped = np.full(rot_steps, -1.0, dtype=np.float64)
	for step_i in range(rot_steps):
		rotation_angle = 2*np.pi * step_i / rot_steps
		c, s = np.cos(rotation_angle), np.sin(rotation_angle)
		rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
		rotated_new_points = np.matmul(rotation_matrix, new_cut_points_trf.T).T
		rotated_new_points_flipped = np.matmul(rotation_matrix, new_cut_points_flipped_trf.T).T
		# This uses list comprehension to find the mean squared error of all the minimums
		# of the pairwise distances from each new_cut point to all template points
		rot_error[step_i] = np.mean(np.array([np.min(scipy_cdist(point.reshape((1, 3)), template_points_trf)) for point in rotated_new_points]) ** 2)
		rot_error_flipped[step_i] = np.mean(np.array([np.min(scipy_cdist(point.reshape((1, 3)), template_points_trf)) for point in rotated_new_points_flipped]) ** 2)
	min_error_pos = np.argmin(rot_error)
	min_error_flipped_pos = np.argmin(rot_error_flipped)
	
	
	if rot_error_flipped[min_error_flipped_pos] < rot_error[min_error_pos]:
		print("using flipped orientation")
		final_angle = 2 * np.pi * min_error_flipped_pos / rot_steps
		#pre_trf_new_points = np.matmul(flip_rotation_matrix, new_cut_points_trf.T).T
		pre_trf_new_points = new_cut_points_flipped_trf
		pre_trf_new_normals = np.matmul(flip_rotation_matrix, new_cut_normals_trf.T).T
		pre_trf_template_normals = np.matmul(flip_rotation_matrix, template_normals_trf.T).T
	else:
		print("using unflipped orientation")
		final_angle = 2*np.pi * min_error_pos / rot_steps
		pre_trf_new_points = new_cut_points_trf
		pre_trf_new_normals = new_cut_normals_trf
		pre_trf_template_normals = template_normals_trf
		
	c, s = np.cos(final_angle), np.sin(final_angle)
	final_rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
		
	final_rotated_new_points = np.matmul(final_rotation_matrix, pre_trf_new_points.T).T
	final_rotated_new_normals = np.matmul(final_rotation_matrix, pre_trf_new_normals.T).T
	# This will produce a 1d list where each element is the index to the closest point from the template
	match_indices = np.argmin(scipy_cdist(final_rotated_new_points, template_points_trf), axis=1)
	aligned_normal_polarities = [1 if (np.dot(cut_norm, temp_norm) > 0) else -1 for cut_norm, temp_norm in zip(final_rotated_new_normals, template_normals_trf[match_indices])]
	#aligned_normals = np.multiply(aligned_normal_polarities, new_cut_normals)
	aligned_normals = np.einsum('i,ij->ij', aligned_normal_polarities, new_cut_normals)
	# This uses advanced slicing to get the list of matching
	# labels. The list comprehension equivalent is included below.
	new_cut_matched_labels = template_labels[match_indices]
	print(new_cut_matched_labels)
	#new_cut_matched_labels = np.array([template_labels[match_pos] for match_pos in match_indices])
	#print('\n\n', new_cut_matched_labels)
	
	new_cut_labeled_dict = {label: {'center': tuple(new_cut_points[i]), 'normal': tuple(aligned_normals[i])} for i, label in enumerate(new_cut_matched_labels)}
	
	if verbose:
		print(rot_error)
		print(min_error_pos)
		print(rot_error[min_error_pos])
		
		print(rot_error_flipped)
		print(min_error_flipped_pos)
		print(rot_error_flipped[min_error_flipped_pos])

		print(match_indices)
		
		print(aligned_normal_polarities)
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.view_init(elev=90, azim=0)
		skobj.Points(final_rotated_new_points).plot_3d(ax, c='c')
		for point, norm in zip(final_rotated_new_points, final_rotated_new_normals):
			skobj.Vector(norm).plot_3d(ax, point=point, scalar=10, c='c')
		skobj.Points(template_points_trf).plot_3d(ax, c='m')
		for point, norm in zip(template_points_trf, template_normals_trf):
			skobj.Vector(norm).plot_3d(ax, point=point, scalar=10, c='m')
		ax.set_aspect('equal', adjustable='box')
		plt.show()
	
	return new_cut_labeled_dict


def graph_all_panels(panels, filepaths):
	num_graphs = len(panels)
	width = int(np.ceil(np.sqrt(num_graphs)))
	height = int(np.ceil(num_graphs / width))
	#sub_plot_pos = lambda nth: int(str(height)+str(width)+str(nth+1))
	fig = plt.figure(figsize=(3*width, 3*height))
	for i, panel_data in enumerate(panels):
		coords = [panel['center'] for panel in panel_data.values()]
		norms = [panel['normal'] for panel in panel_data.values()]
		grid_pos = (int(np.floor(i / width)), int(i % width))
		#ax = fig.add_subplot(sub_plot_pos(i), projection='3d')
		ax = plt.subplot2grid((height, width), grid_pos, projection='3d')
		ax.view_init(elev=15, azim=105)
		skobj.Points(coords).plot_3d(ax, s=25, alpha=1, c='c', zorder=1)
		norm_len = np.linalg.norm(np.std(coords, axis=0))/2
		for j, norm_vec in enumerate(norms):
			skobj.Vector(norm_vec).plot_3d(ax, coords[j], scalar=norm_len, c='k', zorder=2, alpha=0.5)
		for k, label in enumerate(panel_data.keys()):
			ax.text(*np.transpose(coords[k]), label, zorder=3, fontsize=8)
		ax.set_title(get_file_name(filepaths[i]))
		ax.set_xlabel('X Axis')
		ax.set_ylabel('Y Axis')
		ax.set_zlabel('Z Axis')
		ax.set_aspect('equal', adjustable='box')
	
	return fig
	

def get_file_name(filepath):
	# This will match to the name of the input panel array file (without its extension)
	name_extension_pattern = r'([^\\\/]+)(\.[^\\\/]+$)'
	file_match = re.search(name_extension_pattern, filepath)
	if not file_match:
		raise NameError("There was an issue while extracting the name of this template cut")
	return file_match[1]


if __name__ == "__main__":
	main()
