"""
Code written by Trevor K. Carter for Python 3.11, started July 24th, 2024

This program is intended for the CMR burst project PAC, where several
birdsfeet with custom hinges and a standardized overall profile were
scanned many times by a 3D scanning machine. This is intended to section
these birdsfeet in an appropriate manner to have the panels' relative
positions and rotations calculated with code written by Clark.

File naming nomenclature:
- There is a folder containing the .stl files for 4 panels for all 30 scans, 120 files
	in total, each titled mesh%%_panel%.stl according to its scan number and which panel it is.
- There are individual folders for each birdsfoot

"""
from os import scandir
from file_renaming import zero_indexify
import regex as re
from copy import copy
from math import floor, ceil

import numpy as np
from stl import mesh
from matplotlib import pyplot as plt
import pyvista as pv
import skspatial.objects as skobj
import skspatial.transformation as sktrf
from scipy.spatial.transform import Rotation as R


def main():
	#birdsfoot = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\demo\mesh01.stl'
	#birdsfoot = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\demo\mesh23.stl'
	# If the panel is inconsistenly flipped left to right (clips behind the panel), set auto_flip=False and force_flip=True as necessary.
	# If the panel is randomly rotated around different sides, consider setting rot_res to a higher integer value.
	# If the panel is consistently rotated on the wrong side, set rot_clockwise to an integer value (one unit per side) to manually rotate it.
	# The program will normally produce a graph and pause until the graph is closed. Set show_cut=False to disable this.
	#split_panels(birdsfoot, r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation')
	
	#input_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\6 Deg KC'
	#output_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\6 Deg KC Sliced'
	#output_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\6 Deg KC Circle Sliced'
	#input_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\CCLET'
	#output_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\CCLET Sliced'
	#output_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\CCLET Circle Sliced'
	input_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\1 + 5 SLET'
	output_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\1 + 5 SLET Circle Sliced'
	#output_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\demo'
	folder_scanner(input_folder, output_folder)
	
def folder_scanner(read_folder, write_folder, make_zero_ind=False):
	mesh_file_list = []
	for item in scandir(read_folder):
		if item.is_file():
			mesh_file_list.append(item.path)
		else:
			print("WARNING: Non-file located in directory, will be skipped.")
	print(mesh_file_list)
	
	rot_res, rot_clockwise, auto_flip, force_flip, show_cut, cut_type = 36, 0, False, False, True, 'circles'
	reset = True
	for mesh_file_path in mesh_file_list:
		if reset:
			rot_res, rot_clockwise, auto_flip, force_flip, show_cut, cut_type = 36, 0, False, False, True, 'circles'
		while True:
			try:
				result = split_birdsfoot(mesh_file_path, write_folder, rot_res=rot_res, rot_clockwise=rot_clockwise, auto_flip=auto_flip, force_flip=force_flip, show_cut=show_cut, cut_type=cut_type)
				if result is None:
					pass
				else:
					reset = False
					force_flip, rot_clockwise = result
				break
			except Exception as error:
				print("An exception occurred:", error)
				print("\n\nTo run the program again, type ' ' and press enter."
					  "\n\nTo skip this mesh, type 'skip' and press enter."
					  "\n\nTo return to default settings and run this mesh again, type 'reset' and press enter."
					  "\n\nTo run again with new parameters, enter values for each of the following parameters"
					  "\nin the following format (defaults given below, 1 for True, 0 for false). The values"
					  "\nwill be maintained for future meshes."
					  "\nrot_res rot_clockwise auto_flip force_flip show_cut cut_type"
					  f"\n{rot_res}      {rot_clockwise}             {int(auto_flip)}         {int(force_flip)}          {int(show_cut)}          {cut_type}")
				user_response = input()
				responses = user_response.split()
				if user_response == ' ':
					continue
				elif responses[0] == 'reset':
					reset = True
					continue
				elif responses[0] == 'skip':
					break
				else:
					reset = False
					int_responses = [int(val) for val in responses[:5]]
					rot_res = int_responses[0]
					rot_clockwise = int_responses[1]
					auto_flip = bool(int_responses[2])
					force_flip = bool(int_responses[3])
					show_cut = bool(int_responses[4])
					cut_type = responses[5]
					continue
					
	if make_zero_ind:
		zero_indexify(write_folder)
					
				
			
	
	
def split_birdsfoot(full_panel_path, destination, rot_res=12, rot_clockwise=0, auto_flip=False, force_flip=False, show_cut=True, cut_type='circles'):
	mesh_object = pv.read(full_panel_path)
	# Make the point cloud (which can be read as a NumPy array)
	mesh_points = mesh_object.points
	print("Original Mesh Points: ", mesh_points.shape[0])
	print(mesh_points)
	# Make the list of faces (from which we choose only those faces whose
	# points still remain by the end of our cleaning and sectioning process)
	mesh_faces = mesh_object.regular_faces
	print("Original Mesh Faces: ", mesh_faces.shape[0])
	print(mesh_faces)
	
	# Remove outliers along the normal of the plane of best fit of the whole flasher
	full_plane, full_points, indicies_full_points = planar_outlier_removal(mesh_points)
	print("Mesh points after initial outlier removal: ", full_points.shape[0])
	
	# Generate two randomly oriented vectors that are orthonormal with the normal of the plane after outliers are removed
	b1, b2 = generate_orthonormal_basis(full_plane)
	b3 = full_plane.normal.unit()
	
	
	# Transform the cloud into 2D cooridnates within the plane found after removing outliers
	trans_proj_cloud = sktrf.transform_coordinates(full_points, full_plane.point, (b1, b2, b3))[:, 0:2]

	# Look at this many rotational positions to roughly find the bottom edge
	rough_rot_steps = rot_res
	rough_resolution = np.pi/rough_rot_steps
	# Making half of a rotation and considering minimums and maximums for bucket intervals,
	# this finds a standard deviation for the top and bottom edges around the circle
	# TODO: Continue annotation
	rough_rot_std = rotate_solve_flatness(trans_proj_cloud, rot_steps=rough_rot_steps)
	print(rough_rot_std)
	# This finds the four flattest sides and which steps they occurred on
	rough_sides, rough_corners = find_n_sides(4, rough_rot_std)
	
	# This finds the roughest of the flat_scans sides (which should be where the
	# clips are, but it might be one of the sides if our resolution is too low)
	messy_side_index = np.argmax(rough_sides[:, 1])
	# This finds the flattest of the flat_scans sides (which may be the bottom)
	clean_side_index = np.argmin(rough_sides[:, 1])
	# If the average distance from the messiest side to all other sides is significantly larger than the standard
	# deviation amongst those other sides, we can be confident that the messiest side is the side with the clamps.
	# TLDR, this is the right method to use if one side is distinctly rougher than all the others.
	# This value will be high if this is the case. This tends to be the most common case that works well.
	local_min_confidence = np.abs(np.mean(np.delete(rough_sides[:, 1], messy_side_index, axis=0)) - rough_sides[messy_side_index, 1]) / np.std(np.delete(rough_sides[:, 1], messy_side_index, axis=0))
	print("local_min_confidence: ", local_min_confidence)
	# This is the right method to use if one side is distinctly flatter than all others
	absolute_min_confidence = np.abs(np.mean(np.delete(rough_sides[:, 1], clean_side_index, axis=0)) - rough_sides[clean_side_index, 1]) / np.std(np.delete(rough_sides[:, 1], clean_side_index, axis=0))
	print("absolute_min_confidence: ", absolute_min_confidence)
	sorted_corners = np.sort(rough_corners[:, 1])
	# This is the proportion of the difference between the inner two rough corners to the
	# sum of the differences between the first two and the outer two. This will be higher
	# and is the right method to use if two corners are significantly rougher than the other
	# two. This condition has not been fine tuned to take priority over the others when it happens.
	max_valley_confidence = np.abs(sorted_corners[2] - sorted_corners[1]) / (np.abs(sorted_corners[1] - sorted_corners[0]) + np.abs(sorted_corners[3] - sorted_corners[2]))
	print("max_valley_confidence: ", absolute_min_confidence)
	if max_valley_confidence < local_min_confidence > absolute_min_confidence:
		# In this case, our bottom edge will be the opposite side of the roughest side
		bottom_side_index = (messy_side_index + 2) % 4
	elif max_valley_confidence < absolute_min_confidence:
		# In this case, our bottom edge will be the flattest side
		bottom_side_index = clean_side_index
	else:
		# In this case, our bottom edge will be opposite of the side between the two roughest corners
		bottom_side_index = np.min(np.argsort(rough_corners[:, 1])[1:3])
		
	# Knowing how many steps around the entire circle were taken to get roughly to the bottom edge, find the exact angle that translates to
	rough_angle_to_bottom = 2*np.pi*rough_sides[bottom_side_index, 0]/rough_rot_steps
	# This represents the resolution around the circle that our chosen step size resulted in. We're confident
	# that the flattest side is within this range, but we can't be sure exactly where yet.
	rough_sub_range = (rough_angle_to_bottom-rough_resolution, rough_angle_to_bottom+rough_resolution)
	
	# We will now look at more fine rotations within that subdivision to more precisely find the flattest side
	fine_rot_steps = 10
	fine_resolution = rough_resolution / fine_rot_steps
	fine_rot_std = rotate_solve_flatness(trans_proj_cloud, rot_steps=fine_rot_steps, start=rough_sub_range[0], stop=rough_sub_range[1])
	fine_sides, _ = find_n_sides(1, fine_rot_std)
	print(fine_sides)
	flattest_side_ind = fine_sides[0, 0]
	fine_angle_to_bottom = rough_sub_range[0] + np.ptp(rough_sub_range) * flattest_side_ind / fine_rot_steps
	
	# We will now look at even finer rotations within the last subdivision to even more precisely find the flattest side
	extra_fine_rot_steps = 10
	fine_sub_range = (fine_angle_to_bottom-fine_resolution, fine_angle_to_bottom+fine_resolution)
	extra_fine_rot_std = rotate_solve_flatness(trans_proj_cloud, rot_steps=extra_fine_rot_steps, start=fine_sub_range[0], stop=fine_sub_range[1])
	extra_fine_sides, _ = find_n_sides(1, extra_fine_rot_std)
	print(extra_fine_sides)
	extra_flattest_side_ind = extra_fine_sides[0, 0]
	fine_angle_to_bottom = fine_sub_range[0] + np.ptp(fine_sub_range) * extra_flattest_side_ind / extra_fine_rot_steps
	
	while True:
		print("\n\n---------- APPLYING NEW FLIP / ROTATIONAL SETTINGS ----------\n")
		print(fine_angle_to_bottom)
		bottom_rotation = R.from_rotvec((fine_angle_to_bottom + rot_clockwise * np.pi / 2) * b3)
		
		B1 = skobj.Vector(bottom_rotation.apply(b1)).unit()
		B2 = skobj.Vector(bottom_rotation.apply(b2)).unit()
		B3 = b3
		
		# Remove outliers along the line of the y axis (on the bottom edge of the flasher, and the messy edge
		clean_bottom_plane, clean_bottom_cloud, indicies_clean_aligned_final = planar_outlier_removal(full_points, active_indices=indicies_full_points, outlier_axis=skobj.Line(direction=B2, point=full_plane.point), cutoff_scaling=5)
		# Project the entire flasher into the coordinate system where x is along the bottom of
		# the birdsfoot, y towards the messy side and z out of the front of the birdsfoot
		aligned_cloud = sktrf.transform_coordinates(clean_bottom_cloud, full_plane.point, (B1, B2, B3))
		# After this transformation, B1, B2 and B3 are now just the x, y and z axes
		B1 = skobj.Vector((1, 0, 0))
		B2 = skobj.Vector((0, 1, 0))
		B3 = skobj.Vector((0, 0, 1))
		
		# The minimum y coordinate should now be approximately the distance from the middle of the flasher to the edge. Use that as our first distance reference.
		rough_half = abs(np.min(aligned_cloud[:, 1]))
		# Find the total range along the bottom edge of the flasher, after removing outliers, looking only within rough_half/8 of the edge
		edge_map = (1.25*-rough_half < aligned_cloud[:, 1]) & (aligned_cloud[:, 1] < 0.75*-rough_half)
		print(edge_map)
		_, edge_points, _ = planar_outlier_removal(aligned_cloud[edge_map], outlier_axis=skobj.Line(direction=B1, point=clean_bottom_plane.point))
		edge_length = np.ptp(edge_points[:, 0])
		print(rough_half)
		print(edge_length)
		# Get a small square of points on the bottom left panel, in a manner such that forwards or backwards orientation shouldn't matter
		rough_hinge_offset = 15
		rough_hinge_offset_45deg = 1.2 * rough_hinge_offset * np.sqrt(2)
		f1 = lambda x: 1*x - rough_half + 0.5*edge_length - rough_hinge_offset_45deg
		f2 = lambda x: -1*x - rough_half + 0.5*edge_length - rough_hinge_offset_45deg
		f1_aligned_cloud = f1(aligned_cloud[:, 0])
		f2_aligned_cloud = f2(aligned_cloud[:, 0])
		plane_angle_map = (aligned_cloud[:, 1] < f1_aligned_cloud) & (aligned_cloud[:, 1] < f2_aligned_cloud)
		plane_y_map = (-rough_half+rough_hinge_offset/2 < aligned_cloud[:, 1])
		# Use that square to define a new plane with the bottom left panel
		bot_left_panel_plane, _, _ = planar_outlier_removal(aligned_cloud[plane_angle_map & plane_y_map])
		
		
		# These are the new bases, where x is along the bottom edge,  z is orthogonal to the above
		# filtered section of that same bottom panel, and y is orthogonal to both of them
		X = skobj.Vector(B1).unit()
		# We must guarantee that this vector is pointing in the same orientation as B2, our original Z axis
		Z = bot_left_panel_plane.normal.unit()
		if Z[2] < 0:
			Z = -Z
		Y = skobj.Vector(np.cross(Z, X)).unit()
		
		orientation_check_map = (aligned_cloud[:, 0] > -edge_length/4) & (aligned_cloud[:, 0] < edge_length/4) & (aligned_cloud[:, 1] > 0)
		orientation_check_points = sktrf.transform_coordinates(aligned_cloud[orientation_check_map], bot_left_panel_plane.point, (X, Y, Z))
		scan_offset = 2 * edge_length / 140
		points_above = np.count_nonzero(orientation_check_points[:, 2] > scan_offset)
		print("Points Above: ", points_above)
		points_below = np.count_nonzero(orientation_check_points[:, 2] < -scan_offset)
		print("Points Below: ", points_below)
		needs_flip = (points_below > points_above)
		# If the program and user would both flip it, that would bring it back to its unflipped position, so do nothing.
		if (auto_flip and needs_flip) and force_flip:
			was_flipped = 2
		# If only the program or user would flip it, flip it.
		elif (auto_flip and needs_flip) or force_flip:
			was_flipped = 1
			print("Flipping the point cloud left to right")
			new_x = (-1, 0, 0)
			old_y = (0, 1, 0)
			new_z = (0, 0, -1)
			aligned_cloud = sktrf.transform_coordinates(aligned_cloud, (0, 0, 0), (new_x, old_y, new_z))
		else:
			was_flipped = 0
		
		
		# Get only the points in an area in the corner of the bottom left panel
		bot_left_corner_map = (aligned_cloud[:, 0] < -edge_length*35/140) & (aligned_cloud[:, 1] < -edge_length*35/140)
		# Remove outliers along the x axis, since we can't do that for the whole panel (the ruler or clips might interfere if they were included)
		_, bot_left_corner_points, _ = planar_outlier_removal(aligned_cloud[bot_left_corner_map], outlier_axis=skobj.Line(direction=B1, point=clean_bottom_plane.point))
		# Transform these points such that their xy plane is defined by the plane of the bottom left panel
		clean_bot_left_corner_points = sktrf.transform_coordinates(bot_left_corner_points, (0, 0, 0), (X, Y, Z))
		# The most downward and leftward point is will have the highest product of its
		# x and y values (though negative). Find which transformed point produces that.
		print(clean_bot_left_corner_points[:, 0:2])
		print(clean_bot_left_corner_points[:, 0:2].shape)
		print(clean_bot_left_corner_points[:, 0:2].dtype)
		xy_products = np.prod(clean_bot_left_corner_points[:, 0:2], axis=1)
		print(xy_products)
		print(xy_products.shape)
		bot_left_corner_index = np.argmax(xy_products)
		# Use that index to find the untransformed point, and project it onto the bottom left panel plane.
		# This is our bottom left panel position, which we will use as our origin.
		bot_left_corner = bot_left_panel_plane.project_point(bot_left_corner_points[bot_left_corner_index, :])
		print(bot_left_corner)
		
		# An array of points, whose original positions amongst the points are saved in indices_clean_aligned_final
		final_cloud = sktrf.transform_coordinates(aligned_cloud, bot_left_corner, (X*edge_length/140, Y*edge_length/140, Z*edge_length/140))
		
		if cut_type == 'clark':
			hinge_offset = 11
			hinge_offset_45deg = hinge_offset * np.sqrt(2)
			x_map = (final_cloud[:, 0] > hinge_offset) & (final_cloud[:, 0] < 140-hinge_offset)
			y_map_lower = (final_cloud[:, 1] > hinge_offset) & (final_cloud[:, 1] < 68-hinge_offset)
			y_map_upper = (final_cloud[:, 1] > 70+hinge_offset) & (final_cloud[:, 1] < 140-hinge_offset)
			g1 = lambda x: 1*x + 0
			g1_final_cloud = g1(final_cloud[:, 0])
			diagonal_map_upper_left = final_cloud[:, 1] > g1_final_cloud+hinge_offset_45deg
			diagonal_map_upper_right = final_cloud[:, 1] < g1_final_cloud-hinge_offset_45deg
			g2 = lambda x: -1*x + 140
			g2_final_cloud = g2(final_cloud[:, 0])
			diagonal_map_lower_left = final_cloud[:, 1] < g2_final_cloud-hinge_offset_45deg
			diagonal_map_lower_right = final_cloud[:, 1] > g2_final_cloud+hinge_offset_45deg
			g3 = lambda x: -1*x/3 + 140
			g3_final_cloud = g3(final_cloud[:, 0])
			hinge_map_upper_left = final_cloud[:, 1] < g3_final_cloud
			
			ll_map = (x_map & y_map_lower & diagonal_map_lower_left)
			lr_map = (x_map & y_map_lower & diagonal_map_lower_right)
			ul_map = (x_map & y_map_upper & diagonal_map_upper_left & hinge_map_upper_left)
			ur_map = (x_map & y_map_upper & diagonal_map_upper_right)
			
			lower_left_verts_vis = skobj.Points(final_cloud[ll_map][::10])
			lower_right_verts_vis = skobj.Points(final_cloud[lr_map][::10])
			upper_left_verts_vis = skobj.Points(final_cloud[ul_map][::10])
			upper_right_verts_vis = skobj.Points(final_cloud[ur_map][::10])
			clark_points_vis = skobj.Points(np.concatenate([lower_left_verts_vis, lower_right_verts_vis, upper_left_verts_vis, upper_right_verts_vis], axis=0))
			
			lower_left_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[ll_map]), axis=1)]
			lower_right_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[lr_map]), axis=1)]
			upper_left_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[ul_map]), axis=1)]
			upper_right_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[ur_map]), axis=1)]
			
			clark_groups = zip([lower_left_faces, lower_right_faces, upper_left_faces, upper_right_faces],
							   ['_panel4',        '_panel3',         '_panel1',        '_panel2'])
		elif cut_type == 'circles':
			pos_1ref = np.array((61.5, 112.44, 0))
			pos_1 = np.array((83, 112.44, 0))
			pos_2 = np.array((112.44, 83, 0))
			pos_3 = np.array((112.44, 57, 0))
			pos_4 = np.array((83, 27.56, 0))
			pos_4ref = np.array((40, 27.56, 0))
			circle_rad = 5.32
			
			pos_1ref_map = np.linalg.norm(final_cloud-pos_1ref, axis=1) < circle_rad
			pos_1_map = np.linalg.norm(final_cloud-pos_1, axis=1) < circle_rad
			pos_2_map = np.linalg.norm(final_cloud-pos_2, axis=1) < circle_rad
			pos_3_map = np.linalg.norm(final_cloud-pos_3, axis=1) < circle_rad
			pos_4_map = np.linalg.norm(final_cloud-pos_4, axis=1) < circle_rad
			pos_4ref_map = np.linalg.norm(final_cloud-pos_4ref, axis=1) < circle_rad
			
			pos_1ref_vis = skobj.Points(final_cloud[pos_1ref_map][::10])
			pos_1_vis = skobj.Points(final_cloud[pos_1_map][::10])
			pos_2_vis = skobj.Points(final_cloud[pos_2_map][::10])
			pos_3_vis = skobj.Points(final_cloud[pos_3_map][::10])
			pos_4_vis = skobj.Points(final_cloud[pos_4_map][::10])
			pos_4ref_vis = skobj.Points(final_cloud[pos_4ref_map][::10])
			circles_points_vis = skobj.Points(np.concatenate([pos_1ref_vis, pos_1_vis, pos_2_vis, pos_3_vis, pos_4_vis, pos_4ref_vis], axis=0))

			pos_1ref_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[pos_1ref_map]), axis=1)]
			pos_1_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[pos_1_map]), axis=1)]
			pos_2_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[pos_2_map]), axis=1)]
			pos_3_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[pos_3_map]), axis=1)]
			pos_4_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[pos_4_map]), axis=1)]
			pos_4ref_faces = mesh_faces[np.all(np.isin(mesh_faces, indicies_clean_aligned_final[pos_4ref_map]), axis=1)]
			
			circles_faces = np.concatenate([pos_1ref_faces, pos_1_faces, pos_2_faces, pos_3_faces, pos_4_faces, pos_4ref_faces], axis=0)
		
		
		if show_cut:
			print("rot_res, rot_clockwise, auto_flip, force_flip, show_cut, was_flipped")
			print(rot_res, rot_clockwise, auto_flip, force_flip, show_cut, was_flipped)
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.view_init(elev=50, azim=-60)
			
			skobj.Points(final_cloud[::10]).plot_3d(ax, s=0.1, alpha=0.1, c='k')
			if cut_type == 'clark':
				clark_points_vis.plot_3d(ax, s=0.3, alpha=0.1, c='c')
				# lower_left_verts_vis.plot_3d(ax, s=0.3, alpha=0.1, c='c')
				# lower_right_verts_vis.plot_3d(ax, s=0.3, alpha=0.1, c='c')
				# upper_left_verts_vis.plot_3d(ax, s=0.3, alpha=0.1, c='c')
				# upper_right_verts_vis.plot_3d(ax, s=0.3, alpha=0.1, c='c')
			elif cut_type == 'circles':
				circles_points_vis.plot_3d(ax, s=0.3, alpha=1, c='c')
			
			X.plot_3d(ax, (0, 0, 0), scalar=50, c='r')
			Y.plot_3d(ax, (0, 0, 0), scalar=50, c='g')
			Z.plot_3d(ax, (0, 0, 0), scalar=50, c='b')
			
			plt.figtext(0.5, 0.01, "X - B1 (red), Y - B2 (green), Z - B3 (blue)", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
			ax.set_aspect('equal', adjustable='box')
			plt.show()
		
		def save_mesh(faces, filename, vertices=mesh_points):
			# Create the mesh
			this_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
			for i, f in enumerate(faces):
				for j in range(3):
					this_mesh.vectors[i][j] = vertices[f[j], :]
			this_mesh.save(filename)
		
		name_extension_pattern = r'([^\\\/]+)(\.[^\\\/]+)$'
		print(full_panel_path)
		file_match = re.search(name_extension_pattern, full_panel_path)
		print(file_match)
		if file_match:
			file_name = file_match.group(1)
			file_extension = file_match.group(2)
		# print(f"File name: {file_name}")
		# print(f"File extension: {file_extension}")
		else:
			raise FileNotFoundError("No match found")
		
		print(f"\n\nIf the above visualization was satisfactory, each panel will be"
			  f"\nindividually saved as {destination+'/'+file_name}_panel<1-4>.stl"
			  f"\n\nType ' ' and press enter to approve the slicing and continue to the next mesh (not saving settings)"
			  f"\n\nType 'skip' to NOT save these files and ignore this mesh, moving on to the next one."
			  f"\n\nType 'add <text>' to save these files, but prepend '<text>_' to each of their names."
			  f"\n\nType 'more' to get more options for slicing this mesh to try again (including returning to default settings)."
			  f"\n\nType 'save' to accept the settings from this cut, save, and apply them to subsequent cuts"
			  f"\n\nOtherwise, enter a response with integers in the following format to adjust quick settings."
			  f"\n<flip: 1 to flip left to right, 0 to ignore> <rotate_clockwise: 0-3 clockwise rotations>")
		user_approval = input("")
		responses = user_approval.split()
		save = False
		if user_approval == " ":
			if cut_type == 'clark':
				for faces_list, panel_name in clark_groups:
					save_mesh(faces_list, f'{destination}\\{file_name}{panel_name}{file_extension}')
			elif cut_type == 'circles':
				save_mesh(circles_faces, f'{destination}\\{file_name}_circles{file_extension}')
			# save_mesh(lower_left_faces, f'{destination}\\{file_name}_panel4{file_extension}')
			# save_mesh(lower_right_faces, f'{destination}\\{file_name}_panel3{file_extension}')
			# save_mesh(upper_left_faces, f'{destination}\\{file_name}_panel1{file_extension}')
			# save_mesh(upper_right_faces, f'{destination}\\{file_name}_panel2{file_extension}')
			break
		elif responses[0] == 'skip':
			print(f"Skipping {file_name} without saving any of its panels.")
			break
		elif responses[0] == 'add':
			save_mesh(lower_left_faces, f'{destination}\\{responses[1]}_{file_name}_panel4{file_extension}')
			save_mesh(lower_right_faces, f'{destination}\\{responses[1]}_{file_name}_panel3{file_extension}')
			save_mesh(upper_left_faces, f'{destination}\\{responses[1]}_{file_name}_panel1{file_extension}')
			save_mesh(upper_right_faces, f'{destination}\\{responses[1]}_{file_name}_panel2{file_extension}')
			break
		elif responses[0] == 'more':
			raise InterruptedError("Selected 'more' option to prepare set more fine settings for slicing this mesh.")
		if responses[0] == 'save':
			save = True
			save_mesh(lower_left_faces, f'{destination}\\{file_name}_panel4{file_extension}')
			save_mesh(lower_right_faces, f'{destination}\\{file_name}_panel3{file_extension}')
			save_mesh(upper_left_faces, f'{destination}\\{file_name}_panel1{file_extension}')
			save_mesh(upper_right_faces, f'{destination}\\{file_name}_panel2{file_extension}')
			break
		else:
			responses = [int(val) for val in responses]
			print(responses)
			if len(responses) == 1:
				force_flip = bool(responses[0])
			elif len(responses) == 2:
				force_flip = bool(responses[0])
				rot_clockwise = responses[1]
			else:
				print("Invalid number of parameters passed for panel slicing approval. Try again.")

	if save:
		return force_flip, rot_clockwise
	else:
		return








def rotate_solve_flatness(point_cloud_2d, rot_steps, start=0.0, stop=2*np.pi, linear_steps=100, verification_percentile=0.5, verbose=True):

	# For each rotation position, we will calculate the standard deviation across the minimums of the x buckets
	rot_std = np.full(rot_steps, -1, dtype=np.float64)
	# If we're looking at the entire circle, we can check the max values to consider opposite sides simultaneously.
	# We will take half as many actual steps, but save the max values into the second half of the rot_std array
	if stop-start >= 2*np.pi:
		iter_range = range(floor(rot_steps/2))
		save_max = True
	# If we're just looking at some arc, we need to actually take every step, and we can ignore the max values.
	else:
		iter_range = range(rot_steps)
		save_max = False
	for step_i, angle in [(step, -(start+(stop-start)*step/rot_steps)) for step in iter_range]:
		# This is just a progress bar to track how many rotations have been observed
		if verbose:
			if save_max:
				print(f'\nEvaluating 180 degrees of rotation in {rot_steps} increments, checking minimums and maximums of inner {verification_percentile*100}%'
					  f'\nof {linear_steps} x buckets to evaluate edge flatness with standard deviation.')
				print("X"*floor(20*step_i/(rot_steps/2)), "-"*ceil(20-(20*step_i/(rot_steps/2))))
			else:
				print(f'\nEvaluating degrees of rotation from {None} to {None} in {rot_steps} increments, checking minimums of inner {verification_percentile*100}%'
					  f'\nof {linear_steps} x buckets to evaluate edge flatness with standard deviation.')
				print("X"*floor(20*step_i/rot_steps), "-"*ceil(20-(20*step_i/rot_steps)))
			print("Step:", step_i, '/', rot_steps)
			print("Angle: ", angle * 180 / (2 * np.pi))
		rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
									[np.sin(angle), np.cos(angle)]])
		rotated_2d_cloud = point_cloud_2d @ rotation_matrix.T
		# This will produce 2d coordinates, where b1 is the x axis and b2 is the y axis
		#plt.clf()
		#sns.scatterplot(x=rotated_2d_cloud[:, 0], y=rotated_2d_cloud[:, 1])
		#plt.show()
		# This finds how big each x bucket should be (the range on x divided by the number of steps to be taken)
		linear_step_size = np.ptp(rotated_2d_cloud[:, 0]) / linear_steps
		#print(linear_step_size)
		# This makes a 1D array of values from the minimum of our x range (inclusive) to the maximum (exclusive), stepping by our step size as calculated above
		x_intervals = np.arange(np.min(rotated_2d_cloud[:, 0]), np.max(rotated_2d_cloud[:, 0]), linear_step_size)
		#print(x_intervals)
		# This array is identical in shape to the x intervals, and will have 1 y min per x interval
		y_minimums = np.zeros_like(x_intervals)
		y_maximums = np.zeros_like(x_intervals)
		for i, x_pos in enumerate(x_intervals):
			# Make a mask of our entire point list indicating only those points within this x interval
			interval_mask = (x_pos <= rotated_2d_cloud[:, 0]) & (rotated_2d_cloud[:, 0] < x_pos + linear_step_size)
			# Verify that at least one point is in this x range (which is not guaranteed, with a sufficiently high resolution
			if np.count_nonzero(interval_mask) > 0:
				# Get the y values for those points, and remove outliers. Note that this function will never return an empty collection of points
				filtered_y_values = linear_outlier_removal(rotated_2d_cloud[interval_mask, 1], step=0.005, cutoff_scaling=10)
				#print(filtered_y_values.shape)
				# Find the minimum y value for this x interval without outliers
				y_min = np.min(filtered_y_values)
				y_minimums[i] = y_min
				y_max = np.max(filtered_y_values)
				y_maximums[i] = y_max
			# Since the first bucket is guaranteed to include at least one point (since
			# it is only there because of that minimal point), y_min will always be
			# defined before this branch. If a subsequent bucket has no points, it
			# can be set to the most recent value of y_min from the previous bucket
			else:
				y_minimums[i] = y_min
				y_maximums[i] = y_max
		# This makes integer indices for the section that will be observed to determine flatness
		start = ceil(linear_steps/2 - verification_percentile*linear_steps/2)
		end = floor(linear_steps/2 + verification_percentile*linear_steps/2)
		# TODO: Find an implementation for standard deviation that caters to many points close together and isn't disturbed
		#  by a few very far points, thus disregarding positions where there are groups of points and slightly different locations
		# For this rotation step, save the standard deviation of y mins across x buckets in our observation range
		rot_std[step_i] = np.std(y_minimums[start:end])
		if verbose: print("std: ", rot_std[step_i])
		if save_max:
			rot_std[step_i+floor(rot_steps/2)] = np.std(y_maximums[start:end])
			if verbose: print("std + 180 deg: ", rot_std[step_i+floor(rot_steps/2)])
		#plt.clf()
		#sns.scatterplot(x=x_intervals, y=y_minimums)
		#plt.show()
	#plt.clf()
	#sns.lineplot(x=range(rot_steps), y=rot_std)
	#plt.show()
	
	return rot_std

def find_n_sides(n_sides, rot_std):
	sides = np.zeros((n_sides, 2))
	corners = np.zeros((n_sides, 2))
	# This will produce the index for the rotation resulting in minimal standard deviation (which may be any one of the relatively flat_scans sides)
	some_side_pos = np.argmin(rot_std)
	bucket_min_start = 2 * max(rot_std)
	bucket_max_start = min(rot_std)
	rot_steps = rot_std.shape[0]
	
	for i in range(n_sides):
		side_ref = some_side_pos + i * rot_steps / n_sides
		
		bucket_min = bucket_min_start
		side_pos = None
		for step_i in range(floor(side_ref-0.5*rot_steps/n_sides), ceil(side_ref+0.5*rot_steps/n_sides)):
			step_i = step_i % rot_steps
			this_height = rot_std[step_i]
			if this_height < bucket_min:
				bucket_min = this_height
				side_pos = step_i
		
		bucket_max = bucket_max_start
		corner_pos = None
		for step_i in range(floor(side_ref), ceil(side_ref+rot_steps/n_sides)):
			step_i = step_i % rot_steps
			this_height = rot_std[step_i]
			if this_height > bucket_max:
				bucket_max = this_height
				corner_pos = step_i
		
		sides[i, :] = (side_pos, bucket_min)
		corners[i, :] = (corner_pos, bucket_max)
	
	print("\nSides: \n", sides)
	print("Corners: \n", corners)
	return sides, corners


# Following function was generated by ChatGPT then simplified to use skspatial objects and
# methods. It's basic purpose is to generate two vectors that are orthogonal with the normal
# of our plane, and thus in the plane, such that they can act as bases for that plane.
def generate_orthonormal_basis(plane, seeded=12345):
	# Option to set a seed to get repeatable bases for our plane
	if seeded is not None:
		np.random.seed(seeded)
	# Normalize the normal to our plane (get it as a vector with a magnitude of one, a unit vector)
	plane_norm = plane.normal.unit()
	# Generate some completely random vector
	random_vector = np.random.randn(3)
	
	# Make sure the random vector is not parallel to the normal vector of our plane
	# (which is so absurdly unlikely it's basically impossible, but we like being thorough)
	while np.allclose(np.dot(random_vector, plane_norm), 0):
		random_vector = np.random.randn(3)
	
	# Project the random vector onto the plane to make it orthogonal to the normal vector,
	# then normalize it to a unit vector, so that it can qualify as a basis for our plane
	basis_vector_1 = plane.project_vector(skobj.Vector(random_vector)).unit()
	
	# Create the second basis vector by taking the cross product of the normal and the first basis
	basis_vector_2 = skobj.Vector(np.cross(plane_norm, basis_vector_1)).unit()
	
	return basis_vector_1, basis_vector_2


# Code in this function was partially generated with ChatGPT then modified
def iter_best_fit_plane(coordinates):
	chunk_size = 10000  # Adjust as per your memory constraints and data size
	
	# Initialize the plane parameters
	n_accumulated = 0
	centroid_accumulated = np.zeros(3)
	covariance_accumulated = np.zeros((3, 3))
	
	# Process points in chunks
	for i in range(0, len(coordinates), chunk_size):
		chunk_points = coordinates[i:i + chunk_size]
		n_chunk = len(chunk_points)
		
		# Accumulate centroid
		centroid_chunk = np.mean(chunk_points, axis=0)
		# print("\n\n\nIteration ", i)
		# print(n_accumulated)
		# print(centroid_accumulated)
		# print(n_chunk)
		# print(centroid_chunk)
		centroid_accumulated = (n_accumulated * centroid_accumulated + n_chunk * centroid_chunk) / (
				n_accumulated + n_chunk)
		
		# Accumulate covariance matrix
		centered_chunk_points = chunk_points - centroid_chunk
		covariance_chunk = np.dot(centered_chunk_points.T, centered_chunk_points)
		covariance_accumulated += covariance_chunk
		
		# Update number of accumulated points
		n_accumulated += n_chunk
	
	# Compute the plane normal from accumulated covariance matrix
	_, eigen_vectors = np.linalg.eigh(covariance_accumulated)
	normal = eigen_vectors[:, 0]  # Eigenvector corresponding to the smallest eigenvalue
	
	# Create a Plane object with the accumulated normal and centroid
	plane = skobj.Plane(normal=normal, point=centroid_accumulated)
	
	# Now 'plane' contains the fitted plane using incremental updates
	return plane


# This will remove outlier points, relative to the normal of the plane of
# best fit, until the range of refined points has more than the target
# proportion of the previous iteration (i.e. when removing the extreme
# points is no longer significantly affecting the point distribution
# along that direction)
def planar_outlier_removal(original_points, active_indices=None, outlier_axis='norm', target=None, cutoff_scaling=20, step=0.001, iter_limit=20, verbose=False):
	"""
	:param original_points: PyVista object containing all points in the mesh being read
	:param outlier_axis: 'norm' or skspatial.objects.Line object. If left to default 'norm', will remove outliers
		relative to the normal of the plane of best fit. Otherwise, will remove outliers relative to the provided line.
	:param target: Float between 0.0 and 1.0. The proportion of the range along the normal of the new reduced plane
		to the range along the normal of the previous plane at which point the iterative program will stop removing points.
	:param cutoff_scaling: Positive float value. Alternative option to target, sets the target proportion as 1.0 - cutoff_scaling * step.
	:param step: The combined proportion removed equally from the extremities along the normal of the plane of best fit
	:param iter_limit: After this many iterations removing outliers, stop, and return what was obtained
	:return final_plane, final_points: The remaining points and the plane of best fit that was generated from them
	"""
	# ^ is the logical NAND operator: True if exactly one of the inputs are True, False otherwise.
	if not ((target is None) ^ (cutoff_scaling is None)):
		raise UserWarning("You must provide a target removal percentage, or at least leave the default cutoff_scaling.")
	
	# I added the "cutoff_scaling" option because of a way that I found "target" could break.
	# If the step size was very small, target doesn't change at all, and at some point,
	# removing few enough outliers does sufficiently little to shrink the total range
	# that the target is instantly met. Setting our final_target relative to our steps
	# implies that for small steps, the new range must be significantly closer to the
	# previous iteration's range than would be required to stop removing outliers for
	# large steps. Increasing the cutoff_scaling makes the target easier to meet, to avoid cutting
	# out too many points.
	if target:
		final_target = target
	else:
		final_target = 1.0 - (cutoff_scaling * step)
	
	# This does the following to the data:
	#  1) Create a 1d array, matching our cloud_array, of the points by their positions along the normal of the plane
	#  2) Return only the points corresponding to the inner m% of their respective distributions along the normal
	def remove_outliers(old_points, outlier_line):
		# Project all points of the point cloud onto this line
		cloud_line = outlier_line.transform_points(old_points)
		# Get the total range of values along this line
		old_range = max(cloud_line) - min(cloud_line)
		# Get the minimum and maximum values of points to be removed, based on their distribution
		# along the normal line, by setting the percentiles with step/2 for upper and lower bounds each
		lower_bound = np.percentile(cloud_line, 100 * (0.0 + step / 2))
		upper_bound = np.percentile(cloud_line, 100 * (1.0 - step / 2))
		# Solve for the new resulting range of values along the normal line
		new_range = upper_bound - lower_bound
		# Make a active_indices indicating only those points within the lower and upper bounds along the normal
		inner_mask = (cloud_line >= lower_bound) & (cloud_line <= upper_bound)
		# Apply that active_indices to the point cloud itself
		new_points = old_points[inner_mask]
		return new_points, inner_mask, new_range, old_range
	
	iterations = 0
	prev_points = original_points
	original_range = None
	if active_indices is None:
		active_indices = np.arange(original_points.shape[0])
	else:
		assert active_indices.shape[0] == original_points.shape[0]
	
	while True:
		if outlier_axis == 'norm':
			# Get a line through the plane's center and normal
			prev_plane = iter_best_fit_plane(prev_points)
			removal_line = skobj.Line(point=prev_plane.point, direction=prev_plane.normal)
		else:
			assert isinstance(outlier_axis, skobj.Line)
			removal_line = outlier_axis
		filtered_points, this_mask, new_range, old_range = remove_outliers(prev_points, removal_line)
		active_indices = active_indices[this_mask]
		iterations += 1
		# The first time we solve the range of points, save it (for reference of before vs. after removing outliers)
		if original_range is None:
			original_range = copy(old_range)
		# If the removal of these outliers has a sufficiently small effect on the range of points
		# along the normal of the plane such that it meets the target, stop iterating, we have
		# converged on an appropriate solution. If the iteration limit is reached, stop just the same.
		if (new_range / old_range > final_target) or (iterations >= iter_limit):
			final_plane = iter_best_fit_plane(filtered_points)
			final_points = filtered_points
			print("Final Point Range Percentage of original: ", 100 * new_range / original_range, "%")
			print("Remaining Point Percentage: ", 100 * final_points.shape[0] / original_points.shape[0], "%")
			print("Iterations: ", iterations)
			break
		# If we have not converged, make the point cloud resulting from this iteration the previous cloud,
		# and remove outliers from it again. Repeat until target (or iteration limit) is reached.
		else:
			prev_points = filtered_points
	
	if verbose:
		return final_plane, final_points, active_indices, iterations, original_range, new_range
	else:
		return final_plane, final_points, active_indices
	
	
def linear_outlier_removal(positions, step=0.0005, cutoff_scaling=20, iter_limit=25):
	def remove_outliers(points):
		old_range = max(points) - min(points)
		# Get the minimum and maximum values of points to be removed, based on their distribution
		# along the normal line, by setting the percentiles with step/2 for upper and lower bounds each
		lower_bound = np.percentile(points, 100 * (0.0 + step / 2))
		upper_bound = np.percentile(points, 100 * (1.0 - step / 2))
		# Solve for the new resulting range of values along the normal line
		new_range = upper_bound - lower_bound
		# Make a mask indicating only those points within the lower and upper bounds along the normal
		inner_mask = (points >= lower_bound) & (points <= upper_bound)
		if np.count_nonzero(inner_mask) > 0:
			# Apply that mask to the point cloud itself
			new_points = points[inner_mask]
		else:
			new_points = np.array(np.mean(points))
		
		return new_points, new_range, old_range
	
	target = 1.0 - (cutoff_scaling * step)
	prev_points = positions
	iterations = 0
	while True:
		filtered_points, new_range, old_range = remove_outliers(prev_points)
		iterations += 1
		# If the removal of these outliers has a sufficiently small effect on the range of points
		# such that it meets the target, stop iterating, we have converged on an appropriate solution.
		# If the iteration limit is reached, stop just the same.
		if (new_range / old_range > target) or (iterations >= iter_limit):
			#print("Satisfactory Range Percentage: ", 100 * new_range / old_range, "%\nIterations: ", iterations)
			final_points = filtered_points
			break
		# If we have not converged, make the y values resulting from this iteration the previous cloud,
		# and remove outliers from it again. Repeat until target (or iteration limit) is reached.
		else:
			prev_points = filtered_points
	
	return final_points
	
	
	
	

if __name__ == '__main__':
	main()
	