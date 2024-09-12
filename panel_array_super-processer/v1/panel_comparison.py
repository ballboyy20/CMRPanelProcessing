from json import load
from os import scandir

from math import degrees, radians, isclose, tan, pi
import numpy as np
import pandas as pd

import pyvista as pv

import skspatial.objects as skobj
import skspatial.transformation as sktrf

import matplotlib.pyplot as plt
import seaborn as sns

"""
Code written by Trevor Kent Carter, starting August 5th, 2024

	This code is for the PAC project this spring/summer semester at the CMR lab.
	It takes from code I wrote previously to evaluate the positions of two panels
	connected by a single hinge, about that hinge. It will now be updated to make
	comparisons based on already saved point and normal information, rather than
	the entire process of forming planes of best fit itself. Multiple methods of
	evaluating relative positions and repeatability will be implemented. Eventually,
	I would like to implement a method that uses lines representing the positions
	of hinges in order to more accurately gauge repeatability and precision for
	individual hinges.

Axis Definitions (Looking down on two tall planes, side by side with eachother)
	X: Up, along the hinges.
	Y: Left, into the left panel
	X: Into the camera, out of plane of the panels

Target Degrees of Freedom (all for right panel relative to left panel):
	θx: Out of plane bending, rotation around the hinge, folding
	θy: Out of plane twisting, rotation through the hinge, roll
	δz: Out of plane shear, translation along the z axis, piston


"""


def main():
	flat_flasher_file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\autolabel_demo\flat_mesh_cut_labeled.json'
	
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# single_scan_heatmap(ax, flat_flasher_file)
	# plt.show()
	
	birdsfoot_hinges = [('p1', 'p2'), ('p1', 'p4'), ('p2', 'p3'), ('p4', 'p3')]
	slet_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\autolabel_birdsfeet\1 + 5 SLET'
	slet_hinge_stats = multi_scan_hinge_stats(slet_folder, birdsfoot_hinges)
	#graph_multiscan_by_hinge(slet_hinge_stats, array_name='1 + 5 SLET')
	deg6kc_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\autolabel_birdsfeet\6 Deg KC'
	deg6kc_hinge_stats = multi_scan_hinge_stats(deg6kc_folder, birdsfoot_hinges)
	#graph_multiscan_by_hinge(deg6kc_hinge_stats, array_name='6 Deg KC')
	cclet_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\autolabel_birdsfeet\CCLET'
	cclet_hinge_stats = multi_scan_hinge_stats(cclet_folder, birdsfoot_hinges)
	#graph_multiscan_by_hinge(cclet_hinge_stats, array_name='CCLET')
	
	graph_multiarray([slet_hinge_stats, deg6kc_hinge_stats, cclet_hinge_stats], ['1 + 5 SLET', '6 Deg KC', 'CCLET'])
	
	# center_panel = {'center': [-409.65452609004785, 219.959850346595, 511.00468377595865],
	# 			  'normal': [0.10419923157161, 0.9940227196272352, 0.03257841317131838]}
	# a1_panel = {"center": [-453.96916212673744, 224.5664192381814, 515.7688444080179],
	# 			"normal": [0.11053943624572062, 0.9933236071297419, 0.0330037051440677]}
	# left_panel_trf, right_panel_trf = transform_panels(center_panel, a1_panel)
	#
	# theta_x, theta_y, delta_z = solve_positions(left_panel_trf, right_panel_trf, method='hinge')
	# dif = solve_positions(left_panel_trf, right_panel_trf, method='difference')
	# angle = solve_positions(left_panel_trf, right_panel_trf, method='angle')
	# print(theta_x, theta_y, delta_z)
	# print(dif)
	# print(angle)

	# fig = plt.figure()
	# ax = fig.add_subplot(121, projection='3d')
	# ax.view_init(elev=15, azim=-170)
	# ax2 = fig.add_subplot(122, projection='3d')
	# ax2.view_init(elev=15, azim=-150)
	# ax.set_aspect('equal', adjustable='box')
	# plot_circle(ax, left_plane.point, left_plane.normal, radius=40, c='c')
	# plot_circle(ax, right_plane.point, right_plane.normal, radius=40, c='m')
	# left_plane.normal.plot_3d(ax, scalar=20, point=left_plane.point, c='c')
	# right_plane.normal.plot_3d(ax, scalar=20, point=right_plane.point, c='m')
	# ltr_line.direction.plot_3d(ax, point=ltr_line.point, scalar=0.5, c='c')
	# basis_x_vec.plot_3d(ax, point=left_plane.point, scalar=20, c='r')
	# basis_y_vec.plot_3d(ax, point=left_plane.point, scalar=20, c='g')
	# basis_z_vec.plot_3d(ax, point=left_plane.point, scalar=20, c='b')
	# skobj.Vector((1, 0, 0)).plot_3d(ax2, scalar=10, c='r')
	# skobj.Vector((0, 1, 0)).plot_3d(ax2, scalar=10, c='g')
	# skobj.Vector((0, 0, 1)).plot_3d(ax2, scalar=10, c='b')
	# left_plane_trf.plot_3d(ax2, lims_x=[-80, 80], lims_y=[-40, 40], alpha=0.5, color='c')
	# right_plane_trf.plot_3d(ax2, lims_x=[-80, 80], lims_y=[-40, 40], alpha=0.5, color='m')
	# skobj.Vector((0, 0, 1)).plot_3d(ax2, point=(0, 0, 0), scalar=5, c='c')
	# right_norm_trf.plot_3d(ax2, point=right_center_trf, scalar=5, c='m')
	# plt.show()
	
def graph_multiarray(panel_hingedata_list, panel_names, fig=None):
	# This loops through the list of panels, each of which contains a dictionary with hingedata
	panel_variances = []
	panel_hinges = None
	for i, (panel_dict, panel_name) in enumerate(zip(panel_hingedata_list, panel_names)):
		# grid_pos = (int(np.floor(i / width)), int(i % width))
		# ax = plt.subplot2grid((height, width), grid_pos, fig=fig)
		# ax.set_title(panel_name)
		if panel_hinges is None:
			panel_hinges = panel_dict.keys()
		else:
			assert panel_hinges == panel_dict.keys()
		panel_x_variance_vals = []
		# Each dictionary with hingedata has the hinge's name as a key, and a pandas dataframe with columns for theta x, theta y, and delta z for each scan of that panel. This finds the average variance of each hinge's angle
		for hinge_name, hinge_df in panel_dict.items():
			panel_x_variance_vals.append(np.var(hinge_df['theta x'].values * 180 / pi))
		panel_variances.append(panel_x_variance_vals)
	
	# This creates an array (then properly indexed dataframe) where each column contains
	# the x variance of each hinge for a given panel, and rows represent each hinge
	x_hinge_data = np.column_stack(panel_variances)
	x_hinge_df = pd.DataFrame(data=x_hinge_data, index=panel_hinges, columns=panel_names)
	melted_x_df = pd.melt(x_hinge_df, var_name="Panel Type", value_name="Avg. Variance")

	if fig is None:
		fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle("Degree Four Vertex Average Hinge Angle Variance")
	sns.barplot(ax=ax, data=melted_x_df, x="Panel Type", y="Avg. Variance", errorbar=None)
	ax.set(ylabel='Avg. Variance across hinges\n(θ^2, Degrees Squared)')
	plt.show()

	
def graph_multiscan_by_hinge(hinge_dict, fig=None, array_name='Multiple Scans'):
	num_graphs = len(hinge_dict)
	width = int(np.ceil(np.sqrt(num_graphs)))
	height = int(np.ceil(num_graphs / width))
	if fig is None:
		fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle(array_name)
	for i, (hinge_name, hinge_df) in enumerate(hinge_dict.items()):
		grid_pos = (int(np.floor(i / width)), int(i % width))
		ax = plt.subplot2grid((height, width), grid_pos, fig=fig)
		ax.set_title(hinge_name)
		# Making the df use degrees
		hinge_df.iloc[:, 0:2] = hinge_df.iloc[:, 0:2] * 180 / pi
		# Standardizing the df around the averages of that hinge
		hinge_df = hinge_df.apply(lambda x: x - x.mean())
		hinge_df_melted = pd.melt(hinge_df, var_name='hinge type', value_name='value')
		sns.stripplot(hinge_df_melted, ax=ax, x='hinge type', y='value')
		#sns.scatterplot(hinge_df_melted, x='hinge type', y='value', marker="_")
		#sns.violinplot(hinge_df_melted, x='hinge type', y='value')
		ax.set_ylim([-0.25, 0.25])
		#sns.barplot(hinge_df_melted, x='hinge type', y='value')
	plt.tight_layout()
	plt.show()
	


# This function will take a list of jsons and a key showing which pairs of panels represent
# hinges, returning a standard deviation and list of absolute angles, in each axis, for each hinge
def multi_scan_hinge_stats(array_json_folder, hinge_pairs):
	scan_list = {}
	for file in scandir(array_json_folder):
		with open(file.path, 'r') as panel_file:
			array_dict = load(panel_file)
		scan_list[file.name] = array_dict
	#print(scan_list)
	
	hinge_data_dict = {}
	for left, right in hinge_pairs:
		# The columns hold the rotational x, y and translational z values
		# across all scans for this given hinge, where each row represents one scan
		pos_data = np.zeros(shape=(len(scan_list), 3))
		for i, panel_dict in enumerate(scan_list.values()):
			pos_data[i] = direct_compare_panels(panel_dict[left], panel_dict[right], method='hinge')
		pos_data_df = pd.DataFrame(data=pos_data, columns=['theta x', 'theta y', 'delta z'])
		hinge_data_dict[rf'{left} -> {right}'] = pos_data_df
	
	return hinge_data_dict
	
	


def single_scan_heatmap(ax, panel_data_json, method='angle'):
	with open(panel_data_json, 'r') as panel_data_file:
		panel_dict = load(panel_data_file)
	
	panel_names = sorted(panel_dict.keys())
	size = len(panel_names)
	index_array = np.indices((size, size))
	heatmap = np.zeros((size, size))
	for i, panel_i in enumerate(panel_names):
		for j, panel_j in enumerate(panel_names):
			if i == j:
				continue
			if method == 'angle':
				heatmap[i, j] = direct_compare_panels(panel_dict[panel_i], panel_dict[panel_j], method='angle')
			elif method == 'x_angle':
				heatmap[i, j] = direct_compare_panels(panel_dict[panel_i], panel_dict[panel_j], method='hinge')[0]
			elif method == 'y_angle':
				heatmap[i, j] = direct_compare_panels(panel_dict[panel_i], panel_dict[panel_j], method='hinge')[1]
			elif method == 'z_trans':
				heatmap[i, j] = direct_compare_panels(panel_dict[panel_i], panel_dict[panel_j], method='hinge')[2]
			elif method == 'difference':
				heatmap[i, j] = np.linalg.norm(direct_compare_panels(panel_dict[panel_i], panel_dict[panel_j], method='difference'))
	sns.heatmap(ax=ax, data=heatmap, xticklabels=panel_names, yticklabels=panel_names)


def direct_compare_panels(left_panel_dict, right_panel_dict, method):
	# Converting NumPy point clouds into skspatial planes with origins and normals
	left_plane = skobj.Plane(point=left_panel_dict['center'], normal=left_panel_dict['normal'])
	right_plane = skobj.Plane(point=right_panel_dict['center'], normal=right_panel_dict['normal'])
	#print('\n', left_plane)
	#print(right_plane)
	if np.dot(left_plane.normal, right_plane.normal) < 0:
		right_plane = skobj.Plane(point=right_plane.point, normal=-right_plane.normal)
	
	
	left_panel_trf, right_panel_trf = transform_panels(left_plane, right_plane)
	if method == 'hinge':
		theta_x, theta_y, delta_z = solve_positions(left_panel_trf, right_panel_trf, method='hinge')
		return theta_x, theta_y, delta_z
	elif method == 'custom hinge':
		raise ValueError("The provided positional comparison method for these planes has not yet"
						 "been implemented. Please choose one of 'hinge', 'difference' or 'angle'")
	elif method == 'difference':
		dif = solve_positions(left_panel_trf, right_panel_trf, method='difference')
		return dif
	elif method == 'angle':
		angle = solve_positions(left_panel_trf, right_panel_trf, method='angle')
		return angle
	elif method == 'euler':
		raise ValueError("The provided positional comparison method for these planes has not yet"
						 "been implemented. Please choose one of 'hinge', 'difference' or 'angle'")
	else:
		raise ValueError("The provided positional comparison method for these planes is not recognized,"
						 "please choose one of 'hinge', 'custom hinge', 'difference', 'angle' or 'euler'")
	


def transform_panels(left_plane, right_plane, hinge_line=None):
	# Getting the line from the left plane's center to the projection of the right plane's center
	# TODO: Add an option to project this line such that it is perpendicular to another line
	#  in the left_plane, lying between the left and right planes, representing the hinge
	ltr_line = c_to_proj(left_plane, right_plane.point)
	# The y axis will be based on the unit vector opposite the direction of the line described above
	basis_y_vec = -1 * ltr_line.direction.unit()
	# The z axis will be based on the normal of the left plane
	basis_z_vec = left_plane.normal.unit()
	# The x axis will be based on a vector perpendicular to the other two
	basis_x_vec = basis_y_vec.cross(basis_z_vec).unit()
	# These three form an orthogonal basis
	
	# Transform the original central positions such that they use the orthogonal
	# basis derived above, with the left point's center as the origin
	# TODO: For custom hinge-conscious evaluation, define this point as the intersection
	#  of the right plane, a plane with the a normal parallel to the y axis and centered
	#  on the right plane's center, and the plane formed by the y and z axes.
	right_center_trf = sktrf.transform_coordinates(right_plane.point, left_plane.point,
												   vectors_basis=(basis_x_vec, basis_y_vec, basis_z_vec))
	right_norm_trf = skobj.Vector(sktrf.transform_coordinates(right_plane.normal, (0, 0, 0),
															  vectors_basis=(basis_x_vec, basis_y_vec, basis_z_vec))).unit()
	
	# Reforming our planes of best fit from the transformed center and normal
	left_plane_trf = skobj.Plane(point=(0, 0, 0), normal=(0, 0, 1))
	right_plane_trf = skobj.Plane(point=right_center_trf, normal=right_norm_trf)
	
	return left_plane_trf, right_plane_trf


def solve_positions(ground_plane, ref_plane, method, verbose=False, trf_graph=None):
	if method == 'hinge':
		ground_center = ground_plane.point
		ref_center = ref_plane.point
		# Reference center projected onto the ground plane
		rc_on_gp = ground_plane.project_point(ref_center)
		# Line from ground center to the reference center's projection. It must be a line,
		# not a vector, in order to solve the intersection with the reference plane.
		gc_to_rcproj = skobj.Line(point=(0, -1, 0), direction=(0, -1, 0))
		# Reference plane intersection with line from ground center to reference center projection
		rp_intersect = ref_plane.intersect_line(gc_to_rcproj)
		# Line between reference center and the reference plane intersection (described above).
		# It must always be pointing towards negative y (away from the ground plane), so we check if the
		# intersection point lies further from the origin than the projection of the reference
		# plane center. If it's further, we draw the line from the reference center to the
		# intersection point, the other way around if the intersection point is closer.
		if rp_intersect[1] < rc_on_gp[1]:
			rc_to_gcl = skobj.Vector.from_points(ref_center, rp_intersect)
		else:
			rc_to_gcl = skobj.Vector.from_points(rp_intersect, ref_center)
		theta_x = gc_to_rcproj.direction.angle_between(rc_to_gcl)
		# If the direction of rc_to_gcl in the z direction is negative, then theta_x should also
		# be negative. The initial calculation for theta x above cannot account for this.
		if rc_to_gcl[2] <= 0:
			theta_x = -theta_x
		
		# This produces lines in the general direction of the positive x-axis from the centers of both
		# of the planes. These lines should be parallel to the xz plane, assuming the line from ground
		# center to the reference center's projection is perpendicular to the xz plane. Not the inverted
		# order of normal vs. in plane line to maintain the same direction for both orthogonal vectors.
		#ground_orth = ground_plane.normal.cross(gc_to_rcproj.direction)
		# Since our y axis was already defined by the line as described above, we can just use a unit vector (theoretically)
		ground_orth = skobj.Vector((1, 0, 0))
		ref_orth = ref_plane.normal.cross(rc_to_gcl)
		theta_y = ground_orth.angle_between(ref_orth)
		if ref_orth[2] <= 0:
			theta_y = -theta_y
		
		z_rc = ref_center[2]
		y_rc = ref_center[1]
		assert isclose(ref_center[0], 0.0, abs_tol=1e-8)
		delta_z = abs(z_rc) - abs(y_rc) * tan(theta_x / 2)

		if verbose:
			print('\ngc_to_rcproj: ', gc_to_rcproj)
			print('rc_to_gcl: ', rc_to_gcl)
			print('theta_x: ', theta_x)
			print('\nground_orth: ', ground_orth)
			print('ref_orth: ', ref_orth)
			print('theta_y: ', theta_y)
			print('\nz_rc: ', z_rc)
			print('y_rc: ', y_rc)
			print('delta_z: ', delta_z)
		if trf_graph is not None:
			gc_to_rcproj.direction.unit().plot_3d(trf_graph, scalar=50, point=gc_to_rcproj.point, c='c')
			rc_to_gcl.unit().plot_3d(trf_graph, scalar=50, point=ref_center, c='m')
			ground_orth.unit().plot_3d(trf_graph, scalar=50, point=ground_center, c='c')
			ref_orth.unit().plot_3d(trf_graph, scalar=50, point=ref_center, c='m')
		
		return theta_x, theta_y, delta_z
	# The line from the left plane center to the projection of the right
	# plane's center ino the left plane is assumed to be perpendicular
	# with the projection of the hinge_line into the left plane. This is
	# only used in the case that the hinge is closer to one point than to the other.
	elif method == 'custom hinge':
		raise ValueError("The provided positional comparison method for these planes has not yet"
						 "been implemented. Please choose one of 'hinge', 'difference' or 'angle'")
	elif method == 'difference':
		return ref_plane.normal - ground_plane.normal
	elif method == 'angle':
		return ground_plane.normal.angle_between(ref_plane.normal)
	elif method == 'euler':
		raise ValueError("The provided positional comparison method for these planes has not yet"
						 "been implemented. Please choose one of 'hinge', 'difference' or 'angle'")
	else:
		raise ValueError("The provided positional comparison method for these planes is not recognized,"
						 "please choose one of 'hinge', 'custom hinge', 'difference', 'angle' or 'euler'")


def c_to_proj(origin_plane, point_proj):
	origin = origin_plane.point
	end = origin_plane.project_point(point_proj)
	return skobj.Line.from_points(origin, end)


def get_center_plane(point_cloud, plane_normal):
	min_xyz = np.min(point_cloud, axis=0)
	max_xyz = np.max(point_cloud, axis=0)
	center_xyz = (min_xyz + max_xyz) / 2
	centered_plane = skobj.Plane(point=center_xyz, normal=plane_normal)
	return centered_plane


# Code partially inspired by https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list and conversations with ChatGPT
def remove_outliers(cloud_array, cloud_plane, m=90, verbose=False):
	# This does the following to the data:
	#  1) Create a 1d array, matching our cloud_array, of the projected point positions along the normal of the plane
	#  2) Return only the points corresponding to the inner m% of their respective distributions along the normal
	norm_line = skobj.Line(point=cloud_plane.point, direction=cloud_plane.normal)
	cloud_line = norm_line.transform_points(skobj.Points(cloud_array))
	# Get the inner 90% of the points based on their distribution along the normal
	lower_bound = np.percentile(cloud_line, 50 - int(m / 2))
	upper_bound = np.percentile(cloud_line, 50 + int(m / 2))
	# Filter the points to keep only the inner 90%
	inner_90_mask = (cloud_line >= lower_bound) & (cloud_line <= upper_bound)
	filtered_points = cloud_array[inner_90_mask]
	if verbose:
		print("\ncloud_array.shape: ", cloud_array.shape)
		print(f"Positions along normal with outliers -\nMin: {min(cloud_line)}\t\tMax: {max(cloud_line)}")
		print(
			f"Positions along normal after cleaning -\nMin: {min(cloud_line[inner_90_mask])}\t\tMax: {max(cloud_line[inner_90_mask])}")
		print("filtered_points.shape: ", filtered_points.shape)
		print(f"Total cleaned points: {filtered_points.shape[0] - cloud_array.shape[0]} - {m}% of inner points kept")
	return filtered_points


# Code generated with ChatGPT
def iter_best_fit_plane(coordinates):
	points = skobj.Points(coordinates)
	
	chunk_size = 10000  # Adjust as per your memory constraints and data size
	
	# Initialize the plane parameters
	n_accumulated = 0
	centroid_accumulated = np.zeros(3)
	covariance_accumulated = np.zeros((3, 3))
	
	# Process points in chunks
	for i in range(0, len(points), chunk_size):
		chunk_points = points[i:i + chunk_size]
		n_chunk = len(chunk_points)
		
		# Accumulate centroid
		centroid_chunk = np.mean(chunk_points, axis=0)
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


def mesh_to_array(filename):
	# Load the mesh
	mesh = pv.read(filename)
	
	# Get the vertices as a NumPy array
	return np.array(mesh.points)


# ChatGPT generated and modified code
def plot_circle(ax, center, normal, radius=1, num_points=100, c='red'):
	# Normalize the normal vector
	normal = normal.unit()
	
	# Create a basis on the plane
	if np.allclose(normal, [0, 0, 1]):
		u = skobj.Vector([1, 0, 0])
	else:
		u = skobj.Vector([0, 0, 1]).cross(normal).unit()
	v = normal.cross(u).unit()
	
	# Generate points on the circle
	theta = np.linspace(0, 2 * np.pi, num_points)
	circle_points = [
		center + radius * (np.cos(t) * u + np.sin(t) * v)
		for t in theta
	]
	
	# Convert the circle points to a numpy array for plotting
	circle_points = np.array(circle_points)
	
	# Plot the circle
	ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], label='Circle', color=c)


# ax.scatter(*center, color=c, label='Center')

# Plot the normal vector for reference
# normal_end = center + normal
# ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], length=radius, color=c,
#		  label='Normal Vector')


if __name__ == '__main__':
	main()
