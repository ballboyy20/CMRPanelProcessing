from math import degrees, radians, isclose, tan
import numpy as np


import pyvista as pv

import skspatial.objects as skobj
import skspatial.transformation as sktrf
from skspatial.plotting import plot_3d

import matplotlib.pyplot as plt

"""
Code written by Trevor Kent Carter, starting June 17th, 2024

	This code is for the PAC project this spring/summer semester at the CMR lab.
	Its first purpose is to create a new orthogonal basis aligned with the axes
	we're using to evaluate joint configurations. Its next purpose it to find the
	relative positions between the pannels in several degrees of freedom, namely
	out of plane bending (θx), out of plane shear (δz) and out of plane twisting (θy).
	
Axis Definitions (Looking down on two tall planes, side by side with eachother)
	X: Up, along the hinges.
	Y: Left, into the left panel
	X: Into the camera, out of plane of the panels
	
Target Degrees of Freedom (all for right panel relative to left panel):
	θx: Out of plane bending, rotation around the hinge, folding
	θy: Out of plane twisting, rotation through the hinge, roll
	δz: Out of plane shear, translation along the z axis, piston
	
Methodology (OUTDATED):
	1) Convert the .stl files of the left and right panels into NumPy readable array of point cloud coordinates using PyVista
	2) Convert the the NumPy coordinates into Sci-Kit Spatial Points, which are then used to find a plane of best fit and generate a Plane object
	3) Define a new orthogonal basis using the Gram-Schmidt method on the following vectors:
		- the Y axis originates from the opposite of the direction of the line drawn from the center point of
			the left Plane to the projection of the right Plane's center point into the left plane.
		- the Z axis originates from the normal of the left Plane
		- the X axis originates from the cross product of the above described Y and Z axis vectors.
	3) Orthogonalize the points of the left and and right planes, and make the left plane the origin
	4) Resolve for the Points and Plane objects
	5) Find the angle between the two following lines for θx:
		- The Y axis (a line, in plane with the left plane, from the center point of the
			left plane to the projection of the center point of the right plane)
		- A line from the center point of the right plane to the point of intersection of the Y axis with the right plane
		

		
"""

def main():
	
	left_panel = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\mesh1selection_left.stl'
	right_panel = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\mesh1selection_right.stl'

	# Converting 3d files into NumPy arrays with PyVista
	left_cloud_raw = mesh_to_array(left_panel)
	right_cloud_raw = mesh_to_array(right_panel)
	
	# Converting NumPy point clouds into skspatial planes with origins and normals
	left_plane_raw = array_to_plane(left_cloud_raw)
	right_plane_raw = array_to_plane(right_cloud_raw)
	
	# Find the plane of best fit, and project all points onto the normal of the plane of best fit. Use that distribution to remove outliers.
	left_cloud_clean = remove_outliers(left_cloud_raw, left_plane_raw, m=95)
	right_cloud_clean = remove_outliers(right_cloud_raw, right_plane_raw, m=95)
	
	# Converting NumPy point clouds into skspatial planes with origins and normals
	left_plane_clean = array_to_plane(left_cloud_clean)
	right_plane_clean = array_to_plane(right_cloud_clean)
	
	# Getting the centers of the cuboids containing the left and right planes each
	left_plane_centered = get_center(left_cloud_clean, left_plane_clean)
	right_plane_centered = get_center(right_cloud_clean, right_plane_clean)
	
	# Getting the line from the left plane's center to the projection of the right plane's center
	ltr_line = c_to_proj(left_plane_centered, right_plane_centered.point)
	# The y axis is based on the unit vector opposite the direction of that line
	ltr_y_vec = -ltr_line.direction.unit()
	ltr_z_vec = left_plane_centered.normal.unit()
	ltr_x_vec = ltr_y_vec.cross(ltr_z_vec).unit()
	
	# Perform QR decomposition
	new_basis = np.column_stack((ltr_x_vec, ltr_y_vec, ltr_z_vec))
	Q, R = np.linalg.qr(new_basis)
	
	# Transform the original point clouds such that they use the orthogonal
	# basis derived above, with the left point's center as the origin
	left_cloud_trf = sktrf.transform_coordinates(left_cloud_clean, left_plane_centered.point, vectors_basis=Q.T)
	right_cloud_trf = sktrf.transform_coordinates(right_cloud_clean, left_plane_centered.point, vectors_basis=Q.T)

	#left_cloud_raw_trf = sktrf.transform_coordinates(left_cloud_raw, left_plane_raw.point, vectors_basis=Q.T)
	
	left_center_trf = sktrf.transform_coordinates(left_plane_centered.point, left_plane_centered.point, vectors_basis=Q.T)
	right_center_trf = sktrf.transform_coordinates(right_plane_centered.point, left_plane_centered.point, vectors_basis=Q.T)
	
	left_norm_trf = Q.T @ np.array(left_plane_clean.normal).T
	left_plane_trf = skobj.Plane(point=left_center_trf, normal=left_norm_trf)
	right_norm_trf = Q.T @ np.array(right_plane_clean.normal).T
	right_plane_trf = skobj.Plane(point=right_center_trf, normal=right_norm_trf)
	
	theta_x, theta_y, delta_z = solve_positions(left_plane_trf, right_plane_trf)
	print(f"rot X:\t{degrees(theta_x)}\nrot Y:\t{degrees(theta_y)}\ntrans X:\t{delta_z}")
	
	fig = plt.figure()
	#ax2 = fig.add_subplot(121, projection='3d')
	#ax = fig.add_subplot(122, projection='3d')
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev=5, azim=-5)
	#ax2.view_init(elev=5, azim=-100)
	
	#plot_solutions(ax2, points=[left_cloud_raw_trf])
	#plot_solutions(ax, points=[left_cloud], planes=[left_plane_centered])
	plot_solutions(ax, planes=[left_plane_trf, right_plane_trf])
	#plot_solutions(ax, points=[left_cloud_trf, right_cloud_trf], planes=[left_plane_trf, right_plane_trf])
	
	
	ax.set_aspect('equal', adjustable='box')
	plt.show()


def plot_solutions(ax, points=None, planes=None, vectors=None, colors=None, show=False):
	if colors is None:
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
	col_counter = 0
	
	if points is not None:
		if isinstance(points, list):
			for point_listed in points:
				skobj.Points(point_listed).plot_3d(ax, s=0.05, alpha=0.01, c=colors[col_counter])
				col_counter += 1
			col_counter = 0
		else:
			points.plot_3d(ax, s=0.1, alpha=0.05, c=colors[0], depthshade=False)
	if planes is not None:
		if isinstance(planes, list):
			for plane_listed in planes:
				plane_listed.plot_3d(ax, lims_x=[-40, 40], lims_y=[-20, 20], alpha=0.5, color=colors[col_counter])
				plane_listed.point.plot_3d(ax, s=50, c=colors[col_counter])
				plane_listed.normal.plot_3d(ax, point=plane_listed.point, c='b')
				col_counter += 1
			col_counter = 0
		else:
			planes.plot_3d(ax, alpha=1)
	
	if show:
		plt.show()
		
		
def solve_positions(ground_plane, ref_plane):
	ground_center = ground_plane.point
	ref_center = ref_plane.point
	# Reference center projected onto the ground plane
	rc_on_gp = ground_plane.project_point(ref_center)
	# Line from ground center to the reference center projection
	gc_to_rcproj = skobj.Line.from_points(ground_center, rc_on_gp)
	# Reference plane intersection with line from ground center to reference center projection
	rp_intersect = ref_plane.intersect_line(gc_to_rcproj)
	# Line from reference center to the reference plane intersection (described above)
	rc_to_gcl = skobj.Line.from_points(ref_center, rp_intersect)
	theta_x = radians(180) - gc_to_rcproj.direction.angle_between(rc_to_gcl.direction)
	#print("Theta x (degrees): ", degrees(theta_x))
	
	# This produces lines in the general direction of the positive x-axis from the centers of both
	# of the planes. These lines should be parallel to the xz plane, assuming the line from ground
	# center to the reference center's projection is perpendicular to the xz plane. Not the inverted
	# order of normal vs. in plane line to maintain the same direction for both orthogonal vectors.
	ground_orth = gc_to_rcproj.direction.cross(ground_plane.normal)
	ref_orth = ref_plane.normal.cross(rc_to_gcl.direction)
	theta_y = ground_orth.angle_between(ref_orth)
	#print("Theta y (degrees): ", degrees(theta_y))
	
	#print("Ground Center: ", ground_center)
	#print("Reference Center: ", ref_center)
	z_rc = ref_center[2]
	y_rc = ref_center[1]
	assert isclose(ref_center[0], 0.0, abs_tol=1e-8)
	delta_z = z_rc - (-y_rc) * tan(theta_x/2)
	#print("Delta z (translation): ", delta_z)
	
	return theta_x, theta_y, delta_z



def get_center(point_cloud, plane):
	min_xyz = np.min(point_cloud, axis=0)
	max_xyz = np.max(point_cloud, axis=0)
	center_xyz = (min_xyz + max_xyz) / 2
	centered_plane = skobj.Plane(point=center_xyz, normal=plane.normal)
	return centered_plane



def c_to_proj(origin_plane, point_proj):
	origin = origin_plane.point
	end = origin_plane.project_point(point_proj)
	return skobj.Line.from_points(origin, end)


# Code partially inspired by https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list and conversations with ChatGPT
def remove_outliers(cloud_array, cloud_plane, m=90):
	#print(cloud_array.shape)
	# TODO: This needs to do the following to the data:
	#  1) Create a 1d array, matching our cloud_array, of the projected point positions along the normal of the plane
	#  2) Return only the points corresponding to the inner 90% of their respective distributions along the normal
	norm_line = skobj.Line(point=cloud_plane.point, direction=cloud_plane.normal)
	cloud_line = norm_line.transform_points(skobj.Points(cloud_array))
	#print(f"Positions along normal with outliers -\nMin: {min(cloud_line)}\t\tMax: {max(cloud_line)}")
	# Get the inner 90% of the points based on their distribution along the normal
	lower_bound = np.percentile(cloud_line, 50-int(m/2))
	upper_bound = np.percentile(cloud_line, 50+int(m/2))
	# Filter the points to keep only the inner 90%
	inner_90_mask = (cloud_line >= lower_bound) & (cloud_line <= upper_bound)
	filtered_points = cloud_array[inner_90_mask]
	#print(f"Positions along normal after cleaning -\nMin: {min(cloud_line[inner_90_mask])}\t\tMax: {max(cloud_line[inner_90_mask])}")
	#print(filtered_points.shape)
	return filtered_points

# Code generated with ChatGPT
def array_to_plane(coordinates):
	
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

if __name__ == '__main__':
	main()
