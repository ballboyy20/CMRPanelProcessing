from math import degrees, radians, isclose, tan, pi
import numpy as np


import pyvista as pv

import skspatial.objects as skobj
import skspatial.transformation as sktrf

import matplotlib.pyplot as plt

"""
Code written by Trevor Kent Carter, starting June 19th, 2024

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

		
"""

def main():
	slet_4_left = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\_Big Folder of Files for Trevor\3-arm SLET - mesh4 left.stl'
	slet_4_right = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\_Big Folder of Files for Trevor\3-arm SLET - mesh4 right.stl'
	slet_5_left = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\_Big Folder of Files for Trevor\3-arm SLET - mesh5 left.stl'
	slet_5_right = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\_Big Folder of Files for Trevor\3-arm SLET - mesh5 right.stl'
	slet_5_left = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\Mesh Cuts (by person)\_Big Folder of Files for Trevor\3-arm SLET - mesh1 left.stl'
	slet_5_right = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\Mesh Cuts (by person)\_Big Folder of Files for Trevor\3-arm SLET - mesh1 right.stl'
	cymaldo_3_left = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\_Big Folder of Files for Trevor\Cymaldo - mesh3 left.stl'
	cymaldo_3_right = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\_Big Folder of Files for Trevor\Cymaldo - mesh3 right.stl'
	cymaldo_4_left = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\_Big Folder of Files for Trevor\Cymaldo - mesh4 left.stl'
	cymaldo_4_right = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\_Big Folder of Files for Trevor\Cymaldo - mesh4 right.stl'



	
	fig = plt.figure()
	ax = fig.add_subplot(121, projection='3d')
	ax.view_init(elev=15, azim=-170)
	ax2 = fig.add_subplot(122, projection='3d')
	ax2.view_init(elev=15, azim=-150)
	
	#slet_4_data = compare_panels(slet_4_left, slet_4_right, verbose=True,  raw_graph=ax, trf_graph=ax)
	slet_5_data = compare_panels(slet_5_left, slet_5_right, verbose=True, raw_graph=ax, trf_graph=ax2)
	#cymaldo_3_data = compare_panels(cymaldo_3_left, cymaldo_3_right, verbose=True, raw_graph=ax, raw_cloud=True, trf_graph=ax2)
	
	#print(slet_4_data)
	print(slet_5_data)
	#print(cymaldo_3_data)

	ax.set_aspect('equal', adjustable='box')
	
	plt.show()

def compare_panels(left_file, right_file, verbose=False, raw_graph=None, raw_cloud=False, trf_graph=None):
	# Converting 3d files into NumPy arrays with PyVista
	left_cloud_raw = mesh_to_array(left_file)
	right_cloud_raw = mesh_to_array(right_file)
	if verbose:
		print('\nleft_cloud_raw: ', left_cloud_raw)
		print('right_cloud_raw: ', right_cloud_raw)
	
	# Converting NumPy point clouds into skspatial planes with origins and normals
	left_plane_raw = iter_best_fit_plane(left_cloud_raw)
	right_plane_raw = iter_best_fit_plane(right_cloud_raw)
	if verbose:
		print('\nleft_plane_raw: ', left_plane_raw)
		print('right_plane_raw: ', right_plane_raw)
	
	# Find the plane of best fit, and project all points onto the normal of the plane of best fit. Use that distribution to remove outliers.
	left_cloud_clean = remove_outliers(left_cloud_raw, left_plane_raw, m=95)
	right_cloud_clean = remove_outliers(right_cloud_raw, right_plane_raw, m=95)
	if verbose:
		print('\nleft_cloud_clean: ', left_cloud_clean)
		print('right_cloud_clean: ', right_cloud_clean)
	
	# Resolve for the planes now that we've gotten rid of outliers
	left_plane_clean = iter_best_fit_plane(left_cloud_clean)
	right_plane_clean = iter_best_fit_plane(right_cloud_clean)
	if verbose:
		print('\nleft_plane_clean: ', left_plane_clean)
		print('right_plane_clean: ', right_plane_clean)
	
	# Getting the centers of the cuboids containing the left and right point clouds and planes each,
	# then generating a plane (parallel with the existing one) where that is the plane's center
	left_plane_centered = get_center_plane(left_cloud_clean, left_plane_clean.normal)
	right_plane_centered = get_center_plane(right_cloud_clean, right_plane_clean.normal)
	if verbose:
		print('\nleft_plane_centered: ', left_plane_centered)
		print('right_plane_centered: ', right_plane_centered)
	if raw_graph is not None:
		#left_plane_centered.plot_3d(raw_graph, lims_x=[-20, 20], lims_y=[-20, 20], lims_z=[-20, 20], alpha=0.5, color='c')
		#right_plane_centered.plot_3d(raw_graph, lims_x=[-20, 20], lims_y=[-20, 20], lims_z=[-20, 20], alpha=0.5, color='m')
		plot_circle(raw_graph, left_plane_centered.point, left_plane_centered.normal, radius=40, c='c')
		plot_circle(raw_graph, right_plane_centered.point, right_plane_centered.normal, radius=40, c='m')
		left_plane_centered.normal.plot_3d(raw_graph, scalar=20, point=left_plane_centered.point, c='c')
		right_plane_centered.normal.plot_3d(raw_graph, scalar=20, point=right_plane_centered.point, c='m')
		if raw_cloud is True:
			skobj.Points(left_cloud_raw).plot_3d(raw_graph, s=0.03, alpha=0.01, c='c')
			skobj.Points(right_cloud_raw).plot_3d(raw_graph, s=0.03, alpha=0.01, c='m')
	
	# Getting the line from the left plane's center to the projection of the right plane's center
	ltr_line = c_to_proj(left_plane_centered, right_plane_centered.point)
	# The y axis will be based on the unit vector opposite the direction of the line described above
	basis_y_vec = -1*ltr_line.direction.unit()
	# The z axis will be based on the normal of the left plane
	basis_z_vec = left_plane_centered.normal.unit()
	# The x axis will be based on a vector perpendicular to the other two
	basis_x_vec = basis_y_vec.cross(basis_z_vec).unit()
	# These three form an orthogonal basis
	if verbose:
		print('\nltr_line: ', ltr_line)
		print('basis_x_vec: ', basis_x_vec)
		print('basis_y_vec: ', basis_y_vec)
		print('basis_z_vec: ', basis_z_vec)
	if raw_graph is not None:
		ltr_line.direction.plot_3d(raw_graph, point=ltr_line.point, scalar=0.5, c='c')
		basis_x_vec.plot_3d(raw_graph, point=left_plane_centered.point, scalar=20, c='r')
		basis_y_vec.plot_3d(raw_graph, point=left_plane_centered.point, scalar=20, c='g')
		basis_z_vec.plot_3d(raw_graph, point=left_plane_centered.point, scalar=20, c='b')
	
	# Transform the original point clouds such that they use the orthogonal
	# basis derived above, with the left point's center as the origin
	left_cloud_trf = sktrf.transform_coordinates(left_cloud_clean, left_plane_centered.point, vectors_basis=(basis_x_vec, basis_y_vec, basis_z_vec))
	right_cloud_trf = sktrf.transform_coordinates(right_cloud_clean, left_plane_centered.point, vectors_basis=(basis_x_vec, basis_y_vec, basis_z_vec))
	if verbose:
		print('\nleft_cloud_trf: ', left_cloud_trf)
		print('right_cloud_trf: ', right_cloud_trf)

	# Transform the original central positions such that they use the orthogonal
	# basis derived above, with the left point's center as the origin
	# (left_center_trf is redundant, and should always be zero)
	left_center_trf = sktrf.transform_coordinates(left_plane_centered.point, left_plane_centered.point, vectors_basis=(basis_x_vec, basis_y_vec, basis_z_vec))
	right_center_trf = sktrf.transform_coordinates(right_plane_centered.point, left_plane_centered.point, vectors_basis=(basis_x_vec, basis_y_vec, basis_z_vec))
	if verbose:
		print('\nleft_center_trf: ', left_center_trf)
		print('right_center_trf: ', right_center_trf)
	if trf_graph is not None:
		skobj.Vector((1, 0, 0)).plot_3d(trf_graph, scalar=100, c='r')
		skobj.Vector((0, 1, 0)).plot_3d(trf_graph, scalar=100, c='g')
		skobj.Vector((0, 0, 1)).plot_3d(trf_graph, scalar=100, c='b')
	
	# Manually transforming the end points of the planes' normal vectors with our transformation matrix.
	left_norm_trf = skobj.Vector(sktrf.transform_coordinates(left_plane_centered.normal, (0,0,0), vectors_basis=(basis_x_vec, basis_y_vec, basis_z_vec)))
	# The left plane's transformed normal is gauranteed to be a Vector directly (or approximately
	# by computational error) parallel with the z axis, so we could represent it as such.
	#left_norm_trf = skobj.Vector((0, 0, 1))
	right_norm_trf = skobj.Vector(sktrf.transform_coordinates(right_plane_centered.normal, (0,0,0), vectors_basis=(basis_x_vec, basis_y_vec, basis_z_vec)))
	# We must gaurantee that the right plane's normal is always in the general direction of the z axis, or this will
	# create inconsistencies with cross products down the line (particularly impacting the calculation of theta_y
	if right_norm_trf[2] <= 0:
		right_norm_trf = -right_norm_trf
	# Reforming our planes of best fit from the transformed center and normal
	left_plane_trf = skobj.Plane(point=left_center_trf, normal=left_norm_trf)
	right_plane_trf = skobj.Plane(point=right_center_trf, normal=right_norm_trf)
	if verbose:
		print('\nleft_plane_trf: ', left_plane_trf)
		print('right_plane_trf: ', right_plane_trf)
	if trf_graph is not None:
		#left_plane_trf.plot_3d(trf_graph, lims_x=[-80, 80], lims_y=[-40, 40], alpha=0.5, color='c')
		#right_plane_trf.plot_3d(trf_graph, lims_x=[-80, 80], lims_y=[-40, 40], alpha=0.5, color='m')
		left_norm_trf.plot_3d(trf_graph, point=left_center_trf, scalar=5, c='c')
		right_norm_trf.plot_3d(trf_graph, point=right_center_trf, scalar=5, c='m')
	
	# Using various methods contained in this function to solve the relative angles
	# in degrees, and the relative translation in the units of these models
	theta_x, theta_y, delta_z = solve_positions(left_plane_trf, right_plane_trf, verbose=verbose, trf_graph=trf_graph)
	if verbose:
		print('\ntheta_x: ', theta_x)
		print('theta_y: ', theta_y)
		print('delta_z: ', delta_z)
	return theta_x, theta_y, delta_z



def solve_positions(ground_plane, ref_plane, verbose=False, trf_graph=None):
	ground_center = ground_plane.point
	ref_center = ref_plane.point
	# Reference center projected onto the ground plane
	rc_on_gp = ground_plane.project_point(ref_center)
	# Line from ground center to the reference center's projection. It must be a line,
	# not a vector, in order to solve the intersection with the reference plane.
	#gc_to_rcproj = skobj.Line.from_points(ground_center, rc_on_gp)
	# Since our y axis was already defined by the line described above, we could just use any line along the y axis
	gc_to_rcproj = skobj.Line(point=(0, -1, 0), direction=(0, -1, 0))
	# Reference plane intersection point with line from ground center to reference center projection
	rp_intersect = ref_plane.intersect_line(gc_to_rcproj)
	# Line between reference center and the reference plane intersection (described above).
	# It must always be pointing towards negative y (away from the ground plane), so we check if the
	# intersection point lies further from the origin than the projection of the reference
	# plane center. If it's further, we draw the line from the reference center to the
	# intersection point, the other way around if the intersection point is closer.
	#if rp_intersect.distance_point((0,0,0)) > rc_on_gp.distance_point((0,0,0)):
	# The above method could create issues when rp_intersect was further from the origin in the positive
	# y direction than rc_on_gp in the negative y direction. This was found when running this code for
	# an extremely flat sheet of metal that produced much tigheter tolerances than normal use cases.
	if rp_intersect[1] < rc_on_gp[1]:
		rc_to_gcl = skobj.Vector.from_points(ref_center, rp_intersect)
	else:
		rc_to_gcl = skobj.Vector.from_points(rp_intersect, ref_center)
	theta_x = gc_to_rcproj.direction.angle_between(rc_to_gcl)
	# If the direction of rc_to_gcl in the z direction is negative, then theta_x should also
	# be negative. The initial calculation for theta x above cannot account for this.
	if rc_to_gcl[2] <= 0:
		theta_x = -theta_x
	if verbose:
		print('\ngc_to_rcproj: ', gc_to_rcproj)
		print('rc_to_gcl: ', rc_to_gcl)
		print('theta_x: ', theta_x)
	if trf_graph is not None:
		gc_to_rcproj.direction.unit().plot_3d(trf_graph, scalar=50, point=gc_to_rcproj.point, c='c')
		rc_to_gcl.unit().plot_3d(trf_graph, scalar=50, point=ref_center, c='m')
	
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
	if verbose:
		print('\nground_orth: ', ground_orth)
		print('ref_orth: ', ref_orth)
		print('theta_y: ', theta_y)
	if trf_graph is not None:
		ground_orth.unit().plot_3d(trf_graph, scalar=50, point=ground_center, c='c')
		ref_orth.unit().plot_3d(trf_graph, scalar=50, point=ref_center, c='m')
	
	z_rc = ref_center[2]
	y_rc = ref_center[1]
	assert isclose(ref_center[0], 0.0, abs_tol=1e-8)
	delta_z = abs(z_rc) - abs(y_rc) * tan(theta_x/2)
	if verbose:
		print('\nz_rc: ', z_rc)
		print('y_rc: ', y_rc)
		print('delta_z: ', delta_z)
	
	return theta_x, theta_y, delta_z


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
	lower_bound = np.percentile(cloud_line, 50-int(m/2))
	upper_bound = np.percentile(cloud_line, 50+int(m/2))
	# Filter the points to keep only the inner 90%
	inner_90_mask = (cloud_line >= lower_bound) & (cloud_line <= upper_bound)
	filtered_points = cloud_array[inner_90_mask]
	if verbose:
		print("\ncloud_array.shape: ", cloud_array.shape)
		print(f"Positions along normal with outliers -\nMin: {min(cloud_line)}\t\tMax: {max(cloud_line)}")
		print(f"Positions along normal after cleaning -\nMin: {min(cloud_line[inner_90_mask])}\t\tMax: {max(cloud_line[inner_90_mask])}")
		print("filtered_points.shape: ", filtered_points.shape)
		print(f"Total cleaned points: {filtered_points.shape[0]-cloud_array.shape[0]} - {m}% of inner points kept")
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
	#ax.scatter(*center, color=c, label='Center')
	
	# Plot the normal vector for reference
	#normal_end = center + normal
	#ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], length=radius, color=c,
	#		  label='Normal Vector')



if __name__ == '__main__':
	main()
