"""
Code written by Trevor K. Carter for Python 3.11, started on 7-5-2024

Ideas to implement:
Use k-means to group, manually input the number of groups during the pre-scan procedure, and use the k-means++ initialization method.
	OR use HDBSCAN to automatically group them, but it is likely slower and struggles with density
TODO: After groups are made, individually remove outliers per group, using that group's plane of best fit
TODO: Use adjusted rand index for k-means to help find the good initializations after a few loops
TODO: Consider implementing RANSAC algorithm in place of plane of best fit

Universal Panel Pre-Processing:
	1) Iteratively find the plane of best fit for the entire scan
		1.1) Find the plane of best fit for all points
		1.2) Remove the outlier points by their position along the normal to that plane of best fit
		1.3) Redefine the plane of best fit without the outliers
		1.4) Repeat until total range variance between iterations reaches a significantly lower proportion than that of the points being removed.
	2) Use K-Means to make groups for each of the panel surfaces, manually inputting the number of panels.
		2.1) Run K-Means once to get grouped points and centroids
		2.2) Verify that the largest group isn't significantly bigger (in world coordinates)
			than the smallest group. If it is bigger, a group covering	two panels was
			must have somehow been created, so reinitialize K-Means and repeat step 2.1
		2.3) When groups are created, remove outliers within each group
		2.4) Save the central positions (mean of min/max) of each group,
			along with the normal vector of the plane of that group to a json

"""


from copy import copy
from json import dumps as json_dumps
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

import pyvista as pv
import skspatial.objects as skobj
import skspatial.transformation as sktrf
from sklearn.cluster import KMeans, HDBSCAN


def main():
	warped_flasher_file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\warped flasher\mesh_cut.stl'
	warped_flasher_data_directory = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\warped flasher'
	
	find_panel_data(warped_flasher_file, warped_flasher_data_directory, 27, save_posttext='_panelData')


# High level wrapper function where the various steps are executed
def find_panel_data(cut_flasher_filepath, save_directory, panel_num, save_posttext='_panel_data', verbose=False, graph_clusters=True):
	"""
	:param cut_flasher_filepath: Filepath to a .stl file containing uniformly sized selections from flat_scans surfaces on panels of a panel array
	:param save_directory: None or Directory path to which the centers and normals of the panel groups from the panel array will be saved
	:param panel_num: The number of point selection groups, normally representing panels, which will be used by k-means
	:param save_posttext: text that will be added to the end of the name of the original panel array file for the .json file containing its centers and normals
	:param verbose: set to True to print values as they are calculated in real time
	:param graph_clusters: set to True to graph the clustered points in 2d when they are grouped
	:return: A python dictionary with keys for (arbitrary) panel labels corresponding
		to sub-dictionaries, with the keys "center" and "normal", containing the center
		position of this panel and the normal to the best-fit plane for that group of points.
	"""
	# Generate a list of points from the .stl file vertices
	mesh_points = pv.read(cut_flasher_filepath).points
	# After iteratively removing outliers, save the plane of best fit and points
	plane, points, kept_point_indices, iterations, original_range, new_range = planar_outlier_removal(mesh_points, cutoff_scaling=20, step=0.0005, verbose=True)
	
	if verbose:
		print("Iterations: ", iterations)
		print("Original Range: ", original_range)
		print("New Range: ", new_range)
		print("Final Range Percentage: ", 100*new_range/original_range, "%")
		print("Original Size: ", mesh_points.shape)
		print("New Size: ", points.shape)
		print("Removed Points: ", mesh_points.shape[0] - points.shape[0])
		print("Removed Point Percentage: ", 100-100*(mesh_points.shape[0] - points.shape[0])/mesh_points.shape[0], "%")
	
	# Make clusters, considering the points if they were to be projected onto the plane of best fit of the whole array
	labels, centroids, ref_points_2d = get_clusters(points, plane, n_clusters=panel_num)
	
	# Optionally visualizing the 2d groupings
	if graph_clusters:
		plot_clusters(ref_points_2d, labels, centroids)
	
	# Produce a .json compatible structure (lists and dictionaries) saving each panel's arbitrary label, center and normal
	cluster_data = format_clusters(labels, points, plane.normal)
	
	if save_directory is not None:
		# This will match to the name of the input panel array file (without its extension)
		name_extension_pattern = r'([^\\\/]+)(\.[^\\\/]+$)'
		file_match = re.search(name_extension_pattern, cut_flasher_filepath)
		if not file_match:
			raise NameError("There was an issue while extracting the name of this")
		panel_array_name = file_match[1]
		# This will be the path of the saved .json with panel array data
		cluster_filepath = save_directory+'\\'+panel_array_name+save_posttext+'.json'
		if verbose:
			print(cluster_filepath)
		
		# Writing to a .json
		with open(cluster_filepath, "w") as outfile:
			outfile.write(json_dumps(cluster_data, indent=2))
		
	return cluster_data


def format_clusters(cluster_labels, point_cloud, main_vector):
	cluster_data = {}
	for label_i in np.unique(cluster_labels):
		cluster_map = (cluster_labels == label_i)
		cluster_points = point_cloud[cluster_map, :]
		final_plane, final_points, active_indices = planar_outlier_removal(cluster_points)
		# TODO: Also remove outliers in the x and y directions
		min_corner = np.min(final_points, axis=0)
		max_corner = np.max(final_points, axis=0)
		spatial_center = np.mean((min_corner, max_corner), axis=0)
		cluster_center = final_plane.project_point(spatial_center)
		cluster_normal = final_plane.normal.unit()
		# Verify that every normal is facing the same direction as the main vector
		if np.dot(cluster_normal, main_vector) < 0:
			cluster_normal = - cluster_normal
		cluster_data[str(label_i)] = {'center': tuple(cluster_center), 'normal': tuple(cluster_normal)}
		
	return cluster_data
	
		


# This will remove outlier points, relative to the normal of the plane of
# best fit, until the range of refined points has more than the target
# proportion of the previous iteration (i.e. when removing the extreme
# points is no longer significantly affecting the point distribution
# along that direction)
def planar_outlier_removal(original_points, outlier_axis='norm', target=None, cutoff_scaling=20,
						   step=0.001, iter_limit=20, verbose=False):
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
		raise UserWarning("You must provide a target removal percentage, or at make sure cutoff_scaling is not None.")
	
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
	#  2) Return only the points corresponding to some inter % of the points, ordered along the normal
	def remove_outliers(old_points, outlier_line):
		# Project all points of the point cloud onto this line
		cloud_line = outlier_line.transform_points(old_points)
		# Get the total range of values along this line
		old_range = max(cloud_line) - min(cloud_line)
		# Get the minimum and maximum values of the inner range for points to be cut off
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
	active_indices = np.arange(original_points.shape[0])
	
	# Iteratively remove outliers until the ratio of the new range
	# to the old range for this iteration meets the final target
	while True:
		# By default, outliers will be removed along the normal of the plane of best fit of this collection of points
		if outlier_axis == 'norm':
			# Get a line through the plane's center and normal
			prev_plane = iter_best_fit_plane(prev_points)
			removal_line = skobj.Line(point=prev_plane.point, direction=prev_plane.normal)
		# An skspatial.objects.Line object can be provided as an axis for the removal of outliers instead
		else:
			assert isinstance(outlier_axis, skobj.Line)
			removal_line = outlier_axis
		# Get the list of points, the mask that produced them, and the old and new ranges.
		filtered_points, this_mask, new_range, old_range = remove_outliers(prev_points, removal_line)
		# We need to keep track of the original positions of these filtered points, such
		# that the faces of the original shape can still be referenced correctly at the end of this whole process
		active_indices = active_indices[this_mask]
		iterations += 1
		# The first time we solve the range of points, save it (for reference of before vs. after removing outliers)
		if original_range is None:
			original_range = copy(old_range)
		# If the removal of these outliers has a sufficiently small effect on the range of points
		# along the normal of the plane such that it meets the target, stop iterating, we have
		# converged on an appropriate solution. OR, if the iteration limit is reached, stop just the same.
		if (new_range / old_range > final_target) or (iterations >= iter_limit):
			final_plane = iter_best_fit_plane(filtered_points)
			final_points = filtered_points
			break
		# If we have not converged, make the point cloud resulting from this iteration the previous cloud,
		# and remove outliers from it again. Repeat until target (or iteration limit) is reached.
		else:
			prev_points = filtered_points
	
	if verbose:
		return final_plane, final_points, active_indices, iterations, original_range, new_range
	else:
		return final_plane, final_points, active_indices


# This will sort through the complete list of points to form some number of groups
def get_clusters(point_cloud, best_fit_plane, n_clusters, cluster_similarity=2.5, km_iter_lim=20):
	"""
	
	:param point_cloud: skspatial.objects.Points object or a NumPy array where a list of vertices is provided
	:param best_fit_plane: skspatial.objects.Plane object, for this point cloud
	:param n_clusters: Integer value for the number of groups to search for, passed along to k-means
	:param cluster_similarity: Float value, if the ratio of the largest group to the smallest one is greater than this, k-means will be reinitialized
	:param km_iter_lim: After this many iterations, raise a TimeoutError, since a satisfactory solution could not be reached
	:return: labels, centroids, trans_proj_cloud: For all points, labels about which cluster the point would
		be grouped into. Central positions for each cluster. The 2D transformed coordinates that were used.
	"""
	# Randomly produce the x and y components of an orthonormal basis with the normal of the plane of
	b1, b2 = generate_orthonormal_basis(best_fit_plane)
	# Transform the points into a coordinate system using the above unit vectors (and the best fit plane's normal vector) as bases.
	trans_proj_cloud = sktrf.transform_coordinates(point_cloud, best_fit_plane.point, (b1, b2, best_fit_plane.normal))[:, 0:2]
	
	print(f"Using k-means clustering algorithm, searching for {n_clusters} clusters.")
	if not isinstance(n_clusters, int):
		raise UserWarning("k-means requires an integer for the number of clusters, but you provided ", n_clusters)
	iteration = 1
	# Keep reinitializing k-means if the cluster sizes are too different
	while True:
		if iteration > km_iter_lim:
			raise TimeoutError(f"With {iteration} iterations, surpassed the limit set for k-means. Try adjusting"
							   f"cluster_similarity to expand or shrink the expected similarity in size between"
							   f"clusters, or increasing the allowed number of iterations with km_iter_lim")
		# Initializing, fitting and making predictions about labels with the model
		km_model = KMeans(n_clusters=n_clusters)
		# Make a k shaped array, where each value is an integer, representing the
		# label for a point at the corresponding row of points passed to the model
		labels = km_model.fit_predict(trans_proj_cloud)
		# Make a k * n shaped array, where n is the number of dimensions of our coordinates,
		# and k is the number of groups, where each column (n) contains the maximal distance
		# between points of a given group (by row, k) in that particular dimension
		ranges = np.array([np.ptp(trans_proj_cloud[labels == label], axis=0) for label in np.unique(labels)])
		# Make two n shaped array, respectively containing the smallest and largest group size values of each dimension
		min_ranges = np.min(ranges, axis=0)
		max_ranges = np.max(ranges, axis=0)
		print("Smallest cluster size (x, y): ", min_ranges)
		print("Largest cluster size (x, y): ", max_ranges)
		# For all dimensions (columns), if the maximum range is less than 120% of the minimum range, the model must have made even groups that catch all panels.
		if np.all(max_ranges <= min_ranges * cluster_similarity):
			print(f"After {iteration} iterations, found clusters such that the largest group's size is within {100*cluster_similarity}% of the that of the smallest group.")
			break
		else:
			print("There exists a dissimilarity in size greater than the limit of ", 100*cluster_similarity, "%")
			print("Re-initializing k-means to try again...")
			iteration += 1
	# The k-means model also generates centers for each cluster. When we've found a satisfactory grouping, get the centroids of each group.
	#TODO: Verify that the centroids are NOT weighted by number of points
	#centroids = km_model.cluster_centers_
	centroids_list = []
	for label_i in np.unique(labels):
		cluster_map = (labels == label_i)
		cluster_points = trans_proj_cloud[cluster_map, :]
		min_corner = np.min(cluster_points, axis=0)
		max_corner = np.max(cluster_points, axis=0)
		spatial_center = np.mean((min_corner, max_corner), axis=0)
		centroids_list.append(spatial_center)
	centroids = np.array(centroids_list)
	return labels, centroids, trans_proj_cloud
		

	
# Following function was generated by ChatGPT then simplified to use skspatial objects and
# methods. It's basic purpose is to generate two vectors that are orthogonal with the normal
# of our plane, and thus in the plane, such that they can act as bases for that plane.
def generate_orthonormal_basis(plane, seeded=None):
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
		#print("\n\n\nIteration ", i)
		#print(n_accumulated)
		#print(centroid_accumulated)
		#print(n_chunk)
		#print(centroid_chunk)
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


# This code was written for the original panel comparing algorithm, to graph grouped
# collections of points and (optionally) the corresponding center points of each group
def plot_clusters(dataset, cluster_labels, centroids=None):
	plt.clf()
	#color = plt.cm.rainbow(np.linspace(0, 1, cluster_labels.unique.shape[0]))
	cmap = plt.get_cmap('plasma', np.unique(cluster_labels).shape[0])
	for group_i in range(min(cluster_labels), max(cluster_labels) + 1):
		cluster_map = (cluster_labels == group_i)
		this_group = dataset[cluster_map, :]
		plt.scatter(this_group[:, 0], this_group[:, 1], marker='o', c=cmap(group_i))
		if centroids is not None:
			plt.plot(centroids[group_i, 0], centroids[group_i, 1], marker='x', c=cmap(group_i), markersize=30)
	plt.show()


if __name__ == '__main__':
	main()
	#find_panels(r"C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\warped flasher\mesh_cut.stl", 27)
	