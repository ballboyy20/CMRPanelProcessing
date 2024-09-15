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


def find_panel_data(cut_flasher_filepath, save_directory, panel_num, save_posttext='_panel_data', verbose=False, graph_clusters=True):
	"""
	This function starts with the filepath of a preprocessed .stl file representing
	panels. Outliers are removed and k-means is used to generate clusters

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
	# TODO: Integrate the ability to read a variety of file types
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
	
	# Make clusters of the points by how they are projected onto the 2D plane of best fit of the whole array
	labels, centroids, ref_points_2d = get_clusters(points, plane, n_clusters=panel_num)
	
	# Optionally visualizing the 2d groupings
	if graph_clusters:
		plot_clusters(ref_points_2d, labels, centroids)
	
	# Produce a .json compatible structure (lists and dictionaries containing only Python-native objects)
	# saving each panel's randomly assigned unique label, center position and normal vector
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
	
		
def planar_outlier_removal(original_points, outlier_axis='norm', target=None, cutoff_scaling=20,
						   step=0.001, iter_limit=20, verbose=False):
	"""
	This will remove points that are outliers by their positions projected onto the normal
	of the plane of best fit. This is repeated iteratively until removing the extreme
	points no longer significantly affects the distribution of the remaining points.

	:param original_points: PyVista object containing all points in the mesh being read
	:param outlier_axis: 'norm' or skspatial.objects.Line object. If left to default 'norm', will remove outliers
		relative to the normal of the plane of best fit. Otherwise, will remove outliers relative to the provided line.
	:param target: Float between 0.0 and 1.0. The program measures the proportion of a) the range of points projected onto the normal of the plane after outliers have been removed,
		to b) the range of those points including the outliers. The iterative program will stop removing points when this proportion exceeds that of 'target'
	:param cutoff_scaling: Positive float value. Alternative option to target, sets the target proportion as 1.0 - cutoff_scaling * step.
		A bigger target value will be easier to reach, while a small target value will only be reached when a very high density of points is located.
	:param step: The combined proportion removed equally from the extremities along the normal of the plane of best fit per pass. As
		long as no specific 'target' value has been passed, smaller steps means outlier removal will take more iterations and be more
		fine (potentially being less greedy while removing outliers, at the risk of being more sensitive to large collections of outliers)
	:param iter_limit: After this many iterations removing outliers, stop, and return what was obtained
	:return final_plane, final_points: The remaining points and the plane of best fit that was generated from them
	"""
	# ^ is the logical NAND operator: True if exactly one of the inputs are True, False otherwise.
	# Either a manual target must be set, or the cutoff_scaling left (or updated)
	if not ((target is None) ^ (cutoff_scaling is None)):
		raise UserWarning("You must provide a target removal percentage, or at least make sure cutoff_scaling is not None.")

	# I added the "cutoff_scaling" option because of a way that I found "target" could break.
	# If the step size was changed to be very small but target wasn't changed at all,
	# removing few enough outliers does sufficiently little to shrink the total range
	# that the target may be instantly met. Making the final_target directly correlated to step size
	# implies that for smaller steps, the target is higher and harder to hit.
	if target:
		final_target = target
	else:
		final_target = 1.0 - (cutoff_scaling * step)
	
	# This does the following to the data:
	#  1) Create a 1d array, of the same height as the passed points, containing their positions projected onto the normal of the plane
	#  2) Remove some outer percentage of points from both extremes (determined by step)
	# TODO: Visualize the distribution of points along the normal of a plane of best fit using a ____ plot
	def remove_outliers(old_points, outlier_line):
		# Project all points of the point cloud onto this line
		cloud_line = outlier_line.transform_points(old_points)
		# Find the total range of values of all points along the normal line
		old_range = max(cloud_line) - min(cloud_line)
		# Get the values of the upper and lower percentile points to create an inner range of values for the points to be kept
		# TODO: update this to use quantile (to remove scaling by a factor of 100)
		lower_bound = np.percentile(cloud_line, 100 * (0.0 + step / 2))
		upper_bound = np.percentile(cloud_line, 100 * (1.0 - step / 2))
		# Find the total range of values of the points within that inner range along the normal line
		new_range = upper_bound - lower_bound
		# Make a active_indices map correlating only those rows for points within the lower and upper bounds along the normal
		inner_mask = (cloud_line >= lower_bound) & (cloud_line <= upper_bound)
		# Apply that active_indices to the point cloud itself
		new_points = old_points[inner_mask]
		return new_points, inner_mask, new_range, old_range
	
	iterations = 0
	prev_points = original_points
	original_range = None
	active_indices = np.arange(original_points.shape[0])
	

	# Iteratively remove the outer percentiles of outliers until this has a sufficiently small effect on the
	# total value range of points as projected along the normal of the plane of best fit
	while True:
		# By default, outliers will be removed along the normal of the plane of best fit of this collection of points
		if outlier_axis == 'norm':
			# Get a line through the plane's center and normal
			prev_plane = iter_best_fit_plane(prev_points)
			removal_line = skobj.Line(point=prev_plane.point, direction=prev_plane.normal)
		# An skspatial.objects.Line object can be provided as an axis for the
		# removal of outliers instead, to remove outliers in other 1D axes
		else:
			assert isinstance(outlier_axis, skobj.Line)
			removal_line = outlier_axis
		# Get the list of new points without outliers, the mask that produced them, and the old and new ranges each.
		filtered_points, this_mask, new_range, old_range = remove_outliers(prev_points, removal_line)
		# We need to keep track of the index positions of these filtered points from the original array where
		# they were stored. The faces in .stl files are groups of index references to vertex positions. Saving
		# indicies allows us to rebuild those original faces (excluding any faces that included outlier points)
		# TODO: Find a way to accound for points that were ignored because they shared a face with an outlier
		active_indices = active_indices[this_mask]
		iterations += 1
		# The first time we solve the range of points, save it
		if original_range is None:
			original_range = copy(old_range)
		# When the ratio of a) the range of this new iteration to b) the range of the previous
		# iteration meets the final target, stop iterating, we have converged on an appropriate
		# solution. OR, if the iteration limit is reached, stop just the same.
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


def get_clusters(point_cloud, best_fit_plane, n_clusters, cluster_similarity=2.5, km_iter_lim=20, center_type='spatial', verbose=True):
	"""
	Observing a set of points, project them into their own plane of best fit, then use
	k-means to identify the point groups selected by the user during preprocessing.
	Verify that all clusters are similarly sized (to a tolerance), then return them.
	
	:param point_cloud: skspatial.objects.Points object or a NumPy array where a list of vertices is provided
	:param best_fit_plane: skspatial.objects.Plane object, for this point cloud
	:param n_clusters: Integer value for the number of groups to search for, passed along to k-means
	:param cluster_similarity: Float value, if the ratio of the largest group to the smallest one is greater than this, k-means will be reinitialized
	:param km_iter_lim: After this many iterations, raise a TimeoutError, since satisfactory clusters couldn't be formed
	:param center_type: 'spatial' to calculate cluster centers as the middle points of the extremes
		in each dimension, 'centroid' to use the centroids produced by the k-means algorithm itself
	:param verbose: If True, give real time data as clusters are formed
	:return: labels, centroids, trans_proj_cloud: For all points, labels about which cluster the point would
		be grouped into. Central positions for each cluster. The 2D transformed coordinates that were used.
	"""
	# Randomly produce x and y vectors that form an orthonormal basis with the normal of the plane of best fit
	b1, b2 = generate_orthonormal_basis(best_fit_plane)
	# Transform the points into a coordinate system with the previously generated unit vectors (and the
	# plane of best fit's normal vector) as bases and the plane of best fit's point as the origin. Discard
	# the z coordinates (out of the plane), effectively projecting the points onto a 2D plane.
	trans_proj_cloud = sktrf.transform_coordinates(point_cloud, best_fit_plane.point, (b1, b2, best_fit_plane.normal))[:, 0:2]
	
	if verbose:
		print(f"Using k-means clustering algorithm, searching for {n_clusters} clusters.")

	if not isinstance(n_clusters, int):
		raise UserWarning("k-means requires an integer for the number of clusters, but you provided ", n_clusters)
	iteration = 1
	# Keep reinitializing k-means until clusters are formed that are close enough in size
	while True:
		if iteration > km_iter_lim:
			raise TimeoutError(f"With {iteration} iterations, surpassed the limit set for k-means. Try adjusting"
							   f"cluster_similarity to expand or shrink the expected similarity in size between"
							   f"clusters, or increasing the allowed number of iterations with km_iter_lim")
		# Initializing
		km_model = KMeans(n_clusters=n_clusters)
		# Fitting the model to our 2D transformed point cloud and grouping it. Makes a 1D array,
		# where each element is an integer, representing the assigned cluster
		# of the point at the corresponding row of the point cloud passed to the model.
		labels = km_model.fit_predict(trans_proj_cloud)
		# Make a k * n shaped array, where n is the number of dimensions of our coordinates,
		# and k is the number of clusters, where each column (n) contains the maximal distance
		# in that dimension for all points of a given cluster (by row, k)
		ranges = np.array([np.ptp(trans_proj_cloud[labels == label], axis=0) for label in np.unique(labels)])
		# Make two n shaped array, respectively containing the smallest and largest cluster sizes in each dimension
		min_ranges = np.min(ranges, axis=0)
		max_ranges = np.max(ranges, axis=0)
		if verbose:
			print("Smallest cluster size (x, y): ", min_ranges)
			print("Largest cluster size (x, y): ", max_ranges)
		# For all dimensions (columns), if the maximum range is less than 120% of the minimum range, the model must have made
		# even groups that catch all panels. When this is done, simply exit the loop and use the latest assigned group values.
		if np.all(max_ranges <= min_ranges * cluster_similarity):
			if verbose:
				print(f"After {iteration} iterations, found clusters such that the largest group's size is within {100*cluster_similarity}% of the that of the smallest group.")
			break
		else:
			if verbose:
				print("There exists a dissimilarity in size greater than the limit of ", 100*cluster_similarity, "%")
				print("Re-initializing k-means to try again...")
			iteration += 1
	# When we've found a satisfactory grouping, get central values for each group.
	centroids_list = []
	for label_i in np.unique(labels):
		cluster_map = (labels == label_i)
		cluster_points = trans_proj_cloud[cluster_map, :]
		# Find the minimums and maximums in each dimension
		min_corner = np.min(cluster_points, axis=0)
		max_corner = np.max(cluster_points, axis=0)
		# The middle point between the minimums and maximums will be our spatial center for this label
		spatial_center = np.mean((min_corner, max_corner), axis=0)
		centroids_list.append(spatial_center)
	centroids = np.array(centroids_list)

	# The k-means model does produce centroids, but their ordering is unknown (but likely the same
	# order as the labels counting up). It also likely leans towards higher densities of points
	# (which I'm now realizing is mostly irrelevant, since in-plane translation of the selected
	# points relative to the real, physical panel is assumed to be imprecise already)
	#centroids = km_model.cluster_centers_

	return labels, centroids, trans_proj_cloud
		

def generate_orthonormal_basis(plane, seed=None):
	"""
	Following function was generated by ChatGPT then simplified to use skspatial objects and methods. It's
	basic purpose is to generate two random vectors that are orthogonal with the normal of our plane.

	:param plane: An skspatial.objects.Plane object
	:param seed: An integer value to get repeatable results
	:return: basis_vector_1, basis_vector_2: Two vectors that are orthogonal with the normal of the input plane
	"""
	if seed is not None:
		np.random.seed(seed)
	# Normalize the normal to our plane (make it a vector with a magnitude of one, a unit vector)
	plane_norm = plane.normal.unit()

	# Generate a completely random vector in 3d
	random_vector = np.random.randn(3)
	# Make sure the random vector is not parallel to the normal vector of our plane
	# (which is basically impossible as a random float, but we like being thorough)
	while np.allclose(np.dot(random_vector, plane_norm), 0):
		random_vector = np.random.randn(3)
	
	# Project the random vector onto the plane to make it orthogonal
	# to the normal vector, then normalize it to a unit vector
	basis_vector_1 = plane.project_vector(skobj.Vector(random_vector)).unit()
	
	# Create the second basis vector by taking the cross product of the normal and the first basis
	basis_vector_2 = skobj.Vector(np.cross(plane_norm, basis_vector_1)).unit()
	
	return basis_vector_1, basis_vector_2
	
	
# Code in this function was partially generated with ChatGPT then modified
def iter_best_fit_plane(coordinates):
	"""
	For a given set of coordinates, cumulatively find the plane of best fit. This iterative method means the
	plane of best fit can be calculated for extremely large sets of points for which the SKSpatial library struggles.

	:param coordinates: A n*x shaped NumPy array containing coordinates for a point cloud
	:return: plane: An skspatial.objects.Plane object, with a center, normal and various methods
	"""
	# The number of points to be processed at once
	chunk_size = 10000
	
	# Initialize the plane parameters
	n_accumulated = 0
	centroid_accumulated = np.zeros(3)
	covariance_accumulated = np.zeros((3, 3))
	
	# Process chunk_size number of points at once, iterating through all points
	for i in range(0, len(coordinates), chunk_size):
		# Get all points for this chunk
		chunk_points = coordinates[i:i + chunk_size]
		# If we're at the end of the set of points, the number of points
		# in this chunk may be smaller than the normal chunk_size
		n_chunk = len(chunk_points)
		
		# This takes a weighted average, judging weight by the number of points contributing to the
		# existing accumulated centroid vs. the number of new points that are now influencing the centroid
		centroid_chunk = np.mean(chunk_points, axis=0)
		centroid_accumulated = (n_accumulated * centroid_accumulated + n_chunk * centroid_chunk) / (n_accumulated + n_chunk)
		
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
	
	return plane


def plot_clusters(points, cluster_labels, centroids=None):
	"""
	This code was written for the original panel comparing algorithm, to graph grouped
	collections of points and (optionally) the corresponding center points of each group

	:param points: An n*x shaped array, where n is the number of points, and x is the dimension of their coordinates
	:param cluster_labels: An n shaped array, where n is the number of points, containing
		k unique values for the cluster label of a point in the corresponding row from 'points'
	:param centroids: A k*x shaped array, where k is the number of clusters, and x is the dimension of their coordinates
	:return: None
	"""
	plt.clf()
	# This color map produces colors along a spectrum for each of the cluster labels
	cmap = plt.get_cmap('plasma', np.unique(cluster_labels).shape[0])
	for cluster_i in range(min(cluster_labels), max(cluster_labels) + 1):
		# Get all points in this cluster
		cluster_map = (cluster_labels == cluster_i)
		this_group = points[cluster_map, :]
		# Graph all points in this cluster, using the i_th color in the color map
		plt.scatter(this_group[:, 0], this_group[:, 1], marker='o', c=cmap(cluster_i))
		# If cluster centroids were provided, graph them as large Xs
		if centroids is not None:
			plt.plot(centroids[cluster_i, 0], centroids[cluster_i, 1], marker='x', c=cmap(cluster_i), markersize=30)
	plt.show()


if __name__ == '__main__':
	main()
	#find_panels(r"C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\warped flasher\mesh_cut.stl", 27)
