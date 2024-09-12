from math import degrees

import pyvista as pv
import numpy as np

from skspatial.objects import Plane, Points, Line, Vector
from skspatial.plotting import plot_3d

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def main():
	file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\mesh1.stl'
	
	left_panel = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\mesh1selection_left.stl'
	right_panel = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\demo\demo_meshes\mesh1selection_right.stl'
	
	left_points = mesh_to_points(left_panel)
	right_points = mesh_to_points(right_panel)
	
	#plot_solutions(points=[left_points, right_points])
	
	#print(left_points)
	#print(right_points)
	
	left_plane = array_to_plane(left_points)
	right_plane = array_to_plane(right_points)
	
	print(left_plane)
	print(right_plane)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev=-52, azim=25)
	ax.set_box_aspect([1, 1, 1])
	
	plot_solutions(ax, planes=[left_plane, right_plane])
	#plot_solutions(ax, points=[left_points, right_points], planes=[left_plane, right_plane])
	
	
	planar_intersection = left_plane.intersect_plane(right_plane)
	planar_intersection.direction.plot_3d(ax, (left_plane.point + right_plane.point)/2, scalar=7000)
	
	lp_to_intersect = Line.from_points(left_plane.point, planar_intersection.project_point(left_plane.point))
	rp_to_intersect = Line.from_points(right_plane.point, planar_intersection.project_point(right_plane.point))
	
	intersect_angle = 180 - degrees(lp_to_intersect.direction.angle_between(rp_to_intersect.direction))
	print("Intersect Angle: ", intersect_angle)
	
	# Get normal vectors of the planes
	left_normal = left_plane.normal
	right_normal = right_plane.normal
	
	# Example usage
	plot_circle(ax, left_plane.point, left_normal, radius=20)
	plot_circle(ax, right_plane.point, right_normal, radius=20)
	
	planar_angle = degrees(left_normal.angle_between(right_normal))
	
	print("Plane Normal Angle: ", planar_angle)
	
	# the point where the left plane intersects with the right normal
	lp_intersect_rn_vec = Vector.from_points(left_plane.point, left_plane.intersect_line(Line.from_points(right_plane.point, right_normal)))
	#lp_to_rp_vec = Vector.from_points(left_plane.point, right_plane.point)
	rp_intersect_ln_vec = Vector.from_points(right_plane.point, right_plane.intersect_line(Line.from_points(left_plane.point, left_normal)))
	
	lp_intersect_rn_vec.plot_3d(ax, point=left_plane.point, scalar=1.5, c='m')
	rp_intersect_ln_vec.plot_3d(ax, point=left_plane.point, scalar=-1.5, c='y')
	
	bend_angle = 180 - degrees(lp_intersect_rn_vec.angle_between(rp_intersect_ln_vec))
	
	print("Bend Angle: ", bend_angle)
	
	ln_to_lpint = degrees(left_normal.angle_between(lp_to_intersect.direction))
	print("Left normal to line to intersect: ", ln_to_lpint)

	plt.show()
	
	
def plot_solutions(ax, points=None, planes=None, vectors=None, colors=None, show=False):
	if colors is None:
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
	col_counter = 0
	
	if points is not None:
		if isinstance(points, list):
			for point_listed in points:
				point_listed.plot_3d(ax, s=0.05, alpha=0.01, c=colors[col_counter])
				col_counter += 1
			col_counter = 0
		else:
			points.plot_3d(ax, s=0.1, alpha=0.05, c=colors[0], depthshade=False)
	if planes is not None:
		if isinstance(planes, list):
			for plane_listed in planes:
				plane_listed.plot_3d(ax, lims_x=[-2, 2], lims_y=[-20, 20], alpha=0.5, color=colors[col_counter])
				plane_listed.point.plot_3d(ax, s=50, c=colors[col_counter])
				plane_listed.normal.plot_3d(ax, point=plane_listed.point, scalar=20, c='b')
				col_counter += 1
			col_counter = 0
		else:
			planes.plot_3d(ax, alpha=1)
			
	if show:
		plt.show()


# ChatGPT generated and modified code
def plot_circle(ax, center, normal, radius=1, num_points=100):
	# Normalize the normal vector
	normal = Vector(normal).unit()
	
	# Create a basis on the plane
	if np.allclose(normal, [0, 0, 1]):
		u = Vector([1, 0, 0])
	else:
		u = Vector([0, 0, 1]).cross(normal).unit()
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
	ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], label='Circle')
	ax.scatter(*center, color='red', label='Center')
	
	# Plot the normal vector for reference
	normal_end = center + normal
	ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], length=radius, color='blue',
			  label='Normal Vector')


def mesh_to_points(filename):
	
	# Load the mesh
	mesh = pv.read(filename)
	
	# Get the vertices as a NumPy array
	vertices = np.array(mesh.points)
	
	return Points(vertices)

def array_to_plane(points):
	
	chunk_size = 10000  # Adjust as per your memory constraints and data size
	
	# Initialize the plane parameters
	n_accumulated = 0
	centroid_accumulated = np.zeros(3)
	covariance_accumulated = np.zeros((3, 3))
	
	# Process points in chunks
	for i in range(0, len(points), chunk_size):
		chunk_points = points[i:i+chunk_size]
		n_chunk = len(chunk_points)
		
		# Accumulate centroid
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
	plane = Plane(normal=normal, point=centroid_accumulated)
	
	# Now 'plane' contains the fitted plane using incremental updates
	
	return plane
	
if __name__ == '__main__':
	main()
