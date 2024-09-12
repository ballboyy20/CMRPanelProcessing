"""
Code written by Trevor K. Carter near the August 13th, 2024

This was an exploratory attempt at using a .svg file whose lines traced
the hinges of a flasher, where points included on the .svg file
correspond to the panel selections for the auto-labeler, such that
one could accurately evaluate rotation about hinges, rather than just
the angular differences between the normal vectors of each panel.

"""


import ezdxf
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def main():
	#vector_file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\Flasher_Meta_Optics_Rev_1_Feb_23.DXF'
	vector_file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\panel_array_super-processer\v1\references\POINTS Flasher_Meta_Optics_Rev_1_Feb_23.dxf'
	
	with open(vector_file, 'r') as file_object:
		doc = ezdxf.read(file_object)
	
	for layout in doc.layouts:
		print(layout)
		for paperspace in layout:
			print('\t\t', paperspace)
	
	# This will get any isolated vertices and all lines stored in the svg.
	og_vertices, og_lines = og_extract_lines_verts(doc)
	print(og_lines)
	print(og_vertices)
	og_vert_array = np.array(og_vertices)
	print(og_vert_array)
	print(og_vert_array.shape)
	sns.scatterplot(x=og_vert_array[:, 0], y=og_vert_array[:, 1])
	plt.show()
	
	# Code inspired from this post: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
	def ccw(A, B, C):
		return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
	# Return true if line segments AB and CD intersect
	def intersect(A, B, C, D):
		return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
	
	
	
def og_extract_lines_verts(doc):
	# Initialize lists to hold vertices and lines
	vertices = []
	lines = []
	# Iterate through all entities in modelspace
	for entity in doc.modelspace():
		# If it's a line...
		if entity.dxftype() == 'LINE':
			# ... add it to that list as a tuple of start point coordinates and end point coordinates
			start_point = entity.dxf.start
			end_point = entity.dxf.end
			lines.append((start_point, end_point))
			#vertices.append(start_point)
			#vertices.append(end_point)
		# Otherwise, if it's a polyline, which can apparently represent isolated points...
		elif entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']:
			# ... add the isolated points to our list.
			# WARNING: I don't know what other things polylines may represent or include.
			polyline_points = entity.get_points('xyb')
			for point in polyline_points:
				vertices.append(point)
			# if entity.is_closed:
			# 	lines.extend([(polyline_points[i], polyline_points[(i + 1) % len(polyline_points)]) for i in
			# 				  range(len(polyline_points))])
			# else:
			# 	lines.extend([(polyline_points[i], polyline_points[i + 1]) for i in range(len(polyline_points) - 1)])
	return vertices, lines


def OLDextract_vertex_and_line_data(doc):
	# Get the modelspace
	msp = doc.modelspace()
	
	# Initialize lists to hold vertices and lines
	vertices = set()
	line_points = set()
	lines = []
	
	# Iterate through all entities in modelspace
	for entity in msp:
		if entity.dxftype() == 'LINE':
			start_point = entity.dxf.start
			end_point = entity.dxf.end
			lines.append((start_point, end_point))
			line_points.add(start_point)
			line_points.add(end_point)
		
		elif entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']:
			polyline_points = entity.get_points('xyb')
			if entity.is_closed:
				lines.extend([(polyline_points[i], polyline_points[(i + 1) % len(polyline_points)]) for i in
							  range(len(polyline_points))])
			else:
				lines.extend([(polyline_points[i], polyline_points[i + 1]) for i in range(len(polyline_points) - 1)])
			for point in polyline_points:
				line_points.add(point)
	
	# Collect vertices not used in lines
	for entity in msp:
		if entity.dxftype() in ['POINT', 'CIRCLE', 'ARC', 'TEXT']:  # Add other entities as needed
			if entity.dxftype() == 'POINT':
				point = entity.dxf.location
			elif entity.dxftype() == 'CIRCLE' or entity.dxftype() == 'ARC':
				point = entity.dxf.center
			elif entity.dxftype() == 'TEXT':
				point = entity.dxf.insert
			
			if point not in line_points:
				vertices.add(point)
	
	return list(vertices), lines


if __name__ == "__main__":
	main()
