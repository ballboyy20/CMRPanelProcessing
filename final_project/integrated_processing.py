from math import isclose, degrees, radians, tan
from os import scandir
from warnings import warn

import regex as re
import pandas as pd
from matplotlib import pyplot as plt

import planar_comparison

"""
Code written by Trevor Kent Carter, starting June 18th, 2024

Code to integrate bulk file handling and naming with the planar_comparison code that
solves for the relative positions of the panels in the target degrees of freedom.

"""

def main():
	# The program will search this folder for all files of the type.stl that follow a certain formatting to be determined
	#mesh_folder = r"C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\Mesh Cuts (by person)\\"
	mesh_folder = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\flat_sheet\cut'
	filetype = r'.stl'
	left_key = r'left'
	right_key = r'right'
	seperator1 = r'\s-\s'
	#seperator2 = r'\s'
	seperator2 = r'_'
	
	panel_pairs = sort_files(mesh_folder, filetype, seperator1, seperator2, left_key, right_key, search_down=True)
	#print(valid_panel_pairs['3-arm SLET']['1']['right'])
	
	all_joints_table = compare_solve_joints(panel_pairs, left_key, right_key)
	
	csv_output_path = r'C:\\Users\\thetk\\Documents\\BYU\\Work\\pythonProject\\final_project\\position_results\\flat_scans\\'
	csv_output_name = r'panel_positions.csv'
	
	make_csv(all_joints_table, csv_output_path, csv_output_name, types=['all', 'joint'])
	
	
	

def make_csv(data_table, output_path, output_name, types=('all')):
	if isinstance(types, str):
		types = [types]
	for type in types:
		if type == 'all':
			filename = output_path+'all - '+output_name
			data_table.to_csv(path_or_buf=filename, index=False)
		if type == 'joint':
			joint_indexed_table = data_table.set_index('Joint Type')
			joint_groups = joint_indexed_table.groupby(by='Joint Type')
			for name, group in joint_groups:
				filename = output_path+name+' - '+output_name
				group.to_csv(path_or_buf=filename, index=False)

	
	
def compare_solve_joints(panels_dict, l_key, r_key):
	
	col_names = ['Joint Type', 'Scan Iteration', 'theta_x', 'theta_y', 'delta_z']
	row_data = []
	for joint_type in panels_dict:
		this_joint_dict = panels_dict[joint_type]
		for scan_instance in this_joint_dict:
			scan_dict = this_joint_dict[scan_instance]
			if l_key in scan_dict and r_key in scan_dict:
				fig = fig = plt.figure()
				ax1 = fig.add_subplot(121, projection='3d')
				ax2 = fig.add_subplot(111, projection='3d')
				theta_x, theta_y, delta_z = planar_comparison.compare_panels(scan_dict[l_key], scan_dict[r_key], verbose=True, trf_graph=ax2)#, raw_graph=ax1)
				#ax1.set_aspect('equal', adjustable='box')
				ax1.set_xlabel('X Axis')
				ax1.set_ylabel('Y Axis')
				ax1.set_zlabel('Z Axis')
				#ax2.set_aspect('equal', adjustable='box')
				ax2.set_xlabel('X Axis')
				ax2.set_ylabel('Y Axis')
				ax2.set_zlabel('Z Axis')
				plt.show()
				plt.clf()
				row_data.append([joint_type, scan_instance, theta_x, theta_y, delta_z])
			else:
				warn(f"For joint type '{joint_type}', within its scan iteration '{scan_instance}',"
					 f"either the left key ('{l_key}') or right key ('{r_key}') could not be located."
					 f"The elements present are as follows: {scan_dict}")

	panel_data = pd.DataFrame(columns=col_names, data=row_data)
	return panel_data


	
# This will produce a dictionary where joint types are the keys, corresponding each to a
# dictionary with keys for the various scan iterations, containing tuples with the side and
# the filepath for the cut scan of that joint scan iteration.
def sort_files(folder_path, filetype, sep1, sep2, l_key, r_key, search_down=False):
	# Iterate through all elements in directory
	files_list = []
	def search_folder(path, f_list, search):
		for item in scandir(path):
			if item.is_file():
				f_list.append(item)
			if search:
				if item.is_dir():
					search_folder(item.path, f_list, search)
					
	search_folder(folder_path, files_list, search_down)
	
	joints_dict = {}
	#for full_path in scandir(folder_path):
	for full_path in files_list:
		# We only care about files that end with our target file type
		if full_path.is_file() and re.match(rf'.*\{filetype}$', full_path.path):
			# Out of those files, we only care about ones that have:
			# 	- the proper number of separators, and
			# 	- end with one of l_key or r_key
			#full_path_a = full_path.path[0:-len(filetype)]
			filename = full_path.name
		
			# Regex pattern to match the specific format (generated with help from ChatGPT)
			scan_pattern = re.compile(rf'(.+){sep1}mesh(\d+){sep2}({l_key}|{r_key})')
			
			if match_groups := scan_pattern.match(filename):
				joint = match_groups[1]
				scan_num = match_groups[2]
				side = match_groups[3]
				print("Located qualifying file: ", filename)
				print('\tJoint Type: ', joint)
				print('\tMesh Num: ', scan_num)
				print('\tSide: ', side)
				if joint in joints_dict:
					this_joint_dict = joints_dict[joint]
					if scan_num in this_joint_dict:
						scan_dict = this_joint_dict[scan_num]
						if side in scan_dict:
							warn("A duplicate side instance of a particular joint type and scan_num iteration was located.")
							selection = input(f"Which duplicated instance would you like to keep? ('1' for previously existing side)"
											  f"(0) {full_path.path}"
											  f"(1) {scan_dict[side]}")
							if int(selection) == 0:
								print("Previously existing side overwritten by new located side.")
								scan_dict[side] = full_path.path
							else:
								print("Previously existing side kept.")
						else:
							scan_dict[side] = full_path.path
					else:
						this_joint_dict[scan_num] = {side: full_path.path}
				else:
					joints_dict[joint] = {scan_num: {side: full_path.path}}
					
			else:
				print(f"Located file in target directory {folder_path} that"
					  f"doesn't match the correct file naming format: {filename}"
					  f"\nCorrect format is '<joint_name> - mesh<scan_num> <left/right>.stl")
	return joints_dict
				


if __name__ == '__main__':
	main()
	