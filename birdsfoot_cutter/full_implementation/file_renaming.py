from os import scandir, rename
import re

def main():
	deg6_sliced = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\6 Deg KC Sliced'
	#zero_indexify(deg6_sliced)
	cclet_sliced = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\birdsfoot_cutter\full_implementation\CCLET Sliced'
	#zero_indexify(cclet_sliced)


def zero_indexify(directory):
	pattern = re.compile(r'([^\d]+?)(\d+)(.*)')
	one_indexed = False
	for i, item in enumerate(scandir(directory)):
		match = pattern.search(item.name)
		# "abc123xyz456.txt"
		object = match.group(1)  # 'abc'
		instance = match.group(2)  # '123'
		details = match.group(3)  # 'xyz456.txt'
		new_instance = int(instance)
		if i == 0 and new_instance == 1:
			one_indexed = True
		new_name = directory+'\\'+object+str(new_instance-one_indexed)+details
		print("Original Path: ", item.path)
		print("Modified Path: ", new_name)
		rename(item.path, new_name)


if __name__ == '__main__':
	main()
