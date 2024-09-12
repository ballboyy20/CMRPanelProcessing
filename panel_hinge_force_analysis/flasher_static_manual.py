import numpy as np
cos = np.cos
sin = np.sin
atan = np.arctan2
# A quarter rotation
q = np.pi / 2
# A rotation as a slice of a pentagon
p = 2 * np.pi / 5

def main():
	f_ext = 0.35
	r43, r45, r52, r32, r35, r21, r15, r10, r_ext =(
		[0.0]*9)
	#f43, f45, f52, f32, f35, f21, f15, f10, b
	A = np.array([
		[cos(r43+q), cos(r45+q), 0, 0, 0, 0, 0, 0],
		[sin(r43+q), sin(r45+q), 0, 0, 0, 0, 0, 0],
		[cos(r43-q), 0, 0, cos(r32+q), cos(r35+q), 0, 0, 0],
		[sin(r43-q), 0, 0, sin(r32+q), sin(r45+q), 0, 0, 0],
		[0, cos(r45-q), cos(r52+q), 0, cos(r35+q), 0, cos(r15-p-q), 0],
		[0, sin(r45-q), sin(r52+q), 0, sin(r45+q), 0, sin(r15-p-q), 0],
		[0, 0, cos(r52-q), cos(r32-q), 0, cos(r21+q), 0, 0],
		[0, 0, sin(r52-q), sin(r32-q), 0, sin(r21+q), 0, 0],
		[0, 0, 0, 0, 0, cos(r21-q), cos(r15+q), cos(r10+q)],
		[0, 0, 0, 0, 0, sin(r21-q), cos(r15+q), sin(r10+q)]
	])
	A = np.vstack((A[:2], A[4:]))
	b = np.array([f_ext*cos(r_ext), f_ext*sin(r_ext), 0, 0, 0, 0, 0, 0, 0, 0])
	
	print(A.shape)
	print(b.shape)
	
	lambda solve: (A, b), np.linalg.inv(A) @ b


if __name__ == '__main__':
	main()
