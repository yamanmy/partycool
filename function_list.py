def image_reader(image_path):
	'''
	Read the image as the format that can be treated with subsequent functions, typically 2D arrays
	Function needed: image read, resize, data conversion into 2D array

	image_path: imput path for the image
	'''

def image_plotter(image_2d_array):
	'''
	Plot the image
	'''

def edge_detector(imput_image):
	'''
	Detect the edge from the image and plot them on the image.
	Output:1. edge defined image 2. list of the edges/area for the particles
	'''

def scale_detector(input_image):
	'''
	Detect the scale bar from the image and recognize the length of the bar in pixel unit and the real length label
	Output: pixel length of the bar, relating scale label of the bar
	'''

def size_calc(input_scale, input_edge):
	'''
	input_edge from edge_detector
	input_scale from scale_detector

	find longest end from the edge for each particles, match with input scale to calculate the length of the particles
	output: list of particle length/ if needed area for the particles
	'''

def cluster_checker(input_edge):
	'''
	input_edge from edge_detector

	based on the edge list to figure out monomer, dimer, polymer of the particles
	'''

def particle_sum():
	'''
	summarize cluster, edge, size result
	output:list of
	[particle_size, particle_cluster, etc.]
	'''

def partycool_gen():
	'''
	Generate a distribution summary from the particle_sum results
	'''

def 