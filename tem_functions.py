# functions used in tem.ipynb
# Akshay Trikha, Katie Partington
# 4th November, 2020

import cv2 as cv                   # OpenCV for image processing
import matplotlib.pyplot as plt    # Matplotlib for visualizing
import numpy as np                 # NumPy for quick maths
from collections import Counter    # dictionary quick maths

### constants
# nm_per_pixel = 100 / 46
# nm_per_pixel = 100 / 95 	# In Challenge_1.jpg there are 92 pixels per 200nm = 46 pixels per 100 nm
nm_per_pixel = 100 / 113 	# In TES-II-36a.tif there are 113 pixels per 100 nm
expected_radius = 100


def display_images(images, titles, grayscales):
	"""takes in list of images, list of titles, list of boolean grayscales and displays them"""
	if len(images) == 1:
		fig, axs = plt.subplots(1, 1, figsize=(10,10))
		plt.imshow(images[0], cmap=plt.cm.gray)
		axs.set_title(titles[0])
	else:
		fig, axs = plt.subplots(1, len(images), figsize=(15,15))

		# loop through images and display
		for i in range(len(images)):
			# axs[i].imshow(images[i], cmap=plt.cm.gray)
			# axs[i].set_title(titles[i])
			if grayscales[i]:
				axs[i].imshow(images[i], cmap=plt.cm.gray)
				axs[i].set_title(titles[i])
			else:
				axs[i].imshow(images[i])
				axs[i].set_title(titles[i])


# TODO: delete since it's context based?
def save_images(images, titles):
	for i in range(len(images)):
		plt.savefig(str(images[i]) + "_" + titles[i] + ".png", dpi=500)


def get_threshold(threshold):
	"""TODO: returns automatic thresholding of grayscale image"""
	return threshold


def setup(image_name, threshold, extra_display):
	"""perform setup steps: grayscale, gaussian blur, binary threshold, noise removel, dilation"""

	# get color and grayscale images
	color_image = cv.imread(image_name)
	gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

	# apply gaussian blur transformation to grayscale image
	blur = cv.GaussianBlur(gray_image, (0,0) ,cv.BORDER_DEFAULT) 
		
	# apply binary threshold transformation to grayscale image
	ret, thresh = cv.threshold(blur, get_threshold(threshold), 255, cv.THRESH_BINARY_INV)

	# noise removal
	kernel = np.ones((10,10),np.uint8)
	opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

	# sure background area
	sure_bg = cv.dilate(opening, kernel, iterations=3)

	# Finding distance transform - lighter spots mean further away from contours
	dist_transform = cv.distanceTransform(opening,cv.DIST_L2, 5) # mask size needs to be 0 or 3 or 5

	# display all substeps if desired 
	if extra_display:
		display_images([color_image, gray_image], ["Color Image", "Grayscale Image"], [0,1])
		display_images([blur], ["Gaussian Blur"], [1])
		display_images([thresh], ["Binary Threshold of " + str(ret)], [1])
	display_images([dist_transform, sure_bg], ["Distance Transform", "Sure Background"], [0, 1])

	return color_image, dist_transform, sure_bg


def get_watershed_markers(dist_transform, dist_transform_thresh, sure_bg, color_image, display):
	"""obtains watershed markers by thresholding distance transform"""
	threshold, fg = cv.threshold(dist_transform, dist_transform_thresh*dist_transform.max(), 255, 0)

	# Finding unknown region
	fg = np.uint8(fg)
	unknown_2 = cv.subtract(sure_bg, fg)
    
    # Marker labelling
	threshold, pre_watershed_markers = cv.connectedComponents(fg)
	# Add one to all labels so that sure background is not 0, but 1
	pre_watershed_markers = pre_watershed_markers+1
	# Now, mark the region of unknown with zero
	pre_watershed_markers[unknown_2==255] = 0

	# copy input image
	watershed_color_copy = color_image.copy()

	watershed_markers = cv.watershed(watershed_color_copy, pre_watershed_markers)

	if display:
		display_images([pre_watershed_markers, watershed_markers], ["Pre-Watershed Markers", "Watershed Markers"], [0,0])

	return watershed_markers


def get_areas(watershed_markers):
	"""get the areas of the particles"""

	# dictionary mapping colors to their areas
	particle_areas = {}

	# loop through pixels in watershed markers and count the number of pixels for each color
	for row in range(1, len(watershed_markers) - 1):
		for col in range(1, len(watershed_markers[0]) - 1):
			# if pixel not in background
			if watershed_markers[row][col] != 1:
				# get current pixel and its neighbours 
				current = watershed_markers[row][col]
				# add current pixel to dictionary
				if current not in particle_areas:
					particle_areas[current] = 1
				else:
					particle_areas[current] += 1

	# remove -1 key from particle_areas because it represents contours drawn by cv.watershed()
	del particle_areas[-1]

	# loop to adjust areas from number of pixels to nm^2
	for particle in particle_areas:
		current_area = particle_areas[particle] * nm_per_pixel**2
		particle_areas[particle] = current_area

	return particle_areas


# TODO: differentiate paramters and variable names
def get_watershed_threshold(dist_transform, sure_bg, color_image, expected_radius):
	"""outputs optimal threshold for thresholding distance transform to obtain separated particles (not agglomerates)"""
	max_radius = 3*expected_radius
	dist_transform_thresh = 0.25
	while max_radius > (2*expected_radius):

	    watershed_markers = get_watershed_markers(dist_transform, dist_transform_thresh, sure_bg, color_image, 0)
	    
	    # dictionary mapping colors to their pixels
	    particle_colors = {}

	    # loop through pixels in watershed markers
	    for row in range(1, len(watershed_markers) - 1):
	        for col in range(1, len(watershed_markers[0]) - 1):
	            # if pixel not in background
	            if watershed_markers[row][col] != 1:
	                # get current pixel and its neighbours 
	                current = watershed_markers[row][col]
	                # add current pixel to dictionary
	                if current not in particle_colors:
	                    particle_colors[current] = 1
	                else:
	                    particle_colors[current] += 1
	                    
	    # remove -1 key from particle_colors because it represents bad contours drawn by cv.watershed()
	    del particle_colors[-1]
	    
	    # loop to adjust areas from number of pixels to nm^2
	    max_radius = 0
	    for particle in particle_colors:
	        current_area = particle_colors[particle] * nm_per_pixel**2
	        particle_colors[particle] = [current_area, (current_area/np.pi)**0.5]
	        if (current_area/np.pi)**0.5 > max_radius:
	            max_radius = (current_area/np.pi)**0.5
	            
	    # print(dist_transform_thresh, max_radius)
	            
	    dist_transform_thresh += 0.05

	return dist_transform_thresh - 0.05 # TODO this seems bad


def get_contour_colors(watershed_markers, color_image):
	"""returns dictionary mapping colors to their pixels"""
	# copy input image
	chords_color_copy = color_image.copy() 

	# output
	contour_colors = {}

	contour_size = 0
	# loop through pixels in watershed markers
	for row in range(1, len(watershed_markers) - 1):		# TODO: don't ignore border particles
	    for col in range(1, len(watershed_markers[0]) - 1):	# TODO: don't ignore border particles
	        # if pixel not in background
	        if watershed_markers[row][col] != 1:
	            # get current pixel and its neighbours 
	            current = watershed_markers[row][col]	
	            up = watershed_markers[row-1][col]
	            down = watershed_markers[row+1][col]
	            left = watershed_markers[row][col-1]
	            right = watershed_markers[row][col+1]
	            # if not surrounded by uniform pixels
	            if ((up != current) or (down != current) or (left != current) or (right != current)):
	                # if the right pixel is different
	                if (right != current):
	                    # add current pixel to dictionary
	                    if current not in contour_colors:
	                        contour_colors[current] = [(col, row)]	# pixels are appended as (col, row) in order to feed to cv.lines() later on
	                    else:
	                        contour_colors[current] += [(col,row)]
	                    
	                    # if the right pixel is not in background, add it to the dictionary
	                    if (right != 1):
	                        if right not in contour_colors:
	                            contour_colors[right] = [(col+1, row)]
	                        else:
	                            contour_colors[right] += [(col+1,row)]
	                        chords_color_copy[row][col+1] = [255, 0, 0]
	                        contour_size += 1
	                        
	                    chords_color_copy[row][col] = [255, 0, 0]	# TODO: why do we need chord_color_copy if we already have watershed_color_copy
	                    contour_size += 1
	                    
	                # if the down pixel is different
	                elif (down != current):
	                    # add current pixel to dictionary
	                    if current not in contour_colors:
	                        contour_colors[current] = [(col, row)]
	                    else:
	                        contour_colors[current] += [(col,row)]
	                    
	                    # if the down pixel is not in background, add it to the dictionary
	                    if (down != 1):
	                        if down not in contour_colors:
	                            contour_colors[down] = [(col, row+1)]
	                        else:
	                            contour_colors[down] += [(col,row+1)]
	                        chords_color_copy[row+1][col] = [255, 0, 0]
	                        contour_size += 1
	                        
	                    chords_color_copy[row][col] = [255, 0, 0]
	                    contour_size += 1
	                
	                # if the up or left pixel is different than the current pixel and is not a border pixel
	                elif (((up != 255) and (up != current)) or ((left != 255) and (left != current))):
	                    # add current pixel to dictionary
	                    if current not in contour_colors:
	                        contour_colors[current] = [(col, row)]
	                    else:
	                        contour_colors[current] += [(col,row)]
	                        
	                    chords_color_copy[row][col] = [255, 0, 0]                    
	                    contour_size += 1

	# remove -1 key from contour_colors because it represents contours drawn by cv.watershed()
	del contour_colors[-1]

	return contour_colors, chords_color_copy


def find_centerpoints(contour_colors):
	"""calculates the centerpoint for each color"""

	# maps each particle to its extracted information
	particles = {}

	# loop through the particles
	for color in contour_colors:
		color_pixels = contour_colors[color]
		sum_x = 0
		sum_y = 0

		# add x and y coordinates to running sum for particle
		for pixel in color_pixels:
			sum_x += pixel[0]
			sum_y += pixel[1]

        # add x and y centerpoints to particles dictionary, accounting for nm per pixel
		center_x = sum_x / len(color_pixels)
		center_y = sum_y / len(color_pixels)
		particles[color] = [("x", center_x*nm_per_pixel)]
		particles[color] += [("y", center_y*nm_per_pixel)]

	return particles


def match_images(particles, contour_colors, agg_particles, agg_contour_colors, agg_areas):
    """Replaces agglomerates with particles and outputs a single dictionary"""
    out_contour_colors = {}
    out_particles = {}
    max_color = np.max(list(agg_particles.keys()))
    
    # loop through agglomerate particles
    for agg_particle in agg_particles:
    
        # if particle has a radius more than twice expected size
        if agg_areas[agg_particle] > np.pi*(expected_radius*2)**2:
            # then particle is agglomerate
        
            # find max and min x and y for given agglomerate
            # TODO: make this 100000000x better using numpy
            max_x = 0
            max_y = 0
            min_x = 10000
            min_y = 10000
            for pixel in agg_contour_colors[agg_particle]:
                if pixel[0] > max_x:
                    max_x = pixel[0]
                if pixel[1] > max_y:
                    max_y = pixel[1]
                if pixel[0] < min_x:
                    min_x = pixel[0]
                if pixel[1] < min_y:
                    min_y = pixel[1]

            # collect list of particles in particles dictionary that fall within agglomerate
            replacements = []
            for particle in particles:
                x = particles[particle][0][1]
                y = particles[particle][1][1]
                if ((x < max_x) and (x > min_x) and (y < max_y) and (y > min_y)):
                    replacements += [particle]

            # add particles to output dictionaries
            for i in range(len(replacements)):
                out_particles[max_color+i] = particles[replacements[i]]
                out_contour_colors[max_color+i] = contour_colors[replacements[i]]

            # update max color
            max_color += len(replacements)
        
        else:
            out_particles[agg_particle] = agg_particles[agg_particle]
            out_contour_colors[agg_particle] = agg_contour_colors[agg_particle]
        
    return out_particles, out_contour_colors


# input pixels as tuples
def pixel_distance(pixel1, pixel2):
	"""finds the distance between two pixels"""
	return np.power(np.power(pixel1[0] - pixel2[0], 2) + np.power(pixel1[1] - pixel2[1], 2), 0.5)


#TODO: adjust output to account for change in the order in which extracted information is added to the dictionary
def get_long_chord_lengths(particles, contour_colors):
	"""finds the long chord lengths for the contours and returns them as pairs of pixel coordinates"""

	# keep track of maximum chord length found
	max_chord = 0

	# store long pairs as [[color, (start pixel), (end pixel)]]
	long_pairs = []
	# loop through all colors
	for color in contour_colors:
	    # loop through all pixels in a color
	    color_pixels = contour_colors[color]
	    current_max = 0
	    for i in range(len(color_pixels)):
	        for j in range(len(color_pixels[i:])):
	            distance = pixel_distance(color_pixels[i], color_pixels[j])
	            if distance > current_max:
	                current_max = distance
	                long_pair = [color, color_pixels[i], color_pixels[j]]
	    # keep track of long chord length pair for each color 
	    long_pairs += [long_pair]
	    # add to particles dictionary, accounting for nm per pixel
	    particles[color] += [("a", (current_max / 2) * nm_per_pixel)]
	    
	    if current_max > max_chord:
	        max_chord = current_max

	return long_pairs, particles


def get_short_chord_lengths(particles, contour_colors, long_pairs):
	"""finds the short chord lengths for the contours given the long pairs and returns them as pairs of pixel coordinates"""

	# store short chord length pair of points once found
	short_pixels = []
	# store scores (observed perpendicular slope vs. actual perpendicular slope)
	scores = []
	# keep track of each particle's centerpoints
	centerpoints = []
	# keep track of max x,y coordinates
	max_x = 0
	max_y = 0

	# loop through long pairs to get midpoint for each pair and then perpendicular short pairs
	for pair in long_pairs:
	    # get all pixels of a color
	    current_pixels = contour_colors[pair[0]]
	    # get start & end x,y coordinates
	    x1 = pair[1][0]
	    y1 = pair[1][1]
	    x2 = pair[2][0]
	    y2 = pair[2][1]
	    # calculate slope as rise over run
	    slope = (y2-y1) / (x2-x1)
	    # calculate orthogonal slope
	    orthogonal_slope = -1 / slope
	    # check which pixel start, which is end and calculate midpoint accordingly 
	    if x1 > x2:
	        mid_x = x2 + int((x1-x2) / 2)
	    else:
	        mid_x = x1 + int((x2-x1) / 2)
	    if y1 > y2:
	        mid_y = y2 + int((y1-y2) / 2)
	        # calculate rotation in X-Y plane with long length's slope
	        if x1 > x2:
	            theta = 360 - np.arctan((y1-y2)/(x1-x2))*(180/np.pi)
	        else:
	            theta = 180 + np.arctan((y1-y2)/(x2-x1))*(180/np.pi)
	    else:
	        mid_y = y1 + int((y2-y1) / 2)
	        # calculate rotation in X-Y plane with long length's slope
	        if slope > 0:
	            theta = -1*(180 - (np.arctan(slope)*(180/np.pi)))
	        else:
	            theta = np.arctan(slope)*(180/np.pi)
	    
	    # add X-Y rotation angle to particles dictionary
	    particles[pair[0]] += [("theta", theta)]
	    
	    short_pixel = []
	    current_score = []
	    # loop through all pixels for a color
	    for pixel in current_pixels:
	        # calculate change in x and y with respect to midpoint for pixel
	        dx = pixel[0] - mid_x
	        dy = pixel[1] - mid_y
	        # change in x cannot be 0
	        if dx != 0:
	            # compute slope between pixel and midpoint
	            comp_slope = dy/dx
	            # score ~= ratio of pixel slope / orthogonal slope from long chord length
	            score = np.abs(1 - comp_slope / orthogonal_slope)
	            # set threshold that pixel slope / orthogonal slope must be < 0.5
	            if (score < 0.5):
	                # if pixel isn't already in short_pixel
	                if pixel not in short_pixel:
	                    short_pixel += [pixel]
	                    current_score += [score]
	    
	    # record pair of points of short chord length
	    short_pixels += [short_pixel]
	    # record each pair of points' scores
	    scores += [current_score]

	# store min pixel pairs to visualize lines
	short_pairs = []

	# process pairs to keep ones with min score
	for i in range(len(short_pixels)):
	    if len(short_pixels[i]) >= 2:
	    	# first find pixel with overall minimum
	        min_index_1 = np.argmin(scores[i])
	        min_pixel_1 = short_pixels[i][min_index_1]
	        
	        # now loop through rest of pixels to find far away pixel with low score
	        min_distance = 100 # TODO: generalize for different particle sizes
	        for j in range(len(short_pixels[i])):
	            distance = pixel_distance(min_pixel_1, short_pixels[i][j])
	            if distance > min_distance:
	                min_pixel_2 = short_pixels[i][j]
	                
	                # store pixels for cv.line() later
	                short_pairs += [(min_pixel_1, min_pixel_2)]
	                
	                # long_pairs[i][0] is current color
	                # add short distance to particles dictionary, accounting for nm per pixel
	                particles[long_pairs[i][0]] += [("b", (distance/2)*nm_per_pixel)]
	                break

	return short_pairs, particles


def draw_long_lengths(image, long_pairs):
	"""draws long chord lengths on an image"""
	for long_pair in long_pairs:
		cv.line(image, long_pair[1], long_pair[2], [255,255,0])


def draw_short_lengths(image, short_pairs):
	"""draws short chord lengths on an image"""
	for short_pair in short_pairs:
		cv.line(image, short_pair[0], short_pair[1], [0,255,255])


# TODO: account for particles that cross the y-axis into the negative z
def get_prism_dimensions(particles):
	"""finds the dimensions of the prism"""
	