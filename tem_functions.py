# image processing functions used in TEM-pipeline.ipynb
# Akshay Trikha, Katie Partington
# 4th November, 2020

import cv2 as cv                             # OpenCV for image processing
import numpy as np                           # NumPy for quick maths
from collections import Counter, defaultdict # dictionary quick maths
import time                                  # measure function execution times
from numba import jit, njit, types, typeof   # optimization library
from numba.typed import Dict, List           # optimized data structures
from visuals import *                        # visuals.py contains visualization functions
from main import *                           # main.py contains global constants
from scipy.optimize import fsolve            # used for solving system of nonlin eqs. (particle intersections)
import warnings                              # (particle intersections)
warnings.filterwarnings("ignore", category=RuntimeWarning) # (particle intersections)


def get_threshold(threshold):
    """TODO: returns automatic thresholding of grayscale image"""
    return threshold


def setup(image_name, threshold, extra_display):
    """perform setup steps: grayscale, gaussian blur, binary threshold, noise removel, dilation, distance transform"""

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

    # Finding distance transform - lighter spots mean further away from contours
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2, 5) # mask size needs to be 0 or 3 or 5

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

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


@njit
def get_watershed_threshold_helper(particle_colors, watershed_markers, dist_transform_thresh, nm_per_pixel):
    # loop through pixels in watershed markers
    for row in range(1, len(watershed_markers) - 1):
        for col in range(1, len(watershed_markers[0]) - 1):
            # if pixel not in background
            if watershed_markers[row][col] != 1:
                # get current pixel and its neighbours 
                current = types.int64(watershed_markers[row][col])
                # add current pixel to dictionary
                if current not in particle_colors:
                    particle_colors[current] = types.int64(1)
                else:
                    particle_colors[current] += types.int64(1)

    # remove -1 key from particle_colors because it represents bad contours drawn by cv.watershed()
    if -1 in particle_colors:
        del particle_colors[-1]

    # loop to adjust areas from number of pixels to nm^2
    max_radius = 0.0
    for particle in particle_colors:
        current_area = particle_colors[particle] * nm_per_pixel**2  # don't need to square number of pixels because they already represent an area
        if (current_area/np.pi)**0.5 > max_radius:
            max_radius = (current_area/np.pi)**0.5
    return max_radius


# TODO: differentiate parameters and variable names
def get_watershed_threshold(dist_transform, sure_bg, color_image, expected_radius, nm_per_pixel):
    """outputs optimal threshold for thresholding distance transform to obtain separated particles (not agglomerates)"""
    max_radius = 3 * expected_radius # just needs to be something > 2 * expected_radius
    dist_transform_thresh = 0.25
    while max_radius > (2 * expected_radius):

        watershed_markers = get_watershed_markers(dist_transform, dist_transform_thresh, sure_bg, color_image, 0)
        
        # dictionary mapping colors to their pixels
        particle_colors = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )

        max_radius = get_watershed_threshold_helper(particle_colors, watershed_markers, dist_transform_thresh, nm_per_pixel)
                
        dist_transform_thresh += 0.05

    return dist_transform_thresh - 0.05 # TODO this seems bad


@njit
def get_contour_colors_helper(contour_colors, watershed_markers, chords_color_copy):
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
                            contour_colors[current] = List([List([col,row])])	# pixels are appended as (col, row) in order to feed to cv.lines() later on
                        else:
                            contour_colors[current].append(List([col,row]))
                        
                        # if the right pixel is not in background, add it to the dictionary
                        if (right != 1):
                            if right not in contour_colors:
                                contour_colors[right] = List([List([col+1,row])])
                            else:
                                contour_colors[right].append(List([col+1,row]))
                            chords_color_copy[row][col+1] = [255, 0, 0]
                            contour_size += 1
                            
                        chords_color_copy[row][col] = [255, 0, 0]	# TODO: why do we need chord_color_copy if we already have watershed_color_copy
                        contour_size += 1
                        
                    # if the down pixel is different
                    elif (down != current):
                        # add current pixel to dictionary
                        if current not in contour_colors:
                            contour_colors[current] = List([List([col,row])])
                        else:
                            contour_colors[current].append(List([col,row]))
                        
                        # if the down pixel is not in background, add it to the dictionary
                        if (down != 1):
                            if down not in contour_colors:
                                contour_colors[down] = List([List([col,row+1])])
                            else:
                                contour_colors[down].append(List([col,row+1]))
                            chords_color_copy[row+1][col] = [255, 0, 0]
                            contour_size += 1
                            
                        chords_color_copy[row][col] = [255, 0, 0]
                        contour_size += 1
                    
                    # if the up or left pixel is different than the current pixel and is not a border pixel
                    elif (((up != 255) and (up != current)) or ((left != 255) and (left != current))):
                        # add current pixel to dictionary
                        if current not in contour_colors:
                            contour_colors[current] = List([List([col,row])])
                        else:
                            contour_colors[current].append(List([col,row]))
                            
                        chords_color_copy[row][col] = [255, 0, 0]                    
                        contour_size += 1
    return contour_colors, chords_color_copy


# TODO: just loop through the output of watershed markers, where the border pixel are already marked with -1 and see what pixels they neighbor
def get_contour_colors(watershed_markers, color_image):
    """returns dictionary mapping colors to their pixels"""
    # copy input image
    chords_color_copy = color_image.copy() 

    # output
    contour_colors = Dict.empty(
        key_type=types.int64, # don't need int64 but compiler throws warnings otherwise
        value_type=types.ListType(types.ListType(types.int64))
    )

    contour_colors, chords_color_copy = get_contour_colors_helper(contour_colors, watershed_markers, chords_color_copy)

    # remove -1 key from contour_colors because it represents contours drawn by cv.watershed()
    if -1 in contour_colors:
        del contour_colors[-1]

    return contour_colors, chords_color_copy


def find_centerpoints(contour_colors, nm_per_pixel):
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


@njit
def get_areas_helper(watershed_markers, particle_areas, nm_per_pixel):
    # loop through pixels in watershed markers and count the number of pixels for each color
    for row in range(1, len(watershed_markers) - 1):
        for col in range(1, len(watershed_markers[0]) - 1):
            # if pixel not in background
            if watershed_markers[row][col] != 1:
                # get current pixel and its neighbours 
                current = watershed_markers[row][col]
                # add current pixel to dictionary
                if current not in particle_areas:
                    particle_areas[current] = 1.0
                else:
                    particle_areas[current] += 1.0
    
    # remove -1 key from particle_areas because it represents contours drawn by cv.watershed()
    if -1 in particle_areas:
        del particle_areas[-1]
        
    # loop to adjust areas from number of pixels to nm^2
    for particle in particle_areas:
        current_area = particle_areas[particle] * nm_per_pixel**2
        particle_areas[particle] = current_area
                    
    return particle_areas


def get_areas(watershed_markers, nm_per_pixel):
    """get the areas of the particles"""

    # dictionary mapping colors to their areas
    particle_areas = Dict.empty(
        key_type=types.int64, # don't need int64 but compiler throws warnings otherwise
        value_type=types.float64
    )

    particle_areas = get_areas_helper(watershed_markers, particle_areas, nm_per_pixel)

    return particle_areas


def get_replacements(agg_contour, potential_replacement_particles, nm_per_pixel):
    """takes the contour of an agglomerate and returns a list of replacement particle IDs"""

    # convert agg_contour weird Numba list datatype into ndarray to feed to cv functions
    agg_contour = np.asarray(agg_contour)

    replacements = []

    # loop through potential replacement particles for agglomerate
    for particle in potential_replacement_particles:
        
        # TODO replace potential_replacement_particles variable name, too long
        center = (int(potential_replacement_particles[particle][0][1] * 1/nm_per_pixel), int(potential_replacement_particles[particle][1][1] * 1/nm_per_pixel))
        
        # check if their centerpoints lie within the agglomerate's contour
        # draw_point_contour(center, agg_contour) # if you want to visualize the point
        if cv.pointPolygonTest(agg_contour, center, False) == 0.0:  # pointPolygonTest returns 0.0 if pixel in or on contour, -1.0 otherwise
            replacements += [particle]

    return replacements


def match_images(particles, contour_colors, agg_particles, agg_contour_colors, agg_areas, nm_per_pixel):
    """Replaces agglomerates with particles and outputs a single dictionary"""
    out_contour_colors = {}
    out_particles = {}
    max_id = np.max(list(agg_particles.keys()))

    # loop through agglomerate particles
    for agg_particle in agg_particles:

        # if particle has an area more than four times expected size
        if agg_areas[agg_particle] > np.pi * np.power((expected_radius*2), 2):
            # then particle is agglomerate

            agg_contour = agg_contour_colors[agg_particle]
            replacements = get_replacements(agg_contour, particles, nm_per_pixel)

            # add particles to output dictionaries
            for i in range(len(replacements)):
                out_particles[max_id+1+i] = particles[replacements[i]]
                out_contour_colors[max_id+1+i] = contour_colors[replacements[i]]

            # if no replacement particles are found, add the particle to the end of each dictionary
            if replacements == []:
                out_particles[max_id + 1] = agg_particles[agg_particle]
                out_contour_colors[max_id + 1] = agg_contour_colors[agg_particle]
                max_id += 1
            else:
                # update max id
                max_id += len(replacements)
        
        else:
            out_particles[agg_particle] = agg_particles[agg_particle]
            out_contour_colors[agg_particle] = agg_contour_colors[agg_particle]
        
    return out_particles, out_contour_colors


@njit
def pixel_distance(pixel1, pixel2):
    """finds the distance between two pixel tuples"""
    return np.power(np.power(pixel1[0] - pixel2[0], 2) + np.power(pixel1[1] - pixel2[1], 2), 0.5)


@njit
def get_long_chord_lengths_helper(color_pixels, long_pair_pixels):
    cur_max_len = 0
    for i in range(len(color_pixels)):
        for j in range(len(color_pixels[i:])):
            distance = pixel_distance(color_pixels[i], color_pixels[j])
            if distance > cur_max_len:
                cur_max_len = distance
                long_pair_pixels[0][0] = color_pixels[i][0]    # Numba doesn't support long_pair[0] = color_pixels[i] yet
                long_pair_pixels[0][1] = color_pixels[i][1]    # have to manually set list indicies
                long_pair_pixels[1][0] = color_pixels[j][0]
                long_pair_pixels[1][1] = color_pixels[j][1]

    return long_pair_pixels, cur_max_len


#TODO: adjust output to account for change in the order in which extracted information is added to the dictionary
def get_long_chord_lengths(particles, potential_replacement_particles, potential_contour_colors, contour_colors, nm_per_pixel):
    """finds the long chord lengths for the contours and returns them as pairs of pixel coordinates"""

    # store long pairs as [[color, (start pixel), (end pixel)]]
    long_pairs = []
    remaining_agglomerates = []

    # loop through all colors
    for color in contour_colors:
        # loop through all pixels in a color
        color_pixels = contour_colors[color]

        long_pair_pixels = np.zeros((2,2), dtype=int)
        long_pair_pixels, cur_max_len = get_long_chord_lengths_helper(color_pixels, long_pair_pixels)

        # if the particle has a diameter greater than twice the expected diameter, mark it as an agglomerate
        if cur_max_len * nm_per_pixel > 4 * expected_radius:
            remaining_agglomerates += [color]
        
        # if the particle is not an agglomerate, add its long pairs to the list and add its long length to the dictionary
        else:
            # keep track of long chord length pair for each color 
            long_pairs += [[color, (long_pair_pixels[0][0], long_pair_pixels[0][1]), (long_pair_pixels[1][0], long_pair_pixels[1][1])]]

            # add to particles dictionary, accounting for nm per pixel
            particles[color] += [("a", (cur_max_len / 2) * nm_per_pixel)]

    # record the maximum particle ID before breaking up remaining agglomerates
    max_id = np.max(list(particles.keys()))

    # loop through remaining agglomerates
    for color in remaining_agglomerates:

        # find the replacement particles for a given agglomerate
        agg_contour = contour_colors[color]
        replacements = get_replacements(agg_contour, potential_replacement_particles, nm_per_pixel)

        # find the current maximum particle ID
        cur_max_id = np.max(list(particles.keys()))

        # if no replacement particles are found, add the particle to the end of each dictionary
        if replacements == []:
            particles[cur_max_id + 1] = particles[color]
            contour_colors[cur_max_id + 1] = contour_colors[color]
        else:
            # add replacement particles to output dictionaries
            for i in range(len(replacements)):
                particles[cur_max_id + 1 + i] = potential_replacement_particles[replacements[i]]
                contour_colors[cur_max_id + 1 + i] = potential_contour_colors[replacements[i]]

    # loop through all colors
    for color in contour_colors:
        # if particle is a replacement for an agglomerate, find its longest length
        if color > max_id:
            # loop through all pixels in a color
            color_pixels = contour_colors[color]
            
            long_pair_pixels = np.zeros((2,2), dtype=int)
            long_pair_pixels, cur_max_len = get_long_chord_lengths_helper(color_pixels, long_pair_pixels)

            # keep track of long chord length pair for each color 
            long_pairs += [[color, (long_pair_pixels[0][0], long_pair_pixels[0][1]), (long_pair_pixels[1][0], long_pair_pixels[1][1])]]

            # add to particles dictionary, accounting for nm per pixel
            particles[color] += [("a", (cur_max_len / 2) * nm_per_pixel)]

    return long_pairs, particles, contour_colors


def get_short_chord_lengths(particles, contour_colors, long_pairs, nm_per_pixel):
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
        
        #deal with edge case where x1 and x2 are the same
        if x2-x1 == 0:
            denominator = 0.01
        else:
            denominator = x2-x1

        # calculate slope as rise over run
        slope = (y2-y1) / denominator
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
                theta = 360 - np.arctan((y1-y2)/denominator)*(180/np.pi)
            else:
                theta = 180 + np.arctan((y1-y2)/denominator)*(180/np.pi)
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
            if dx == 0:
                dx = 0.01
            
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
            
            # if pixel is right next to midpoint, add it to short_pixel regardless of score
            elif abs(mid_x - pixel[0]) < 2 and abs(mid_y - pixel[1]) < 2:
                short_pixel += [pixel]
                current_score += [score]
        
        # record pair of points of short chord length
        short_pixels += [short_pixel]
        # record each pair of points' scores
        scores += [current_score]
        
    # store min pixel pairs to visualize lines
    short_pairs = []
    
    # create list of indices to delete from long_chords
    to_be_deleted = []

    # process pairs to keep ones with min score
    for i in range(len(short_pixels)):
        if len(short_pixels[i]) >= 2:

            # first find pixel with overall minimum score
            min_index = np.argmin(scores[i])
            min_pixel = short_pixels[i][min_index]
            
            # calculate distances from pixel with minimum score
            distances_1 = []
            for pixel in short_pixels[i]:
                distances_1 += [pixel_distance(min_pixel, pixel)]
            
            # find the furthest pixel from the pixel with the minimum score
            max_dist_index = np.argmax(distances_1)
            max_dist_pixel = short_pixels[i][max_dist_index]
            
            # calculate distances from furthest pixel
            distances_2 = []
            for pixel in short_pixels[i]:
                distances_2 += [pixel_distance(max_dist_pixel, pixel)]
                
            pixels_1 = []
            pixels_2 = []
            scores_1 = []
            scores_2 = []
            # loop through the pixels and group the pixels into 2 groups
            for k in range(len(short_pixels[i])):
                
                # group pixels closest to the min pixel
                if distances_1[k] < distances_2[k]:
                    pixels_1 += [short_pixels[i][k]]
                    scores_1 += [scores[i][k]]
                    
                # group pixels closest to the furthest pixel
                else:
                    pixels_2 += [short_pixels[i][k]]
                    scores_2 += [scores[i][k]]
            
            # find the two pixels by finding the minimum score for each group
            min_index_1 = np.argmin(scores_1)
            min_index_2 = np.argmin(scores_2)
            min_pixel_1 = pixels_1[min_index_1]
            min_pixel_2 = pixels_2[min_index_2]
            
            # find b radius, accounting for nm per pixel
            b_radius = (pixel_distance(min_pixel_1, min_pixel_2)/2)*nm_per_pixel
            
            # if b_radius is bigger than 15% of the expected radius
            if b_radius > 0.15 * expected_radius:
                # add b radius to particles dictionary
                particles[long_pairs[i][0]] += [("b", b_radius)]
                
                # store pixels for cv.line() later
                short_pairs += [(min_pixel_1, min_pixel_2)]

            # delete corresponding long chord and contour
            else:
                # list the indices of particles to be deleted from largest to smallest so deletions do not affect subsequent indices
                to_be_deleted = [i] + to_be_deleted
                del contour_colors[long_pairs[i][0]]

    # delete long chords for particles that are too small
    for index in to_be_deleted:
        del long_pairs[index]

    return short_pairs, particles


# this function was taken from https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
def rotate_around_point(point, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    x, y = point
    ox, oy = origin

    qx = ox + np.cos(radians) * (x - ox) + np.sin(radians) * (y - oy)
    qy = oy + -np.sin(radians) * (x - ox) + np.cos(radians) * (y - oy)

    return qx, qy


def break_up_agglomerates(particles, agg_contours, nm_per_pixel, long_pairs, short_pairs):
    """if match_images is unsuccessful at breaking up agglomerates, this can be used to replace agglomerates with spheres"""
    new_particles = {}
    max_area = np.square(expected_radius*2)*np.pi
    max_id = np.max(list(particles.keys()))
    for particle in particles:
        particle_data = particles[particle]
        if len(particle_data) == 5:
            a = particle_data[2][1]
            b = particle_data[4][1]
            x = particle_data[0][1]
            y = particle_data[1][1]
            angle = -1*particle_data[3][1]
            if a*b*np.pi > max_area:
                num_particles_length = int(np.floor(2*a/(2*expected_radius+1)))
                num_particles_width = int(np.floor(2*b/(2*expected_radius+1)))
                agg_contour = np.asarray(agg_contours[particle])
                if num_particles_length != 0:
                    if num_particles_width == 0:
                        num_particles_width = 1
                    length_increment = 2*a/num_particles_length
                    width_increment = 2*b/num_particles_width
                    cur_x = x + a
                    cur_y = y + b
                    for i in range(num_particles_length):
                        for k in range(num_particles_width):
                            rot_x, rot_y = rotate_around_point([cur_x - length_increment/2, cur_y - width_increment/2], np.deg2rad(angle), [x, y])
                            center = (int(rot_x * 1/nm_per_pixel), int(rot_y * 1/nm_per_pixel))
                            if cv.pointPolygonTest(agg_contour, center, False) == 0.0:
                                new_particles[max_id+1] = [('x', rot_x), ('y', rot_y), ('a', expected_radius), ('theta', 0), ('b', expected_radius)]
                                max_id = max_id + 1
                                long_pairs += [[max_id+1, (int((rot_x+expected_radius)/nm_per_pixel), int(rot_y/nm_per_pixel)), (int((rot_x-expected_radius)/nm_per_pixel), int(rot_y/nm_per_pixel))]]
                                short_pairs += [([int(rot_x/nm_per_pixel), int((rot_y+expected_radius)/nm_per_pixel)], [int(rot_x/nm_per_pixel), int((rot_y-expected_radius)/nm_per_pixel)])]
                            cur_y = cur_y - width_increment
                        cur_y = y + b
                        cur_x = cur_x - length_increment
            else:
                new_particles[particle] = particles[particle]
                    
    return new_particles


def get_c(particles):
    """sets the c radius for each of the particles to be the average of the a and b radii for that particle or expected_radius"""
    # loop through all the particles
    for particle in particles:
        # get particle data
        particle_data = particles[particle]
        # if the particle has x and y positions, a and b radii, and an angle
        if len(particle_data) == 5:
            if particle_data[2][1] < expected_radius:
                # set c radius to be the expected radius
                c_radius = expected_radius
            else:
                # set c radius to be average of a and b radii
                c_radius = (particle_data[2][1] + particle_data[4][1]) / 2
            
            # add c radius to the particles dictionary
            particles[particle] += [("c", c_radius)]

    return particles


def set_max_c(particles):
    """sets the c radius for each of the particles to be the maximum average of the a and b radii for all of the particles"""
    # keep track of maximum c radius
    max_c = 0
    
    # loop through particles
    for particle in particles:
        # get particle data
        particle_data = particles[particle]
        # if the particle has an x and y position, a and b radii, and an angle
        if len(particle_data) == 5:
            # set c radius to be average of a and b radii
            c_radius = (particle_data[2][1] + particle_data[4][1]) / 2
            # if c radius is the biggest yet
            if c_radius > max_c:
                # update the maximum c radius
                max_c = c_radius
    
    # loop through particles
    for particle in particles:
        particle_data = particles[particle]
        # if the particle has an x and y position, a and b radii, and an angle, set each particle to have the maximum c radius
        if len(particle_data) == 5:
            # add c radius to the particles dictionary
            particles[particle] += [("c", max_c)]
    return particles


# TODO: find root cause of duplicate particles and fix there, this is a temporary fix
def delete_duplicates(particles):
    """deletes duplicate particles based on centerpoints"""
    # dictionary that maps (x,y) centerpoints to particle ids
    centerpoints = {}
    for id in list(particles):
        centerpoint = (particles[id][0][1], particles[id][1][1])
        if centerpoint not in centerpoints:
            centerpoints[(particles[id][0][1], particles[id][1][1])] = id
        else:
            del particles[id]
    return particles
    

def double_solve(f1, f2, x0, y0):
    """solves the system of equations"""
    func = lambda x: [f1(x[0], x[1]), f2(x[0], x[1])]
    return fsolve(func, [x0, y0])


def check_intersection(a1, b1, cx1, cy1, theta1, a2, b2, cx2, cy2, theta2):
    """checks for intersections between two particles"""
    phi1 = theta1 * np.pi / 180
    phi2 = theta2 * np.pi / 180

    test1 = lambda x, y: x ** 2 + 1 - y
    test2 = lambda x, y: y - 3
    res_test = double_solve(test1, test2, 1, 0)


    eq1 = lambda x, y: ((x - cx1) * np.cos(phi1) + (y - cy1) * np.sin(phi1)) ** 2 / a1 ** 2 + ((x - cx1) * np.sin(phi1) - (y - cy1) * np.cos(phi1)) ** 2 / b1 ** 2 - 1
    eq2 = lambda x, y: ((x - cx2) * np.cos(phi2) + (y - cy2) * np.sin(phi2)) ** 2 / a2 ** 2 + ((x - cx2) * np.sin(phi2) - (y - cy2) * np.cos(phi2)) ** 2 / b2 ** 2 - 1
    startx = min(cx1, cx2) + abs(cx1 - cx2) / 2
    starty = min(cy1, cy2) + abs(cy1 - cy2) / 2

    res = double_solve(eq1, eq2, startx, starty)

    return np.all((abs(eq1(res[0], res[1])) < 0.000001, abs(eq2(res[0], res[1])) < 0.000001), axis=0)


def layer_check_intersections(particles):
    """returns the particles that intersect for a given layer"""

    intersecting_particles = []

    for particle1 in particles:
        for particle2 in particles:
            if particle2 > particle1:
                # get particle data
                particle_data1 = particles[particle1]
                particle_data2 = particles[particle2]
                # if the particle has an x and y position, a, b, and c radii, and an angle
                if len(particle_data1) == 6 and len(particle_data2) == 6:
                    # extract particle info
                    x1 = particle_data1[0][1]
                    y1 = particle_data1[1][1]
                    a1 = particle_data1[2][1]
                    b1 = particle_data1[4][1]
                    theta1 = particle_data1[3][1]

                    x2 = particle_data2[0][1]
                    y2 = particle_data2[1][1]
                    a2 = particle_data2[2][1]
                    b2 = particle_data2[4][1]
                    theta2 = particle_data2[3][1]

                    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    max_diff = a1 + a2
                    if dist < max_diff:
                        solve = check_intersection(a1, b1, x1, y1, theta1, a2, b2, x2, y2, theta2)
                        if solve:
                            intersecting_particles += [[particle1, particle2]]
                        else:
                            continue

    return intersecting_particles


def layer_scale_particles(particles):
    """scales particles to get rid of intersections"""

    intersections_found = layer_check_intersections(particles)

    # loop through pairs of intersecting particles
    for i in range(len(intersections_found)):
        particle1 = intersections_found[i][0]
        particle_data1 = particles[particle1]
        particle2 = intersections_found[i][1]
        particle_data2 = particles[particle2]

        # extract particle info
        x1 = particle_data1[0][1]
        y1 = particle_data1[1][1]
        a1 = particle_data1[2][1]
        b1 = particle_data1[4][1]
        theta1 = particle_data1[3][1]

        x2 = particle_data2[0][1]
        y2 = particle_data2[1][1]
        a2 = particle_data2[2][1]
        b2 = particle_data2[4][1]
        theta2 = particle_data2[3][1]

        # keep scaling down the a and b radii until the particles no longer intersect
        while check_intersection(a1, b1, x1, y1, theta1, a2, b2, x2, y2, theta2):
            a1 = a1 * 0.99
            a2 = a2 * 0.99
            b1 = b1 * 0.99
            b2 = b2 * 0.99

        # adjust the dictionary to reflect new a and b radii
        particles[particle1][2] = ("a", a1)
        particles[particle2][2] = ("a", a2)
        particles[particle1][4] = ("b", b1)
        particles[particle2][4] = ("b", b2)

    return particles

# TODO: account for the tuples inside of the dictionary and particles with multiple intersections or multiple potential intersections
def layer_xy_rotate_particles(particles):
    """rotates particles to minimize particle intersections"""

    intersections_found = layer_check_intersections(particles)

    # loop through pairs of intersecting particles
    for i in range(len(intersections_found)):
        particle1 = intersections_found[i][0]
        particle_data1 = particles[particle1]
        particle2 = intersections_found[i][1]
        particle_data2 = particles[particle2]

        # extract particle info
        x1 = particle_data1[0][1]
        y1 = particle_data1[1][1]
        a1 = particle_data1[2][1]
        b1 = particle_data1[4][1]
        theta1 = particle_data1[3][1]

        x2 = particle_data2[0][1]
        y2 = particle_data2[1][1]
        a2 = particle_data2[2][1]
        b2 = particle_data2[4][1]
        theta2 = particle_data2[3][1]

        # keep scaling down the a and b radii until the particles no longer intersect
        total_angle_rotated = 0
        # while check_intersection(a1, b1, x1, y1, theta1, a2, b2, x2, y2, theta2) and total_angle_rotated < 180:
            

        # adjust the dictionary to reflect new a and b radii
        particles[particle1][2] = ("theta", theta1)
        particles[particle2][2] = ("theta", theta2)

    return particles

    # rotations = 0
    # tot_intersections = []
    # while len(intersections_found) > 0 and rotations < 90:
    #     tot_intersections += [len(intersections_found)]

    #     for i in range(len(intersections_found)):
    #         particle1 = intersections_found[i][0]
    #         particle2 = intersections_found[i][1]
    #         particle_data1 = particles[particle1]
    #         particle_data2 = particles[particle2]

    #         # extract particle data
    #         a1 = particle_data1[2]
    #         b1 = particle_data1[4]
    #         a2 = particle_data2[2]
    #         b2 = particle_data2[4]

    #         area1 = a1 * b1 * np.pi
    #         area2 = a2 * b2 * np.pi
    #         if area1 < area2:
    #             to_rotate = particle1
    #         else:
    #             to_rotate = particle2

    #         particles[to_rotate][3] = particles[to_rotate][3] - 2  # decreases theta by 2 degrees


    #     if rotations == 89:
    #         if len(intersections_found) <= min(tot_intersections):
    #             print(len(intersections_found))
    #             rotations = 90
    #             continue
    #         else:
    #             rotations = rotations - 1

    #     print(len(intersections_found))
    #     rotations = rotations + 1
    #     intersections_found = check_intersections(particles)

    # return particles


def get_layer_info(particles):
    """gets the relevant info from a particle dictionary representing a single layer so that it can be combined with other layers"""
    # keeping track of maximum and minimum x and y positions
    max_x = 0
    min_x = 10000
    max_y = 0
    min_y = 10000
    # keeping track of maximum c radius and the volume of particles in the layer
    max_c = 0
    layer_volume = 0

    # loop through particles
    for particle in particles:
        # get particle data
        particle_data = particles[particle]
        # if the particle has an x and y position, a, b, and c radii, and an angle
        if len(particle_data) == 6:
            # extract particle info
            x = particle_data[0][1]
            y = particle_data[1][1]
            a = particle_data[2][1]
            b = particle_data[4][1]
            c = particle_data[5][1]
            # add the volume of the particle to the layer volume
            layer_volume += (4/3)*np.pi*a*b*c
            
            # update minimum and maximum x and y values
            if (x - a) < min_x:
                min_x = x-a
            if (x + a) > max_x:
                max_x = x+a
            if (y - a) < min_y:
                min_y = y-a
            if (y + a) > max_y:
                max_y = y+a
            # update maximum c radius
            if c > max_c:
                max_c = c

    # set x and y lengths of the layer 
    x_length = max_x - min_x
    y_length = max_y - min_y
    
    # find volume fraction of the layer
    volume_fraction = layer_volume/(x_length*y_length*max_c*2)

    return [max_x, max_y, min_x, min_y, max_c, layer_volume, volume_fraction]


def combine_layers(particle_layers, layer_infos):
    """creating a particles dictionary and info list for a composite from a list of particle dictionaries and list of layer infos"""
    electrode_and_layer_spacing = 0.0001

    # keeping track of height, volume, and minimum and maximum x and y lengths of prism
    total_height = electrode_and_layer_spacing
    total_volume = 0

    max_x = 0
    max_y = 0
    
    # keeping track of the height of each layer
    layer_heights = []
    # TODO: do this better
    x_position_prism = 100000000000
    y_position_prism = 100000000000
    # looping through the layer(s)
    for info in layer_infos:
        total_height += ((info[4]*2) + electrode_and_layer_spacing)
        layer_heights += [info[4]*2]
        total_volume += info[5]
        if info[0] > max_x:
            max_x = info[0]
        if info[1] > max_y:
            max_y = info[1]
        # if x and/or y position are the lowest yet, update the prism x and/or y postion 
        if info[2] < x_position_prism:
            x_position_prism = info[2]
        if info[3] < y_position_prism:
            y_position_prism = info[3]

    x_length_prism = max_x - x_position_prism
    y_length_prism = max_y - y_position_prism

    # calculate prism volume fraction
    volume_fraction = total_volume/(x_length_prism*y_length_prism*total_height)

    composite_info = [total_volume, x_length_prism, y_length_prism, total_height, x_position_prism, y_position_prism, volume_fraction]

    composite_particles = {}
    particleID = 1
    layer_counter = 0
    height_adjustment = electrode_and_layer_spacing
    # loop through the layers
    for layer in particle_layers:
        # loop through the particles
        for particle in layer:
            # get particle data
            particle_data = layer[particle]
            # if the particle has an x and y position, a, b, and c radii, and an angle
            if len(particle_data) == 6:

                # calculating a randomized current height
                leftover_space = layer_heights[layer_counter] - (particle_data[5][1]*2)
                if leftover_space > 0.001:
                    rand_offset = np.random.uniform(0, leftover_space)
                    current_height = height_adjustment + particle_data[5][1] + rand_offset
                else:
                    current_height = height_adjustment + (layer_heights[layer_counter]/2)

                composite_particles[particleID] = [particle_data[0], particle_data[1], particle_data[2], particle_data[3], particle_data[4], particle_data[5]]
                composite_particles[particleID] += [("z", current_height)]
                
                particleID += 1

        # add the height of the layer we just looped through and a space between layers to the height adjustment
        height_adjustment += (layer_heights[layer_counter] + electrode_and_layer_spacing)
        # increment layer counter
        layer_counter += 1

    return composite_particles, composite_info


def get_electrode_particle_metric(composite_particles, composite_info):
    """find the number of electrode particle interactions and the average distance between the particle and electrode for those interactions"""
    num_interactions = 0
    total_dist = 0
    for particle in composite_particles:
        particle_data = composite_particles[particle]
        z = particle_data[6][1]
        c = particle_data[5][1]
        if (z-c) < 10: 
            num_interactions += 1
            total_dist += z-c
        if (z+c) > (composite_info[3]-10):
            num_interactions += 1
            total_dist += composite_info[3]-(z+c)
    if num_interactions != 0:
        average_dist = total_dist/num_interactions
    else:
        average_dist = 0
    composite_info += [num_interactions, average_dist]


def decrease_vol_loading(composite_particles, composite_info, desired_vol_loading):
    """takes a particle dictionary and randomly deletes particles until it hits the target volume loading"""
    vol_loading = composite_info[6]
    total_particles_vol = composite_info[0]
    vol_prism = composite_info[1] * composite_info[2] * composite_info[3]
    particles_deleted = []
    max_ID = np.max(list(composite_particles.keys()))
    while vol_loading > desired_vol_loading and (vol_loading-desired_vol_loading) >= 0.005:
        random_id = rand_int = np.random.randint(1, max_ID)
        while random_id in particles_deleted:
            random_id = rand_int = np.random.randint(1, max_ID)
        particle_data = composite_particles[random_id]
        a = particle_data[2][1]
        b = particle_data[4][1]
        c = particle_data[5][1]
        total_particles_vol -= (4/3)*np.pi*a*b*c
        vol_loading = total_particles_vol/vol_prism
        particles_deleted += [random_id]
        
    new_particles = {}
    for particle in composite_particles:
        if particle not in particles_deleted:
            particle_data = composite_particles[particle]
            new_particles[particle] = [particle_data[0], particle_data[1], particle_data[2], particle_data[3], particle_data[4], particle_data[5], particle_data[6]]
    
    new_info = [total_particles_vol, composite_info[1], composite_info[2], composite_info[3], composite_info[4], composite_info[5], vol_loading]
    
    return new_particles, new_info


def generate_text_file(composite_particles, composite_info, filename):
    """creating a text file from dictionary of particles"""
    # open the file to write in
    with open(filename, "w") as output_file:
        particleID = 1
        # loop through the particles
        for particle in composite_particles:
            # get particle data
            particle_data = composite_particles[particle]
            # if the particle has x, y, and z positions, a, b, and c radii, and an angle
            if len(particle_data) == 7:

                # write all the data for the particle to the text file
                output_file.writelines(particle_data[2][0] + str(particleID) + " " + str(particle_data[2][1]) + "[nm]" + "\n")       # a
                output_file.writelines(particle_data[4][0] + str(particleID) + " " + str(particle_data[4][1]) + "[nm]" + "\n")       # b
                output_file.writelines(particle_data[5][0] + str(particleID) + " " + str(particle_data[5][1]) + "[nm]" + "\n")       # c
                output_file.writelines(particle_data[0][0] + str(particleID) + " " + str(particle_data[0][1]) + "[nm]" + "\n")       # x
                output_file.writelines(particle_data[1][0] + str(particleID) + " " + str(particle_data[1][1]) + "[nm]" + "\n")       # y
                output_file.writelines(particle_data[6][0] + str(particleID) + " " + str(particle_data[6][1]) + "[nm]" + "\n")       # z
                output_file.writelines(particle_data[3][0] + str(particleID) + " " + str(particle_data[3][1]) + "[degrees]" + "\n")  # theta
                # increment particleID
                particleID += 1

        # write the information for the composite to the end of the file
        output_file.writelines("*****************\n")
        output_file.writelines("total_particles " + str(particleID-1) + "\n")                                   # number of particles
        output_file.writelines("total_volume_ellipsoids " + str(composite_info[0]) + "[nm^3]" + "\n")           # total volume
        output_file.writelines("x_length_prism " + str(composite_info[1]) + "[nm]" + "\n")                      # x length prism
        output_file.writelines("y_length_prism " + str(composite_info[2]) + "[nm]" + "\n")                      # y length prism
        output_file.writelines("z_length_prism " + str(composite_info[3]) + "[nm]" + "\n")                      # z length prism
        output_file.writelines("x_position_prism " + str(composite_info[4]) + "[nm]" + "\n")                    # x position prism
        output_file.writelines("y_position_prism " + str(composite_info[5]) + "[nm]" + "\n")                    # x position prism
        output_file.writelines("volume_fraction " + str(composite_info[6]) + "\n")                              # volume fraction
        output_file.writelines("number_particles_within_10nm_of_electrode " + str(composite_info[7])+ "\n")     # number of particle electrode interactions
        output_file.writelines("average_particle_electrode_distance " + str(composite_info[8])+ "[nm]"+ "\n")   # particle electrode average distance
        # output_file.writelines("volume_loading_dielectric_LDPE " + str(composite_info[9])+ "\n")                # dielectric based off volume loading in LDPE
        # output_file.writelines("volume_loading_dielectric_epoxy " + str(composite_info[10]))                    # dielectric based off volume loading in epoxy

        # close output file
        output_file.close()