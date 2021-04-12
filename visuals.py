# visualizations used in TEM-pipeline.ipynb
# Gio Ferro, Katie Partington, Akshay Trikha
# 1st January, 2021

import cv2 as cv                        # OpenCV for image processing
import matplotlib.pyplot as plt         # Matplotlib for visualizing
from mpl_toolkits.mplot3d import Axes3D # Axes3D for 3D visualization
import numpy as np                      # NumPy for quick maths


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


def draw_point_contour(point, contour):
    """takes in a point and a contour and returns true if point is within contour and false otherwise"""
    # load new images
    color = cv.imread("./inputs/TES-36b-cropped.tif")
    gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    
    cv.drawContours(color, [contour], -1, (100, 255, 100), 2)
    color = cv.circle(color, point, 15, (255, 100, 100), 15)
    if cv.pointPolygonTest(contour, point, False) == 0.0:
        return True
    else:
        return False
    plt.imshow(color);


def draw_long_lengths(image, long_pairs):
    """draws long chord lengths on an image"""
    for long_pair in long_pairs:
        # cv.line(image, long_pair[1], long_pair[2], [255,255,0])
        cv.line(image, (long_pair[1][0], long_pair[1][1]), (long_pair[2][0], long_pair[2][1]), [255,255,0])
        

def draw_short_lengths(image, short_pairs):
    """draws short chord lengths on an image"""
    for short_pair in short_pairs:
        # cv.line(image, short_pair[0], short_pair[1], [0,255,255])
        cv.line(image, (short_pair[0][0], short_pair[0][1]), (short_pair[1][0], short_pair[1][1]), [0,255,255])

        
def split_image(gray_image, shape, display=False):
    """splits image into shape (# of vertical segments, # of horizontal segments) 
       and returns image chunk boundaries (clockwise starting from top left), and modified image"""
    # shape = (1,1) means no transformation is performed
    # shape = (2,2) means image is split into 4 quadrants
    # TODO: only works for equal-sized chunks
    
    image_copy = gray_image.copy()
    
    rows = len(image_copy)
    cols = len(image_copy[0])
    
    # check if image is divisible equally into shape 
    mod_rows = rows % shape[0]
    mod_cols = cols % shape[1]
    
    # if isn't crop rows and cols so that it is
    if mod_rows != 0:
        rows -= mod_rows
    if mod_cols != 0:
        cols -= mod_cols
    
    # calculate vertical and horizintal segment amounts
    vert_seg = int(rows/shape[0])
    hor_seg = int(cols/shape[1])
    
    # calculate and return sub-image boundaries
    boundaries = []
    for vert_i in range(0, shape[0]):
        for hor_i in range(0, shape[1]):
            boundaries += [[(hor_i * hor_seg, vert_i * vert_seg), 
                           ((hor_i + 1) * hor_seg, vert_i * vert_seg),
                           ((hor_i + 1) * hor_seg, (vert_i + 1) * vert_seg),
                           (hor_i * hor_seg, (vert_i + 1) * vert_seg)]]

    # draw vertical lines
    for vert_i in range(1, shape[0]+1):
        cv.line(image_copy, (vert_i * hor_seg, 0), (vert_i * hor_seg, rows), [100,0,100], 10)
    # draw horizontal lines
    for hor_i in range(1, shape[1]+1):
        cv.line(image_copy, (0, hor_i * vert_seg), (cols, hor_i * vert_seg), [100,0,100], 10)
        
    if display:
        display_images([image_copy], ["Partitioned Image"], [1])
        
    return boundaries, image_copy


def layer_render(particles, layer_info, rendering_type="wireframe"):
    """renders a dictionary of particles"""
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)

    ax = fig1.add_subplot(111, projection='3d')
    # ax = Axes3D(fig1)
    ax.set_xlabel("$X$", fontsize = 20, rotation = 150)
    ax.set_ylabel("$Y$", fontsize = 20, rotation = 150)
    ax.set_zlabel("$Z$", fontsize = 20, rotation = 150)

    ay = fig2.add_subplot(111, projection='3d')
    # ay = Axes3D(fig2)
    ay.set_xlabel("$X$", fontsize = 20, rotation = 150)
    ay.set_ylabel("$Y$", fontsize = 20, rotation = 150)
    ay.set_zlabel("$Z$", fontsize = 20, rotation = 150)

    az = fig3.add_subplot(111, projection='3d')
    # az = Axes3D(fig3)
    az.set_xlabel("$X$", fontsize = 20, rotation = 150)
    az.set_ylabel("$Y$", fontsize = 20, rotation = 150)
    az.set_zlabel("$Z$", fontsize = 20, rotation = 150)

    ax.view_init(elev=0, azim=0)
    ay.view_init(elev=0, azim=-90)
    az.view_init(elev=90, azim=-90)

    for particle in particles:
        # get particle data
        particle_data = particles[particle]
        # if the particle has an x and y position, a, b, and c radii, and an angle
        if len(particle_data) == 6:
            # extract particle info
            center_x = particle_data[0][1]
            center_y = particle_data[1][1]
            a_radius = particle_data[2][1]
            b_radius = particle_data[4][1]
            c_radius = particle_data[5][1]
            theta = particle_data[3][1]
            center_z = layer_info[4]+1
            
            # plot the ellipsoid
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)

            # converting theta from degrees to radians
            theta = theta/180*np.pi

            x1_1 = np.outer(a_radius * np.cos(u), np.sin(v)) + center_x
            y1_1 = np.outer(b_radius * np.sin(u), np.sin(v)) + center_y
            z = np.outer(c_radius * np.ones(np.size(u)), np.cos(v)) + center_z
            x = np.cos(theta) * (x1_1 - center_x) - np.sin(theta) * (y1_1 - center_y) + center_x
            y = np.sin(theta) * (x1_1 - center_x) + np.cos(theta) * (y1_1 - center_y) + center_y

            if rendering_type == "surface":
            #colored ellipsoid -- here the particles look like colored balloons: pretty but takes a couple extra seconds to run.
                ax.plot_surface(x, y, z, linewidth=0.0)
                ay.plot_surface(x, y, z, linewidth=0.0)
                az.plot_surface(x, y, z, linewidth=0.0)

            if rendering_type == "wireframe":
            #wireframe rendering -- this is a mesh sort of look. Runs more quickly.
                ax.plot_wireframe(x, y, z, rstride=25, cstride=25, color='b', alpha=0.3)
                ay.plot_wireframe(x, y, z, rstride=25, cstride=25, color='b', alpha=0.3)
                az.plot_wireframe(x, y, z, rstride=25, cstride=25, color='b', alpha=0.3)

    ax.set_zlim(layer_info[2], layer_info[0])
    ay.set_zlim(layer_info[3], layer_info[0])
    az.set_zlim(0, layer_info[0])
    plt.xlim([layer_info[2], layer_info[0]])
    plt.ylim([layer_info[3], layer_info[0]])

    plt.show()

def composite_render(composite_particles, composite_info, rendering_type="wireframe"):
    """renders a dictionary of particles"""
    fig4 = plt.figure(4)
    fig5 = plt.figure(5)
    fig6 = plt.figure(6)

    ax2 = fig4.add_subplot(111, projection='3d')
    # ax = Axes3D(fig1)
    ax2.set_xlabel("$X$", fontsize = 20, rotation = 150)
    ax2.set_ylabel("$Y$", fontsize = 20, rotation = 150)
    ax2.set_zlabel("$Z$", fontsize = 20, rotation = 150)

    ay2 = fig5.add_subplot(111, projection='3d')
    # ay = Axes3D(fig2)
    ay2.set_xlabel("$X$", fontsize = 20, rotation = 150)
    ay2.set_ylabel("$Y$", fontsize = 20, rotation = 150)
    ay2.set_zlabel("$Z$", fontsize = 20, rotation = 150)

    az2 = fig6.add_subplot(111, projection='3d')
    # az = Axes3D(fig3)
    az2.set_xlabel("$X$", fontsize = 20, rotation = 150)
    az2.set_ylabel("$Y$", fontsize = 20, rotation = 150)
    az2.set_zlabel("$Z$", fontsize = 20, rotation = 150)

    ax2.view_init(elev=0, azim=0)
    ay2.view_init(elev=0, azim=-90)
    az2.view_init(elev=90, azim=-90)

    for particle in composite_particles:
        # get particle data
        particle_data = composite_particles[particle]
        # if the particle has an x and y position, a, b, and c radii, and an angle
        if len(particle_data) == 7:
            # extract particle info
            center_x = particle_data[0][1]
            center_y = particle_data[1][1]
            center_z = particle_data[6][1]
            a_radius = particle_data[2][1]
            b_radius = particle_data[4][1]
            c_radius = particle_data[5][1]
            theta = particle_data[3][1]
            
            # plot the ellipsoid
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)

            # converting theta from degrees to radians
            theta = theta/180*np.pi

            x1_1 = np.outer(a_radius * np.cos(u), np.sin(v)) + center_x
            y1_1 = np.outer(b_radius * np.sin(u), np.sin(v)) + center_y
            z = np.outer(c_radius * np.ones(np.size(u)), np.cos(v)) + center_z
            x = np.cos(theta) * (x1_1 - center_x) - np.sin(theta) * (y1_1 - center_y) + center_x
            y = np.sin(theta) * (x1_1 - center_x) + np.cos(theta) * (y1_1 - center_y) + center_y

            if rendering_type == "surface":
            #colored ellipsoid -- here the particles look like colored balloons: pretty but takes a couple extra seconds to run.
                ax2.plot_surface(x, y, z, linewidth=0.0)
                ay2.plot_surface(x, y, z, linewidth=0.0)
                az2.plot_surface(x, y, z, linewidth=0.0)

            if rendering_type == "wireframe":
            #wireframe rendering -- this is a mesh sort of look. Runs more quickly.
                ax2.plot_wireframe(x, y, z, rstride=25, cstride=25, color='b', alpha=0.3)
                ay2.plot_wireframe(x, y, z, rstride=25, cstride=25, color='b', alpha=0.3)
                az2.plot_wireframe(x, y, z, rstride=25, cstride=25, color='b', alpha=0.3)

    ax2.set_zlim(composite_info[4], (composite_info[4]+composite_info[1]))
    ay2.set_zlim(composite_info[5], (composite_info[4]+composite_info[1]))
    az2.set_zlim(0, (composite_info[4]+composite_info[1]))
    plt.xlim([composite_info[4], composite_info[4]+composite_info[1]])
    plt.ylim([composite_info[5], composite_info[4]+composite_info[1]])

    plt.show()