# visualizations used in TEM-pipeline.ipynb
# Gio Ferro
# 1st January, 2021

import matplotlib.pyplot as plt         # Matplotlib for visualizing
from mpl_toolkits.mplot3d import Axes3D # Axes3D for 3D visualization
from scipy.optimize import fsolve       # used for solving system of nonlin eqs. (particle intersections)
import warnings                         # (particle intersections)
warnings.filterwarnings("ignore", category=RuntimeWarning) # (particle intersections)


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

def draw_long_lengths(image, long_pairs):
    """draws long chord lengths on an image"""
    for long_pair in long_pairs:
        cv.line(image, long_pair[1], long_pair[2], [255,255,0])


def draw_short_lengths(image, short_pairs):
    """draws short chord lengths on an image"""
    for short_pair in short_pairs:
        cv.line(image, short_pair[0], short_pair[1], [0,255,255])


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


    eq1 = lambda x, y: ((x - cx1) * np.cos(phi1) + (y - cy1) * np.sin(phi1)) ** 2 / a1 ** 2 + (
            (x - cx1) * np.sin(phi1) - (y - cy1) * np.cos(phi1)) ** 2 / b1 ** 2 - 1
    eq2 = lambda x, y: ((x - cx2) * np.cos(phi2) + (y - cy2) * np.sin(phi2)) ** 2 / a2 ** 2 + (
            (x - cx2) * np.sin(phi2) - (y - cy2) * np.cos(phi2)) ** 2 / b2 ** 2 - 1
    startx = min(cx1, cx2) + abs(cx1 - cx2) / 2
    starty = min(cy1, cy2) + abs(cy1 - cy2) / 2

    res = double_solve(eq1, eq2, startx, starty)

    return np.all((abs(eq1(res[0], res[1])) < 0.000001, abs(eq2(res[0], res[1])) < 0.000001), axis=0)


def layer_check_intersections(particles):
    """returns the particles that intersect for a given layer"""

    intersecting_particles = []

    for particle1 in particles:
        particle_counter = particle1
        for particle2 in particles:
            if particle2 > particle_counter:
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
                    c1 = particle_data1[5][1]
                    theta1 = particle_data1[3][1]

                    x2 = particle_data2[0][1]
                    y2 = particle_data2[1][1]
                    a2 = particle_data2[2][1]
                    b2 = particle_data2[4][1]
                    c2 = particle_data2[5][1]
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

    ax.set_zlim(0, (layer_info[2]+layer_info[0]))
    ay.set_zlim(0, (layer_info[2]+layer_info[0]))
    az.set_zlim(0, (layer_info[2]+layer_info[0]))
    plt.xlim([layer_info[2], layer_info[2]+layer_info[0]])
    plt.ylim([layer_info[3], layer_info[3]+layer_info[1]])

    plt.show()