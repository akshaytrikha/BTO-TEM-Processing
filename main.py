# run image processing pipeline as script from command line
# Akshay Trikha
# 8th February, 2021

from tem_functions import *     # tem_functions.py contains image processing functions
from visuals import *           # visuals.py contains visualization functions               

### constants
# nm_per_pixel = 100 / 95 	
# nm_per_pixel = 1000 / 131 # In 500nm_epoxy_2.jpg there are 131 pixels per 1 micrometer 
nm_per_pixel = 100 / 113 	# In TES-II-36a.tif there are 113 pixels per 100 nm
# nm_per_pixel = 500 / 291 	# In TES-II-36h.tif there are 291 pixels per 500 nm
# nm_per_pixel = 500 / 108 	# In 500nm_epoxy_15.jpg there are 108 pixels per 0.5 micrometer

expected_radius = 100 # in nm


def pipeline(image_names, thresholds, output_file, debug=False):
    """combines all functions to create image processing pipeline, prints each function's execution time if debug=True"""

    start_pipe = time.perf_counter()

    # debug mode runs pipeline for first given image
    if debug:
        # setup, finding optimal watershed threshold, populating particles dictionary
        start = time.perf_counter()
        color_image_1, dist_transform_1, sure_bg_1 = setup(image_names[0], thresholds[0], False)
        end = time.perf_counter()
        print("setup() ran in", str(end - start) + "s")
        start = time.perf_counter()
        dist_transform_thresh_1 = get_watershed_threshold(dist_transform_1, sure_bg_1, color_image_1, expected_radius)
        end = time.perf_counter()
        print("get_watershed_threshold() ran in", str(end - start) + "s")
        start = time.perf_counter()
        watershed_markers_1 = get_watershed_markers(dist_transform_1, dist_transform_thresh_1, sure_bg_1, color_image_1, False)
        agg_watershed_markers_1 = get_watershed_markers(dist_transform_1, 0.1, sure_bg_1, color_image_1, False)
        end = time.perf_counter()
        print("get_watershed_markers() for particles and agglomerates ran in", str(end - start) + "s")
        start = time.perf_counter()
        contour_colors_1, chords_color_copy_1 = get_contour_colors(watershed_markers_1, color_image_1)
        agg_contour_colors_1, agg_chords_color_copy_1 = get_contour_colors(agg_watershed_markers_1, color_image_1)
        end = time.perf_counter()
        print("get_contour_colors() for particles and agglomerates ran in", str(end - start) + "s")

        # find particle centerpoints
        start = time.perf_counter()
        particles_1 = find_centerpoints(contour_colors_1)
        agg_particles_1 = find_centerpoints(agg_contour_colors_1)
        end = time.perf_counter()
        print("find_centerpoints() for particles and agglomerates ran in", str(end - start) + "s")

        # calculate particle areas
        start = time.perf_counter()
        particle_areas_1 = get_areas(watershed_markers_1)
        agg_areas_1 = get_areas(agg_watershed_markers_1)
        end = time.perf_counter()
        print("get_areas() for particles and agglomerates ran in", str(end - start) + "s")

        # merge dictionaries of particles and agglomerates
        start = time.perf_counter()
        merge_particles_1, merge_contour_colors_1 = match_images(particles_1, contour_colors_1, agg_particles_1, agg_contour_colors_1, agg_areas_1)
        end = time.perf_counter()
        print("match_images() ran in", str(end - start) + "s")

        # long and short chord lengths
        start = time.perf_counter()
        long_pairs_1, merge_particles_1, merge_contour_colors_1 = get_long_chord_lengths(merge_particles_1, particles_1, contour_colors_1, merge_contour_colors_1)
        end = time.perf_counter()
        print("get_long_chord_lengths() ran in", str(end - start) + "s")
        start = time.perf_counter()
        short_pairs_1, merge_particles_1 = get_short_chord_lengths(merge_particles_1, merge_contour_colors_1, long_pairs_1)
        end = time.perf_counter()
        print("get_short_chord_lengths() ran in", str(end - start) + "s")

        # calculate c radii for merged particles
        start = time.perf_counter()
        merge_particles_1 = get_c(merge_particles_1)
        end = time.perf_counter()
        print("get_c() ran in", str(end - start) + "s")

        # get layer stats
        start = time.perf_counter()
        info_1 = get_layer_info(merge_particles_1)
        end = time.perf_counter()
        print("get_layer_info() ran in", str(end - start) + "s")

        # output multiple layer data into .txt
        start = time.perf_counter()
        combine_layers([merge_particles_1], [info_1], output_file)
        end = time.perf_counter()
        print("combine_layers() ran in", str(end - start) + "s")

    else:
        particle_layers = []
        layer_infos = []

        for i in range(len(image_names)):
            # setup, finding optimal watershed threshold, populating particles dictionary
            color_image, dist_transform, sure_bg = setup(image_names[i], thresholds[i], False)
            dist_transform_thresh = get_watershed_threshold(dist_transform, sure_bg, color_image, expected_radius)
            watershed_markers = get_watershed_markers(dist_transform, dist_transform_thresh, sure_bg, color_image, False)
            agg_watershed_markers = get_watershed_markers(dist_transform, 0.1, sure_bg, color_image, False)
            contour_colors, chords_color_copy = get_contour_colors(watershed_markers, color_image)
            agg_contour_colors, agg_chords_color_copy = get_contour_colors(agg_watershed_markers, color_image)

            # find particle centerpoints
            particles = find_centerpoints(contour_colors)
            agg_particles = find_centerpoints(agg_contour_colors)

            # calculate particle areas
            particle_areas = get_areas(watershed_markers)
            agg_areas = get_areas(agg_watershed_markers)

            # merge dictionaries of particles and agglomerates
            merge_particles, merge_contour_colors = match_images(particles, contour_colors, agg_particles, agg_contour_colors, agg_areas)

            # long and short chord lengths
            long_pairs, merge_particles, contour_colors = get_long_chord_lengths(merge_particles, particles, contour_colors, merge_contour_colors)
            short_pairs, merge_particles = get_short_chord_lengths(merge_particles, merge_contour_colors, long_pairs)

            # calculate c radii for merged particles
            merge_particles = get_c(merge_particles)

            # scale the particles to get rid of intersections
            merge_particles = layer_scale_particles(merge_particles)

            # get layer stats
            info = get_layer_info(merge_particles)

            # add layer data to lists
            particle_layers += [merge_particles]
            layer_infos += [info]

        # output multiple layer data into .txt
        combine_layers(particle_layers, layer_infos, output_file)

    end_pipe = time.perf_counter()

    print("\npipeline ran succesfully in", str(end_pipe - start_pipe) + "s")


# run pipeline as script for given image(s) and thresholds
def main():
    pipeline(["./inputs/TES-36a-cropped.tif", "./inputs/TES-36b-cropped.tif", "./inputs/TES-36b-cropped.tif"], [55, 35, 45], "./outputs/12_02_21_scaled_abe.txt")
    
if __name__ == "__main__":
    main()